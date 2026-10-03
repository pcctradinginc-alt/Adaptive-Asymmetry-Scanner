"""
modules/analysis_cache.py – Analyse-Hash-Cache für die Deep Analysis (Kostenoptimierung, Prio 2).

Schlüssel = Hash über ALLE entscheidungsrelevanten Eingaben des Prompts:
  Ticker, News-Cluster (normalisiert, dedupliziert, sortiert), 48h-Bewegung (gebuckelt),
  Quick-MC-Hit-Rate (gebuckelt), EPS-Abweichung, Earnings-Datum, Makro-Regime/-Kontext, Sektor,
  Prescreen-Kategorie, Mega-Cap/Anomalie-Flags, Prompt-Version (Hash der Prompt-Vorlagen) und Modell.
Ändert sich irgendetwas davon (neue Nachricht, Kurssprung, Earnings, Regime, Prompt, Modell),
entsteht ein neuer Schlüssel = Invalidierung. Zusätzlich TTL (config/cost_policy.yaml).

Modus (analysis_cache.mode):
  observe  Treffer werden nur protokolliert (kind=cache, mode=observe) – Sonnet läuft trotzdem.
  active   Treffer liefern das gespeicherte Rohergebnis; kein Sonnet-Call.
Ein fehlerhafter Cache ist nie ein Fehler der Analyse: jede Ausnahme -> kein Treffer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from modules import cost_telemetry as ct

log = logging.getLogger(__name__)

CACHE_PATH = Path("outputs/costs/analysis_cache.json")
_WORD = re.compile(r"[a-z0-9]+")
_PENDING: dict[str, dict] = {}
COMPARE_FIELDS = ("direction", "impact", "surprise", "time_to_materialization")


def _cfg() -> dict:
    return (ct.policy().get("analysis_cache") or {})


def mode() -> str:
    m = str(_cfg().get("mode", "observe")).lower()
    return m if m in ("off", "observe", "active") else "observe"


# ── News-Dedup/Clustering (Prio 3) ──────────────────────────────────────────
def _tokens(h: str) -> set[str]:
    return set(_WORD.findall(h.lower()))


def cluster_headlines(headlines: list[str], threshold: float = 0.8) -> list[str]:
    """Fasst (nahezu) gleiche Schlagzeilen zusammen (Jaccard der Wortmengen >= threshold).
    Rückgabe: je Cluster der erste Vertreter in Originalreihenfolge."""
    reps: list[tuple[str, set[str]]] = []
    for h in headlines or []:
        if not isinstance(h, str) or not h.strip():
            continue
        t = _tokens(h)
        if any(t and r and len(t & r) / len(t | r) >= threshold for _, r in reps):
            continue
        reps.append((h, t))
    return [h for h, _ in reps]


def _bucket(x, step: float) -> Any:
    if not isinstance(x, (int, float)):
        return None
    return round(round(float(x) / step) * step, 6)


def make_key(components: dict) -> str:
    raw = json.dumps(components, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def deep_analysis_key(*, ticker: str, news: list[str], move_48h, mc_hit_rate, eps_deviation, earnings_date,
                      macro_text: str, sector: str, prescreen_category, is_mega_cap: bool, data_anomaly: bool,
                      prompt_version: str, model: str) -> str:
    step = float(_cfg().get("price_move_bucket_pct", 1.0)) / 100.0
    clusters = sorted(" ".join(sorted(_tokens(h))) for h in cluster_headlines(news[:8]))
    return make_key({
        "ticker": ticker, "news": clusters, "move_48h": _bucket(move_48h, step),
        "mc_hit_rate": _bucket(mc_hit_rate, 0.05), "eps_deviation": eps_deviation,
        "earnings_date": earnings_date, "macro": macro_text, "sector": sector,
        "prescreen_category": prescreen_category, "mega_cap": bool(is_mega_cap), "anomaly": bool(data_anomaly),
        "prompt_version": prompt_version, "model": model,
    })


def prompt_version(*templates: str) -> str:
    return hashlib.sha256("\x1f".join(templates).encode("utf-8")).hexdigest()[:12]


# ── Speicher ────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _prune(store: dict, now: datetime, ttl_h: float) -> dict:
    keep = {}
    for k, e in store.items():
        try:
            if now - datetime.fromisoformat(e["ts"]) <= timedelta(hours=ttl_h):
                keep[k] = e
        except Exception as ex:  # noqa: BLE001 – defekter Eintrag gilt als abgelaufen
            log.debug(f"analysis_cache: Eintrag {k} verworfen: {ex}")
            continue
    return keep


def lookup(key: str, *, workflow: str, ticker: str | None = None, now: datetime | None = None,
           path: Path | None = None) -> dict | None:
    """Gibt im active-Modus das gespeicherte Ergebnis zurück, sonst None. Protokolliert jeden Lookup."""
    try:
        m = mode()
        if m == "off":
            return None
        now = now or datetime.now(timezone.utc)
        ttl = float(_cfg().get("ttl_hours", 26))
        store = _prune(_load(Path(path or CACHE_PATH)), now, ttl)
        e = store.get(key)
        ct.record({"kind": "cache", "workflow": workflow, "ticker": ticker, "mode": m, "hit": e is not None,
                   "key": key, "saved_usd": (e or {}).get("cost_usd")})
        if e is not None and m == "active":
            return copy.deepcopy(e["result"])
        if e is not None:
            _PENDING[key] = copy.deepcopy(e["result"])   # observe: später gegen frisches Ergebnis prüfen
        return None
    except Exception as ex:  # noqa: BLE001
        log.debug(f"analysis_cache.lookup Fehler: {ex}")
        return None


def store(key: str, result: dict, *, cost_usd: float | None, model: str | None, now: datetime | None = None,
          path: Path | None = None) -> None:
    try:
        if mode() == "off" or not isinstance(result, dict):
            return
        p = Path(path or CACHE_PATH)
        now = now or datetime.now(timezone.utc)
        data = _prune(_load(p), now, float(_cfg().get("ttl_hours", 26)))
        data[key] = {"ts": now.isoformat(timespec="seconds"), "model": model, "cost_usd": cost_usd,
                     "result": result}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str))
        tmp.replace(p)
    except Exception as ex:  # noqa: BLE001
        log.debug(f"analysis_cache.store Fehler: {ex}")


def compare_observed(key: str, fresh: dict, *, workflow: str, ticker: str | None = None) -> dict | None:
    """A/B im Beobachtungsmodus: Cache-Ergebnis vs. frische Analyse bei identischen Eingaben.
    Protokolliert kind=cache_check; Grundlage für die Freigabe von mode=active."""
    try:
        old = _PENDING.pop(key, None)
        if old is None or not isinstance(fresh, dict):
            return None
        verdict = lambda r: ((r.get("red_team") or {}).get("red_team_verdict"))  # noqa: E731
        imp_o, imp_f = old.get("impact"), fresh.get("impact")
        row = {"kind": "cache_check", "workflow": workflow, "ticker": ticker, "key": key,
               "direction_equal": old.get("direction") == fresh.get("direction"),
               "verdict_equal": verdict(old) == verdict(fresh),
               "ttm_equal": old.get("time_to_materialization") == fresh.get("time_to_materialization"),
               "impact_abs_diff": (abs(float(imp_o) - float(imp_f))
                                   if isinstance(imp_o, (int, float)) and isinstance(imp_f, (int, float)) else None),
               "surprise_abs_diff": (abs(float(old["surprise"]) - float(fresh["surprise"]))
                                     if isinstance(old.get("surprise"), (int, float))
                                     and isinstance(fresh.get("surprise"), (int, float)) else None)}
        ct.record(row)
        return row
    except Exception as ex:  # noqa: BLE001
        log.debug(f"analysis_cache.compare_observed Fehler: {ex}")
        return None


def activation_report(rows: list[dict], min_checks: int = 30) -> dict:
    """Freigabekriterium für mode=active (Entscheidung bleibt beim Menschen):
    >= min_checks Vergleiche, Richtung und Red-Team-Verdikt je >= 95 % gleich,
    mittlere |Impact-Differenz| <= 1."""
    checks = [r for r in rows if r.get("kind") == "cache_check"]
    n = len(checks)

    def share(f):
        return round(sum(1 for r in checks if r.get(f)) / n, 3) if n else None

    diffs = [r["impact_abs_diff"] for r in checks if r.get("impact_abs_diff") is not None]
    mean_diff = round(sum(diffs) / len(diffs), 3) if diffs else None
    rep = {"checks": n, "direction_agreement": share("direction_equal"), "verdict_agreement": share("verdict_equal"),
           "ttm_agreement": share("ttm_equal"), "mean_impact_abs_diff": mean_diff}
    if n < min_checks:
        rep["decision"] = "PENDING"
    elif (rep["direction_agreement"] >= 0.95 and rep["verdict_agreement"] >= 0.95
          and mean_diff is not None and mean_diff <= 1.0):
        rep["decision"] = "KEEP"
    else:
        rep["decision"] = "REJECT"
    return rep
