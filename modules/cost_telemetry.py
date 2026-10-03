"""
modules/cost_telemetry.py – zentrale Kosten-Telemetrie für LLM- und kostenrelevante API-Calls.

Ledger (append-only, maschinenlesbar): outputs/costs/ledger-YYYY-MM.jsonl
  kind=llm    ein Eintrag je Anthropic-Call (auch fehlgeschlagene): Zeitpunkt, workflow, stage,
              ticker, scope (production|shadow|research), provider, model, input/output/cache-Tokens,
              cost_usd (gemessene usage x offizielle Preise; unbekannt -> null), runtime_s, success.
  kind=cache  Analyse-Hash-Cache-Ereignis (observe: möglicher Treffer, active: echter Treffer)
              mit geschätzter Ersparnis aus dem Originalcall.
  kind=api    Zähler je Fremd-API und Lauf (Requests, Fehler, 429) – Kosten nur aus Konfiguration.

Grundsatz: Telemetrie bricht nie die Funktion. Jeder Fehler beim Erfassen wird verschluckt;
fehlende Werte bleiben null ("nicht verfügbar") und werden nie als 0 gezählt.
Preise/Budgets: config/cost_policy.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

POLICY_PATH = Path("config/cost_policy.yaml")
LEDGER_DIR = Path("outputs/costs")
RUN_ID = os.getenv("GITHUB_RUN_ID") or uuid.uuid4().hex[:12]
_LOCK = threading.Lock()
_POLICY_CACHE: dict | None = None


# ── Konfiguration ───────────────────────────────────────────────────────────
def policy(path: Path | None = None) -> dict:
    global _POLICY_CACHE
    if path is None and _POLICY_CACHE is not None:
        return _POLICY_CACHE
    try:
        import yaml
        data = yaml.safe_load(Path(path or POLICY_PATH).read_text()) or {}
    except Exception as e:  # noqa: BLE001 – ohne Policy: keine Preise, keine Budgets
        log.debug(f"cost_policy nicht ladbar: {e}")
        data = {}
    if path is None:
        _POLICY_CACHE = data
    return data


def scope_of(workflow: str, pol: dict | None = None) -> str:
    return str(((pol or policy()).get("scopes") or {}).get(workflow, "research"))


def model_family(model: str | None, pol: dict | None = None) -> str:
    if not model:
        return "UNKNOWN"
    spec = ((pol or policy()).get("models") or {}).get(model)
    if spec and spec.get("family"):
        return spec["family"]
    m = model.lower()
    for fam in ("sonnet", "haiku", "opus"):
        if fam in m:
            return fam.upper()
    return "OTHER"


# ── Kostenrechnung ──────────────────────────────────────────────────────────
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def usage_from_response(resp: Any) -> dict:
    """usage-Felder der Antwort; fehlende Felder bleiben None (nicht 0)."""
    u = getattr(resp, "usage", None)
    out: dict[str, int | None] = {}
    for f in USAGE_FIELDS:
        v = getattr(u, f, None) if u is not None else None
        if v is None and isinstance(u, dict):
            v = u.get(f)
        out[f] = int(v) if isinstance(v, (int, float)) else None
    return out


def compute_cost(model: str | None, usage: dict, batch: bool = False, cache_ttl: str = "5m",
                 pol: dict | None = None) -> float | None:
    """USD aus gemessener usage. None, wenn Modellpreis oder input/output-Tokens fehlen."""
    pol = pol or policy()
    spec = (pol.get("models") or {}).get(model or "")
    if not spec or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    disc = float(pol.get("batch_discount", 0.5)) if batch else 1.0
    write_rate = spec.get("cache_write_1h" if cache_ttl == "1h" else "cache_write_5m", spec["input"])
    cost = (usage["input_tokens"] * spec["input"] * disc
            + usage["output_tokens"] * spec["output"] * disc
            + (usage.get("cache_creation_input_tokens") or 0) * write_rate
            + (usage.get("cache_read_input_tokens") or 0) * spec["cache_read"]) / 1e6
    return round(cost, 6)


def cache_savings(model: str | None, usage: dict, pol: dict | None = None) -> float | None:
    """Prompt-Caching-Netto-Ersparnis gegenüber ungecachtem Input (Leseersparnis - Schreibaufschlag)."""
    pol = pol or policy()
    spec = (pol.get("models") or {}).get(model or "")
    if not spec:
        return None
    read = usage.get("cache_read_input_tokens") or 0
    write = usage.get("cache_creation_input_tokens") or 0
    return round((read * (spec["input"] - spec["cache_read"])
                  - write * (spec["cache_write_5m"] - spec["input"])) / 1e6, 6)


# ── Ledger ──────────────────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.now(timezone.utc)


def ledger_path(ts: datetime, ledger_dir: Path | None = None) -> Path:
    return Path(ledger_dir or LEDGER_DIR) / f"ledger-{ts:%Y-%m}.jsonl"


def record(event: dict, ledger_dir: Path | None = None) -> None:
    """Hängt ein Ereignis an. Wirft nie."""
    try:
        ts = event.get("ts")
        when = datetime.fromisoformat(ts) if isinstance(ts, str) else _now()
        event = {"ts": when.isoformat(timespec="seconds"), "run_id": RUN_ID, **{k: v for k, v in event.items()
                                                                                  if k != "ts"}}
        p = ledger_path(when, ledger_dir)
        with _LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # noqa: BLE001 – Telemetrie bricht nie die Funktion
        log.debug(f"cost_telemetry.record fehlgeschlagen: {e}")


def tracked_create(client: Any, *, workflow: str, stage: str | None = None, ticker: str | None = None,
                   batch: bool = False, ledger_dir: Path | None = None, **kwargs) -> Any:
    """client.messages.create(**kwargs) mit Telemetrie. Ergebnis/Exception unverändert durchgereicht."""
    model = kwargs.get("model")
    t0 = time.monotonic()
    base = {"kind": "llm", "workflow": workflow, "stage": stage or workflow, "ticker": ticker,
            "scope": scope_of(workflow), "provider": "anthropic", "model": model,
            "family": model_family(model), "batch": batch}
    try:
        resp = client.messages.create(**kwargs)
    except Exception as e:
        record({**base, **{f: None for f in USAGE_FIELDS}, "cost_usd": None,
                "runtime_s": round(time.monotonic() - t0, 3), "success": False,
                "error": type(e).__name__}, ledger_dir)
        raise
    try:
        usage = usage_from_response(resp)
        served = getattr(resp, "model", None)
        record({**base, **usage, "served_model": served if isinstance(served, str) else None,
                "cost_usd": compute_cost(model, usage, batch=batch),
                "cache_savings_usd": cache_savings(model, usage),
                "stop_reason": getattr(resp, "stop_reason", None),
                "runtime_s": round(time.monotonic() - t0, 3), "success": True}, ledger_dir)
    except Exception as e:  # noqa: BLE001
        log.debug(f"cost_telemetry: usage nicht erfassbar: {e}")
    return resp


# ── Fremd-API-Zähler (Transport-Ebene, ohne Eingriff in die Aufrufer) ────────
_API_COUNTS: dict[str, dict[str, int]] = defaultdict(lambda: {"requests": 0, "errors": 0, "rate_limited": 0})
_HOST_MAP: dict[str, str] = {}
_INSTALLED = False


def _provider_for(url: str) -> str | None:
    try:
        from urllib.parse import urlparse
        return _HOST_MAP.get((urlparse(url).hostname or "").lower())
    except Exception:  # noqa: BLE001
        return None


def install_http_counter() -> bool:
    """Zählt requests-Aufrufe an konfigurierte kostenrelevante Hosts (keine Keys, keine URLs gespeichert)."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import requests
        for name, spec in (policy().get("paid_apis") or {}).items():
            for h in spec.get("hosts") or []:
                _HOST_MAP[str(h).lower()] = name
        orig = requests.Session.request

        def counted(self, method, url, *a, **kw):
            prov = _provider_for(str(url))
            try:
                resp = orig(self, method, url, *a, **kw)
            except Exception:
                if prov:
                    with _LOCK:
                        _API_COUNTS[prov]["requests"] += 1
                        _API_COUNTS[prov]["errors"] += 1
                raise
            if prov:
                with _LOCK:
                    c = _API_COUNTS[prov]
                    c["requests"] += 1
                    code = getattr(resp, "status_code", 200) or 200
                    if code == 429:
                        c["rate_limited"] += 1
                    elif code >= 400:
                        c["errors"] += 1
            return resp

        requests.Session.request = counted
        _INSTALLED = True
        return True
    except Exception as e:  # noqa: BLE001
        log.debug(f"install_http_counter fehlgeschlagen: {e}")
        return False


def flush_api_counts(workflow: str, ledger_dir: Path | None = None) -> list[dict]:
    """Schreibt je Provider eine kind=api-Zeile für diesen Lauf und setzt die Zähler zurück."""
    pol = policy()
    rows = []
    with _LOCK:
        snapshot = {k: dict(v) for k, v in _API_COUNTS.items()}
        _API_COUNTS.clear()
    for prov, c in sorted(snapshot.items()):
        spec = (pol.get("paid_apis") or {}).get(prov, {})
        lim = spec.get("free_daily_limit")
        row = {"kind": "api", "workflow": workflow, "scope": scope_of(workflow), "provider": prov, **c,
               "plan": spec.get("plan", "unknown"), "free_daily_limit": lim,
               "quota_usage": round(c["requests"] / lim, 3) if lim else None, "cost_usd": None}
        record(row, ledger_dir)
        rows.append(row)
    return rows


# ── Lesen + Aggregation ─────────────────────────────────────────────────────
def load_ledger(start: date | None = None, end: date | None = None, ledger_dir: Path | None = None) -> list[dict]:
    """Alle Ereignisse mit start <= Datum < end (UTC). Defekte Zeilen werden übersprungen."""
    d = Path(ledger_dir or LEDGER_DIR)
    rows: list[dict] = []
    if not d.exists():
        return rows
    for p in sorted(d.glob("ledger-*.jsonl")):
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception as e:  # noqa: BLE001
            log.warning(f"Kosten-Ledger {p} nicht lesbar: {e}")
            continue
        for line in lines:
            try:
                r = json.loads(line)
                day = datetime.fromisoformat(r["ts"]).date()
            except Exception as e:  # noqa: BLE001 – defekte Zeile: überspringen, nie raten
                log.debug(f"Kosten-Ledger: defekte Zeile in {p.name}: {e}")
                continue
            if (start and day < start) or (end and day >= end):
                continue
            r["_day"] = day
            rows.append(r)
    return rows


def _add(acc: dict, key: str, v) -> None:
    if v is not None:
        acc[key] = acc.get(key, 0) + v


def aggregate(rows: Iterable[dict]) -> dict:
    """Summen über LLM-Calls; Kosten nur aus Zeilen mit bekanntem cost_usd (unknown separat gezählt)."""
    tot: dict[str, Any] = {"calls": 0, "failed": 0, "cost_usd": 0.0, "unknown_cost_calls": 0,
                           "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0, "cache_savings_usd": 0.0}
    by: dict[str, dict] = {k: defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
                           for k in ("family", "workflow", "stage", "scope", "model", "day", "week")}
    analysis_cache = {"hits": 0, "would_hit": 0, "saved_usd": 0.0, "potential_saved_usd": 0.0, "lookups": 0}
    api: dict[str, dict] = defaultdict(lambda: {"requests": 0, "errors": 0, "rate_limited": 0})
    for r in rows:
        kind = r.get("kind")
        if kind == "cache":
            analysis_cache["lookups"] += 1
            if r.get("hit") and r.get("mode") == "active":
                analysis_cache["hits"] += 1
                analysis_cache["saved_usd"] += r.get("saved_usd") or 0.0
            elif r.get("hit"):
                analysis_cache["would_hit"] += 1
                analysis_cache["potential_saved_usd"] += r.get("saved_usd") or 0.0
            continue
        if kind == "api":
            a = api[r.get("provider", "?")]
            for k in ("requests", "errors", "rate_limited"):
                a[k] += int(r.get(k) or 0)
            continue
        if kind != "llm":
            continue
        tot["calls"] += 1
        if not r.get("success", True):
            tot["failed"] += 1
        for f in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                  "cache_savings_usd"):
            _add(tot, f, r.get(f))
        cost = r.get("cost_usd")
        if cost is None and r.get("success", True):
            tot["unknown_cost_calls"] += 1
        day = r.get("_day") or datetime.fromisoformat(r["ts"]).date()
        iso = day.isocalendar()
        keys = {"family": r.get("family") or model_family(r.get("model")), "workflow": r.get("workflow", "?"),
                "stage": r.get("stage", "?"), "scope": r.get("scope", "?"), "model": r.get("model") or "?",
                "day": day.isoformat(), "week": f"{iso[0]}-W{iso[1]:02d}"}
        for dim, k in keys.items():
            b = by[dim][k]
            b["calls"] += 1
            if cost is not None:
                b["cost_usd"] += cost
        if cost is not None:
            tot["cost_usd"] += cost
    tot["cost_usd"] = round(tot["cost_usd"], 4)
    tot["cache_savings_usd"] = round(tot["cache_savings_usd"], 4)
    for k in ("saved_usd", "potential_saved_usd"):
        analysis_cache[k] = round(analysis_cache[k], 4)
    return {"totals": tot, "by": {dim: {k: {"calls": v["calls"], "cost_usd": round(v["cost_usd"], 4)}
                                        for k, v in sorted(d.items())} for dim, d in by.items()},
            "analysis_cache": analysis_cache, "api": {k: dict(v) for k, v in sorted(api.items())}}


# ── Budgets / Guards ────────────────────────────────────────────────────────
def budget_status(now: datetime | None = None, ledger_dir: Path | None = None, pol: dict | None = None) -> dict:
    pol = pol or policy()
    b = pol.get("budgets") or {}
    now = now or _now()
    today = now.date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    rows = load_ledger(min(week_start, month_start), today + timedelta(days=1), ledger_dir)
    week = aggregate(r for r in rows if r["_day"] >= week_start)["totals"]["cost_usd"]
    month = aggregate(r for r in rows if r["_day"] >= month_start)["totals"]["cost_usd"]
    warn = float(b.get("warn_fraction", 0.8))
    out = {"week_usd": week, "month_usd": month, "weekly_budget_usd": b.get("weekly_llm_usd"),
           "monthly_budget_usd": b.get("monthly_llm_usd"), "level": "OK", "throttled_scopes": [], "warnings": []}
    fracs = []
    for label, spent, budget in (("Woche", week, b.get("weekly_llm_usd")), ("Monat", month, b.get("monthly_llm_usd"))):
        if not budget:
            continue
        f = spent / float(budget)
        fracs.append(f)
        if f >= 1.0:
            out["warnings"].append(f"LLM-Budget {label} überschritten: ${spent:.2f} von ${float(budget):.2f}")
        elif f >= warn:
            out["warnings"].append(f"LLM-Budget {label} bei {f:.0%}: ${spent:.2f} von ${float(budget):.2f}")
    top = max(fracs, default=0.0)
    if top >= 1.0:
        out["level"] = "EXCEEDED"
        # erst Research/Shadow (throttle_order), Produktion nie
        out["throttled_scopes"] = [s for s in b.get("throttle_order", ["research", "shadow"]) if s != "production"]
    elif top >= warn:
        out["level"] = "WARN"
        order = [s for s in b.get("throttle_order", ["research", "shadow"]) if s != "production"]
        out["throttled_scopes"] = order[:1]          # ab 80 %: zuerst Research reduzieren
    return out


def allow(workflow: str, now: datetime | None = None, ledger_dir: Path | None = None) -> bool:
    """False nur für drosselbare Bereiche (research/shadow) bei Budgetdruck. Produktion: immer True."""
    try:
        scope = scope_of(workflow)
        if scope == "production":
            return True
        st = budget_status(now, ledger_dir)
        if scope in st["throttled_scopes"]:
            log.warning(f"Kosten-Guard: {workflow} ({scope}) pausiert – {'; '.join(st['warnings'])}")
            return False
        return True
    except Exception as e:  # noqa: BLE001 – im Zweifel nicht blockieren, Produktion unberührt
        log.debug(f"cost_telemetry.allow Fehler: {e}")
        return True


def production_warning(now: datetime | None = None, ledger_dir: Path | None = None) -> str | None:
    """Warntext, wenn Budget überschritten – Produktion läuft trotzdem weiter (nie still überspringen)."""
    try:
        st = budget_status(now, ledger_dir)
        return "; ".join(st["warnings"]) if st["warnings"] else None
    except Exception:  # noqa: BLE001
        return None


# ── Monatsauswertung (für monthly_report.py) ────────────────────────────────
def _month_bounds(key: str) -> tuple[date, date]:
    y, m = map(int, key.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), m % 12 + 1, 1)
    return start, end


def _scan_stats(key: str, reports_dir: Path) -> dict:
    """Scan-Läufe und finale Trades aus den vorhandenen Tagesreports (keine Parallelstruktur)."""
    runs, trades = 0, 0
    trades_known = False
    for p in sorted(Path(reports_dir).glob(f"{key}-*.json")):
        try:
            s = (json.loads(p.read_text()) or {}).get("stats") or {}
        except Exception as e:  # noqa: BLE001
            log.debug(f"Tagesreport {p.name} nicht lesbar: {e}")
            continue
        runs += 1
        if isinstance(s.get("trades"), (int, float)):
            trades += int(s["trades"])
            trades_known = True
    return {"scan_runs": runs, "final_trades": trades if trades_known else None}


def _cache_validation(rows: list[dict]) -> dict | None:
    try:
        from modules import analysis_cache
        return analysis_cache.activation_report(rows)
    except Exception:  # noqa: BLE001
        return None


def _paid_api_cost(api: dict, pol: dict | None = None) -> float | None:
    """Monatskosten der genutzten Fremd-APIs aus der Konfiguration (Pläne sind Fixkosten).
    None, sobald ein genutzter Provider keinen bekannten Kostenwert hat."""
    specs = (pol or policy()).get("paid_apis") or {}
    total = 0.0
    for prov, c in api.items():
        if not c.get("requests"):
            continue
        est = (specs.get(prov) or {}).get("estimated_cost_usd_month")
        if est is None:
            return None
        total += float(est)
    return round(total, 2)


def month_summary(key: str, ledger_dir: Path | None = None,
                  reports_dir: Path = Path("outputs/daily_reports")) -> dict:
    start, end = _month_bounds(key)
    prev_key = f"{start.year - (start.month == 1)}-{(start.month - 2) % 12 + 1:02d}"
    p_start, _ = _month_bounds(prev_key)
    rows = load_ledger(p_start, end, ledger_dir)
    cur = aggregate(r for r in rows if r["_day"] >= start)
    prev = aggregate(r for r in rows if r["_day"] < start)
    scans = _scan_stats(key, reports_dir)
    t = cur["totals"]
    n_cur, n_prev = t["calls"], prev["totals"]["calls"]
    cost = t["cost_usd"] if n_cur else None
    prev_cost = prev["totals"]["cost_usd"] if n_prev else None
    fam = cur["by"]["family"]
    analyzed = cur["by"]["stage"].get("deep_analysis", {}).get("calls", 0)
    missing_days = 0
    if scans["scan_runs"]:
        llm_days = {k for k in cur["by"]["day"]}
        for p in Path(reports_dir).glob(f"{key}-*.json"):
            try:
                s = (json.loads(p.read_text()) or {}).get("stats") or {}
            except Exception as e:  # noqa: BLE001
                log.debug(f"Tagesreport {p.name} nicht lesbar: {e}")
                continue
            if (s.get("prescreened") or s.get("analyzed")) and p.stem not in llm_days:
                missing_days += 1
    complete = bool(n_cur) and t["unknown_cost_calls"] == 0 and missing_days == 0

    def per(x):
        return round(cost / x, 4) if cost is not None and x else None

    return {
        "month": key, "prev_month": prev_key, "telemetry_calls": n_cur, "complete": complete,
        "days_without_telemetry": missing_days, "unknown_cost_calls": t["unknown_cost_calls"],
        "cost_usd": cost, "prev_cost_usd": prev_cost,
        "delta_usd": round(cost - prev_cost, 4) if cost is not None and prev_cost is not None else None,
        "delta_pct": (round((cost - prev_cost) / prev_cost, 4)
                      if cost is not None and prev_cost not in (None, 0) else None),
        "by_family": fam, "by_workflow": cur["by"]["workflow"], "by_scope": cur["by"]["scope"],
        "prev_by_workflow": prev["by"]["workflow"],
        "sonnet_calls": fam.get("SONNET", {}).get("calls", 0), "haiku_calls": fam.get("HAIKU", {}).get("calls", 0),
        "failed_calls": t["failed"], "input_tokens": t["input_tokens"], "output_tokens": t["output_tokens"],
        "cache_read_input_tokens": t["cache_read_input_tokens"], "prompt_cache_savings_usd": t["cache_savings_usd"],
        "analysis_cache": cur["analysis_cache"], "api": cur["api"],
        "paid_api_cost_usd": _paid_api_cost(cur["api"]),
        "analysis_cache_validation": _cache_validation([r for r in rows if r["_day"] >= start]),
        "scan_runs": scans["scan_runs"], "final_trades": scans["final_trades"],
        "cost_per_scan": per(scans["scan_runs"]), "cost_per_final_trade": per(scans["final_trades"]),
        "cost_per_sonnet_analysis": (round(cur["by"]["stage"]["deep_analysis"]["cost_usd"] / analyzed, 4)
                                     if analyzed else None),
    }


def explain(s: dict) -> str:
    """Kurze Erklärung ausschließlich aus Telemetriewerten (keine Vermutungen)."""
    if not s.get("telemetry_calls"):
        return "Keine Kosten-Telemetrie für diesen Monat erfasst – Kosten nicht verfügbar."
    parts = []
    wf = {k: v["cost_usd"] for k, v in (s.get("by_workflow") or {}).items()}
    if wf and s.get("cost_usd"):
        top, val = max(wf.items(), key=lambda kv: kv[1])
        parts.append(f"Größter Kostenblock: {top} mit ${val:.2f} ({val / s['cost_usd']:.0%}).")
    if s.get("delta_usd") is not None:
        prev = {k: v["cost_usd"] for k, v in (s.get("prev_by_workflow") or {}).items()}
        deltas = {k: wf.get(k, 0.0) - prev.get(k, 0.0) for k in set(wf) | set(prev)}
        if deltas:
            k, d = max(deltas.items(), key=lambda kv: abs(kv[1]))
            richtung = "gestiegen" if s["delta_usd"] > 0 else "gesunken"
            parts.append(f"Gesamtkosten um ${abs(s['delta_usd']):.2f} {richtung}; "
                         f"stärkste Veränderung: {k} ({d:+.2f} $).")
    else:
        parts.append("Kein Vormonatsvergleich möglich (Vormonat ohne Telemetrie).")
    if s.get("failed_calls"):
        parts.append(f"{s['failed_calls']} fehlgeschlagene LLM-Calls.")
    ac = s.get("analysis_cache") or {}
    if ac.get("would_hit"):
        parts.append(f"Analyse-Cache (Beobachtungsmodus): {ac['would_hit']} identische Wiederholungen, "
                     f"potenzielle Ersparnis ${ac['potential_saved_usd']:.2f}.")
    return " ".join(parts)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "budget"
    if cmd == "budget":
        print(json.dumps(budget_status(), indent=2, ensure_ascii=False))
    elif cmd == "month":
        key = sys.argv[2] if len(sys.argv) > 2 else f"{_now():%Y-%m}"
        print(json.dumps(month_summary(key), indent=2, ensure_ascii=False, default=str))
