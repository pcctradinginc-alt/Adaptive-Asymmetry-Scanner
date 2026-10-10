"""modules/system_state.py – kanonischer, versionierter SystemState (Single Source of Truth).

    python -m modules.system_state            (berechnen + persistieren, Ausgabe JSON)

Vorher (Audit 2026-10-03) lag "Safe Mode" an vier Stellen: outputs/research/safe_mode.json
(Meta-Cognition), meta_state.json.safe_mode (immer false), machine_state.json.safe_mode (Kopie)
und outputs/health (Data Health). Der Montagsbericht las nur die erste, der Scanner nur die
letzte – beide konnten gleichzeitig Verschiedenes melden.

Jetzt: GENAU EINE Ableitung (`derive`) aus den Komponenten-Eingaben, persistiert und
versioniert in outputs/state/system_state.json (+ append-only Historie). ALLE Verbraucher –
Pipeline, High-Confidence Scanner, PromotionController, ProductionIntelligenceAdapter,
Weekly Report, Alerts, Meta-Learning – lesen `current()`. Die Eingabedateien sind KEINE
Wahrheit, nur Komponenten:

  model_health  outputs/research/safe_mode.json (meta_cognition: Kalibrierung, Disagreement,
                World Model, Performance, defekte Artefakte; Drift-/Daten-Gründe werden hier
                NICHT übernommen, sie kommen aus den eigenen Komponenten)
  drift_state   modules/drift.py (Feature/Modell aus meta_learning.json, Daten aus Health)
  data_health   outputs/health/source_health_snapshot.json (täglicher Source Health Check)
  promotion     outputs/intelligence/promotion_state.json
  champion      config/model_registry.yaml (champion) + Hash von config.yaml (Regelwerk)

Fail-closed: fehlt/unlesbar eine Pflichtkomponente (model_health, data_health) -> safe_mode = true
mit Grund "unbekannt". Unbekannt ist nie gesund.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

OUT = Path("outputs/state")
STATE = OUT / "system_state.json"
HISTORY = OUT / "system_state_history.jsonl"
DEFAULT_INPUTS = {
    "model_health": Path("outputs/research/safe_mode.json"),
    "meta_learning": Path("outputs/research/meta_learning.json"),
    "data_health": Path("outputs/health/source_health_snapshot.json"),
    "promotion": Path("outputs/intelligence/promotion_state.json"),
    "model_registry": Path("config/model_registry.yaml"),
    "champion_config": Path("config.yaml"),
    "universe_v1_frozen": Path("outputs/universe/universe_v1_frozen.json"),
    "universe_v2_snapshots": Path("outputs/universe/v2_snapshots"),
    "commodity_status": Path("outputs/research/commodity_intelligence.json"),
    "root": Path("."),
}
SCHEMA = "system-state-v2"
MODEL_HEALTH_MAX_AGE_DAYS = 14          # Meta-Cognition wöchentlich (ml_research.yml) + Puffer
DRIFT_INPUT_MAX_AGE_DAYS = 8          # wöchentlicher Meta-Lauf + 1 Tag Toleranz (nur Kennzeichnung)
COMMODITY_SOURCES = ("eia_petroleum_weekly", "eia_natural_gas", "fred_commodities", "fred_regime_macro", "cftc_cot")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read(p: Path):
    try:
        if p.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001 – yaml.YAMLError u.ä.: unlesbar = unbekannt
        return None


def _hash_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def code_version() -> str:
    v = os.environ.get("GITHUB_SHA")
    if v:
        return v[:12]
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def derive(inputs: dict[str, Path] | None = None, now: datetime | None = None) -> dict:
    """EINZIGE Ableitung des Systemzustands (rein aus Dateien, deterministisch)."""
    from modules import drift as dr
    from modules.source_health import load_snapshot
    now = now or _now()
    inp = {**DEFAULT_INPUTS, **(inputs or {})}
    reasons, unknown = [], []

    # Data Health (täglich)
    snap = load_snapshot(inp["data_health"], now)
    if snap.get("unknown"):
        unknown.append(f"DATA HEALTH unbekannt: {snap.get('reason')}")
        data_health = {"status": "UNKNOWN", "data_quality": None, "blocked_decisions": [], "disabled_signals": [],
                       "unavailable_features": [], "stale_features": [], "global_reasons": [], "generated": None,
                       "counts": None}
    else:
        dsm = snap.get("safe_mode") or {}
        data_health = {"status": "SAFE_MODE" if dsm.get("active") else "OK", "data_quality": dsm.get("data_quality"),
                       "blocked_decisions": dsm.get("blocked_decisions") or [],
                       "disabled_signals": dsm.get("disabled_signals") or [],
                       "unavailable_features": dsm.get("unavailable_features") or [],
                       "stale_features": dsm.get("stale_features") or [],
                       "global_reasons": dsm.get("global_reasons") or [], "generated": snap.get("generated"),
                       "counts": snap.get("counts"), "fallbacks": snap.get("fallbacks") or []}
        reasons += [f"DATA: {r}" for r in data_health["global_reasons"]]

    # Model Health (Meta-Cognition; nur Modell-Gründe, Drift/Daten aus eigenen Komponenten)
    mh = _read(inp["model_health"])
    mh_age = None
    if isinstance(mh, dict) and mh.get("updated"):
        try:
            _u = datetime.fromisoformat(str(mh["updated"]).replace("Z", "+00:00"))
            mh_age = (now - (_u if _u.tzinfo else _u.replace(tzinfo=timezone.utc))).total_seconds() / 86400
        except ValueError:
            mh_age = None
    if not isinstance(mh, dict) or "active" not in mh:
        unknown.append("SAFE MODE unbekannt: Model Health (safe_mode.json) fehlt oder ist unlesbar")
        model_health = {"status": "UNKNOWN", "reasons": [], "updated": None}
    elif mh_age is not None and mh_age > MODEL_HEALTH_MAX_AGE_DAYS:
        # Audit 2026-10-04: Meta-Cognition läuft wöchentlich; ein alter Befund ist kein aktueller Befund
        unknown.append(f"SAFE MODE unbekannt: Model Health veraltet ({mh_age:.0f} T > {MODEL_HEALTH_MAX_AGE_DAYS} T)")
        model_health = {"status": "UNKNOWN", "reasons": [], "updated": mh.get("updated")}
    else:
        comp = mh.get("components")
        model_reasons = (comp or {}).get("model") if isinstance(comp, dict) else \
            [r for r in mh.get("reasons") or [] if not any(k in r for k in ("DRIFT", "STALE DATA", "PIPELINE FAILURE"))]
        model_health = {"status": "SAFE_MODE" if model_reasons else "OK", "reasons": model_reasons or [],
                        "updated": mh.get("updated")}
        reasons += [f"MODEL: {r}" for r in model_reasons or []]

    # Drift (abgestuft)
    meta = _read(inp["meta_learning"]) or {}
    drift_state = dr.assess(meta, data_health["data_quality"])
    # Freshness des Drift-Inputs (nur Kennzeichnung, 2026-10-09; Policy/Grenzen unverändert):
    # meta_learning.json wird wöchentlich berechnet – die Promotion-Pause basiert ggf. auf älteren Werten.
    _gen = meta.get("generated") if isinstance(meta, dict) else None
    _age = None
    if _gen:
        try:
            _g = datetime.fromisoformat(str(_gen).replace("Z", "+00:00"))
            _age = round((now - (_g if _g.tzinfo else _g.replace(tzinfo=timezone.utc))).total_seconds() / 86400, 1)
        except ValueError:
            _age = None
    drift_state["drift_input_timestamp"] = _gen
    drift_state["drift_input_age_days"] = _age
    drift_state["drift_input_status"] = ("UNKNOWN" if _age is None else
                                         "STALE_DRIFT_INPUT" if _age > DRIFT_INPUT_MAX_AGE_DAYS else "FRESH")
    if drift_state["consequences"]["safe_mode"]:
        reasons += [f"DRIFT: {r}" for r in drift_state["reasons"] if "SEVERE" in r]

    # Champion + Promotion
    reg = _read(inp["model_registry"]) or {}
    champion = {"ml_champion": reg.get("champion"), "rule_champion": "scanner_rules",
                "config_hash": _hash_file(inp["champion_config"]),
                "version": f"{reg.get('champion') or 'scanner_rules'}@{_hash_file(inp['champion_config'])}"}
    ps = _read(inp["promotion"]) or {}
    hy = ps.get("hypotheses") or {}
    counts: dict = {}
    for h in hy.values():
        counts[h.get("state")] = counts.get(h.get("state"), 0) + 1
    promotion_state = {"generated": ps.get("generated"), "policy_version": ps.get("policy_version"),
                       "counts": counts, "with_influence": sorted(k for k, h in hy.items()
                                                                  if (h.get("influence_level") or "NONE") != "NONE"),
                       "max_automatic_influence": ps.get("max_automatic_influence")}

    # Erlaubter Produktionseinfluss (einzige Quelle: promotion_state.json des PromotionControllers)
    allowed_influence = {"max_automatic_influence": ps.get("max_automatic_influence") or "ABSTENTION_ONLY",
                         "by_hypothesis": {k: h.get("influence_level") for k, h in sorted(hy.items())
                                           if (h.get("influence_level") or "NONE") != "NONE"},
                         "commodity_max": "SCORE_LIMITED", "default": "NONE"}

    # Universe: V1 produktiv (eingefroren), V2 Shadow; Segment-Stufen nur aus promotion_state
    frozen = _read(inp["universe_v1_frozen"]) or {}
    v1_ok = None
    try:
        from modules import universe_v2 as _uv
        v1_ok = _uv.v1_unchanged(inp["universe_v1_frozen"]) if frozen else None
        snap_v2 = _uv.latest_snapshot(inp["universe_v2_snapshots"])
    except Exception as e:  # noqa: BLE001 – unbekannt statt Absturz
        snap_v2 = None
        unknown.append(f"UNIVERSE unbekannt: {type(e).__name__}")
    seg = {k.split("@")[0].replace("UNIV-V2-SEG-", ""): (h.get("influence_level") or "NONE")
           for k, h in hy.items() if "UNIV-V2-SEG-" in k}
    universe_version = {"production": "V1", "v1_definition_hash": frozen.get("definition_hash"),
                        "v1_unchanged": v1_ok, "v2_status": "PARTIALLY_PROMOTED" if any(v != "NONE" for v in seg.values())
                        else "SHADOW", "v2_segments": seg, "v2_latest_snapshot": (snap_v2 or {}).get("as_of")}
    if frozen and v1_ok is False:
        reasons.append("UNIVERSE: V1-Definition weicht vom eingefrorenen Hash ab")

    # Commodity-Datenqualität: nur Research-Quellen -> NIE Safe-Mode-Grund, nur abhängige Features betroffen
    srcs = (snap.get("sources") or {}) if not snap.get("unknown") else {}
    cs = {sid: (srcs.get(sid) or {}).get("status", "UNVALIDATED") for sid in COMMODITY_SOURCES}
    vals = set(cs.values())
    overall = ("BROKEN" if "BROKEN" in vals else "STALE" if "STALE" in vals else "DEGRADED" if "DEGRADED" in vals
               else "UNVALIDATED" if "UNVALIDATED" in vals else "HEALTHY")
    cst = _read(inp["commodity_status"]) or {}
    commodity_data_health = {"overall": overall, "sources": cs, "research_status": cst.get("research_status") or "RESEARCH_ONLY",
                             "unavailable_series": sorted((cst.get("unavailable") or {}).keys()),
                             "safe_mode_relevant": False}

    from modules import learning_health as _lh
    lh = _lh.assess(inp["root"], now, data_health={"commodity": commodity_data_health})
    learning = {"overall": lh["overall"], "stalled_or_broken": lh["stalled_or_broken"],
                "paths": {k: v["status"] for k, v in lh["paths"].items()}}

    all_reasons = unknown + reasons
    content = {"schema": SCHEMA, "safe_mode": bool(all_reasons), "safe_mode_reason": all_reasons,
               "data_health": data_health, "model_health": model_health, "drift_state": drift_state,
               "champion_version": champion, "promotion_state": promotion_state, "known": not unknown,
               "allowed_influence": allowed_influence, "universe_version": universe_version,
               "commodity_data_health": commodity_data_health, "learning_health": learning,
               "inputs": {k: _hash_file(Path(v)) for k, v in inp.items() if Path(v).is_file()}}
    return content


def _fingerprint(content: dict) -> str:
    core = {k: v for k, v in content.items() if k not in ("updated_at", "code_version", "state_version",
                                                           "fingerprint")}
    if isinstance(core.get("drift_state"), dict):    # Alter ändert sich laufend -> keine neue State-Version
        core["drift_state"] = {k: v for k, v in core["drift_state"].items() if k != "drift_input_age_days"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def current(*, inputs: dict[str, Path] | None = None, now: datetime | None = None, persist: bool = True,
            state_path: Path | None = None, history_path: Path | None = None) -> dict:
    """Kanonischer Zustand. Berechnet aus den Komponenten; ist der Inhalt unverändert, bleibt
    die Version gleich, sonst Version + 1 und Eintrag in der Historie."""
    now = now or _now()
    state_path, history_path = state_path or STATE, history_path or HISTORY
    content = derive(inputs, now)
    fp = _fingerprint(content)
    prev = _read(state_path) if persist else None
    if isinstance(prev, dict) and prev.get("fingerprint") == fp:
        return prev
    ver = int((prev or {}).get("state_version") or 0) + 1
    state = {**content, "state_version": ver, "fingerprint": fp, "updated_at": now.isoformat(timespec="seconds"),
             "code_version": code_version()}
    if persist:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(state_path)                       # atomar: Leser sehen nie einen halben Zustand
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"state_version": ver, "updated_at": state["updated_at"], "safe_mode": state["safe_mode"],
                                 "safe_mode_reason": state["safe_mode_reason"], "drift_level": state["drift_state"]["level"],
                                 "data_health": state["data_health"]["status"], "model_health": state["model_health"]["status"],
                                 "champion": state["champion_version"]["version"], "fingerprint": fp},
                                ensure_ascii=False) + "\n")
    return state


def safe_mode_view(state: dict) -> dict:
    """Kompakte Sicht für Verbraucher (gleiche Felder überall)."""
    dh, ds = state.get("data_health") or {}, state.get("drift_state") or {}
    return {"active": bool(state.get("safe_mode")), "reasons": state.get("safe_mode_reason") or [],
            "known": bool(state.get("known")), "state_version": state.get("state_version"),
            "drift_level": ds.get("level"), "consequences": ds.get("consequences") or {},
            "blocked_decisions": dh.get("blocked_decisions") or [], "disabled_signals": dh.get("disabled_signals") or [],
            "unavailable_features": dh.get("unavailable_features") or [], "data_quality": dh.get("data_quality")}


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    st = current()
    print(json.dumps({k: st[k] for k in ("state_version", "safe_mode", "safe_mode_reason", "known", "updated_at",
                                         "code_version")}, indent=1, ensure_ascii=False))
    print("drift:", st["drift_state"]["level"], "| data:", st["data_health"]["status"], "| model:",
          st["model_health"]["status"], "| champion:", st["champion_version"]["version"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
