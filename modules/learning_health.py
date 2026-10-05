"""modules/learning_health.py – LEARNING_HEALTH: läuft jeder Lernpfad tatsächlich? (Audit 2026-10-04)

Rein lesend aus den Artefakten, die die Workflows schreiben – kein eigenes Lernen, keine
Produktionswirkung. Ein Pfad, der dauerhaft keine Daten erhält, erscheint als STALLED; ein Pfad,
der noch nie Daten hatte, als UNVALIDATED; "Code vorhanden" zählt nie als aktiv.

Status je Pfad: OK | ACTIVE | SHADOW | RESEARCH | NEED_MORE_DATA | UNVALIDATED | STALLED | BROKEN.
Kanonisch eingebunden in modules/system_state.py (Feld learning_health) -> Reports und Produktion
lesen denselben Stand.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Erwartete Kadenz je Pfad (Tage) – darüber ohne neues Artefakt: STALLED
MAX_AGE = {"outcome_ingestion": 45, "shadow_lifecycle": 60, "research_generation": 40, "hypothesis_testing": 40,
           "promotion_pipeline": 5, "universe_v2": 9, "commodity_intelligence": 40, "rl": 40, "research_memory": 40,
           "calibration": 10}
OK_STATES = ("OK", "ACTIVE", "SHADOW", "RESEARCH", "NEED_MORE_DATA")


def _js(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return "CORRUPT"


def _jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for x in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            continue
    return out


def _ts(x) -> datetime | None:
    if not x:
        return None
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _age(x, now: datetime) -> float | None:
    d = _ts(x)
    return None if d is None else (now - d).total_seconds() / 86400


def _fresh(gen, key: str, now: datetime, ok: str = "OK") -> tuple[str, str]:
    a = _age(gen, now)
    if a is None:
        return "UNVALIDATED", "noch nie gelaufen"
    if a > MAX_AGE[key]:
        return "STALLED", f"letztes Artefakt vor {a:.0f} T (> {MAX_AGE[key]} T)"
    return ok, f"vor {a:.1f} T"


# Zwei Evidenzebenen, nie vermischen (Reporting 2026-10-05):
#  HISTORICAL_WALK_FORWARD_CALIBRATION – historische/Walk-Forward-OOS-Daten (Brier, ECE, Coverage, Buckets);
#    keine Production-/Forward-Evidenz, auch wenn "calibrated=True".
#  LIVE_FORWARD_CALIBRATION – ausschließlich prospektive RELIABLE Paper-Trades (prequentiell);
#    unter LIVE_FORWARD_CAL_MIN_N: NEED_MORE_DATA / UNCALIBRATED, unabhängig vom historischen Wert.
HISTORICAL_WF_LABEL = "HISTORICAL_WALK_FORWARD_CALIBRATION"
LIVE_FORWARD_LABEL = "LIVE_FORWARD_CALIBRATION"
LIVE_FORWARD_CAL_MIN_N = 30


def live_forward_calibration(pp: dict | None) -> dict:
    """Status der LIVE_FORWARD_CALIBRATION aus paper_performance_analysis.calibration_oos."""
    from modules.outcomes import artifact_is_current
    cal = (pp or {}).get("calibration_oos") or {} if isinstance(pp, dict) else {}
    n = int(cal.get("n_evaluated") or 0) if artifact_is_current(pp) else 0
    if n < LIVE_FORWARD_CAL_MIN_N:
        status = "NEED_MORE_DATA / UNCALIBRATED"
    else:
        status = "CALIBRATED" if cal.get("calibrated_better") else "NOT_CALIBRATED"
    return {"label": LIVE_FORWARD_LABEL, "status": status, "n": n, "min_n": LIVE_FORWARD_CAL_MIN_N,
            "ece": cal.get("ece_calibrated") if n else None}


def assess(root: Path | None = None, now: datetime | None = None, data_health: dict | None = None) -> dict:
    root = Path(root or ".")
    o = root / "outputs"
    now = now or datetime.now(timezone.utc)
    out: dict[str, dict] = {}

    def put(k, status, detail):
        out[k] = {"status": status, "detail": detail}

    hist = _js(o / "history.json")
    if hist == "CORRUPT":
        put("outcome_ingestion", "BROKEN", "history.json unlesbar")
        put("shadow_lifecycle", "BROKEN", "history.json unlesbar")
    elif not isinstance(hist, dict):
        put("outcome_ingestion", "UNVALIDATED", "history.json fehlt")
        put("shadow_lifecycle", "UNVALIDATED", "history.json fehlt")
    else:
        from modules.outcomes import is_reliable_outcome
        closed = hist.get("closed_trades") or []
        rel = [t for t in closed if is_reliable_outcome(t)]
        last = max((str(t.get("close_date") or "") for t in closed), default="")
        st, d = _fresh(last, "outcome_ingestion", now, "OK")
        if st == "STALLED" and not (hist.get("active_trades") or []):
            st, d = "NEED_MORE_DATA", "keine offenen Trades, nichts zu schließen"
        put("outcome_ingestion", st, f"{len(closed)} geschlossen, {len(rel)} RELIABLE; letzte Schließung {last or 'nie'} ({d})")
        sh = hist.get("shadow_trades") or []
        done = [t for t in sh if t.get("outcome") is not None]
        last_s = max((str(t.get("close_date") or "") for t in done), default="")
        st, d = _fresh(last_s, "shadow_lifecycle", now, "ACTIVE")
        if st == "UNVALIDATED" and sh:
            st, d = "NEED_MORE_DATA", "Shadow-Kandidaten offen, noch kein Horizont erreicht"
        put("shadow_lifecycle", st, f"{len(sh)} Shadow-Kandidaten, {len(done)} mit Outcome ({d})")

    rows = []
    d = o / "intelligence" / "final_mc_ledger"
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            rows += _jl(f)
    outs = _jl(o / "intelligence" / "final_mc_outcomes.jsonl")
    down = _jl(o / "intelligence" / "final_mc_downstream.jsonl")
    if not rows:
        put("gate_counterfactuals", "UNVALIDATED", "Final-MC-Survivor-Ledger noch leer (Start 2026-10-03, erster Scanner-Lauf ausstehend)")
    else:
        old = [r for r in rows if (_age(r.get("timestamp"), now) or 0) > 45]
        resolved = {e.get("observation_id") for e in outs}
        miss = [r for r in old if r.get("observation_id") not in resolved]
        st = "STALLED" if old and len(miss) == len(old) else "ACTIVE" if outs else "NEED_MORE_DATA"
        put("gate_counterfactuals", st, f"{len(rows)} Survivor, {len(down)} Downstream-Gate-Befunde, {len(outs)} Outcomes"
                                        + (f", {len(miss)} überfällig" if miss else ""))

    plan = _js(o / "research" / "factory_plan.json")
    st, dd = _fresh((plan or {}).get("generated") if isinstance(plan, dict) else None, "research_generation", now)
    put("research_generation", st, f"{(plan or {}).get('n_ideas') if isinstance(plan, dict) else 0} Ideen ({dd})")
    res = _js(o / "research" / "factory_results.json")
    db = _js(o / "research" / "hypothesis_db.json")
    gen = max(str((res or {}).get("generated") or "") if isinstance(res, dict) else "",
              str((db or {}).get("generated") or "") if isinstance(db, dict) else "")
    st, dd = _fresh(gen, "hypothesis_testing", now)
    put("hypothesis_testing", st, f"{(db or {}).get('n_tested_total') if isinstance(db, dict) else 0} getestet ({dd})")

    ps = _js(o / "intelligence" / "promotion_state.json")
    hy = (ps or {}).get("hypotheses") or {} if isinstance(ps, dict) else {}
    st, dd = _fresh((ps or {}).get("generated") if isinstance(ps, dict) else None, "promotion_pipeline", now)
    integ = (ps or {}).get("integrity") or {} if isinstance(ps, dict) else {}
    if any(integ.get(k) for k in ("registry", "transitions")):
        st, dd = "BROKEN", f"Integritätsproblem {integ}"
    states: dict = {}
    for h in hy.values():
        states[h.get("state")] = states.get(h.get("state"), 0) + 1
    put("promotion_pipeline", st, f"{states or 'keine Verträge'} ({dd})")
    active = [k for k, h in hy.items() if h.get("state") in ("GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")]
    put("demotion_pipeline", out["promotion_pipeline"]["status"] if out["promotion_pipeline"]["status"] in ("BROKEN", "STALLED",
                                                                                                         "UNVALIDATED") else "OK",
        f"läuft im selben Controller-Lauf (_monitor); {len(active)} aktive Hypothesen überwacht")
    chall = [k for k, h in hy.items() if h.get("state") in ("PROSPECTIVE_CHALLENGER", "FORWARD_VALIDATED")]
    led = o / "intelligence" / "decision_ledger"
    n_led = sum(len(_jl(f)) for f in sorted(led.glob("*.jsonl"))) if led.exists() else 0
    put("forward_validation", "ACTIVE" if chall and n_led else "NEED_MORE_DATA" if chall else "UNVALIDATED",
        f"{len(chall)} Challenger im Forward, {n_led} Decision-Ledger-Zeilen")

    pp = _js(o / "research" / "paper_performance_analysis.json")
    lfc = live_forward_calibration(pp if isinstance(pp, dict) else None)
    if pp is None:
        st = "UNVALIDATED"
    elif pp == "CORRUPT":
        st = "BROKEN"
    else:
        st = "OK" if lfc["status"] == "CALIBRATED" else "NEED_MORE_DATA"
    put("calibration", st, f"{LIVE_FORWARD_LABEL} {lfc['status']}: prospektive RELIABLE Paper-Trades "
                           f"n={lfc['n']} (min {LIVE_FORWARD_CAL_MIN_N}, ECE {lfc['ece']}); "
                           f"{HISTORICAL_WF_LABEL} ist keine Forward-Evidenz")

    try:
        from modules import universe_v2 as uv
        v1_ok = uv.v1_unchanged(o / "universe" / "universe_v1_frozen.json")
        put("universe_v1", "OK" if v1_ok else "BROKEN", "Definition eingefroren und unverändert" if v1_ok
            else "V1-Definition weicht vom eingefrorenen Hash ab")
        snap = uv.latest_snapshot(o / "universe" / "v2_snapshots")
    except Exception as e:  # noqa: BLE001 – Gesundheitsanzeige darf nie werfen
        put("universe_v1", "BROKEN", f"Prüfung fehlgeschlagen: {e}")
        snap = None
    st, dd = _fresh((snap or {}).get("as_of"), "universe_v2", now, "SHADOW")
    put("universe_v2", st, f"letzter PIT-Snapshot {(snap or {}).get('as_of', 'keiner')} ({dd})")

    cst = _js(o / "research" / "commodity_intelligence.json")
    st, dd = _fresh((cst or {}).get("generated") if isinstance(cst, dict) else None, "commodity_intelligence", now,
                    "RESEARCH")
    put("commodity_intelligence", st, f"Feature-Store/Status ({dd}); Einfluss NONE")
    srcs = ((data_health or {}).get("commodity") or {})
    put("commodity_sources", srcs.get("overall", "UNVALIDATED"), str(srcs.get("sources") or "kein Health-Snapshot"))

    rl = _js(o / "models" / "ppo_robust_shadow_meta.json")
    st, dd = _fresh((rl or {}).get("trained_at") if isinstance(rl, dict) else None, "rl", now, "SHADOW")
    put("rl", st, f"{(rl or {}).get('status') if isinstance(rl, dict) else 'kein Modell'}; Veto aus ({dd})")

    mem = _jl(o / "research" / "research_memory.jsonl")
    st, dd = _fresh(max((str(e.get("recorded_at") or "") for e in mem), default=""), "research_memory", now, "ACTIVE")
    put("research_memory", st, f"{len(mem)} Einträge ({dd})")

    bad = sorted(k for k, v in out.items() if v["status"] in ("STALLED", "BROKEN"))
    return {"paths": out, "stalled_or_broken": bad, "overall": "DEGRADED" if bad else "OK"}


def render_lines(lh: dict) -> list[tuple[str, str]]:
    names = {"outcome_ingestion": "Outcome ingestion", "shadow_lifecycle": "Shadow lifecycle",
             "gate_counterfactuals": "Gate learning", "research_generation": "Research generation",
             "hypothesis_testing": "Hypothesis testing", "forward_validation": "Forward validation",
             "promotion_pipeline": "Promotion pipeline", "demotion_pipeline": "Demotion pipeline",
             "calibration": "Calibration", "universe_v1": "Universe V1", "universe_v2": "Universe V2",
             "commodity_intelligence": "Commodity Intelligence", "commodity_sources": "Commodity Sources",
             "rl": "RL", "research_memory": "Research Memory"}
    return [(names.get(k, k), f"{v['status']} – {v['detail']}") for k, v in (lh.get("paths") or {}).items()]
