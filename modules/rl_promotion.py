"""
modules/rl_promotion.py – Bewertung, ob der robuste PPO-Agent ein RL PROMOTION CANDIDATE ist.

Der RL-Agent erhält nie autonom produktiven Einfluss. Dieses Modul MELDET nur:
  NOT_READY               Policy kollabiert (eine Aktion >= 95 %) im Walk-Forward, in-sample
                          oder in den prospektiven Shadow-Aktionen, oder keine Diagnose vorhanden
  ACCUMULATING            nicht kollabiert, aber Stabilität oder Forward-Mehrwert noch nicht belegt
  RL PROMOTION CANDIDATE  (1) kein Kollaps in den letzten STABLE_RUNS Trainingsläufen,
                          (2) stabile Aktionsverteilung zwischen Läufen (L1-Abstand <= MAX_SHIFT),
                          (3) prospektiver Challenger `ppo_robust_shadow` (challenger.py, Bootstrap-CI,
                              Alpha-Spending) empfiehlt Promotion = inkrementeller Nutzen gegenüber
                              Champion-only auf reinen Forward-Daten.
Auch bei RL PROMOTION CANDIDATE bleibt `rl.veto_enabled` aus: Freigabe nur durch einen Menschen
(PR auf config.yaml). Ausgabe: outputs/research/rl_promotion.json (+ Verlauf je Trainingslauf).

    python -m modules.rl_promotion
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

META_PATH = Path("outputs/models/ppo_robust_shadow_meta.json")
OUT_PATH = Path("outputs/research/rl_promotion.json")
HISTORY_PATH = Path("outputs/research/rl_promotion_history.jsonl")
CHALLENGER_ID = "ppo_robust_shadow"
COLLAPSE_SHARE = 0.95
STABLE_RUNS = 3
MAX_SHIFT = 0.30
MIN_LIVE_ACTIONS = 20
ACTIONS = ("SKIP", "NORMAL", "BOOST")


def _shares(counts: dict | None) -> dict | None:
    counts = counts or {}
    n = sum(int(counts.get(a) or 0) for a in ACTIONS)
    return {a: round(int(counts.get(a) or 0) / n, 3) for a in ACTIONS} if n else None


def _collapsed(shares: dict | None) -> bool | None:
    return None if shares is None else max(shares.values()) >= COLLAPSE_SHARE


def live_actions(ledger_rows: list[dict]) -> dict:
    """Prospektive Shadow-Aktionen aus dem Candidate Ledger (features.rl_robust_action)."""
    counts = {a: 0 for a in ACTIONS}
    for r in ledger_rows:
        a = (r.get("features") or {}).get("rl_robust_action")
        if a in counts:
            counts[a] += 1
    n = sum(counts.values())
    sh = _shares(counts)
    return {"n": n, "counts": counts, "shares": sh,
            "collapsed": _collapsed(sh) if n >= MIN_LIVE_ACTIONS else None}


def run_entry(meta: dict) -> dict:
    wf = (meta.get("walk_forward") or {}).get("test") or {}
    ins = meta.get("in_sample") or {}
    return {"trained_at": meta.get("trained_at"), "n_trades": meta.get("n_trades"),
            "wf_shares": _shares(wf.get("action_counts")), "in_sample_shares": _shares(ins.get("action_counts")),
            "wf_collapsed": bool(wf.get("collapsed")), "in_sample_collapsed": bool(ins.get("collapsed"))}


def _l1(a: dict | None, b: dict | None) -> float | None:
    if not a or not b:
        return None
    return round(sum(abs(a[k] - b[k]) for k in ACTIONS), 3)


def assess(meta: dict | None, history: list[dict], challenger: dict | None, ledger_rows: list[dict]) -> dict:
    reasons: list[str] = []
    live = live_actions(ledger_rows)
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "veto_enabled": False,
           "live_actions": live, "challenger": challenger or {}, "requires_human_approval": True}
    if not meta:
        out.update(status="NOT_READY", summary="keine Walk-Forward-Diagnose des robusten PPO vorhanden",
                   reasons=["ppo_robust_shadow_meta.json fehlt"])
        return out
    cur = run_entry(meta)
    out["current_run"] = cur
    if cur["wf_collapsed"]:
        reasons.append(f"Walk-Forward kollabiert ({cur['wf_shares']})")
    if cur["in_sample_collapsed"]:
        reasons.append(f"In-Sample kollabiert ({cur['in_sample_shares']})")
    if live["collapsed"]:
        reasons.append(f"prospektive Shadow-Aktionen kollabiert ({live['shares']}, n={live['n']})")
    if reasons:
        out.update(status="NOT_READY", reasons=reasons,
                   summary="Policy kollabiert – kein produktiver Einfluss, keine Promotion-Prüfung")
        return out
    runs = [h for h in history if h.get("trained_at")][-STABLE_RUNS:]
    stable_runs = len(runs) >= STABLE_RUNS and not any(h.get("wf_collapsed") or h.get("in_sample_collapsed") for h in runs)
    shifts = [_l1(a.get("wf_shares"), b.get("wf_shares")) for a, b in zip(runs, runs[1:])]
    stable_actions = bool(shifts) and all(s is not None and s <= MAX_SHIFT for s in shifts)
    out["stability"] = {"runs_considered": len(runs), "no_collapse": stable_runs, "action_shifts_l1": shifts,
                        "stable_actions": stable_actions}
    verdict = (challenger or {}).get("verdict")
    forward_ok = verdict == "promote_recommended"
    if not stable_runs:
        reasons.append(f"Stabilität: {len(runs)}/{STABLE_RUNS} kollapsfreie Trainingsläufe")
    if not stable_actions:
        reasons.append(f"Aktionsverteilung zwischen Läufen nicht stabil (L1 {shifts}, max {MAX_SHIFT})")
    if not forward_ok:
        reasons.append(f"Forward-Mehrwert vs. Champion-only nicht belegt (Challenger-Verdikt {verdict or 'n/a'}, "
                       f"n {challenger.get('n_challenger') if challenger else 'n/a'}, CI "
                       f"[{challenger.get('ci_lower') if challenger else None}, {challenger.get('ci_upper') if challenger else None}])")
    if stable_runs and stable_actions and forward_ok:
        out.update(status="RL PROMOTION CANDIDATE", reasons=[],
                   summary="Kriterien erfüllt – Empfehlung an Menschen; rl.veto_enabled bleibt aus bis zur PR-Freigabe")
    else:
        out.update(status="ACCUMULATING", reasons=reasons, summary="nicht kollabiert; Evidenz wird gesammelt")
    return out


def _load_json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"rl_promotion: {p} nicht lesbar ({e})")
        return None


def _history(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.debug(f"rl_promotion: defekte Verlaufszeile übersprungen: {line[:60]}")
    return out


def run(meta_path: Path = META_PATH, out_path: Path = OUT_PATH, history_path: Path = HISTORY_PATH,
        today: date | None = None) -> dict:
    meta = _load_json(meta_path)
    hist = _history(history_path)
    if meta and meta.get("trained_at") not in {h.get("trained_at") for h in hist}:
        entry = run_entry(meta)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        hist.append(entry)
    challenger, rows = None, []
    try:
        from modules import challenger as ch
        rows = ch.load_ledger_rows()
        challenger = next((r for r in ch.evaluate_all(today=today) if r.get("id") == CHALLENGER_ID), None)
    except Exception as e:  # noqa: BLE001 – ohne Challenger-Auswertung kein Forward-Nachweis
        log.warning(f"rl_promotion: Challenger-Auswertung nicht verfügbar ({e})")
    res = assess(meta, hist, challenger, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = run()
    print(f"RL-Status: {r['status']} – {r.get('summary')}")
    for x in r.get("reasons") or []:
        print(f"  - {x}")
