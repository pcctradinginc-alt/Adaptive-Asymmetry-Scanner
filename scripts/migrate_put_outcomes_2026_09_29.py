"""
Einmalige Datenkorrektur (Audit 2026-09-29): Long-Put-Outcomes aus der
Delta-Näherung waren vorzeichenverkehrt (feedback.compute_outcome ohne
Put-Vorzeichen). Deterministische Neuberechnung aus gespeicherten Eingaben:

    stock_return = close_price / simulation.current_price - 1
    leverage     = current_price / entry_debit * 0.65
    alt (falsch) = clip( stock_return * leverage, -1, 5)
    neu          = clip(-stock_return * leverage, -1, 5)

Nur Zeilen, deren gespeichertes Outcome EXAKT der falschen Formel entspricht,
werden korrigiert (Original bleibt in outcome_uncorrected). Alle Long-Option-
Outcomes, die der Näherung entsprechen, werden als
outcome_method_reconstructed="delta_approx" markiert (geschätzt, kein Quote).
Danach wird history.feature_stats deterministisch aus den geschlossenen
Trades neu aufgebaut (die Bins speisen das Produktions-Scoring).

    python scripts/migrate_put_outcomes_2026_09_29.py [--apply]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HISTORY = Path("outputs/history.json")
TAG = "put_sign_delta_approx_2026-09-29"


def _approx(t: dict, sign: float) -> float | None:
    sim, o = t.get("simulation") or {}, t.get("option") or {}
    s0 = float(sim.get("current_price") or 0)
    s1 = float(t.get("close_price") or 0)
    deb = float(t.get("entry_debit") or o.get("ask") or 0)
    if s0 <= 0 or s1 <= 0 or deb <= 0:
        return None
    return max(-1.0, min(sign * (s1 / s0 - 1.0) * (s0 / deb) * 0.65, 5.0))


def migrate(history: dict) -> dict:
    from feedback import update_bin
    report = {"put_corrected": [], "delta_approx_marked": 0}
    for key in ("closed_trades", "shadow_trades"):
        for t in history.get(key) or []:
            strat = str(t.get("strategy") or "").upper()
            out = t.get("outcome")
            if not isinstance(out, (int, float)) or "SPREAD" in strat or not t.get("close_date"):
                continue
            wrong = _approx(t, +1.0)
            if wrong is None or abs(wrong - out) > 0.002:
                continue
            t["outcome_method_reconstructed"] = "delta_approx"
            report["delta_approx_marked"] += 1
            if "PUT" in strat and t.get("outcome_correction") != TAG:
                fixed = round(_approx(t, -1.0), 4)
                t["outcome_uncorrected"] = out
                t["outcome"] = fixed
                t["outcome_correction"] = TAG
                report["put_corrected"].append({"list": key, "ticker": t.get("ticker"),
                                                "entry": t.get("entry_date"), "old": out, "new": fixed})
    # feature_stats deterministisch aus closed_trades neu aufbauen
    fs: dict = {}
    for t in history.get("closed_trades") or []:
        feat = t.get("features") or {}
        for f_name, bin_key in (("impact", "bin_impact"), ("mismatch", "bin_mismatch"),
                                ("eps_drift", "bin_eps_drift")):
            if feat.get(bin_key) and isinstance(t.get("outcome"), (int, float)):
                update_bin(fs, f_name, feat[bin_key], t["outcome"])
    report["feature_stats_before"] = history.get("feature_stats")
    history["feature_stats"] = fs
    report["feature_stats_after"] = fs
    return report


if __name__ == "__main__":
    h = json.loads(HISTORY.read_text())
    rep = migrate(h)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("feature_stats")}, indent=1))
    print("feature_stats vorher:", json.dumps(rep["feature_stats_before"]))
    print("feature_stats nachher:", json.dumps(rep["feature_stats_after"]))
    if "--apply" in sys.argv:
        HISTORY.write_text(json.dumps(h, indent=2, default=str))
        print("angewendet.")
