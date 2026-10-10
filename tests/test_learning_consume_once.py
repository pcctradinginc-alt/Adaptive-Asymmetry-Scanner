"""Consume-once-Garantie (modules/learning_ledger): Jede Beobachtung je (id, Horizont, Lernziel) genau einmal.
Zweimal lernen -> kein Doppel-Update; Crash/Retry -> kein Duplikat; verschiedene Horizonte getrennt erlaubt;
keine stille Mehrfachgewichtung alter Outcomes (Duplikat-Quarantäne in closed_trades)."""
from __future__ import annotations

import copy
import json

import pytest

import feedback
from modules import learning_ledger as ll


def _trade(ticker="AAPL", strike=280.0, outcome=None, close_date=None, ext=True):
    t = {"ticker": ticker, "entry_date": "2026-09-01", "strategy": "LONG_CALL",
         "option": {"expiry": "2027-01-15", "strike": strike},
         "features": {"bin_impact": "high", "bin_mismatch": "mid", "bin_eps_drift": "low"}}
    if ext:
        t["external_context_entry"] = {"primitives": {"shipping_breadth": 0.4}, "divergence_z": 1.2}
    if outcome is not None:
        t["outcome"], t["close_date"] = outcome, close_date or "2026-10-01"
    return t


@pytest.fixture(autouse=True)
def _reset():
    ll.run_stats(reset=True)
    yield


def test_key_and_trade_id():
    assert ll.key("cand1", 60, "calibration") == "cand1:60d:calibration"
    assert ll.key("cand1", "close", "feature_stats") == "cand1:close:feature_stats"
    assert ll.trade_id(_trade()) == ll.trade_id(_trade())
    assert ll.trade_id(_trade()) != ll.trade_id(_trade(strike=300.0))
    with pytest.raises(ValueError):
        ll.key("", 20, "x")


def test_second_learning_run_is_skipped():
    state, calls = {}, []
    assert ll.consume(state, "c1", 60, "calibration", lambda: calls.append(1)) is True
    assert ll.consume(state, "c1", 60, "calibration", lambda: calls.append(1)) is False
    assert calls == [1] and ll.run_stats()["duplicate_skips"] == 1
    assert state[ll.META]["duplicate_skips"] == 1 and "c1:60d:calibration" in state[ll.FIELD]


def test_different_horizons_and_targets_are_separate():
    state, calls = {}, []
    for h, tgt in ((60, "calibration"), (120, "calibration"), (60, "feature_stats"), (120, "feature_stats")):
        assert ll.consume(state, "c1", h, tgt, lambda: calls.append((h, tgt)))
    assert len(calls) == 4 and len(state[ll.FIELD]) == 4


def test_crash_before_save_then_retry_processes_exactly_once(tmp_path):
    """Markierung und Update liegen im selben atomar gespeicherten Dokument."""
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"feature_stats": {}}))
    trade = _trade(outcome=0.2)
    # Lauf 1: lernt im Speicher, stürzt VOR save_history ab -> nichts persistiert
    s1 = json.loads(path.read_text())
    feedback.learn_on_close(s1, trade, 0.2, True)
    assert s1["feature_stats"]["impact"]["high"]["count"] == 1
    # Retry: lädt den unveränderten Stand und lernt genau einmal
    s2 = json.loads(path.read_text())
    feedback.learn_on_close(s2, trade, 0.2, True)
    from modules.atomic_io import atomic_write_json
    atomic_write_json(path, s2)
    # Lauf 3 (z. B. Workflow-Retry nach erfolgreichem Speichern): kein zweites Update
    s3 = json.loads(path.read_text())
    res = feedback.learn_on_close(s3, trade, 0.2, True)
    assert res == {"feature_stats": False, "feature_stats_external": False}
    assert s3["feature_stats"]["impact"]["high"]["count"] == 1
    ext = s3["feature_stats_external"]
    assert all(b["count"] == 1 for dim in ext.values() for b in dim.values())


def test_failure_inside_update_rolls_back_and_is_not_marked():
    state = {"feature_stats": {"impact": {"high": {"count": 3, "avg_return": 0.1}}}}
    before = copy.deepcopy(state["feature_stats"])

    def half_then_fail():
        feedback.update_bin(state["feature_stats"], "impact", "high", 0.5)        # halbes Update ...
        raise RuntimeError("Abbruch")                                              # ... dann Fehler
    with pytest.raises(RuntimeError):
        ll.consume(state, "c9", "close", "feature_stats", half_then_fail, sections=("feature_stats",))
    assert state["feature_stats"] == before and not ll.is_processed(state, "c9:close:feature_stats")
    assert ll.consume(state, "c9", "close", "feature_stats",
                      lambda: feedback.update_bin(state["feature_stats"], "impact", "high", 0.5),
                      sections=("feature_stats",))
    assert state["feature_stats"]["impact"]["high"]["count"] == 4


def test_unreliable_outcome_is_not_learned_and_not_marked():
    state = {}
    assert feedback.learn_on_close(state, _trade(outcome=0.3), 0.3, False) == {
        "feature_stats": False, "feature_stats_external": False}
    assert not state.get(ll.FIELD)


def test_quarantine_exact_duplicates_idempotent():
    a, b = _trade(outcome=0.8166, close_date="2026-05-26"), _trade("META", 700.0, -0.49, "2026-05-26")
    state = {"closed_trades": [a, b, copy.deepcopy(a), copy.deepcopy(a), copy.deepcopy(b),
                               _trade(outcome=0.10, close_date="2026-06-30")]}      # gleiche ID, anderer Close
    assert ll.quarantine_duplicate_trades(state) == 3
    assert len(state["closed_trades"]) == 3 and len(state["closed_trades_quarantine"]) == 3
    assert all(q["_quarantine"]["reason"] == "exact_duplicate_closed_trade" for q in state["closed_trades_quarantine"])
    assert ll.quarantine_duplicate_trades(state) == 0                                # idempotent


def test_quarantine_removes_silent_multi_weighting_in_weights():
    """Pearson-Gewichte (voll neu berechnet) sehen jeden Outcome nur einmal."""
    base = [dict(_trade(f"T{i}", 100.0 + i, outcome=0.05 * (i % 5 - 2), close_date="2026-07-01"),
                 features={"impact": i % 7, "mismatch": (i * 3) % 5, "eps_drift": i % 4}) for i in range(40)]
    clean = {"closed_trades": copy.deepcopy(base)}
    dup = {"closed_trades": copy.deepcopy(base) + [copy.deepcopy(base[0]) for _ in range(6)]}
    ll.quarantine_duplicate_trades(dup)
    assert feedback.compute_pearson_weights(dup) == feedback.compute_pearson_weights(clean)


def test_already_closed_detects_duplicate_active_entry():
    state = {"closed_trades": [_trade(outcome=0.2)]}
    assert ll.already_closed(state, _trade()) and not ll.already_closed(state, _trade(strike=999.0))


def test_shadow_feature_stats_external_consumed_once():
    state = {}
    st = _trade("XYZ", outcome=0.1)
    for _ in range(3):
        ll.consume(state, "shadow:abc123", 45, "feature_stats_external",
                   lambda: feedback.update_feature_stats_external(state, st), sections=("feature_stats_external",))
    assert all(b["count"] == 1 for dim in state["feature_stats_external"].values() for b in dim.values())
    assert ll.run_stats()["duplicate_skips"] == 2


def test_ea_outcomes_and_evaluation_count_each_outcome_once(tmp_path):
    """Ledger-Outcomes: erster Eintrag je Schlüssel gilt; Evaluation zweimal = identisch (keine Akkumulation)."""
    from modules.expectation_alpha import evaluation as ev
    from modules.expectation_alpha import ledger as eal
    root = tmp_path / "ea"
    (root / "candidates").mkdir(parents=True)
    rows = [{"observation_id": f"o{i}", "date": f"2026-11-{1 + i % 25:02d}", "ticker": f"T{i}", "status": "TRADE",
             "group": "A", "stage": "EA_NEWS_CANDIDATE", "expressions": [{"expression": "UNDERLYING"}],
             "selected_expression": {"selected_expression": "UNDERLYING"}} for i in range(40)]
    eal.append_jsonl(root / "candidates" / "x.jsonl", rows)
    out = [{"observation_id": f"o{i}", "expression": "UNDERLYING", "variant": "immediate", "horizon": 60,
            "outcome": 0.01 * i, "outcome_net": 0.01 * i, "mfe": 0.0, "mae": 0.0, "max_drawdown": 0.0} for i in range(40)]
    dup = [{**o, "outcome_net": 9.9} for o in out[:10]]                        # spätere Duplikate (z. B. Retry)
    eal.append_jsonl(root / "outcomes.jsonl", out + dup)
    assert len(eal.read_outcomes(root)) == 40
    r1 = ev.run(root=root, promotion_state={}, write=False)
    r2 = ev.run(root=root, promotion_state={}, write=False)
    g1, g2 = r1["groups"]["60"]["A"], r2["groups"]["60"]["A"]
    assert g1 == g2 and g1["n"] == 40 and g1["mean"] == pytest.approx(0.195, abs=1e-6)
