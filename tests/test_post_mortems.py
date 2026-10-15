"""C3/C2: Post-mortems je geschlossenem Trade (deterministisch, PIT) und Wochenbericht-Block."""
from __future__ import annotations

from datetime import date

from modules import trade_memory as tm
from reports import weekly as w


def case(cid, entry, close, outcome, und, primary=None, labels=(), reliable=True, feats=None, direction=1):
    return {"case_id": cid, "source": "closed_trades", "ticker": cid.upper(), "entry_date": entry, "close_date": close,
            "holding_days": 5, "strategy": "LONG_CALL", "direction": direction, "outcome": outcome,
            "outcome_reliable": reliable, "underlying_return_signed": und, "max_upside": 0.1, "exit_reason": "x",
            "features": feats or {"impact": 5, "surprise": 1, "z_score": 2, "sigma_30d": 0.02},
            "model_predictions": {"sim_hit_rate": 0.7, "impact": 5},
            "failure": {"primary": primary, "labels": list(labels)} if primary else None}


def test_post_mortem_lessons_and_results():
    loss = tm.post_mortem(case("a", "2026-10-01", "2026-10-06", -0.5, -0.06, "signal_wrong", ["signal_wrong"]), [])
    assert loss["result"] == "LOSS" and loss["lesson"] == tm.LESSONS["signal_wrong"]
    assert loss["option_vs_underlying"] == round(-0.5 / -0.06, 2)
    win = tm.post_mortem(case("b", "2026-10-01", "2026-10-06", 0.4, 0.03), [])
    assert win["result"] == "WIN" and "Signalrichtung" in win["lesson"]
    win2 = tm.post_mortem(case("c", "2026-10-01", "2026-10-06", 0.4, -0.01), [])
    assert "nicht das Signal" in win2["lesson"]
    unrel = tm.post_mortem(case("d", "2026-10-01", "2026-10-06", -0.3, None, "unreliable_outcome", reliable=False), [])
    assert unrel["lesson"] == tm.LESSONS["unreliable_outcome"] and unrel["option_vs_underlying"] is None


def test_similar_cases_only_before_entry():
    past = [case(f"p{i}", "2026-08-01", "2026-08-10", -0.2 if i % 2 else 0.3, 0.0,
                 feats={"impact": 4 + i, "surprise": 1 + i, "z_score": 2 - i, "sigma_30d": 0.02 + i / 100})
            for i in range(4)]
    later = [case("late", "2026-09-20", "2026-10-20", 0.9, 0.1)]                    # nach dem Entry geschlossen
    q = case("q", "2026-10-01", "2026-10-06", -0.5, -0.06, "signal_wrong", ["signal_wrong"])
    pm = tm.post_mortem(q, past + later + [q])
    assert pm["similar_before_entry"]["n_pool"] == 4 and pm["similar_before_entry"]["win_share"] == 0.5


def test_post_mortems_only_closed_trades_sorted_and_rendered():
    cs = [case("x", "2026-09-01", "2026-09-05", 0.1, 0.02), case("y", "2026-09-02", "2026-09-04", -0.2, -0.05,
                                                                  "signal_wrong", ["signal_wrong"]),
          dict(case("s", "2026-09-01", "2026-09-03", 0.1, 0.0), source="shadow_trades")]
    pms = tm.post_mortems(cs)
    assert [p["ticker"] for p in pms] == ["Y", "X"]
    md = tm.render_post_mortems(pms)
    assert md.index("## X") < md.index("## Y") and "Lehre:" in md and "keine Signifikanzaussage" in md


def test_weekly_learning_progress_block():
    data = {"post_mortems_week": [tm.post_mortem(case("a", "2026-10-01", "2026-10-06", -0.5, -0.06, "signal_wrong",
                                                      ["signal_wrong"]), [])],
            "learning_eta": {"meta_label": {"have": 0, "need": 200, "eta": "2026-12-01"},
                             "rl_arming": {"have": 9, "need": 30, "eta": None}}}
    blocks = w.learning_progress_blocks(data)
    table = next(b for b in blocks if b[0] == "table")
    assert table[2][0][0] == "A" and table[2][0][2] == "LOSS" and table[3] == ["bad"]
    eta = next(b for b in blocks if b[0] == "list" and any("Meta-Label" in x for x in b[1]))
    assert "voraussichtlich ab 2026-12-01" in eta[1][0] and "nicht absehbar" in eta[1][1]
    empty = w.learning_progress_blocks({"post_mortems_week": [], "learning_eta": {}})
    assert ("list", ["keine geschlossenen Trades in dieser Woche"]) in empty
    assert w._recent("2026-10-08", date(2026, 10, 12)) and not w._recent("2026-10-01", date(2026, 10, 12))
