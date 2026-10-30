from datetime import date, timedelta

from modules.research_evidence_discovery import discover, eligible


def row(day, ticker, value, ret=None):
    return {"date": day, "ticker": ticker, "features": {"momentum": value},
            "outcomes": {} if ret is None else {"real_strat_ret_20d": ret}}


def test_never_substitute_stock_returns_for_options():
    r = row("2026-08-01", "AAPL", 1)
    r["outcomes"]["ret_20d"] = 0.8
    assert not eligible(r, date(2026, 10, 30))
    out = discover([r], "2026-10-30")
    assert out["genuine_option_labels"] == 0
    assert out["missing_or_immature_option_labels"] == 1


def test_immature_option_mark_not_used():
    assert not eligible(row("2026-10-25", "AAPL", 1, 0.5), date(2026, 10, 30))


def test_insufficient_evidence_blocks_discovery():
    out = discover([row("2026-08-01", "AAPL", 1, 0.2)], "2026-10-30")
    assert out["status"] == "INSUFFICIENT_EVIDENCE"
    assert out["discoveries"] == []
    assert out["production_influence"] == "NONE"


def test_holdout_can_disagree_with_training_without_promotion():
    rows = []
    start = date(2026, 1, 1)
    for i in range(70):
        day = (start + timedelta(days=i)).isoformat()
        for j in range(4):
            signal = j % 2
            # Training relation is positive, but late holdout reverses.
            ret = (0.2 if signal else -0.1) if i < 49 else (-0.2 if signal else 0.1)
            rows.append(row(day, f"T{j}", signal, ret))
    out = discover(rows, "2026-10-30")
    assert out["status"] == "HOLDOUT_OBSERVED"
    assert out["discoveries"][0]["train_delta"] > 0
    assert out["discoveries"][0]["holdout_delta"] < 0
    assert out["discoveries"][0]["promotion_eligible"] is False


def test_deduplicate_same_signal():
    rows = [row("2026-08-01", "AAPL", 1, 0.2)] * 3
    assert discover(rows, "2026-10-30")["genuine_option_labels"] == 1
