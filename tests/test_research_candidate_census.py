import json
from modules.research_candidate_census import audit, run


def test_all_candidates_including_budget_deferrals_are_counted():
    rows = [
        {"date": "2026-10-29", "ticker": "AAA", "reject_reason": "economics_first_quote_probe_budget", "outcomes": {"ret_5d": 0.1}},
        {"date": "2026-10-29", "ticker": "BBB", "reject_reason": "quick_mc", "outcomes": {"real_strat_ret_5d": -0.2}},
        {"date": "2026-10-28", "ticker": "CCC"},
    ]
    a = audit(rows, "2026-10-29")
    assert a["signals"] == 2 and a["unique_tickers"] == 2
    assert a["underlying_outcome_rows"] == 1
    assert a["actual_option_outcome_rows"] == 1
    assert a["reasons"]["economics_first_quote_probe_budget"] == 1
    assert audit(list(reversed(rows)), "2026-10-29")["audit_sample"] == a["audit_sample"]


def test_missing_marks_never_fabricated(tmp_path):
    root = tmp_path / "ledger"
    root.mkdir()
    (root / "2026-10.jsonl").write_text(json.dumps({
        "date": "2026-10-29", "ticker": "AAA", "outcomes": {}
    }) + "\n")
    target = run("2026-10-29", root=root, out=tmp_path / "reports")
    result = json.loads(target.read_text())
    assert result["actual_option_outcome_rows"] == 0
    assert result["missing_option_marks_are_not_zero_returns"]
