from datetime import date
from modules import research_quote_reconciliation as rq


def test_p0_orphan_marks_are_not_learning_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(rq.rl, "_journal_read", lambda p: [] if p.name == "cohorts" else [
        {"cohort_id": "missing", "horizon_days": 20}])
    report = rq.reconcile(tmp_path, date(2026, 10, 30))
    assert report["counts"]["orphan_or_invalid_mark"] == 1
    assert report["verified_research_outcomes"] == []
    assert report["closed_loop_verified"] is False


def test_p0_missing_archived_quotes_fail_closed(monkeypatch, tmp_path):
    cohort = {"cohort_id": "a", "signal_date": "2026-09-01", "ticker": "ABC",
              "legs": [{"occ_symbol": "ABC261120C00100000"}]}
    mark = {"cohort_id": "a", "horizon_days": 20,
            "entry_quote_ids": ["one"], "mark_quote_ids": ["two"]}
    monkeypatch.setattr(rq.rl, "_journal_read",
                        lambda p: [cohort] if p.name == "cohorts" else [mark])
    monkeypatch.setattr(rq.oq, "by_ids", lambda *a, **kw: {})
    report = rq.reconcile(tmp_path, date(2026, 10, 30))
    assert report["counts"]["missing_or_unusable_quotes"] == 1
    assert not report["verified_research_outcomes"]
