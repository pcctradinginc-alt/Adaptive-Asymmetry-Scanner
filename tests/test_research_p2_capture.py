from datetime import date, datetime, timezone
from pathlib import Path

from modules import roi_learning_shadow as rl


def test_p2_due_capture_targets_preserve_spread_and_priority(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "roi_learning_shadow.yaml").write_text(
        (Path("config") / "roi_learning_shadow.yaml").read_text())
    decision = "2026-10-01T15:00:00+00:00"
    def cohort(name, day, legs):
        return {"cohort_id": name, "signal_date": day, "decision_at": decision,
                "legs": [{"occ_symbol": s} for s in legs]}
    rows = [cohort("old", "2026-10-01", ["A", "B"]),
            cohort("new", "2026-10-24", ["C", "D"])]
    monkeypatch.setattr(rl, "_journal_read",
                        lambda p: rows if p.name == "cohorts" else [])
    targets = rl.due_capture_targets(today=date(2026, 10, 26), root=tmp_path)
    assert list(targets) == ["A", "B", "C", "D"]
    assert targets["A"] == datetime(2026, 10, 1, 15, tzinfo=timezone.utc)


def test_p2_missing_decision_timestamp_skips_research_capture(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "roi_learning_shadow.yaml").write_text(
        (Path("config") / "roi_learning_shadow.yaml").read_text())
    monkeypatch.setattr(rl, "_journal_read", lambda p: [
        {"cohort_id": "x", "signal_date": "2026-10-25",
         "legs": [{"occ_symbol": "A"}]}] if p.name == "cohorts" else [])
    assert rl.due_capture_targets(today=date(2026, 10, 26), root=tmp_path) == {}
