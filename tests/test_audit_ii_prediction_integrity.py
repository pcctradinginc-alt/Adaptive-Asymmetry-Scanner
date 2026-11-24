import json
import pandas as pd
import pytest
from modules import prediction_memory as pm
from modules.forward_cohort_audit import audit


def test_pending_distinct_from_invalid():
    result = audit([{"prediction_id":"a","signal_date":"2026-11-02","actual_return":None},
                    {"prediction_id":"b","signal_date":"2026-11-02","actual_return":float("nan")}])
    assert result["pending_outcome_rows"] == 1
    assert result["invalid_or_conflicting_rows"] == 1
    assert result["duplicate_rows"] == 0


def test_immutable_conflict_fails_before_join(tmp_path):
    a = {"type":"prediction","prediction_id":"a","signal_date":"2026-11-02","ticker":"A"}
    (tmp_path/"2026-11.jsonl").write_text(json.dumps(a)+"\n"+json.dumps({**a,"ticker":"B"})+"\n")
    with pytest.raises(ValueError, match="conflicting immutable"):
        pm.load_memory(tmp_path)


def test_identical_event_deduplicated(tmp_path):
    a = {"type":"prediction","prediction_id":"a","signal_date":"2026-11-02","ticker":"A"}
    (tmp_path/"2026-11.jsonl").write_text((json.dumps(a)+"\n")*2)
    assert len(pm.load_memory(tmp_path)) == 1


def test_conflicting_panel_labels_never_book(tmp_path):
    pm.record_predictions([{"signal_date":"2026-11-02","ticker":"A"}],tmp_path)
    rows = [{"date":pd.Timestamp("2026-11-02"),"ticker":"A","fwd_xs_20":x} for x in (0.1,-0.1)]
    assert pm.record_outcomes(pd.DataFrame(rows),tmp_path)==0
    assert pm.record_outcomes(pd.DataFrame(rows[::-1]),tmp_path)==0


def test_corrupt_event_does_not_silently_disappear(tmp_path):
    (tmp_path/"2026-11.jsonl").write_text("{broken\n")
    with pytest.raises(ValueError,match="corrupt JSON"):
        pm.load_memory(tmp_path)
