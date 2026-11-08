"""FD3 Phase-2 validation: immutable paired cohorts, placeholders never count as alpha."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json

import pytest
import numpy as np

from modules import research_feature_discovery as fd
from modules import research_feature_validation as v


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item, sort_keys=True) + "\n")


def fixture(root: Path, *, windows: int = 8, n: int = 35,
            as_of: date = date(2026, 11, 8)):
    p, digest = fd.load_protocol()
    (root / fd.REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / fd.REPORT).write_text(json.dumps({
        "protocol_sha256": digest, "protocol_version": p["version"],
        "hypotheses": {h["id"]: {} for h in p["hypotheses"]},
    }))
    preds, labels = [], []
    start = date(2025, 2, 3)
    for m in range(windows):
        d = start + timedelta(days=35 * m)
        matured = d + timedelta(days=28)
        for i in range(n):
            tk = f"S{i:03d}"
            y = (0.1 if i >= n - 5 else -0.02) + .003 * (m % 2)
            b = float((i * 7 + 13) % n) / n
            c = float(i) / n
            for h in p["hypotheses"]:
                preds.append({"protocol_sha256": digest, "hypothesis_id": h["id"],
                              "date": d.isoformat(), "ticker": tk,
                              "frozen_at": (d + timedelta(days=1)).isoformat() + "T00:00:00+00:00",
                              "baseline_pred": b, "challenger_pred": c,
                              "research_only": True, "label": "FUTURE_UNOBSERVED"})
            labels.append({"protocol_sha256": digest, "date": d.isoformat(),
                           "ticker": tk, "label_end": matured.isoformat(),
                           "fwd_xs_20": y,
                           "recorded_at": (matured + timedelta(days=1)).isoformat()})
    _write(root / fd.JOURNAL, preds)
    _write(root / fd.OUTCOMES, labels)
    return p, digest


def test_no_upstream_never_claims_alpha(tmp_path):
    r = v.run(root=tmp_path, now=datetime(2026, 11, 8, tzinfo=timezone.utc))
    assert r["status"] == "MISSING_FD3_SOURCE"
    assert r["new_looks"] == 0


def test_8_independent_looks_are_frozen_and_idempotent(tmp_path):
    p, digest = fixture(tmp_path, windows=8)
    asof = datetime(2026, 11, 8, tzinfo=timezone.utc)
    a = v.run(root=tmp_path, now=asof)
    assert a["new_looks"] == 4
    assert a["status"] == "DIAGNOSTIC_SHADOW_ONLY"
    assert a["global_alpha_ledger_slot_allocated"] is False
    assert a["option_economics_verified"] is False
    for h in p["hypotheses"]:
        z = a["hypotheses"][h["id"]]
        assert z["independent_forward_windows"] == 8
        assert z["confirmed_alpha"] is False
        assert len(z["looks"]) == 1
        assert z["looks"][0]["look_n"] == 8
        assert z["looks"][0]["local_bonferroni_diagnostic_p"] >= 0
    b = v.run(root=tmp_path, now=asof)
    assert b["new_looks"] == 0
    records, _ = v.read_look_journal(tmp_path / "paired_looks.jsonl")
    assert len(records) == 4


def test_fewer_than_8_dates_is_not_a_test(tmp_path):
    fixture(tmp_path, windows=7)
    result = v.run(root=tmp_path, now=datetime(2026, 11, 8, tzinfo=timezone.utc))
    assert result["new_looks"] == 0
    assert all(vv["status"] == "PENDING_INDEPENDENT_WINDOWS"
               for vv in result["hypotheses"].values())


def test_duplicate_prediction_or_label_fails_closed(tmp_path):
    fixture(tmp_path, windows=1)
    original = v._read_jsonl(tmp_path / fd.JOURNAL) if hasattr(v, "_read_jsonl") else fd._read_jsonl(tmp_path / fd.JOURNAL)
    with (tmp_path / fd.JOURNAL).open("a") as f:
        f.write(json.dumps(original[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate immutable prediction"):
        v.run(root=tmp_path, now=datetime(2026, 11, 8, tzinfo=timezone.utc))


def test_tampered_look_journal_refuses_re_evaluation(tmp_path):
    fixture(tmp_path, windows=8)
    v.run(root=tmp_path, now=datetime(2026, 11, 8, tzinfo=timezone.utc))
    journal = tmp_path / "paired_looks.jsonl"
    text = journal.read_text()
    assert "record_hash" in text
    journal.write_text(text.replace("NO_AUTOMATIC_PROMOTION", "AUTOMATIC_PROMOTION", 1))
    with pytest.raises(ValueError, match="integrity"):
        v.run(root=tmp_path, now=datetime(2026, 11, 8, tzinfo=timezone.utc))


def test_future_unmatured_labels_cannot_enter_look(tmp_path):
    p, digest = fixture(tmp_path, windows=8)
    data = fd._read_jsonl(tmp_path / fd.OUTCOMES)
    for row in data:
        if row["date"] > "2025-09-01":
            row["label_end"] = "2027-01-01"
    _write(tmp_path / fd.OUTCOMES, data)
    result = v.run(root=tmp_path, now=datetime(2026, 11, 8, tzinfo=timezone.utc))
    assert result["new_looks"] == 0


def test_no_signal_is_a_valid_negative_control():
    _, digest = fd.load_protocol()
    data = []
    for i in range(35):
        data.append((f"X{i}", float(i), float(i), float(i), "2026-02-01"))
    dated = {"2026-01-01": data}
    p, _ = fd.load_protocol()
    window = v.independent_days(dated, p)
    assert len(window) == 1
    placebo = v._null_placebo(window, 40, np.random.default_rng(11))
    assert placebo["observed_delta"] == 0
    assert placebo["placebo_p_two_sided"] == 1.0


def test_no_production_import_in_modules():
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline.py").read_text()
    assert "research_feature_validation" not in source
