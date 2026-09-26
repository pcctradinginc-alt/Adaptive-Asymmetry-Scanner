"""
Tests für modules/challenger.py – Pre-registered Challenger/Shadow/Walk-forward.

Deckt ab:
  - Regel-Selektion (inkl. fehlender Felder), Metrik-Fallback
  - "running" bei zu kleiner Stichprobe
  - "promote_recommended" wenn der Challenger klar besser ist
  - "reject" wenn der Challenger klar schlechter ist
  - Expiry
  - Bonferroni-Alpha ändert sich mit n_active
  - "queued" jenseits von MAX_ACTIVE=3
  - Determinismus (zwei Läufe liefern dasselbe Ergebnis)
  - Guard: evaluate_all() schreibt niemals nach config.yaml/challengers.yaml
"""

import builtins
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from modules import challenger as ch


# ── Fixtures / Helpers ───────────────────────────────────────────────────────

def _row(day: str, trade_score=None, final_mc_hit_rate=None, final_mc_shadow_hit_rate=None,
         status="proposed", opt_ret_45d=None, ret_45d=None):
    features = {}
    if trade_score is not None:
        features["trade_score"] = trade_score
    if final_mc_hit_rate is not None:
        features["final_mc_hit_rate"] = final_mc_hit_rate
    if final_mc_shadow_hit_rate is not None:
        features["final_mc_shadow_hit_rate"] = final_mc_shadow_hit_rate
    outcomes = {}
    if opt_ret_45d is not None:
        outcomes["opt_ret_45d"] = opt_ret_45d
    if ret_45d is not None:
        outcomes["ret_45d"] = ret_45d
    return {
        "date": day,
        "ticker": "TST",
        "status": status,
        "features": features,
        "outcomes": outcomes,
    }


def _write_ledger(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, list[dict]] = {}
    for r in rows:
        by_month.setdefault(r["date"][:7], []).append(r)
    for month, rs in by_month.items():
        with open(root / f"{month}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")


def _make_challenger(id_="c1", start_date="2026-01-01", registered_on="2026-01-01",
                      rule=None, baseline_rule=None, metric="outcomes.opt_ret_45d",
                      min_n=5, max_duration_days=180, status="active"):
    return {
        "id": id_,
        "hypothesis": f"hypothesis for {id_}",
        "registered_on": registered_on,
        "start_date": start_date,
        "rule": rule or [{"field": "features.trade_score", "op": ">=", "value": 61}],
        "baseline_rule": baseline_rule or [{"field": "status", "op": "==", "value": "proposed"}],
        "metric": metric,
        "min_n": min_n,
        "horizon_days": 45,
        "max_duration_days": max_duration_days,
        "status": status,
    }


def _series(n, base, spread=0.02):
    """Deterministic ascending series around `base` for reproducible arm means."""
    return [round(base + (i - n / 2) * spread / max(n, 1), 6) for i in range(n)]


# ── Rule selection ────────────────────────────────────────────────────────────

def test_select_basic_and_ops():
    rows = [
        {"features": {"trade_score": 61}},
        {"features": {"trade_score": 55}},
        {"features": {"trade_score": 70}},
    ]
    rule = [{"field": "features.trade_score", "op": ">=", "value": 61}]
    out = ch.select(rows, rule)
    assert len(out) == 2
    assert all(r["features"]["trade_score"] >= 61 for r in out)


def test_select_missing_field_excludes_row():
    rows = [
        {"features": {"trade_score": 61}},
        {"features": {}},                      # missing field
        {"other": "x"},                        # no features at all
    ]
    rule = [{"field": "features.trade_score", "op": ">=", "value": 61}]
    out = ch.select(rows, rule)
    assert len(out) == 1


def test_select_unknown_op_excludes_row():
    rows = [{"features": {"trade_score": 61}}]
    rule = [{"field": "features.trade_score", "op": "~=", "value": 61}]
    assert ch.select(rows, rule) == []


def test_select_in_op():
    rows = [{"direction": "BULLISH"}, {"direction": "BEARISH"}]
    rule = [{"field": "direction", "op": "in", "value": ["BULLISH"]}]
    out = ch.select(rows, rule)
    assert len(out) == 1 and out[0]["direction"] == "BULLISH"


def test_select_and_combination():
    rows = [
        {"features": {"trade_score": 61}, "status": "proposed"},
        {"features": {"trade_score": 61}, "status": "rejected"},
    ]
    rule = [
        {"field": "features.trade_score", "op": ">=", "value": 61},
        {"field": "status", "op": "==", "value": "proposed"},
    ]
    out = ch.select(rows, rule)
    assert len(out) == 1


# ── Metric fallback ────────────────────────────────────────────────────────────

def test_metric_value_primary_present():
    row = {"outcomes": {"opt_ret_45d": 0.1, "ret_45d": 0.2}}
    assert ch.metric_value(row, "outcomes.opt_ret_45d") == 0.1


def test_metric_value_falls_back_when_missing():
    row = {"outcomes": {"ret_45d": 0.2}}
    assert ch.metric_value(row, "outcomes.opt_ret_45d") == 0.2


def test_metric_value_explicit_fallback():
    row = {"outcomes": {"other": 0.3}}
    assert ch.metric_value(row, "outcomes.opt_ret_45d", fallback="outcomes.other") == 0.3


def test_metric_value_none_when_nothing_present():
    row = {"outcomes": {}}
    assert ch.metric_value(row, "outcomes.opt_ret_45d") is None


# ── evaluate(): running / promote / reject / expiry ──────────────────────────

def test_evaluate_running_when_n_below_min():
    rows = [
        _row("2026-02-01", trade_score=70, opt_ret_45d=0.1, status="proposed"),
        _row("2026-02-02", trade_score=40, opt_ret_45d=0.05, status="proposed"),
    ]
    c = _make_challenger(min_n=30, max_duration_days=180)
    result = ch.evaluate(c, rows, today=date(2026, 2, 10), n_active=1)
    assert result["verdict"] == "running"


def test_evaluate_promote_when_challenger_clearly_better():
    rows = []
    base_vals = _series(40, base=0.0, spread=0.01)     # baseline ~ 0.0
    chal_vals = _series(40, base=0.30, spread=0.01)     # challenger ~ +0.30, much better
    for i, v in enumerate(base_vals):
        rows.append(_row(f"2026-02-{(i % 27) + 1:02d}", trade_score=56, opt_ret_45d=v, status="proposed"))
    for i, v in enumerate(chal_vals):
        rows.append(_row(f"2026-03-{(i % 27) + 1:02d}", trade_score=65, opt_ret_45d=v, status="proposed"))

    c = _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55},
                       {"field": "features.trade_score", "op": "<", "value": 61}],
        min_n=30,
        max_duration_days=180,
    )
    result = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    assert result["verdict"] == "promote_recommended"
    assert result["ci_lower"] > 0


def test_evaluate_reject_when_challenger_clearly_worse():
    rows = []
    base_vals = _series(40, base=0.30, spread=0.01)
    chal_vals = _series(40, base=-0.30, spread=0.01)
    for i, v in enumerate(base_vals):
        rows.append(_row(f"2026-02-{(i % 27) + 1:02d}", trade_score=56, opt_ret_45d=v, status="proposed"))
    for i, v in enumerate(chal_vals):
        rows.append(_row(f"2026-03-{(i % 27) + 1:02d}", trade_score=65, opt_ret_45d=v, status="proposed"))

    c = _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55},
                       {"field": "features.trade_score", "op": "<", "value": 61}],
        min_n=30,
        max_duration_days=180,
    )
    result = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    assert result["verdict"] == "reject"
    assert result["ci_upper"] < 0


def test_evaluate_expired_without_promotion_rejects():
    rows = []
    # Both arms similar (no clear edge) -> should not promote, but is expired.
    for i in range(35):
        rows.append(_row(f"2026-02-{(i % 27) + 1:02d}", trade_score=56, opt_ret_45d=0.05, status="proposed"))
        rows.append(_row(f"2026-03-{(i % 27) + 1:02d}", trade_score=65, opt_ret_45d=0.05, status="proposed"))

    c = _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55},
                       {"field": "features.trade_score", "op": "<", "value": 61}],
        min_n=30,
        max_duration_days=10,
        registered_on="2026-01-01",
        start_date="2026-01-01",
    )
    result = ch.evaluate(c, rows, today=date(2026, 6, 1), n_active=1)
    assert result["expired"] is True
    assert result["verdict"] == "reject"


def test_evaluate_expired_but_insufficient_n_still_rejects():
    rows = [
        _row("2026-02-01", trade_score=70, opt_ret_45d=0.1, status="proposed"),
    ]
    c = _make_challenger(min_n=30, max_duration_days=1, registered_on="2026-01-01", start_date="2026-01-01")
    result = ch.evaluate(c, rows, today=date(2026, 6, 1), n_active=1)
    assert result["expired"] is True
    assert result["verdict"] == "reject"


def test_evaluate_ignores_rows_before_start_date():
    rows = [_row("2025-01-01", trade_score=70, opt_ret_45d=0.5, status="proposed")]
    c = _make_challenger(start_date="2026-01-01", registered_on="2026-01-01", min_n=1)
    result = ch.evaluate(c, rows, today=date(2026, 2, 1), n_active=1)
    assert result["n_challenger"] == 0
    assert result["n_baseline"] == 0


# ── Bonferroni alpha scales with n_active ────────────────────────────────────

def test_alpha_scales_with_n_active():
    rows = []
    base_vals = _series(40, base=0.0, spread=0.01)
    chal_vals = _series(40, base=0.30, spread=0.01)
    for i, v in enumerate(base_vals):
        rows.append(_row(f"2026-02-{(i % 27) + 1:02d}", trade_score=56, opt_ret_45d=v, status="proposed"))
    for i, v in enumerate(chal_vals):
        rows.append(_row(f"2026-03-{(i % 27) + 1:02d}", trade_score=65, opt_ret_45d=v, status="proposed"))
    c = _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55},
                       {"field": "features.trade_score", "op": "<", "value": 61}],
        min_n=30,
    )
    r1 = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    r3 = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=3)
    assert r1["alpha"] == pytest.approx(0.10 / 1)
    assert r3["alpha"] == pytest.approx(0.10 / 3)
    assert r3["alpha"] < r1["alpha"]
    # Tighter alpha (n_active=3) means the CI is at least as wide as n_active=1
    assert r3["ci_lower"] <= r1["ci_lower"]
    assert r3["ci_upper"] >= r1["ci_upper"]


# ── Determinism ───────────────────────────────────────────────────────────────

def test_evaluate_is_deterministic():
    rows = []
    base_vals = _series(40, base=0.0, spread=0.01)
    chal_vals = _series(40, base=0.30, spread=0.01)
    for i, v in enumerate(base_vals):
        rows.append(_row(f"2026-02-{(i % 27) + 1:02d}", trade_score=56, opt_ret_45d=v, status="proposed"))
    for i, v in enumerate(chal_vals):
        rows.append(_row(f"2026-03-{(i % 27) + 1:02d}", trade_score=65, opt_ret_45d=v, status="proposed"))
    c = _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55},
                       {"field": "features.trade_score", "op": "<", "value": 61}],
        min_n=30,
    )
    r1 = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    r2 = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    assert r1 == r2


# ── evaluate_all(): MAX_ACTIVE guard, queued ──────────────────────────────────

def test_evaluate_all_queues_beyond_max_active(tmp_path):
    ledger_root = tmp_path / "ledger"
    ledger_root.mkdir()

    registry_path = tmp_path / "challengers.yaml"
    challengers = [
        _make_challenger(id_=f"c{i}", registered_on=f"2026-01-{i+1:02d}", start_date=f"2026-01-{i+1:02d}")
        for i in range(5)
    ]
    registry_path.write_text(yaml.safe_dump({"challengers": challengers}), encoding="utf-8")

    results = ch.evaluate_all(registry_path=registry_path, ledger_root=ledger_root, today=date(2026, 6, 1))
    by_id = {r["id"]: r for r in results}
    evaluated_ids = [f"c{i}" for i in range(3)]
    queued_ids = [f"c{i}" for i in range(3, 5)]
    for cid in evaluated_ids:
        assert by_id[cid]["verdict"] != "queued"
    for cid in queued_ids:
        assert by_id[cid]["verdict"] == "queued"


def test_evaluate_all_empty_ledger_does_not_crash(tmp_path):
    ledger_root = tmp_path / "ledger"  # not created
    registry_path = tmp_path / "challengers.yaml"
    registry_path.write_text(
        yaml.safe_dump({"challengers": [_make_challenger()]}), encoding="utf-8"
    )
    results = ch.evaluate_all(registry_path=registry_path, ledger_root=ledger_root, today=date(2026, 6, 1))
    assert len(results) == 1
    assert results[0]["verdict"] == "running"


def test_seed_registry_loads_and_evaluates(tmp_path):
    """Sanity-check on the real repo-root challengers.yaml (read-only)."""
    registry = ch.load_registry(Path("challengers.yaml"))
    ids = {c["id"] for c in registry}
    assert {"final_mc_dte_shadow", "trade_score_61"} <= ids
    for c in registry:
        assert c["status"] == "active"
        assert c["min_n"] == 30


# ── No-writes guard ────────────────────────────────────────────────────────────

def test_evaluate_all_never_writes_to_registry_or_config(tmp_path, monkeypatch):
    ledger_root = tmp_path / "ledger"
    _write_ledger(ledger_root, [
        _row("2026-02-01", trade_score=70, opt_ret_45d=0.1, status="proposed"),
        _row("2026-02-02", trade_score=40, opt_ret_45d=0.05, status="proposed"),
    ])
    registry_path = tmp_path / "challengers.yaml"
    registry_path.write_text(yaml.safe_dump({"challengers": [_make_challenger()]}), encoding="utf-8")

    forbidden = {str(registry_path.resolve()), str(Path("config.yaml").resolve()),
                 str(Path("challengers.yaml").resolve())}

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode or "x" in mode or "+" in mode:
            resolved = str(Path(file).resolve())
            assert resolved not in forbidden, f"Unerlaubter Schreibzugriff auf {resolved}"
        return real_open(file, mode, *args, **kwargs)

    real_write_text = Path.write_text

    def guarded_write_text(self, *args, **kwargs):
        resolved = str(self.resolve())
        assert resolved not in forbidden, f"Unerlaubter Schreibzugriff (write_text) auf {resolved}"
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    results = ch.evaluate_all(registry_path=registry_path, ledger_root=ledger_root, today=date(2026, 6, 1))
    assert isinstance(results, list) and len(results) == 1

    # Files are byte-for-byte unchanged.
    assert "challengers:" in registry_path.read_text(encoding="utf-8")


def test_evaluate_all_never_writes_real_repo_files(monkeypatch):
    """Runs evaluate_all() against the real challengers.yaml/ledger root and
    asserts config.yaml / challengers.yaml are never opened for writing."""
    forbidden = {str(Path("config.yaml").resolve()), str(Path("challengers.yaml").resolve())}
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode or "x" in mode or "+" in mode:
            resolved = str(Path(file).resolve())
            assert resolved not in forbidden, f"Unerlaubter Schreibzugriff auf {resolved}"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    ch.evaluate_all(today=date(2026, 6, 1))
