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
  - Policy-Bootstrap auf genesteten/überlappenden Armen (kein False-Promote,
    schmaleres CI als die alte unabhängige Resampling-Methode, Determinismus)
  - Pre-Registrierungs-Guard: start_date strikt nach registered_on,
    registered_at gate über signal_timestamp
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
         status="proposed", opt_ret_45d=None, ret_45d=None, signal_timestamp=None):
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
    row = {
        "date": day,
        "ticker": "TST",
        "status": status,
        "features": features,
        "outcomes": outcomes,
    }
    if signal_timestamp is not None:
        row["signal_timestamp"] = signal_timestamp
    return row


def _write_ledger(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, list[dict]] = {}
    for r in rows:
        by_month.setdefault(r["date"][:7], []).append(r)
    for month, rs in by_month.items():
        with open(root / f"{month}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")


def _make_challenger(id_="c1", start_date="2026-01-02", registered_on="2026-01-01",
                      registered_at=None,
                      rule=None, baseline_rule=None, metric="outcomes.opt_ret_45d",
                      min_n=5, max_duration_days=180, status="active"):
    return {
        "id": id_,
        "hypothesis": f"hypothesis for {id_}",
        "registered_on": registered_on,
        "registered_at": registered_at,
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


def test_metric_value_no_implicit_stock_fallback():
    # Options- und Aktienrenditen dürfen nicht stillschweigend gemischt werden
    row = {"outcomes": {"ret_45d": 0.2}}
    assert ch.metric_value(row, "outcomes.opt_ret_45d") is None


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
        start_date="2026-01-02",
    )
    result = ch.evaluate(c, rows, today=date(2026, 6, 1), n_active=1)
    assert result["expired"] is True
    assert result["verdict"] == "reject"


def test_evaluate_expired_but_insufficient_n_still_rejects():
    rows = [
        _row("2026-02-01", trade_score=70, opt_ret_45d=0.1, status="proposed"),
    ]
    c = _make_challenger(min_n=30, max_duration_days=1, registered_on="2026-01-01", start_date="2026-01-02")
    result = ch.evaluate(c, rows, today=date(2026, 6, 1), n_active=1)
    assert result["expired"] is True
    assert result["verdict"] == "reject"


def test_evaluate_ignores_rows_before_start_date():
    rows = [_row("2025-01-01", trade_score=70, opt_ret_45d=0.5, status="proposed")]
    c = _make_challenger(start_date="2026-01-02", registered_on="2026-01-01", min_n=1)
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
        _make_challenger(id_=f"c{i}", registered_on=f"2026-01-{i+1:02d}", start_date=f"2026-01-{i+2:02d}")
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


# ── Policy (paired) bootstrap on nested/overlapping arms ────────────────────
#
# The registered baseline_rule/rule pairs in challengers.yaml are NESTED
# (e.g. baseline score>=55 contains all challenger score>=61 rows). The old
# implementation resampled both arms *independently*, ignoring the shared
# rows between them and thereby overestimating the variance of the
# difference of means (an artificially wide CI). The current implementation
# resamples the eligible population once per replicate and re-applies both
# rules to that same replicate ("policy bootstrap"), preserving the
# covariance between the two overlapping arms.

def _old_independent_bootstrap_diff_ci(challenger_vals, baseline_vals, alpha, n_boot, seed):
    """The PRE-FIX independent-resampling bootstrap, kept here only for
    comparison in tests. Not used anywhere in production code."""
    import random
    import statistics as _stats

    rng = random.Random(seed)
    diffs = []
    nc, nb = len(challenger_vals), len(baseline_vals)
    for _ in range(n_boot):
        c_sample = [challenger_vals[rng.randrange(nc)] for _ in range(nc)]
        b_sample = [baseline_vals[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(_stats.mean(c_sample) - _stats.mean(b_sample))
    diffs.sort()
    lower_idx = max(0, min(n_boot - 1, int(n_boot * alpha)))
    upper_idx = max(0, min(n_boot - 1, int(n_boot * (1 - alpha))))
    return diffs[lower_idx], diffs[upper_idx]


def _nested_rows(n_base_only=30, n_shared=30, base_val=0.05, spread=0.02, better_shared=None):
    """Builds nested-arm rows: baseline_rule (score>=55) matches BOTH groups;
    rule (score>=61) matches only the 'shared' group. Optionally makes the
    shared group's values clearly better than the base-only group."""
    rows = []
    base_only_vals = _series(n_base_only, base=base_val, spread=spread)
    shared_vals = _series(n_shared, base=better_shared if better_shared is not None else base_val, spread=spread)
    for i, v in enumerate(base_only_vals):
        rows.append(_row(f"2026-02-{(i % 27) + 1:02d}", trade_score=56, opt_ret_45d=v, status="proposed"))
    for i, v in enumerate(shared_vals):
        rows.append(_row(f"2026-03-{(i % 27) + 1:02d}", trade_score=65, opt_ret_45d=v, status="proposed"))
    return rows


def _nested_challenger(min_n=25):
    return _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55}],
        min_n=min_n,
        max_duration_days=180,
    )


def test_paired_bootstrap_no_false_promote_on_identical_nested_distribution():
    """(a) Challenger subset drawn from the SAME distribution as the rest of
    the (nested) baseline arm: the paired CI must contain 0 (no false
    promote), and must be narrower than the old independent-resampling CI
    on the same data."""
    rows = _nested_rows(n_base_only=30, n_shared=30, base_val=0.05, spread=0.02)
    c = _nested_challenger(min_n=25)

    result = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    assert result["verdict"] != "promote_recommended"
    assert result["ci_lower"] <= 0 <= result["ci_upper"]

    challenger_vals = [ch.metric_value(r, "outcomes.opt_ret_45d") for r in ch.select(rows, c["rule"])]
    baseline_vals = [ch.metric_value(r, "outcomes.opt_ret_45d") for r in ch.select(rows, c["baseline_rule"])]
    alpha = ch.ALPHA_BASE / 1
    seed = ch._deterministic_seed(c["id"])
    old_lower, old_upper = _old_independent_bootstrap_diff_ci(challenger_vals, baseline_vals, alpha, ch.N_BOOT, seed)

    paired_width = result["ci_upper"] - result["ci_lower"]
    old_width = old_upper - old_lower
    assert paired_width < old_width


def test_paired_bootstrap_promotes_when_nested_subset_clearly_better():
    """(b) Clearly better nested subset -> promote."""
    rows = _nested_rows(n_base_only=30, n_shared=30, base_val=0.0, spread=0.01, better_shared=0.30)
    c = _nested_challenger(min_n=25)

    result = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    assert result["verdict"] == "promote_recommended"
    assert result["ci_lower"] > 0


def test_paired_bootstrap_is_deterministic():
    """(c) Determinism: two evaluations of the same nested data yield the
    exact same CI/verdict."""
    rows = _nested_rows(n_base_only=30, n_shared=30, base_val=0.0, spread=0.01, better_shared=0.30)
    c = _nested_challenger(min_n=25)

    r1 = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    r2 = ch.evaluate(c, rows, today=date(2026, 4, 1), n_active=1)
    assert r1 == r2


# ── Pre-registration guard (registered_on / registered_at) ──────────────────

def test_load_registry_marks_invalid_when_start_date_not_after_registered_on(tmp_path):
    registry_path = tmp_path / "challengers.yaml"
    c = _make_challenger(registered_on="2026-01-05", start_date="2026-01-05")  # equal -> invalid
    registry_path.write_text(yaml.safe_dump({"challengers": [c]}), encoding="utf-8")
    registry = ch.load_registry(registry_path)
    assert registry[0]["_invalid_reason"] is not None


def test_load_registry_valid_entry_has_no_invalid_reason():
    c = _make_challenger(registered_on="2026-01-01", start_date="2026-01-02")
    assert ch._validate_challenger(c) is None


def test_evaluate_all_reports_invalid_verdict_and_excludes_from_n_active(tmp_path):
    ledger_root = tmp_path / "ledger"
    registry_path = tmp_path / "challengers.yaml"

    invalid_c = _make_challenger(id_="bad", registered_on="2026-01-05", start_date="2026-01-04")  # before -> invalid
    good_c = _make_challenger(id_="good", registered_on="2026-01-01", start_date="2026-01-02", min_n=1)
    registry_path.write_text(yaml.safe_dump({"challengers": [invalid_c, good_c]}), encoding="utf-8")

    _write_ledger(ledger_root, [
        _row("2026-02-01", trade_score=70, opt_ret_45d=0.1, status="proposed"),
        _row("2026-02-02", trade_score=40, opt_ret_45d=0.05, status="proposed"),
    ])

    results = ch.evaluate_all(registry_path=registry_path, ledger_root=ledger_root, today=date(2026, 6, 1))
    by_id = {r["id"]: r for r in results}

    assert by_id["bad"]["verdict"] == "invalid"
    assert by_id["bad"]["invalid_reason"]
    assert by_id["bad"]["n_baseline"] is None

    # n_active only counts the valid challenger ("good") -> alpha = 0.10 / 1
    assert by_id["good"]["alpha"] == pytest.approx(0.10 / 1)


def test_registered_at_gates_rows_by_signal_timestamp():
    """Rows carrying signal_timestamp are gated purely by comparing that
    timestamp to registered_at (ignoring `date`); rows without a
    signal_timestamp fall back to the date >= start_date check."""
    rows = []
    registered_at = "2026-09-26T07:00:00Z"

    # Group A: signal_timestamp BEFORE registered_at, date AFTER start_date
    # -> must be excluded (timestamp check wins over date).
    for i in range(30):
        rows.append(_row("2026-09-29", trade_score=65, opt_ret_45d=0.5,
                          signal_timestamp="2026-09-26T06:00:00Z"))

    # Group B: signal_timestamp AFTER registered_at, date BEFORE start_date
    # -> must be included (timestamp check wins over date).
    for i in range(30):
        rows.append(_row("2026-09-01", trade_score=65, opt_ret_45d=0.5,
                          signal_timestamp="2026-09-27T00:00:00Z"))

    # Group C: no signal_timestamp, date AFTER start_date -> included via
    # the date fallback (baseline-only, trade_score below the challenger rule).
    for i in range(30):
        rows.append(_row("2026-09-29", trade_score=56, opt_ret_45d=0.5))

    # Group D: no signal_timestamp, date BEFORE start_date -> excluded via
    # the date fallback.
    for i in range(30):
        rows.append(_row("2026-09-01", trade_score=56, opt_ret_45d=0.5))

    c = _make_challenger(
        rule=[{"field": "features.trade_score", "op": ">=", "value": 61}],
        baseline_rule=[{"field": "features.trade_score", "op": ">=", "value": 55}],
        min_n=1,
        registered_on="2026-09-26",
        registered_at=registered_at,
        start_date="2026-09-28",
    )
    result = ch.evaluate(c, rows, today=date(2026, 10, 1), n_active=1)
    assert result["n_challenger"] == 30       # only Group B
    assert result["n_baseline"] == 60          # Group B + Group C
