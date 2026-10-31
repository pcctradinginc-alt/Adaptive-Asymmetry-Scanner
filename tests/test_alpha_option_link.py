"""Alpha-to-Options link (Phase A): zero-network tests.

Fixtures are synthetic quotes/predictions that exercise integrity rules. They
are NEVER evidence of economic success.
"""
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from modules import market_snapshot as ms
from modules import option_quotes as oq
from modules.autonomous_alpha import forward_options
from modules.autonomous_alpha import option_link as al
from modules.autonomous_alpha.registry import EvidenceIntegrityError

REPO = Path(__file__).resolve().parents[1]
NOW_REG = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)        # Saturday research run
PRED_DAY = "2026-11-06"
CREATED = "2026-11-07T14:45:00+00:00"
DECIDE = datetime(2026, 11, 9, 16, 0, tzinfo=timezone.utc)          # Monday 11:00 ET, in mark window
TICKERS = [a + b + c for a in "ABCDE" for b in "FGH" for c in "JKLM"]  # 60 names


def _setup(root: Path, **cfg_over):
    (root / "config").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((REPO / al.CONFIG).read_text())
    for k, v in cfg_over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    (root / al.CONFIG).write_text(yaml.safe_dump(cfg))
    shutil.copy(REPO / al.ML_REGISTRY, root / al.ML_REGISTRY)
    log = {"momentum_12_1": {"locked_result": {"status": "ok", "evaluated_at": "2026-10-02T21:11:56+00:00",
                                               "base": {"mean": 0.02}}},
           "hgb_xs20_v1": {"locked_result": {"status": "ok", "evaluated_at": "2026-10-02T21:12:00+00:00",
                                             "base": {"mean": 0.004}}}}
    (root / al.ML_REGISTRY_LOG).parent.mkdir(parents=True, exist_ok=True)
    (root / al.ML_REGISTRY_LOG).write_text(json.dumps(log))
    return cfg


def _spec(root, mid):
    return al._ml_registry(root)[mid]


def _prediction(root, mid, *, created=CREATED, day=PRED_DAY, reverse=False, spec_hash=None):
    ranks = {t: round((i + 1) / len(TICKERS), 4) for i, t in enumerate(sorted(TICKERS, reverse=reverse))}
    row = {"prediction_date": day, "model_id": mid, "spec_hash": spec_hash or al.ml_spec_hash(_spec(root, mid)),
           "train_cutoff": "2026-10-08", "params": {}, "code_sha": "abc123", "features_hash": "f00",
           "created_at": created, "n": len(ranks), "rank_pct": ranks, "realized": None}
    d = root / al.ML_PRED_DIR
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{day[:7]}.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _panel():
    rows = []
    for day, vix, mom in (("2026-11-06", 18.0, 0.1), ("2026-11-09", 35.0, 9.9)):
        for t in TICKERS:
            rows.append({"date": pd.Timestamp(day), "ticker": t, "mom_12_1": mom, "vix": vix,
                         "spy_trend_200": 0.05, "fwd_xs_20": 0.5, "label_end_20": pd.Timestamp("2026-12-07")})
    return pd.DataFrame(rows)


def _registered(tmp_path, **cfg_over):
    _setup(tmp_path, **cfg_over)
    _prediction(tmp_path, "momentum_12_1")
    _prediction(tmp_path, "hgb_xs20_v1", reverse=True)
    res = al.register(tmp_path, panel=_panel(), now=NOW_REG)
    return res, al._read_kind(tmp_path, "signals")[0]


# ── registration ─────────────────────────────────────────────────────────────

def test_register_freezes_signals_with_full_pit_context(tmp_path):
    res, sigs = _registered(tmp_path)
    assert res["n_new_signals"] == len(sigs) == 2 * 6 + 5
    top = sorted((s for s in sigs if s["hypothesis_id"] == "momentum_12_1" and s["role"] == "TOP"),
                 key=lambda s: s["rank_position"])
    assert [s["ticker"] for s in top] == sorted(TICKERS)[-1:-4:-1]            # best rank first
    s = top[0]
    assert s["comparison_group"] == "REFERENCE" and s["direction"] == "BULLISH" and s["tradable"]
    assert s["hypothesis_version"] == al.ml_spec_hash(_spec(tmp_path, "momentum_12_1"))
    assert s["prediction_ref"]["prediction_date"] == PRED_DAY and s["signal_time"] == "2026-11-07T14:45:00+00:00"
    assert s["signal_time"] < s["registered_at"] == "2026-11-07T15:00:00+00:00"
    assert s["expected_excess_return_20d"] == 0.02 and "locked_result" in s["expected_return_source"]
    assert s["structures"] == ["LONG_CALL", "BULL_CALL_SPREAD"] and s["order_placed"] is False
    # PIT: features/regime from the prediction date only, never the later row; no labels
    assert s["features"]["mom_12_1"] == 0.1 and s["regime"]["vix"] == 18.0
    assert s["regime"]["label"] == "vix_lt_20|spy_uptrend"
    assert not any(k.startswith(("fwd_", "label_")) for k in s["features"])
    bottom = [x for x in sigs if x["role"] == "BOTTOM"]
    assert bottom and all(not x["tradable"] and x["untradable_reason"] == "NO_ALLOWED_STRUCTURE"
                          and x["direction"] == "BEARISH" and x["structures"] == [] for x in bottom)
    hgb = [x for x in sigs if x["hypothesis_id"] == "hgb_xs20_v1"]
    assert {x["comparison_group"] for x in hgb} == {"TREATMENT"}


def test_register_is_idempotent_with_deterministic_ids(tmp_path):
    _, first = _registered(tmp_path)
    again = al.register(tmp_path, panel=_panel(), now=NOW_REG + timedelta(hours=2))
    assert again["n_new_signals"] == 0
    assert al._read_kind(tmp_path, "signals")[0] == first
    other = tmp_path / "other"
    _, second = _registered(other)
    assert {s["signal_id"] for s in first} == {s["signal_id"] for s in second}


@pytest.mark.parametrize("created, stat", [
    ("2026-11-01T10:00:00+00:00", "ml_pre_link_rows"),          # before link_start: never linked retroactively
    ("2026-11-02T15:00:00+00:00", "ml_late_not_registered"),    # older than the registration window
    ("2026-11-08T15:00:00+00:00", "ml_future_rows"),            # not yet known at registration time
])
def test_register_never_uses_stale_pre_link_or_future_predictions(tmp_path, created, stat):
    _setup(tmp_path)
    _prediction(tmp_path, "momentum_12_1", created=created)
    res = al.register(tmp_path, now=NOW_REG)
    assert res["n_new_signals"] == 0 and res["source_stats"][stat] == 1


def test_register_rejects_spec_drift_and_pre_registration(tmp_path):
    _setup(tmp_path)
    _prediction(tmp_path, "momentum_12_1", spec_hash="deadbeefdeadbeef")
    assert al.register(tmp_path, now=NOW_REG)["source_stats"]["ml_spec_hash_mismatch"] == 1
    reg = yaml.safe_load((tmp_path / al.ML_REGISTRY).read_text())
    reg["models"][0]["registered_at"] = "2026-11-08T00:00:00Z"   # registered after the prediction
    (tmp_path / al.ML_REGISTRY).write_text(yaml.safe_dump(reg))
    shutil.rmtree(tmp_path / al.ML_PRED_DIR)
    _prediction(tmp_path, "momentum_12_1")
    assert al.register(tmp_path, now=NOW_REG)["source_stats"]["ml_pre_registration"] == 1


def test_control_group_is_deterministic_and_disjoint(tmp_path):
    _, sigs = _registered(tmp_path)
    ctrl = [s for s in sigs if s["comparison_group"] == "CONTROL"]
    treated = {s["ticker"] for s in sigs if s["comparison_group"] != "CONTROL"}
    assert len(ctrl) == 5 and not ({s["ticker"] for s in ctrl} & treated)
    assert all(s["hypothesis_id"] == al.CONTROL_ID and s["tradable"] for s in ctrl)
    expected = al._control_sample(TICKERS, treated, PRED_DAY, 5, "alpha-option-control-v1")
    assert [s["ticker"] for s in sorted(ctrl, key=lambda s: s["rank_position"])] == expected


def test_signal_budget_overflow_is_documented_not_dropped(tmp_path):
    _, sigs = _registered(tmp_path, max_signals_per_date=7)      # 5 control + 2 tradable
    top = [s for s in sigs if s["role"] == "TOP"]
    assert len(top) == 6
    assert [s["hypothesis_id"] for s in top if s["tradable"]] == ["momentum_12_1"] * 2   # reference first
    assert sum(s["untradable_reason"] == "SIGNAL_BUDGET_EXCEEDED" for s in top) == 4


# ── journals ─────────────────────────────────────────────────────────────────

def test_tampered_journal_fails_closed_union_duplicates_tolerated(tmp_path):
    _registered(tmp_path)
    f = next((tmp_path / al.ROOT / "signals").glob("*.jsonl"))
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines + lines[:2]) + "\n")              # git union merge duplicate
    rows, integ = al._read_kind(tmp_path, "signals")
    assert len(rows) == 17 and integ["conflicting_duplicates"] == 0
    bad = json.loads(lines[0])
    bad["payload"]["ticker"] = "ZZZ"                                  # edited without re-hashing
    f.write_text("\n".join(lines + [json.dumps(bad)]) + "\n")
    with pytest.raises(EvidenceIntegrityError, match="hash"):
        al._read_kind(tmp_path, "signals")


def test_conflicting_logical_duplicate_keeps_earliest_and_is_reported(tmp_path):
    _registered(tmp_path)
    rows, _ = al._read_kind(tmp_path, "signals")
    later = dict(rows[0], registered_at="2026-11-07T18:00:00+00:00", score=0.99)
    f = next((tmp_path / al.ROOT / "signals").glob("*.jsonl"))
    with f.open("a") as fh:
        fh.write(json.dumps({"sha256": al.digest(later), "payload": later}) + "\n")
    rows2, integ = al._read_kind(tmp_path, "signals")
    assert integ["conflicting_duplicates"] == 1
    assert next(r for r in rows2 if r["signal_id"] == rows[0]["signal_id"]) == rows[0]
    cov = al.coverage(tmp_path, now=NOW_REG, persist=False)
    assert cov["integrity"]["status"] == "CONFLICT_DETECTED"


# ── RTH decisions ────────────────────────────────────────────────────────────

EXPIRY = "2027-01-15"   # DTE 67 on 2026-11-09 (window 45..90)


def _occ(t, k):
    return f"{t}270115C{round(k * 1000):08d}"


def _chain(t, spot=100.0, *, spread=0.2, oi=500, iv=0.3, price=None):
    ms_epoch = int(DECIDE.timestamp() * 1000)
    out = []
    for k in (80, 90, 95, 100, 105, 110, 115, 125):
        mid = price if price is not None else max(2.0, spot - k + 8)
        out.append({"symbol": _occ(t, k), "option_type": "call", "strike": k, "expiration_date": EXPIRY,
                    "bid": round(mid - spread / 2, 2), "ask": round(mid + spread / 2, 2), "open_interest": oi,
                    "volume": 10, "greeks": {"mid_iv": iv, "delta": 0.5}, "underlying": t,
                    "bid_date": ms_epoch, "ask_date": ms_epoch})
    return out


@pytest.fixture
def market(monkeypatch):
    return install_market(monkeypatch)


def install_market(monkeypatch):
    """Offline Tradier stand-in (synthetic chain); shared with the Phase B tests."""
    state = {"chain": {}, "calls": [], "expirations": ["2026-11-20", "2026-12-18", EXPIRY, "2027-03-19"]}
    monkeypatch.setattr(ms, "_use_tradier", lambda: True)
    monkeypatch.setattr(ms, "fetch_underlying_quotes",
                        lambda ts: {t: {"bid": 99.95, "ask": 100.05, "last": 100.0, "quote_ts": "x",
                                        "source": "tradier"} for t in ts})

    def exps(t):
        state["calls"].append(("exp", t))
        return list(state["expirations"])

    def chain(t, e):
        state["calls"].append(("chain", t, e))
        return state["chain"].get(t) or _chain(t)
    monkeypatch.setattr(ms, "_fetch_expirations", exps)
    monkeypatch.setattr(ms, "_fetch_chain", chain)
    return state


def _decisions(root):
    return al._read_kind(root, "decisions")[0]


def test_decide_records_legs_quotes_and_economics(tmp_path, market):
    _, sigs = _registered(tmp_path)
    res = al.decide(tmp_path, now=DECIDE)
    tradable = [s for s in sigs if s["tradable"]]
    ds = _decisions(tmp_path)
    assert res["n_new_decisions"] == len(ds) == 2 * len(tradable)
    d = next(x for x in ds if x["structure"] == "BULL_CALL_SPREAD" and x["comparison_group"] == "REFERENCE")
    assert d["decision"] == "TRADE" and d["abstain_reasons"] == [] and d["order_placed"] is False
    long, short = d["legs"]
    assert (long["side"], long["strike"], short["side"], short["strike"]) == ("long", 100.0, "short", 110.0)
    from modules.options_designer import pick_spread_leg_strike
    assert short["strike"] == pick_spread_leg_strike([80, 90, 95, 100, 105, 110, 115, 125], 100.0, "call")
    assert d["expiry"] == EXPIRY and d["dte"] == 67
    # buy at ask, sell at bid; fees per contract and side
    assert d["net_debit_ask"] == pytest.approx(long["ask"] - short["bid"])
    assert d["capital_usd"] == pytest.approx(d["net_debit_ask"] * 100 + 2 * 0.65)
    assert d["max_loss_usd"] == pytest.approx(d["net_debit_ask"] * 100 + 4 * 0.65)
    assert d["breakeven_move"] == pytest.approx((100 + d["net_debit_ask"]) / 100 - 1)
    assert d["predicted_edge"]["status"] == "AVAILABLE" and d["predicted_edge"]["gating"] == "NONE"
    assert d["decided_at"] >= d["signal_registered_at"]
    # quote provenance: every leg references an archived, usable RTH quote
    arch = oq.by_ids([long["quote_record_id"], short["quote_record_id"]], "2026-11-09",
                     root=tmp_path / oq.ROOT)
    assert set(arch) == {long["quote_record_id"], short["quote_record_id"]}
    assert all(oq.usable_rth_quote(r) and r["context"] == "alpha_link_entry" for r in arch.values())
    assert d["evidence_class_at_entry"] == "QUOTE_VERIFIED_PAPER"
    lc = next(x for x in ds if x["structure"] == "LONG_CALL" and x["signal_id"] == d["signal_id"])
    assert [leg["strike"] for leg in lc["legs"]] == [100.0] and lc["net_debit_ask"] == long["ask"]
    # one chain observation per underlying, shared by all hypotheses
    tickers = {s["ticker"] for s in tradable}
    assert sum(c[0] == "chain" for c in market["calls"]) == len(tickers)


@pytest.mark.parametrize("kw, reason", [
    ({"spread": 3.0}, "LONG_SPREAD_TOO_WIDE"),
    ({"oi": 3}, "LONG_OPEN_INTEREST_BELOW_MIN"),
    ({"iv": 2.5}, "IV_ABOVE_MAX"),
    ({"price": 80.0}, "CAPITAL_ABOVE_MAX"),
])
def test_decide_gates_lead_to_documented_abstain(tmp_path, market, kw, reason):
    _, sigs = _registered(tmp_path)
    t = next(s["ticker"] for s in sigs if s["comparison_group"] == "REFERENCE")
    market["chain"][t] = _chain(t, **kw)
    al.decide(tmp_path, now=DECIDE)
    d = next(x for x in _decisions(tmp_path) if x["ticker"] == t and x["structure"] == "LONG_CALL")
    assert d["decision"] == "ABSTAIN" and reason in d["abstain_reasons"]


def test_no_expiry_in_window_is_binding_abstain(tmp_path, market):
    _registered(tmp_path)
    market["expirations"] = ["2026-11-20", "2027-06-18"]
    al.decide(tmp_path, now=DECIDE)
    ds = _decisions(tmp_path)
    assert ds and all(d["decision"] == "ABSTAIN" and d["abstain_reasons"] == ["NO_EXPIRY_IN_DTE_WINDOW"] for d in ds)


def test_decision_is_binding_and_idempotent(tmp_path, market):
    _registered(tmp_path)
    al.decide(tmp_path, now=DECIDE)
    first = _decisions(tmp_path)
    market["chain"] = {t: _chain(t, spread=0.02) for t in TICKERS}       # better quotes later: ignored
    res = al.decide(tmp_path, now=DECIDE + timedelta(hours=1))
    assert res["n_new_decisions"] == 0 and _decisions(tmp_path) == first


def test_transient_failure_retries_then_missed_window_without_network(tmp_path, market, monkeypatch):
    _registered(tmp_path)
    market["expirations"] = []                                            # API hiccup -> retry later
    res = al.decide(tmp_path, now=DECIDE)
    assert res["n_new_decisions"] == 0 and len(res["underlyings_retry_later"]) > 0
    monkeypatch.setattr(ms, "_fetch_chain", lambda *a: pytest.fail("no network after the window"))
    late = datetime(2026, 11, 13, 16, 0, tzinfo=timezone.utc)            # > 3 trading days after registration
    res = al.decide(tmp_path, now=late)
    ds = _decisions(tmp_path)
    assert res["window_missed"] == len(ds) > 0
    assert all(d["decision"] == "ABSTAIN" and d["abstain_reasons"] == ["DECISION_WINDOW_MISSED"] for d in ds)


def test_no_decision_before_registration_outside_window_or_without_source(tmp_path, market, monkeypatch):
    _registered(tmp_path)
    assert al.decide(tmp_path, now=NOW_REG - timedelta(hours=1))["n_new_decisions"] == 0
    sunday = datetime(2026, 11, 8, 16, 0, tzinfo=timezone.utc)
    assert al.decide(tmp_path, now=sunday)["status"] == "OUTSIDE_MARK_WINDOW"
    monkeypatch.setattr(ms, "_use_tradier", lambda: False)
    assert al.decide(tmp_path, now=DECIDE)["status"] == "NO_DATA_SOURCE"
    assert _decisions(tmp_path) == []


def test_not_configured_is_a_quiet_noop(tmp_path):
    assert al.decide(tmp_path, now=DECIDE) == {"status": "NOT_CONFIGURED"}
    assert al.due_capture_targets(today=date(2026, 12, 8), root=tmp_path) == {}


# ── exits and settlement ─────────────────────────────────────────────────────

def _exit_quote(root, sym, when, bid, ask, context="candidate_mark_rth"):
    raw = {"symbol": sym, "bid": bid, "ask": ask, "open_interest": 100, "iv": 0.3,
           "bid_date": int(when.timestamp() * 1000), "ask_date": int(when.timestamp() * 1000)}
    rec = oq.normalize(raw, underlying=sym[:3], source="tradier", context=context, retrieved_at=when)
    oq.archive([rec], root=root / oq.ROOT)
    return rec


def _one_trade(tmp_path, market):
    _registered(tmp_path)
    al.decide(tmp_path, now=DECIDE)
    return next(d for d in _decisions(tmp_path) if d["structure"] == "BULL_CALL_SPREAD"
                and d["decision"] == "TRADE" and d["comparison_group"] == "REFERENCE")


def test_exit_targets_are_due_only_in_window_and_budgeted(tmp_path, market):
    d = _one_trade(tmp_path, market)
    target = date.fromisoformat(d["exit_target_day"])
    assert target == al.add_trading_days(date(2026, 11, 9), 20) == date(2026, 12, 8)   # skips Thanksgiving
    assert al.due_capture_targets(today=target - timedelta(days=1), root=tmp_path) == {}
    due = al.due_capture_targets(today=target, root=tmp_path)
    assert {leg["occ_symbol"] for leg in d["legs"]} <= set(due)
    assert len(due) <= al.config(tmp_path)["max_capture_symbols_per_run"]
    cfg = yaml.safe_load((tmp_path / al.CONFIG).read_text())
    cfg["max_capture_symbols_per_run"] = 1                 # a spread needs 2 legs -> never half a position
    (tmp_path / al.CONFIG).write_text(yaml.safe_dump(cfg))
    small = al.due_capture_targets(today=target, root=tmp_path)
    whole = [dd for dd in _decisions(tmp_path) if dd["decision"] == "TRADE"
             and {x["occ_symbol"] for x in dd["legs"]} == set(small)]
    assert len(small) == 1 and whole and all(dd["structure"] == "LONG_CALL" for dd in whole)


def test_settle_uses_first_paired_exit_quote_inside_window_only(tmp_path, market):
    d = _one_trade(tmp_path, market)
    long_sym, short_sym = (leg["occ_symbol"] for leg in d["legs"])
    before = datetime(2026, 12, 7, 16, 0, tzinfo=timezone.utc)                  # day before target: ignored
    _exit_quote(tmp_path, long_sym, before, 50.0, 50.2)
    _exit_quote(tmp_path, short_sym, before, 0.1, 0.2)
    at = datetime(2026, 12, 8, 16, 0, tzinfo=timezone.utc)
    q1 = _exit_quote(tmp_path, long_sym, at, 7.0, 7.2)
    q2 = _exit_quote(tmp_path, short_sym, at + timedelta(minutes=1), 2.0, 2.1)
    res = al.settle(tmp_path, now=datetime(2026, 12, 12, 15, 0, tzinfo=timezone.utc))
    assert res["n_new_outcomes"] >= 1
    o = next(x for x in al._read_kind(tmp_path, "outcomes")[0] if x["decision_id"] == d["decision_id"])
    assert o["status"] == "EXIT_OBSERVED" and o["exit_day"] == "2026-12-08"
    assert [x["quote_record_id"] for x in o["exit_legs"]] == [q1["record_id"], q2["record_id"]]
    assert o["evidence_class"] == "QUOTE_VERIFIED_PAPER"
    again = al.settle(tmp_path, now=datetime(2026, 12, 19, 15, 0, tzinfo=timezone.utc))
    assert all(x["decision_id"] != d["decision_id"] for x in al._read_kind(tmp_path, "outcomes")[0][1:]) or \
        again["n_new_outcomes"] >= 0


def test_missing_exit_is_unverifiable_never_a_result(tmp_path, market):
    d = _one_trade(tmp_path, market)
    pending = al.settle(tmp_path, now=datetime(2026, 12, 9, 15, 0, tzinfo=timezone.utc))
    assert pending["n_new_outcomes"] == 0 and pending["n_pending_exit"] > 0
    res = al.settle(tmp_path, now=datetime(2026, 12, 17, 15, 0, tzinfo=timezone.utc))   # after the deadline
    o = next(x for x in al._read_kind(tmp_path, "outcomes")[0] if x["decision_id"] == d["decision_id"])
    assert res["n_exit_missing"] > 0
    assert o["status"] == "EXIT_MISSING" and o["evidence_class"] == "UNVERIFIABLE" and o["exit_legs"] == []


# ── coverage, corrections, rule versions ─────────────────────────────────────

def test_coverage_reports_need_more_data_gaps_and_pre_link(tmp_path, market):
    _setup(tmp_path)
    _prediction(tmp_path, "momentum_12_1", created="2026-10-24T12:56:00+00:00", day="2026-10-23")
    _prediction(tmp_path, "momentum_12_1")
    al.register(tmp_path, panel=_panel(), now=NOW_REG)
    al.decide(tmp_path, now=DECIDE)
    cov = al.coverage(tmp_path, now=datetime(2026, 11, 10, 12, tzinfo=timezone.utc))
    assert cov["status"] == "NEED_MORE_DATA" and cov["economic_result"] == "NOT_EVALUATED_IN_PHASE_A"
    assert cov["pre_link"]["ml_prediction_rows"] == 1 and cov["pre_link"]["status"] == "NOT_LINKABLE_RETROACTIVELY"
    h = cov["hypotheses"]["momentum_12_1"]
    assert h["untradable"] == {"NO_ALLOWED_STRUCTURE": 3}
    assert h["by_structure"]["LONG_CALL"]["TRADE"] + h["by_structure"]["LONG_CALL"]["ABSTAIN"] == 3
    assert h["by_structure"]["LONG_CALL"]["EXIT_PENDING"] == h["by_structure"]["LONG_CALL"]["TRADE"]
    assert h["status"] == "NEED_MORE_DATA"
    assert cov["executed"]["n_attributed_ex_ante"] == 0
    assert json.loads((tmp_path / al.REPORT).read_text())["schema"] == "alpha-option-link-coverage-v1"


def test_corrections_are_versioned_and_never_delete(tmp_path, market):
    d = _one_trade(tmp_path, market)
    path = next((tmp_path / al.DECISIONS_ROOT / "decisions").glob("*.jsonl"))
    raw_before = path.read_text()
    c1 = al.record_correction(tmp_path, kind="decisions", target_id=d["decision_id"],
                              action="DOWNGRADE_EVIDENCE", reason="quote vendor glitch", recorded_by="human")
    c2 = al.record_correction(tmp_path, kind="decisions", target_id=d["decision_id"],
                              action="EXCLUDE_DATA_ERROR", reason="corporate action", recorded_by="human")
    assert (c1["version"], c2["version"]) == (1, 2)
    assert path.read_text() == raw_before                                   # original untouched
    view = next(x for x in al.read_all(tmp_path)["decisions"] if x["decision_id"] == d["decision_id"])
    assert view["excluded_by_correction"] and view["evidence_class_at_entry"] == "UNVERIFIABLE"
    assert [c["version"] for c in view["corrections"]] == [1, 2]
    cov = al.coverage(tmp_path, now=DECIDE, persist=False)
    assert cov["hypotheses"]["momentum_12_1"]["excluded_by_correction"] == 1
    assert cov["integrity"]["corrections"] == 2
    with pytest.raises(EvidenceIntegrityError):
        al.record_correction(tmp_path, kind="decisions", target_id="nope", action="EXCLUDE_DATA_ERROR",
                             reason="x", recorded_by="human")
    with pytest.raises(EvidenceIntegrityError):
        al.record_correction(tmp_path, kind="decisions", target_id=d["decision_id"], action="UPGRADE",
                             reason="x", recorded_by="human")


def test_rule_change_under_used_version_fails_closed(tmp_path, market):
    _one_trade(tmp_path, market)
    cfg = yaml.safe_load((tmp_path / al.CONFIG).read_text())
    cfg["rule"]["max_leg_spread_pct"] = 0.5                                   # same rule_version
    (tmp_path / al.CONFIG).write_text(yaml.safe_dump(cfg))
    with pytest.raises(EvidenceIntegrityError, match="rule"):
        al.settle(tmp_path, now=DECIDE)
    cfg["rule"]["rule_version"] = "alpha-option-rule-v2"                     # new version is fine
    (tmp_path / al.CONFIG).write_text(yaml.safe_dump(cfg))
    al.settle(tmp_path, now=DECIDE)


@pytest.mark.parametrize("path, value", [
    (("rule", "structures"), ["LONG_CALL", "LONG_PUT"]),
    (("rule", "edge_gate"), "EDGE_POSITIVE"),
    (("rule", "contracts_per_position"), 3),
    (("max_signals_per_date",), 200),
    (("production_influence",), "SHADOW"),
    (("coverage", "min_verified_outcomes"), 2),
])
def test_config_hard_caps(tmp_path, path, value):
    cfg = yaml.safe_load((REPO / al.CONFIG).read_text())
    target = cfg
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    (tmp_path / "config").mkdir()
    (tmp_path / al.CONFIG).write_text(yaml.safe_dump(cfg))
    with pytest.raises(EvidenceIntegrityError):
        al.config(tmp_path)


def test_repo_config_is_valid():
    cfg = al.config(REPO)
    assert cfg["rule"]["structures"] == ["LONG_CALL", "BULL_CALL_SPREAD"] and cfg["production_influence"] == "NONE"


# ── executed trades: attribution only ex ante ────────────────────────────────

def test_executed_trades_attribute_only_with_ex_ante_signal(tmp_path):
    _, sigs = _registered(tmp_path)
    s = next(x for x in sigs if x["tradable"] and x["comparison_group"] == "REFERENCE")
    by_id = {x["signal_id"]: x for x in sigs}
    base = {"ticker": s["ticker"], "entry_date": "2026-11-10"}
    assert al.ex_ante_attribution(base, by_id) == (None, "NO_EX_ANTE_SIGNAL_ID")
    assert al.ex_ante_attribution({**base, "alpha_signal_id": s["signal_id"]}, by_id) == (
        "momentum_12_1", "ATTRIBUTED_EX_ANTE")
    early = {**base, "alpha_signal_id": s["signal_id"], "entry_date": "2026-11-06"}
    assert al.ex_ante_attribution(early, by_id)[1] == "SIGNAL_NOT_REGISTERED_BEFORE_ENTRY"
    wrong = {**base, "alpha_signal_id": s["signal_id"], "ticker": "OTHER"}
    assert al.ex_ante_attribution(wrong, by_id)[1] == "TICKER_MISMATCH"


def test_forward_options_verifies_attribution_per_trade(tmp_path):
    (tmp_path / forward_options.POLICY).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / forward_options.POLICY, tmp_path / forward_options.POLICY)
    src = tmp_path / forward_options.SOURCE
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps({"schema": "trade-outcome-v1", "trade_id": "T-1", "ticker": "AAA",
                               "entry_date": "2026-11-04", "close_date": "2026-11-20"}) + "\n")
    rep = forward_options.evaluate(tmp_path, today=date(2026, 11, 30), persist=False)
    assert rep["strategy_alpha_status"] == "NOT_ATTRIBUTABLE"
    assert rep["alpha_attribution_checks"] == {"NO_EX_ANTE_SIGNAL_ID": 1}
    assert rep["n_verified_attributed_ex_ante"] == 0


# ── mirrors and capture-path weight ──────────────────────────────────────────

def test_ml_spec_hash_and_short_leg_rule_mirror_production():
    from modules import ml_research
    from modules.options_designer import pick_spread_leg_strike
    for spec in al._ml_registry(REPO).values():
        assert al.ml_spec_hash(spec) == ml_research.spec_hash(spec)
    strikes = [x / 2 for x in range(100, 400)]
    for long_k in (55.0, 100.0, 123.5, 180.0):
        assert al.short_strike(strikes, long_k, (1.05, 1.20), 1.10) == pick_spread_leg_strike(strikes, long_k, "call")


def test_capture_path_imports_without_pandas(tmp_path):
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from modules.autonomous_alpha import option_link as al;"
            "from datetime import date;"
            "print(al.decide(sys.argv[2]), al.due_capture_targets(today=date(2026,12,8), root=sys.argv[2]));"
            "assert 'pandas' not in sys.modules and 'numpy' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code, str(REPO), str(tmp_path)], capture_output=True, text=True,
                       timeout=60, check=False)
    assert r.returncode == 0, r.stderr[-1500:]


def test_rth_marks_capture_adds_alpha_exit_targets(tmp_path, monkeypatch, market):
    d = _one_trade(tmp_path, market)
    sys.path.insert(0, str(REPO / "scripts"))
    import rth_marks

    from modules import candidate_ledger as cl
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cl, "due_mark_symbols", lambda today: [])
    monkeypatch.setattr(cl, "pending_entry_targets", lambda today: {})
    asked = []
    monkeypatch.setattr(ms, "fetch_option_quotes", lambda syms, context="": asked.extend(syms) or {})
    exit_day = datetime(2026, 12, 8, 16, 0, tzinfo=timezone.utc)
    res = rth_marks.run(exit_day, tmp_path / "h.json", capture_only=True)
    assert res["alpha_link"]["status"] == "OK"
    assert {leg["occ_symbol"] for leg in d["legs"]} <= set(asked)


# ── audit: EXECUTED only with documented broker executions ───────────────────

FILL = {"broker": "tradier", "entry": {"order_id": "E1", "fill_price": 3.1, "filled_at": "2026-11-10T15:00:00+00:00"},
        "exit": {"order_id": "X1", "fill_price": 4.0, "filled_at": "2026-12-08T15:00:00+00:00"}}


@pytest.mark.parametrize("execution, expected", [
    (None, "NO_BROKER_EXECUTION_RECORD"),
    ({"broker": "tradier", "entry": FILL["entry"]}, "EXIT_FILL_INCOMPLETE"),
    ({**FILL, "broker": ""}, "BROKER_MISSING"),
    ({**FILL, "exit": {**FILL["exit"], "filled_at": "2026-11-09T15:00:00+00:00"}}, "FILL_ORDER_INVALID"),
    ({**FILL, "entry": {**FILL["entry"], "fill_price": "n/a"}}, "ENTRY_FILL_INVALID"),
    (FILL, "BROKER_EXECUTION_DOCUMENTED"),
])
def test_executed_requires_documented_broker_fills(execution, expected):
    row = {"broker_execution": execution} if execution is not None else {}
    ok, reason = al.broker_execution_documented(row)
    assert reason == expected and ok == (expected == "BROKER_EXECUTION_DOCUMENTED")


def test_quote_verified_paper_trade_is_never_upgraded_to_executed():
    paper = {"price_type": "OBSERVED_QUOTE", "reliable": True, "entry_quote_record_ids": ["a"],
             "exit_quote_record_ids": ["b"], "outcome_net": 0.4}
    assert al.trade_evidence_class(paper, quote_verifiable=True) == "QUOTE_VERIFIED_PAPER"
    assert al.trade_evidence_class(paper, quote_verifiable=False) == "UNVERIFIABLE"
    assert al.trade_evidence_class({**paper, "broker_execution": FILL}, quote_verifiable=False) == "EXECUTED"


def test_real_trade_outcomes_stay_paper_and_unchanged(tmp_path):
    _, sigs = _registered(tmp_path)
    s = next(x for x in sigs if x["tradable"] and x["comparison_group"] == "REFERENCE")
    shutil.copy(REPO / forward_options.POLICY, tmp_path / forward_options.POLICY)
    src = tmp_path / forward_options.SOURCE
    src.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"schema": "trade-outcome-v1", "trade_id": "P-1", "ticker": s["ticker"], "entry_date": "2026-11-10",
             "close_date": "2026-11-20", "alpha_signal_id": s["signal_id"], "price_type": "OBSERVED_QUOTE"},
            {"schema": "trade-outcome-v1", "trade_id": "B-1", "ticker": s["ticker"], "entry_date": "2026-11-10",
             "close_date": "2026-11-20", "alpha_signal_id": s["signal_id"], "broker_execution": FILL}]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = src.read_bytes()
    rep = forward_options.evaluate(tmp_path, today=date(2026, 11, 30), persist=False)
    assert rep["evidence_classes"] == {"EXECUTED": 1, "UNVERIFIABLE": 1}
    assert rep["n_executed_attributed_ex_ante"] == 0                  # B-1 is not quote-verifiable in this fixture
    cov = al.coverage(tmp_path, now=datetime(2026, 11, 30, tzinfo=timezone.utc), persist=False)
    ex = cov["executed"]
    assert ex["n_documented_broker_executions"] == 1 and ex["n_executed_evidence"] == 1
    assert ex["broker_execution_checks"]["NO_BROKER_EXECUTION_RECORD"] == 1
    assert src.read_bytes() == before                                  # historical records never rewritten


def test_repository_trade_outcomes_are_paper_not_executed():
    rep = forward_options.evaluate(REPO, persist=False)
    assert "EXECUTED" not in rep["evidence_classes"] and rep["n_executed_attributed_ex_ante"] == 0
