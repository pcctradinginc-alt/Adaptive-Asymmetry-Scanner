"""Familienbewusste Evidenz-Aggregation (modules/evidence_families) und ihre Anbindung an die EA-Bestätigung."""
from __future__ import annotations

import itertools

import pytest

from modules import evidence_families as ef
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import cross_asset_confirmation as cac
from modules.expectation_alpha import timing as tm

CFG = eacfg.load()


def _it(i, fam, st):
    return {"id": i, "family": fam, "state": st}


def test_first_full_then_dampened_and_raw_kept():
    a = ef.aggregate([_it("XLF", "equity_banks", 1), _it("KRE", "equity_banks", 1), _it("JPM", "equity_banks", 1),
                      _it("BAC", "equity_banks", 1)], 0.5)
    assert a["raw_confirmation_count"] == 4 and a["family_count"] == 1
    assert a["effective_confirmation"] == pytest.approx(1 + 0.5 + 0.25 + 0.125)      # nicht 4
    assert a["effective_confirmation_ratio"] == 1.0
    fb = a["family_breakdown"]["equity_banks"]
    assert fb["n_available"] == 4 and fb["weight"] == pytest.approx(1.875) and fb["signals"] == ["XLF", "KRE", "JPM", "BAC"]


def test_dampening_factor_is_config_parameter():
    items = [_it("a", "rates", 1), _it("b", "rates", 1), _it("c", "credit", 1)]
    assert ef.aggregate(items, 0.0)["effective_confirmation"] == 2.0      # je Familie höchstens 1
    assert ef.aggregate(items, 1.0)["effective_confirmation"] == 3.0      # keine Dämpfung = roh
    assert ef.aggregate(items, 0.5)["effective_confirmation"] == 2.5
    with pytest.raises(ValueError):
        ef.aggregate(items, 1.5)
    assert CFG["confirmation"]["family_dampening"] == 0.5


def test_multiple_families_and_redundancy_shift_ratio():
    # 4 Banken-Signale bestätigen, 1 Credit-Signal widerspricht: roh 80 %, effektiv deutlich weniger eindeutig
    items = [_it(t, "equity_banks", 1) for t in ("XLF", "KRE", "JPM", "BAC")] + [_it("HYG", "credit", -1)]
    a = ef.aggregate(items, 0.5)
    assert a["raw_confirmation_count"] == 4 and a["raw_conflict_count"] == 1
    assert a["effective_confirmation_ratio"] == pytest.approx(1.875 / 2.875, abs=1e-4)
    assert a["effective_conflict_share"] == pytest.approx(1 / 2.875, abs=1e-4)
    assert a["family_count"] == 2


def test_missing_evidence_never_negative():
    base = [_it("a", "rates", 1), _it("b", "credit", 1)]
    with_missing = base + [_it("c", "rates", None), _it("d", "volatility", None)]
    a, b = ef.aggregate(base, 0.5), ef.aggregate(with_missing, 0.5)
    assert a["effective_confirmation_ratio"] == b["effective_confirmation_ratio"] == 1.0
    assert b["effective_conflict"] == 0.0 and b["families_missing"] == ["volatility"]
    assert b["family_breakdown"]["volatility"]["status"] == "MISSING" and b["family_count"] == 2
    empty = ef.aggregate([_it("x", "rates", None)], 0.5)
    assert empty["effective_confirmation_ratio"] is None and empty["family_count"] == 0   # nie 0 % Bestätigung


def test_order_independent():
    items = [_it("a", "rates", 1), _it("b", "rates", -1), _it("c", "rates", 0), _it("d", "credit", 1)]
    res = {ef.aggregate(list(p), 0.5)["effective_confirmation"] for p in itertools.permutations(items)}
    assert len(res) == 1


def test_ea_spec_families_merge_identical_and_correlated_signals():
    spec, _amb = cac.candidate_spec(1, "XLE", {"oil": 1, "inflation": 1}, CFG)
    fam = {s["signal"]: s["family"] for s in spec}
    assert fam["sector_rs_20d:XLE"] == fam["xle_spy_20d"] == "equity_energy"         # identischer Wert, eine Familie
    assert fam["wti_ret_20d"] == fam["gld_20d"] == "commodities"
    spec_r, _ = cac.candidate_spec(-1, "XLRE", {"policy_rates": -1}, CFG)
    rates = [s["signal"] for s in spec_r if s["family"] == "rates"]
    assert set(rates) == {"irx_20d", "tnx_20d", "tlt_20d"}


def test_all_configured_signals_have_a_family():
    cc = CFG["confirmation"]
    names = {s["signal"] for s in cc["candidate_signals"] if s["signal"] != "sector_rs_20d"}
    names |= {s["signal"] for sigs in cc["domain_signals"].values() for s in sigs}
    assert not [n for n in names if cac.family_of(n, CFG).startswith("unmapped")]
    for etf in cac.SECTOR_ETFS:
        assert not cac.family_of(f"sector_rs_20d:{etf}", CFG).startswith("unmapped"), etf
    assert cac.family_of("unknown_signal", CFG) == "unmapped:unknown_signal"


def test_confirm_reports_raw_and_effective():
    spec = [{"signal": s, "expected": 1, "deadband": 0.0, "weight": 1.0, "family": "rates"}
            for s in ("tnx_20d", "irx_20d", "tlt_20d")] + \
           [{"signal": "hyg_lqd_20d", "expected": 1, "deadband": 0.0, "weight": 1.0, "family": "credit"}]
    c = cac.confirm(spec, {"tnx_20d": 0.1, "irx_20d": 0.1, "tlt_20d": 0.02, "hyg_lqd_20d": None}, dampening=0.5)
    for k in ("raw_confirmation_count", "effective_confirmation", "family_count", "family_breakdown",
              "effective_confirmation_ratio", "effective_available"):
        assert k in c
    assert c["n_available"] == 3 and c["raw_confirmation_count"] == 3 and c["family_count"] == 1
    assert c["effective_confirmation"] == pytest.approx(1.75) and c["missing_inputs"] == ["hyg_lqd_20d"]
    assert c["family_breakdown"]["credit"]["status"] == "MISSING"


def test_decision_needs_independent_families_not_redundant_signals():
    """Drei bestätigende Zins-Signale sind EINE Familie -> kein Research-TRADE, sondern WAIT."""
    spec = [{"signal": s, "expected": 1, "deadband": 0.0, "weight": 1.0, "family": "rates"}
            for s in ("tnx_20d", "irx_20d", "tlt_20d")]
    conf = cac.confirm(spec, {"tnx_20d": 0.1, "irx_20d": 0.1, "tlt_20d": 0.1}, dampening=0.5)
    assert conf["confirmation_ratio"] == 1.0 and conf["n_available"] == 3            # roh: sähe wie TRADE aus
    d = tm.decide(news={"strength": "STRONG"}, context={"context_status": 1}, confirmation=conf,
                  regime_uncertainty=0.3, errors=[], cfg=CFG)
    assert d["status"] == "WAIT" and d["reasons"] == ["CONFIRMATION_INSUFFICIENT"]
