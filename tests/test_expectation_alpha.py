"""Expectation Alpha – Kernlogik: Gap-Mathematik/Einheiten, RoC, Regime-Grenzen, Confirmation, Status,
Kill Conditions, Expressions, Ledger, Outcomes, Evidenz, Verträge. Synthetische Daten (tests/ea_fixtures.py)."""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import tests.ea_fixtures as fx
from modules import hypothesis_contract as hc
from modules import expectation_alpha as ea
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import cross_asset_confirmation as cac
from modules.expectation_alpha import data as eadata
from modules.expectation_alpha import expectation_gap as eg
from modules.expectation_alpha import expression as ex
from modules.expectation_alpha import ledger as eal
from modules.expectation_alpha import thesis as th
from modules.expectation_alpha import timing as tm
from modules.expectation_alpha.future_state import roc_frame, roc_set
from modules.expectation_alpha.regime_change import state_of
from modules.expectation_alpha.schemas import ABSTAIN, ERROR, FeatureValue, TRADE, UNAVAILABLE, WAIT

UTC = timezone.utc
DT = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)
CFG = eacfg.load()


@pytest.fixture(scope="module")
def data():
    px = fx.make_px(end="2027-06-30", years=7)
    for t, etf, k in (("AAPL", "XLK", 1.3), ("XOM", "XLE", 0.8), ("JPM", "XLF", 1.1), ("PG", "XLP", 0.9)):
        px[t] = px[etf] * k
    return {"px": px, "ar": fx.make_archive(), "cm": fx.make_commodity()}


# ── Config / Schemas ────────────────────────────────────────────────────────
def test_config_shadow_and_hash_stable():
    assert CFG["mode"] == "shadow"
    assert eacfg.config_hash(CFG) == eacfg.config_hash(eacfg.load())
    c2 = copy.deepcopy(CFG)
    c2["decision"]["trade_min_confirmation_ratio"] = 0.5
    assert eacfg.config_hash(c2) != eacfg.config_hash(CFG)


def test_invalid_mode_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("mode: production\n")
    with pytest.raises(ValueError):
        eacfg.load(p)


def test_feature_value_missing_is_unavailable_never_zero():
    fv = FeatureValue("x", float("nan"), "pct", "src")
    assert fv.value is None and fv.status == UNAVAILABLE
    assert fv.to_dict()["value"] is None


# ── Gap-Mathematik und Einheiten ────────────────────────────────────────────
def _cpi_archive(growth: float, be: float, years: int = 8):
    end = pd.Timestamp("2026-10-09")
    months = pd.date_range(end - pd.DateOffset(years=years), end, freq="MS")
    cpi = [250.0 * (1 + growth) ** i for i in range(len(months))]
    days = pd.bdate_range(end - pd.DateOffset(years=years), end)
    return {"fred_regime_macro": fx.make_obs("fred_regime_macro", "us_cpi", list(months), cpi, unit="index",
                                             release_lag_days=45),
            "fred_market_expectations": fx.make_obs("fred_market_expectations", "us_breakeven_5y", list(days),
                                                    [be] * len(days), unit="percent", release_lag_days=1)}


def test_level_gap_inflation_math_and_units(data):
    ar = _cpi_archive(0.0025, 2.0)
    inp = eadata.Inputs(decision_time=DT, archive_obs=ar, px=eadata.pit_prices(data["px"], DT))
    g = eg.domain_gaps(inp, eadata.weekly_grid(inp.px), CFG)["inflation"]
    expected = ((1.0025 ** 3) ** 4 - 1.0) * 100.0 - 2.0          # Prozentpunkte
    assert g["unit"] == "pct_points" and g["kind"] == "level_gap"
    assert g["gap_raw"] == pytest.approx(expected, abs=1e-3)
    assert g["model"]["value"] == pytest.approx(expected + 2.0, abs=1e-3)
    assert g["market"]["value"] == pytest.approx(2.0)
    # konstanter Gap -> Std 0 -> z nicht belastbar (INSUFFICIENT_DATA), Rohwert bleibt
    assert g["status"] == "INSUFFICIENT_DATA" and g["gap_z"] is None
    comp = g["model"]["components"]["cpi_3m_ann"]
    for k in ("value", "unit", "source", "observed_at", "published_at", "available_at", "retrieved_at", "vintage",
              "transformation", "freshness_days", "confidence"):
        assert k in comp


def test_missing_market_side_is_unavailable_not_zero(data):
    ar = fx.make_archive(include_expectations=False)
    inp = eadata.Inputs(decision_time=DT, archive_obs=ar, px=eadata.pit_prices(data["px"], DT),
                        commodity=pd.DataFrame())
    gaps = eg.domain_gaps(inp, eadata.weekly_grid(inp.px), CFG)
    for d in ("inflation", "policy_rates", "oil"):
        assert gaps[d]["status"] == UNAVAILABLE and gaps[d]["gap_raw"] is None, d
        assert gaps[d]["reason"]
    assert gaps["growth"]["status"] in ("OK", "INSUFFICIENT_DATA")
    assert gaps["equity_earnings"]["status"] == UNAVAILABLE and "nie simuliert" in gaps["equity_earnings"]["reason"]


def test_z_gap_sign_follows_model_minus_market(data):
    inp = eadata.load_inputs(DT, CFG, **{f"{k}_fn": v for k, v in
                                         fx.loaders(data["px"], data["ar"], data["cm"]).items()})
    dates = eadata.weekly_grid(inp.px)
    fr = eg.component_frame(inp, dates, CFG)
    gaps = eg.domain_gaps(inp, dates, CFG)
    g = gaps["oil"]
    m = eg._side(fr, ["neg_crude_stocks_vs_5y"], 52).iloc[-1]
    k = eg._side(fr, ["wti_ret_60d"], 52).iloc[-1]
    assert g["gap_raw"] == pytest.approx(m - k, abs=1e-4)
    assert g["sign"] in (-1, 0, 1) and (g["gap_z"] is None or -4 <= g["gap_z"] <= 4)
    assert 0 <= g["gap_percentile"] <= 1
    assert g["history"]["n"] > 52 and g["history"]["window_end"] <= "2026-10-08"


# ── Rate of Change ──────────────────────────────────────────────────────────
def test_roc_linear_and_quadratic():
    idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    lin = pd.Series(np.arange(60, dtype=float), index=idx)
    r = roc_frame(lin).iloc[-1]
    assert r["delta_1m"] == 4 and r["delta_3m"] == 13 and r["velocity"] == pytest.approx(4.0)
    assert r["acceleration"] == pytest.approx(0.0) and r["change_of_change"] == pytest.approx(0.0)   # linear
    quad = pd.Series(np.arange(60, dtype=float) ** 2, index=idx)
    q = roc_frame(quad).iloc[-1]
    assert q["acceleration"] == pytest.approx(36.0)                  # konstante 2. Ableitung -> konstante Beschl.
    assert q["change_of_change"] == pytest.approx(0.0)
    neg = roc_frame(-quad).iloc[-1]
    assert neg["acceleration"] == pytest.approx(-36.0)


def test_roc_set_fields_and_missing():
    idx = pd.date_range("2018-01-05", periods=200, freq="W-FRI")
    s = pd.Series(np.sin(np.arange(200) / 7.0), index=idx)
    out = roc_set(s, CFG, unit="z", min_hist=52)
    for k in ("level", "delta_1m", "delta_3m", "velocity", "acceleration", "change_of_change", "percentile",
              "regime_state", "regime_transition_probability", "uncertainty", "z"):
        assert k in out
    assert out["status"] == "OK" and 0 <= out["uncertainty"] <= 1
    s.iloc[-1] = np.nan
    assert roc_set(s, CFG, unit="z", min_hist=52)["status"] == UNAVAILABLE


# ── Regime ──────────────────────────────────────────────────────────────────
def test_state_boundaries():
    assert state_of(0.5, 0.5) == "neutral" and state_of(0.5001, 0.5) == "high"
    assert state_of(-0.5, 0.5) == "neutral" and state_of(-0.51, 0.5) == "low"
    assert state_of(float("nan"), 0.5) is None


def test_regime_from_existing_world_model(data):
    ctx = ea.build_context(DT, CFG, loaders=fx.loaders(data["px"], data["ar"], data["cm"]))
    reg = ctx["regime"]
    from modules import world_model as wm
    assert set(reg["dimensions"]) == set(wm.DIMENSIONS)
    assert reg["dimensions"]["earnings_momentum"]["status"] == UNAVAILABLE
    assert 0 <= reg["regime_uncertainty"] <= 1
    g = reg["dimensions"]["growth"]
    assert {"state", "previous_state", "changed", "velocity", "acceleration", "transition"} <= set(g)


# ── Confirmation ────────────────────────────────────────────────────────────
def test_confirmation_direction_conflict_missing():
    spec, amb = cac.candidate_spec(1, "XLK", {"growth": 1, "policy_rates": -1}, CFG)
    assert "tnx_20d" in amb                                        # growth erwartet +, Zinsen − -> mehrdeutig
    names = {s["signal"] for s in spec}
    assert "sector_rs_20d:XLK" in names and "tnx_20d" not in names
    vals = {s["signal"]: None for s in spec}
    vals.update({"spy_ret_20d": 0.03, "sector_rs_20d:XLK": -0.02, "hyg_lqd_20d": 0.001})
    c = cac.confirm(spec, vals, amb)
    assert c["n_available"] == 3 and c["n_confirming"] == 1 and c["n_conflicting"] == 1
    assert c["confirmation_ratio"] == pytest.approx(1 / 3, abs=1e-4)
    assert c["confidence"] == pytest.approx(3 / c["n_expected"], abs=1e-4)
    assert len(c["missing_inputs"]) == c["n_expected"] - 3          # fehlend != widersprechend
    short, _ = cac.candidate_spec(-1, "XLK", {"growth": 1, "policy_rates": -1}, CFG)
    cs = cac.confirm(short, vals)
    assert cs["n_confirming"] == 1 and cs["n_conflicting"] == 1     # Vorzeichen gespiegelt
    assert cac.confirm(spec, {})["confirmation_ratio"] is None


def test_unmapped_sector_has_no_sector_signal():
    spec, _ = cac.candidate_spec(1, None, {}, CFG)
    assert all(not s["signal"].startswith("sector_rs") for s in spec)


# ── Entscheidung ────────────────────────────────────────────────────────────
NEWS_S, NEWS_W = {"strength": "STRONG"}, {"strength": "WEAK"}
CONF_OK = {"n_available": 4, "confirmation_ratio": 0.75, "conflict_share": 0.0}
CONF_LO = {"n_available": 4, "confirmation_ratio": 0.5, "conflict_share": 0.25}


def test_decision_statuses_are_distinct():
    d = lambda **k: tm.decide(**{"news": NEWS_S, "context": {"context_status": 1}, "confirmation": CONF_OK,
                                 "regime_uncertainty": 0.3, "errors": [], "cfg": CFG, **k})
    assert d()["status"] == TRADE
    w = d(confirmation=CONF_LO)
    assert w["status"] == WAIT and w["wait_trigger"]["any_of"] and w["wait_trigger"]["trigger_hash"]
    assert d(context={"context_status": -1})["status"] == ABSTAIN
    assert d(news=NEWS_W, context={"context_status": 0})["status"] == ABSTAIN
    assert d(regime_uncertainty=0.9)["reasons"] == ["REGIME_UNCERTAIN"]
    assert d(confirmation={"n_available": 4, "confirmation_ratio": 0.25, "conflict_share": 0.5})["status"] == ABSTAIN
    e = d(errors=["DATENFEHLER"])
    assert e["status"] == ERROR and e["status"] != ABSTAIN and e["wait_trigger"] is None
    # kein Kontext überhaupt -> ERROR (Datenfehler), nie ABSTAIN
    assert d(context={"context_status": None}, confirmation={"n_available": 0}, regime_uncertainty=None)["status"] == ERROR
    # weniger als 3 verfügbare Signale -> nie TRADE
    assert d(confirmation={"n_available": 2, "confirmation_ratio": 1.0, "conflict_share": 0.0})["status"] == WAIT


def test_kill_conditions_immutable_and_ttm():
    k = tm.kill_conditions(ttm="4-8 Wochen", primary_domain="growth", gap_sign=1, cfg=CFG)
    assert tm.verify_kill(k)
    assert k["conditions"]["catalyst_failure"]["deadline_trading_days"] == 40
    k2 = copy.deepcopy(k)
    k2["conditions"]["risk_stop"]["return_le"] = -0.5
    assert not tm.verify_kill(k2)
    assert tm.ttm_upper_days("2-3 Monate") == 90 and tm.ttm_upper_days(None) is None
    assert tm.kill_conditions(ttm="bald", primary_domain=None, gap_sign=None, cfg=CFG)["conditions"][
        "catalyst_failure"]["source"] == "default"


# ── Expressions ─────────────────────────────────────────────────────────────
def test_expressions_preregistered_and_rule():
    exprs = ex.registered("AAPL", "XLK", CFG)
    assert [e["expression"] for e in exprs] == ["UNDERLYING", "SECTOR_ETF", "SPY", "QQQ", "IWM"]
    assert ex.select({"strength": "STRONG"}, {"context_status": 1}, exprs, CFG)["selected_expression"] == "UNDERLYING"
    assert ex.select({"strength": "WEAK"}, {"context_status": 1}, exprs, CFG)["selected_expression"] == "SECTOR_ETF"
    no_etf = ex.registered("ZZZ", None, CFG)
    assert ex.select({"strength": "WEAK"}, {"context_status": 1}, no_etf, CFG)["selected_expression"] == "UNDERLYING"
    assert not [e for e in no_etf if e["expression"] == "SECTOR_ETF"][0]["available"]


# ── These / Gruppen ─────────────────────────────────────────────────────────
def test_groups():
    assert th.group_of("STRONG", 1) == "A" and th.group_of("STRONG", 0) == "B" and th.group_of("STRONG", -1) == "C"
    assert th.group_of("WEAK", 1) == "D" and th.group_of("WEAK", -1) == "E" and th.group_of("WEAK", None) == "X"


def test_enrich_never_mutates_input_and_records(tmp_path, data):
    an = [fx.make_analysis("AAPL"), fx.make_analysis("XOM", sector="Energy", impact=2, surprise=1),
          fx.make_analysis("JPM", direction="BEARISH", sector="Financial Services"),
          fx.make_analysis("ZZZ", sector=None, impact=None)]
    before = copy.deepcopy(an)
    s = ea.enrich_candidates(an, decision_time=DT, cfg=CFG, loaders=fx.loaders(data["px"], data["ar"], data["cm"]),
                             root=tmp_path, contracts=[], registry=tmp_path / "r.jsonl")
    assert an == before
    assert s["enriched_count"] == 4 and s["recorded"] == 4 and sum(s["status_counts"].values()) == 4
    assert s["production_influence"] == "NONE"
    rows = eal.read_rows(tmp_path)
    req = ("thesis_id", "decision_time", "candidate_id", "asset", "direction", "horizon", "news_edge",
           "market_expectation", "model_expectation", "expectation_gap", "evidence_for", "evidence_against",
           "regime_support", "cross_asset_confirmation", "timing_state", "catalyst", "status", "wait_trigger",
           "kill_conditions", "data_snapshot", "code_version", "config_hash", "macro_alignment", "sector_alignment",
           "regime_state", "regime_uncertainty", "context_status", "group", "expressions", "selected_expression")
    for r in rows:
        assert not [k for k in req if k not in r], r["ticker"]
    zzz = [r for r in rows if r["ticker"] == "ZZZ"][0]
    assert zzz["status"] == ERROR and zzz["group"] == "X"
    # Wiederholter Lauf am selben Tag erzeugt keine Pseudo-Stichprobe
    s2 = ea.enrich_candidates(an, decision_time=DT, cfg=CFG, loaders=fx.loaders(data["px"], data["ar"], data["cm"]),
                              root=tmp_path, contracts=[], registry=tmp_path / "r.jsonl")
    assert s2["recorded"] == 0 and len(eal.read_rows(tmp_path)) == 4
    runs = [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    for k in ("candidate_count", "enriched_count", "status_counts", "missing_counts", "stale_components", "gap_z",
              "confirmation_ratio", "regime_uncertainty", "errors", "runtime_seconds"):
        assert k in runs[-1]


def test_mode_off_writes_nothing(tmp_path, data):
    off = copy.deepcopy(CFG)
    off["mode"] = "off"
    s = ea.enrich_candidates([fx.make_analysis()], decision_time=DT, cfg=off, root=tmp_path)
    assert s == {"mode": "off", "enabled": False} and not any(tmp_path.iterdir())


def test_data_failure_is_error_not_abstain(tmp_path):
    def boom(cfg, t):
        raise ConnectionError("Kursquelle aus")
    s = ea.enrich_candidates([fx.make_analysis()], decision_time=DT, cfg=CFG, root=tmp_path, contracts=[],
                             registry=tmp_path / "r.jsonl",
                             loaders={"archive": lambda c: {}, "prices": boom, "commodity": lambda c, t: pd.DataFrame()})
    assert s["status_counts"][ERROR] == 1 and s["status_counts"][ABSTAIN] == 0
    assert any(e["where"] == "prices" for e in s["errors"])


def test_budget_exceeded_is_error(tmp_path):
    import time
    s = ea.enrich_candidates([fx.make_analysis()], decision_time=DT, cfg=CFG, root=tmp_path, contracts=[],
                             registry=tmp_path / "r.jsonl", deadline=time.monotonic() + 1.0)
    assert s["status_counts"][ERROR] == 1 and "LAUFZEITBUDGET_UEBERSCHRITTEN" in s["run_errors"]


# ── Ledger / Outcomes ───────────────────────────────────────────────────────
def test_path_math_direction_cost():
    bars = [(date(2026, 1, d), 100 * (1 + 0.01 * d) * 1.01, 100 * (1 + 0.01 * d) * 0.99, 100 * (1 + 0.01 * d))
            for d in range(1, 11)]
    up = eal._path(bars, 0, 5, 1.0, 0.001)
    assert up["outcome"] == pytest.approx(106 / 101 - 1, abs=1e-6)
    assert up["outcome_net"] == pytest.approx(up["outcome"] - 0.002, abs=1e-6)
    dn = eal._path(bars, 0, 5, -1.0, 0.001)
    assert dn["outcome"] == pytest.approx(-(106 / 101 - 1), abs=1e-6) and dn["mae"] < 0 and dn["max_drawdown"] < 0


def test_resolve_outcomes_variants_pending_idempotent(tmp_path, data):
    ea.enrich_candidates([fx.make_analysis("AAPL"), fx.make_analysis("PG", sector="Consumer Defensive")],
                         decision_time=DT, cfg=CFG, loaders=fx.loaders(data["px"], data["ar"], data["cm"]),
                         root=tmp_path, contracts=[], registry=tmp_path / "r.jsonl")
    rows = eal.read_rows(tmp_path)
    w = dict(rows[0], observation_id="wait0000000000001", status=WAIT, wait_trigger=tm.wait_trigger(CFG))
    eal.append_jsonl(eal.paths(tmp_path)["candidates"] / "2026-10-09.jsonl", [w], sort_keys=True, default=str)
    st = eal.resolve_outcomes(today=date(2027, 1, 15), root=tmp_path, bars_fn=fx.bars_fn_from(data["px"]), cfg=CFG)
    outs = eal.read_outcomes(tmp_path)
    assert st["pending"] > 0                                      # 120/250 Handelstage noch nicht erreicht
    assert all(k[3] in (20, 60) for k in outs)
    trig = [e for k, e in outs.items() if k[2] == "triggered"]
    assert trig and all(k[1] == "UNDERLYING" for k in outs if k[2] == "triggered")
    for e in trig:
        assert e["entry"] in ("ENTERED", "NO_ENTRY")
        if e["entry"] == "NO_ENTRY":
            assert e["outcome_net"] == 0.0 and "Cash" in e["note"]
        else:
            assert e["entry_day"] >= 1                           # nie Signal und Ausführung am selben Schluss
    assert eal.resolve_outcomes(today=date(2027, 1, 15), root=tmp_path, bars_fn=fx.bars_fn_from(data["px"]),
                                cfg=CFG)["resolved"] == 0
    n1 = len((tmp_path / "outcomes.jsonl").read_text().splitlines())
    eal.resolve_outcomes(today=date(2027, 6, 30), root=tmp_path, bars_fn=fx.bars_fn_from(data["px"]), cfg=CFG)
    n2 = len((tmp_path / "outcomes.jsonl").read_text().splitlines())
    assert n2 > n1                                               # nur neue Horizonte angehängt


def test_tampered_kill_conditions_are_data_bad(tmp_path, data):
    ea.enrich_candidates([fx.make_analysis("AAPL")], decision_time=DT, cfg=CFG,
                         loaders=fx.loaders(data["px"], data["ar"], data["cm"]), root=tmp_path, contracts=[],
                         registry=tmp_path / "r.jsonl")
    f = next((tmp_path / "candidates").glob("*.jsonl"))
    r = json.loads(f.read_text())
    r["kill_conditions"]["conditions"]["risk_stop"]["return_le"] = -0.99
    f.write_text(json.dumps(r) + "\n")
    st = eal.resolve_outcomes(today=date(2027, 1, 15), root=tmp_path, bars_fn=fx.bars_fn_from(data["px"]), cfg=CFG)
    assert st["kill_hash_mismatch"] >= 1
    km = [e for k, e in eal.read_outcomes(tmp_path).items() if k[2] == "kill_managed"]
    assert km and all(e["entry"] == "DATA_BAD" and e["outcome_net"] is None for e in km)


def test_failure_classification_rules():
    from modules.expectation_alpha.evaluation import classify_failure
    r = {"observation_id": "o", "status": TRADE, "date": "2026-10-09", "regime_dimension": "growth",
         "regime_state": {"growth": "high"}, "selected_expression": {"selected_expression": "UNDERLYING"},
         "expressions": [{"expression": "UNDERLYING"}, {"expression": "SPY"}]}
    o = lambda v, **k: {"outcome_net": v, "outcome": v, "mfe": k.get("mfe", 0.0), "exit_date": "2027-01-05", **k}
    outs = {("o", "UNDERLYING", "immediate", 60): o(-0.05), ("o", "SPY", "immediate", 60): o(-0.02)}
    assert classify_failure(r, outs, 60, []) == "THESIS_WRONG"
    outs[("o", "UNDERLYING", "immediate", 60)] = o(-0.05, mfe=0.08)
    assert classify_failure(r, outs, 60, []) == "TIMING_WRONG"
    ctx = [{"date": "2026-12-01", "regime": {"dimensions": {"growth": {"state": "low"}}}}]
    assert classify_failure(r, outs, 60, ctx) == "REGIME_CHANGED"
    outs[("o", "SPY", "immediate", 60)] = o(0.10)
    outs[("o", "QQQ", "immediate", 60)] = o(0.08)
    r["expressions"].append({"expression": "QQQ"})
    assert classify_failure(r, outs, 60, []) == "EXPRESSION_WRONG"
    assert classify_failure({**r, "status": ERROR}, outs, 60, []) == "DATA_BAD"
    outs[("o", "UNDERLYING", "immediate", 60)] = o(0.01)
    assert classify_failure(r, outs, 60, []) is None


# ── Verträge EA001–EA007 ────────────────────────────────────────────────────
def test_ea_contracts_valid_and_research_only():
    cs = [c for c in hc.load() if c.get("eligible_stage") == "EA_NEWS_CANDIDATE"]
    assert sorted(c["hypothesis_id"] for c in cs) == [
        "EA001_EXPECTATION_GAP", "EA002_CONFIRMATION", "EA003_ACCELERATION", "EA004_WAIT", "EA005_CONTEXT_FILTER",
        "EA006_ABSTENTION", "EA007_EXPRESSION"]
    pol = hc.load_policy()
    for c in cs:
        assert hc.validate(c, pol) == [], c["hypothesis_id"]
        assert c["production_class"] == "research_only" and c["maximum_initial_influence"] == "NONE"
        assert c["forward_start"] > c["registered_at"] and c["h1"] and c["population_filter"]
        assert c["minimum_sample_size"] >= 50 and c["minimum_calendar_span"] >= 90 and c["minimum_regimes"] >= 2
        for f in ("primary_metric", "secondary_metrics", "horizon_days", "secondary_horizons_days",
                  "failure_condition", "promotion_criteria", "demotion_criteria"):
            assert c.get(f), (c["hypothesis_id"], f)


def test_validate_ea_rejects_unsafe_variants():
    c = [c for c in hc.load() if c["hypothesis_id"] == "EA001_EXPECTATION_GAP"][0]
    for patch, frag in (({"production_class": "abstention", "maximum_initial_influence": "ABSTENTION_ONLY"},
                         "research_only"),
                        ({"maximum_initial_influence": "RERANK_ONLY"}, "NONE"),
                        ({"population_filter": None}, "population_filter"),
                        ({"population_filter": "ea_unknown > 0"}, "nicht deklarierte"),
                        ({"ea_outcome_kind": "paired_wait_delta", "direction": -1}, "direction +1"),
                        ({"ea_outcome_kind": "magic"}, "ea_outcome_kind")):
        errs = hc.validate({**c, **patch}, hc.load_policy())
        assert any(frag in e for e in errs), (patch, errs)


def test_population_filter_missing_feature_is_out_of_scope():
    c = [c for c in hc.load() if c["hypothesis_id"] == "EA002_CONFIRMATION"][0]
    assert eal.in_population(c, {"ea_gap_abs_z": 1.5, "ea_confirmation_available": 3})
    assert not eal.in_population(c, {"ea_gap_abs_z": None, "ea_confirmation_available": 3})
    assert not eal.in_population(c, {"ea_gap_abs_z": 0.5, "ea_confirmation_available": 3})


def test_stats_block_ci_and_need_more_data():
    from modules.expectation_alpha.evaluation import stats_block
    rng = np.random.default_rng(5)
    items = [{"date": f"2027-01-{1 + i % 28:02d}", "outcome_net": float(v), "outcome": float(v) + 0.002,
              "mae": -0.01, "mfe": 0.02, "max_drawdown": -0.01} for i, v in enumerate(rng.normal(0.01, 0.02, 120))]
    s = stats_block(items, CFG)
    assert s["status"] == "OK" and s["ci95"][0] < s["mean"] < s["ci95"][1]
    assert 0 <= s["hit_rate"] <= 1 and s["risk_adjusted"] is not None
    assert stats_block(items[:10], CFG) == {"n": 10, "independent_dates": 10, "status": "NEED_MORE_DATA"}


def test_price_failure_with_archive_ok_is_error(tmp_path, data):
    def boom(cfg, t):
        raise ConnectionError("Kursquelle aus")
    s = ea.enrich_candidates([fx.make_analysis()], decision_time=DT, cfg=CFG, root=tmp_path, contracts=[],
                             registry=tmp_path / "r.jsonl",
                             loaders={"archive": lambda c: data["ar"], "prices": boom,
                                      "commodity": lambda c, t: data["cm"]})
    assert s["status_counts"] == {"TRADE": 0, "WAIT": 0, "ABSTAIN": 0, "ERROR": 1}
