"""UNIVERSE_V2: Discovery, Optionability, Liquidität, Buckets, Kosten, Netto-Outcomes, Shadow-Ledger,
segmentweise Promotion/Demotion, V1-Fallback und strikte V1/V2-Trennung (alles ohne Netz)."""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from modules import final_mc_ledger as fml
from modules import hypothesis_contract as hc
from modules import promotion_controller as pc
from modules import universe_v2 as uv
from modules import universe_v2_ledger as v2l
from modules import universe_v2_scan as v2s

CFG = uv.load_cfg()
TODAY = date(2026, 10, 12)
CTX = {"safe_mode_active": 0, "blind_spot_sectors": [], "ml_cards": {}, "system_state_version": 7, "drift_level": "NORMAL"}


def _chain(price=20.0, oi=1000, vol=100, spread=0.02, today=TODAY, n_exp=4, sym="X"):
    rows = []
    for k in range(n_exp):
        e = (today + timedelta(days=45 + 30 * k)).isoformat()
        for st in (0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 0.85):
            mid = max(0.2, price * 0.08 * (1.2 - st))
            rows.append({"symbol": f"{sym}{e}C{st}", "expiry": e, "strike": round(price * st, 2), "option_type": "call",
                         "bid": round(mid * (1 - spread / 2), 3), "ask": round(mid * (1 + spread / 2), 3),
                         "open_interest": oi, "volume": vol})
    return rows


def _quote(price=20.0, avg=2_000_000, vol=4_000_000, spread=0.001, last=TODAY):
    return {"price": price, "bid": price * (1 - spread / 2), "ask": price * (1 + spread / 2), "avg_volume": avg,
            "volume": vol, "last_trade": last.isoformat(), "closes": [price] * 25}


# ── Buckets / Discovery ─────────────────────────────────────────────────────
@pytest.mark.parametrize("mc,b", [(10e6, "ULTRA_MICRO"), (50e6, "MICRO"), (299e6, "MICRO"), (300e6, "SMALL"),
                                  (2e9, "MID"), (10e9, "LARGE"), (250e9, "MEGA"), (None, "UNKNOWN"), (0, "UNKNOWN"),
                                  (float("nan"), "UNKNOWN")])
def test_market_cap_buckets_and_missing_cap(mc, b):
    assert uv.market_cap_bucket(mc, CFG) == b


def test_listing_parse_keeps_regular_exchanges_and_equity_tickers():
    js = {"fields": ["cik", "name", "ticker", "exchange"], "data": [
        [1, "A Corp", "AAA", "Nasdaq"], [2, "B Corp", "BBB", "NYSE"], [3, "Otc Co", "OTCX", "OTC"],
        [4, "Warrant", "ABCDW", "Nasdaq"], [5, "Unit", "SPAC-U", "NYSE"], [6, "Dup", "AAA", "Nasdaq"],
        [7, "Class B", "BRK-B", "NYSE"], [8, "No exch", "NOEX", None]]}
    rows, skipped = uv.parse_sec_listings(js, CFG)
    assert [r["ticker"] for r in rows] == ["AAA", "BBB", "BRK-B"]
    assert skipped == {"exchange": 2, "ticker_pattern": 2, "duplicate": 1}


def test_delisted_or_suspended_is_excluded():
    assert uv.listing_status(TODAY.isoformat(), TODAY, CFG) == "ACTIVE"
    assert uv.listing_status((TODAY - timedelta(days=21)).isoformat(), TODAY, CFG) == "STALE"
    assert uv.listing_status(None, TODAY, CFG) == "UNKNOWN"
    a = uv.assess("OLD", _quote(last=TODAY - timedelta(days=30)), _chain(), 500e6, TODAY, CFG)
    assert not a["research_ok"] and not a["tradeable_ok"] and any("STALE" in r for r in a["research_fail"])


def test_option_availability_filter_and_missing_expiries():
    a = uv.assess("NOOPT", _quote(), [], 500e6, TODAY, CFG)
    assert not a["optionable"] and not a["research_ok"] and "keine gelisteten Optionen" in a["research_fail"]
    far = [dict(r, expiry=(TODAY + timedelta(days=600)).isoformat()) for r in _chain()]
    b = uv.assess("FAR", _quote(), far, 500e6, TODAY, CFG)
    assert b["optionable"] and not b["research_ok"] and any("Verfälle" in r for r in b["research_fail"])


def test_research_vs_tradeable_and_chain_never_automatically_tradeable():
    good = uv.assess("GOOD", _quote(), _chain(), 5e9, TODAY, CFG)
    assert good["research_ok"] and good["tradeable_ok"] and good["execution_quality"] in ("GOOD", "FAIR")
    thin = uv.assess("THIN", _quote(), _chain(oi=5, vol=0), 200e6, TODAY, CFG)   # OI summiert ~140
    assert thin["optionable"] and thin["research_ok"] and not thin["tradeable_ok"]
    assert thin["options_liquidity_bucket"] == "THIN" and thin["market_cap_bucket"] == "MICRO"


def test_extreme_spreads_and_high_slippage():
    wide = uv.assess("WIDE", _quote(), _chain(spread=0.8), 1e9, TODAY, CFG)
    assert not wide["research_ok"] and any("Spread" in r for r in wide["research_fail"])
    deep = uv.execution_cost(1.95, 2.05, 5000, CFG)
    thin = uv.execution_cost(1.95, 2.05, 20, CFG)
    assert thin["estimated_slippage"] == pytest.approx(2 * deep["estimated_slippage"])
    assert uv.execution_quality(0.03, CFG) == "GOOD" and uv.execution_quality(0.5, CFG) == "UNTRADEABLE"
    assert uv.execution_cost(None, 2.0, 100, CFG)["estimated_execution_cost"] is None


def test_penny_and_manipulation_flags_block_production():
    q = _quote(price=0.8, avg=3_000_000, vol=40_000_000)
    a = uv.assess("PUMP", q, _chain(price=0.8), 20e6, TODAY, CFG)
    assert {"PENNY_STOCK", "MANIPULATION_RISK"} <= set(a["risk_flags"]) and not a["tradeable_ok"]


def test_net_outcome_theoretical_vs_realizable():
    r = uv.net_outcome(raw_underlying_return=0.1, entry_bid=1.9, entry_ask=2.1, exit_mid=3.0, open_interest=1000, cfg=CFG)
    assert r["option_theoretical_return"] == pytest.approx(0.5)
    assert r["net_realizable_return"] < r["option_theoretical_return"] and r["estimated_execution_cost"] > 0
    assert r["outcome_status"] == "ESTIMATED_EXIT_SPREAD" and r["liquid_option_available"]
    q = uv.net_outcome(raw_underlying_return=0.1, entry_bid=1.9, entry_ask=2.1, exit_bid=2.9, exit_ask=3.1, cfg=CFG)
    assert q["outcome_status"] == "QUOTED_EXIT"
    d = uv.net_outcome(raw_underlying_return=None, entry_bid=1.9, entry_ask=2.1, delisted=True, cfg=CFG)
    assert d["net_realizable_return"] == -1.0 and d["outcome_status"] == "DELISTED_WORST_CASE"
    n = uv.net_outcome(raw_underlying_return=0.1, entry_bid=None, entry_ask=2.1, exit_mid=3.0, cfg=CFG)
    assert n["outcome_status"] == "NO_LIQUID_OPTION" and n["net_realizable_return"] is None


def test_discovery_snapshot_survivorship_and_rolling(tmp_path):
    js = {"fields": ["cik", "name", "ticker", "exchange"],
          "data": [[i, f"C{i}", f"T{i}", "Nasdaq"] for i in range(6)]}
    caps = {"T0": 20e6, "T1": 100e6, "T2": 1e9, "T3": 5e9, "T4": 50e9, "T5": None}
    recs, meta = uv.discover(TODAY, CFG, listings_js=js, quotes_fn=lambda ts: {t: _quote() for t in ts},
                             chain_fn=lambda t, d, c: (_chain(sym=t), "fake"), cap_fn=lambda t: (caps[t], "fake"))
    snap = uv.save_snapshot(recs, TODAY, meta, snap_dir=tmp_path / "s", delistings=tmp_path / "d.jsonl")
    sm = snap["summary"]
    assert sm["n_optionable"] == 6 and sm["research_by_bucket"]["UNKNOWN"] == 1
    assert set(sm["research_by_bucket"]) == {"ULTRA_MICRO", "MICRO", "SMALL", "MID", "LARGE", "UNKNOWN"}
    # eine Woche später: T1 delisted (fehlt), T2 suspendiert (STALE) -> Delisting-Ereignisse, nie still entfernt
    js2 = {"fields": js["fields"], "data": [r for r in js["data"] if r[2] != "T1"]}
    d2 = TODAY + timedelta(days=7)
    q2 = lambda ts: {t: (_quote(last=d2) if t != "T2" else _quote(last=TODAY - timedelta(days=20))) for t in ts}
    recs2, _ = uv.discover(d2, CFG, listings_js=js2, quotes_fn=q2, chain_fn=lambda t, d, c: (_chain(sym=t, today=d2), "f"),
                           cap_fn=lambda t: (caps[t], "f"), prev=snap)
    uv.save_snapshot(recs2, d2, {}, snap_dir=tmp_path / "s", delistings=tmp_path / "d.jsonl")
    ev = {e["ticker"]: e["event"] for e in fml.read_jsonl(tmp_path / "d.jsonl")}
    assert ev == {"T1": "MISSING", "T2": "STALE"}
    assert uv.latest_snapshot(tmp_path / "s")["as_of"] == d2.isoformat()


def test_first_run_over_budget_marks_unchecked_and_next_run_prioritises_them(tmp_path):
    """Live 2026-10-04: ohne Vor-Snapshot griff das Budget nie -> Job-Timeout, kein Snapshot."""
    js = {"fields": ["cik", "name", "ticker", "exchange"],
          "data": [[i, f"C{i}", f"T{i}", "Nasdaq"] for i in range(5)]}
    calls = []

    def chain(t, d, c):
        calls.append(t)
        return _chain(sym=t), "fake"
    recs, meta = uv.discover(TODAY, CFG, listings_js=js, quotes_fn=lambda ts: {t: _quote() for t in ts},
                             chain_fn=chain, cap_fn=lambda t: (5e9, "fake"), budget_s=-1)
    assert meta["checked_this_run"] == 0 and meta["unchecked_new"] == 5 and calls == []
    assert all(r["status"] == "UNCHECKED" and not r["research_ok"] and not r["tradeable_ok"]
               and not r["optionable"] and r["market_cap"] is None for r in recs)
    snap = uv.save_snapshot(recs, TODAY, meta, snap_dir=tmp_path / "s", delistings=tmp_path / "d.jsonl")
    assert snap["summary"]["n_unchecked"] == 5 and snap["summary"]["n_listed_checked"] == 0
    # Folgelauf mit Budget: UNCHECKED zuerst geprüft, echte Bewertung ersetzt den Platzhalter
    recs2, meta2 = uv.discover(TODAY + timedelta(days=7), CFG, listings_js=js,
                               quotes_fn=lambda ts: {t: _quote(last=TODAY + timedelta(days=7)) for t in ts},
                               chain_fn=lambda t, d, c: (_chain(sym=t, today=d), "f"), cap_fn=lambda t: (5e9, "f"),
                               prev=snap, budget_s=3600)
    assert meta2["checked_this_run"] == 5 and meta2["unchecked_new"] == 0
    assert all(r["status"] != "UNCHECKED" for r in recs2)


def test_v1_frozen_and_unchanged():
    frozen = json.loads(uv.V1_FROZEN.read_text())
    assert frozen["universe_version"] == "V1" and frozen["hard_filters"]["min_market_cap_usd"] == 2_000_000_000
    assert uv.v1_unchanged()
    assert uv.universe_version_of({}) == "V1" and uv.universe_version_of({"universe_version": "V2"}) == "V2"


# ── Verträge / Trennung ─────────────────────────────────────────────────────
def _seg(b):
    return next(c for c in hc.load() if hc.key(c) == f"UNIV-V2-SEG-{b}@v1")


def test_segment_contracts_valid_and_v1_untouched():
    cs = hc.load()
    seg = [c for c in cs if c.get("universe_version") == "V2"]
    assert {c["segment"]["market_cap_bucket"] for c in seg} == {"ULTRA_MICRO", "MICRO", "SMALL", "MID"}
    for c in seg:
        assert hc.validate(c) == [] and c["primary_outcome"] == "net_realizable_return"
        assert c["maximum_initial_influence"] == "NONE" and c["eligible_stage"] == v2l.STAGE
    base = json.loads(open("outputs/state/baselines/post-pr91_2026-10-03.json").read())
    assert {x["key"]: x["spec_hash"] for x in base["contracts"]} == \
        {hc.key(c): hc.spec_hash(c) for c in cs if hc.key(c).startswith("PROM-ABST") and c["version"] == 1}
    bad = copy.deepcopy(seg[0]); bad["primary_outcome"] = "raw_underlying_return"; bad.pop("spec_hash")
    assert any("net_realizable_return" in e for e in hc.validate(bad))


def test_registry_accepts_all_segments_without_similarity_block(tmp_path):
    st = hc.register(hc.load(), tmp_path / "r.jsonl", hc.load_policy())
    assert all(v["status"] == "VALID" for v in st.values())


def test_adapter_never_applies_v2_contracts_to_v1_champion_trades(tmp_path):
    from modules import production_intelligence_adapter as pia
    cs = hc.load()
    reg = tmp_path / "r.jsonl"
    hc.register(cs, reg, hc.load_policy())
    props = [{"ticker": "AAA", "features": {}, "trade_score": {"total": 60}, "simulation": {"hit_rate": 0.7}}]
    kept, blocked, recs = pia.apply_to_proposals(props, vix=18, today="2026-10-13", context=dict(CTX), contracts=cs,
                                                 state_path=tmp_path / "s.json", registry=reg,
                                                 transitions=tmp_path / "t.jsonl", ledger_dir=tmp_path / "l", policy={})
    assert not any(k.startswith("UNIV-V2") for k in recs[0]["intelligence"]) and not blocked
    assert pia.v2_segment_levels(state_path=tmp_path / "missing.json", contracts=cs, registry=reg,
                                 transitions=tmp_path / "t.jsonl") == {}          # fail-safe: V1-Fallback


# ── Shadow-Scan + Ledger ────────────────────────────────────────────────────
def _snapshot(tickers_caps: dict, today=TODAY):
    recs = [uv.assess(t, _quote(last=today), _chain(sym=t, today=today), mc, today, CFG, "fake")
            for t, mc in tickers_caps.items()]
    return {"as_of": today.isoformat(), "records": recs}


def test_scan_uses_champion_quant_stages_and_records_all_fields(tmp_path):
    snap = _snapshot({"S1": 5e9, "S2": 5e9, "LOWRV": 5e9, "NONEWS": 5e9, "LOWMC": 5e9})
    quotes = {t: _quote(last=TODAY) for t in ("S1", "S2", "NONEWS", "LOWMC")} | {"LOWRV": _quote(vol=100_000)}
    cands, stats = v2s.scan(TODAY, snap, vix=20, quotes_fn=lambda ts: {t: quotes[t] for t in ts},
                            news_fn=lambda t: [] if t == "NONEWS" else ["a", "b"],
                            premc_fn=lambda t, p: 0.2 if t == "LOWMC" else 0.6,
                            chain_fn=lambda t, d, c: (_chain(sym=t), "fake"))
    assert sorted(c["ticker"] for c in cands) == ["S1", "S2"] and stats["premc_pass"] == 2
    reg = tmp_path / "r.jsonl"
    hc.register(hc.load(), reg, hc.load_policy())
    rows = v2l.record_candidates(cands, today=TODAY.isoformat(), vix=20, ctx=dict(CTX), registry=reg,
                                 ledger_dir=tmp_path / "led", v1_members={"S2"},
                                 now=datetime(2026, 10, 12, 15, tzinfo=timezone.utc))
    again = v2l.record_candidates(cands, today=TODAY.isoformat(), vix=20, ctx=dict(CTX), registry=reg,
                                  ledger_dir=tmp_path / "led", v1_members={"S2"},
                                  now=datetime(2026, 10, 12, 19, tzinfo=timezone.utc))
    assert len(rows) == 2 and again == []
    r = next(x for x in rows if x["ticker"] == "S1")
    for f in ("universe_version", "market_cap", "market_cap_bucket", "underlying_liquidity", "liquidity_bucket",
              "options_liquidity", "options_liquidity_bucket", "spread_metrics", "estimated_slippage",
              "execution_cost", "execution_quality", "reference_option", "risk_flags", "system_state"):
        assert f in r, f
    assert r["universe_version"] == "V2" and r["in_universe_v1"] == 0 and r["production_effect"] == "NONE"
    assert r["contracts"]["UNIV-V2-SEG-SMALL@v1"]["fired"] is False and r["market_cap_bucket"] == "MID"
    assert r["contracts"]["UNIV-V2-SEG-MID@v1"]["fired"] is True
    s2 = next(x for x in rows if x["ticker"] == "S2")
    assert s2["contracts"]["UNIV-V2-SEG-MID@v1"]["fired"] is False            # V1-Mitglied = Referenz


def _bars(effect):
    def f(t, s, e):
        out, d = [], s
        while d <= e:
            k = min(60, (d - s).days)
            px = 100 * (1 + effect.get(t, 0.0) * k / 60)
            out.append((d, px * 1.01, px * 0.99, px))
            d += timedelta(days=1)
        return out, []
    return f


def test_corporate_action_and_delisting_outcomes(tmp_path, monkeypatch):
    reg = tmp_path / "r.jsonl"
    hc.register(hc.load(), reg, hc.load_policy())
    snap = _snapshot({"SPLT": 1e9, "GONE": 1e9})
    cands = [{"ticker": t, "assessment": next(r for r in snap["records"] if r["ticker"] == t), "hit_rate": 0.6}
             for t in ("SPLT", "GONE")]
    v2l.record_candidates(cands, today="2026-10-12", vix=18, ctx=dict(CTX), registry=reg, ledger_dir=tmp_path / "led",
                          now=datetime(2026, 10, 12, 15, tzinfo=timezone.utc))
    monkeypatch.setattr(uv, "latest_snapshot", lambda *a: {"records": [{"ticker": "GONE", "status": "STALE"}]})

    def bars(t, s, e):
        if t == "GONE":
            return [(s, 101, 99, 100)], []
        b, _ = _bars({})(t, s, e)
        return b, [s + timedelta(days=10)]
    n = v2l.resolve_outcomes(today=date(2027, 1, 1), ledger_dir=tmp_path / "led", path=tmp_path / "o.jsonl",
                             bars_fn=bars, option_close_fn=lambda sym, d: 1.0)
    oc = v2l.read_outcomes(tmp_path / "o.jsonl")
    st = {r["ticker"]: oc[(r["observation_id"], 45)]["outcome_status"] for r in fml.read_rows(tmp_path / "led")}
    assert n == 6 and st == {"SPLT": "CORPORATE_ACTION", "GONE": "DELISTED_WORST_CASE"}


# ── Promotion / Demotion je Segment ─────────────────────────────────────────
def _forward_data(tmp_path, days=120, effects=None):
    """Täglich je Segment-Ticker (eindeutig = eigene Cluster) und V1-Referenz; Netto über Optionsexits."""
    effects = effects or {"SMALL": 0.6, "MICRO": -0.6, "REF": -0.1}
    reg, led, outs = tmp_path / "r.jsonl", tmp_path / "led", tmp_path / "o.jsonl"
    hc.register(hc.load(), reg, hc.load_policy())
    exit_mid = {}
    for k in range(days):
        d = date(2026, 10, 12) + timedelta(days=k)
        caps = {f"SM{k}": 1e9, f"MI{k}": 100e6, f"RF{k}": 1e9}
        snap = _snapshot(caps, today=d)
        cands = []
        for r in snap["records"]:
            cands.append({"ticker": r["ticker"], "assessment": r, "hit_rate": 0.7})
            grp = {"SM": "SMALL", "MI": "MICRO", "RF": "REF"}[r["ticker"][:2]]
            ref = r["reference_option"]
            mid = (ref["bid"] + ref["ask"]) / 2
            exit_mid[ref["symbol"]] = mid * (1 + effects[grp] + 0.05 * ((k % 5) - 2) / 2)
        for c in cands:      # Segment-Zuordnung: SMALL/MICRO über Market Cap, Referenz = V1-Mitglied
            if c["ticker"].startswith("SM"):
                c["assessment"] = dict(c["assessment"], market_cap_bucket="SMALL")
        v2l.record_candidates(cands, today=d.isoformat(), vix=18.0 if k % 2 else 30.0, ctx=dict(CTX), registry=reg,
                              ledger_dir=led, v1_members={f"RF{k}"}, now=datetime(d.year, d.month, d.day, 15, tzinfo=timezone.utc))
    v2l.resolve_outcomes(today=date(2027, 4, 1), ledger_dir=led, path=outs, bars_fn=_bars({}),
                         option_close_fn=lambda sym, on: exit_mid.get(sym))
    return reg, led, outs


def _run(tmp_path, reg, led, outs, now, approvals=None):
    return pc.run(contracts=hc.load(), policy=hc.load_policy(), now=now, registry=reg, transitions=tmp_path / "tr.jsonl",
                  ledger_dir=tmp_path / "champ", outcomes_path=tmp_path / "co.jsonl", looks_path=tmp_path / "looks.jsonl",
                  state_path=tmp_path / "state.json", history={}, approvals=approvals or {}, safe_mode={"active": False},
                  final_mc_dir=tmp_path / "fm", final_mc_outcomes=tmp_path / "fmo.jsonl", v2_dir=led, v2_outcomes=outs)


def test_segmentwise_promotion_needs_human_approval_and_v1_evidence_untouched(tmp_path):
    reg, led, outs = _forward_data(tmp_path)
    s1 = _run(tmp_path, reg, led, outs, datetime(2027, 4, 1, tzinfo=timezone.utc))
    H = s1["hypotheses"]
    small, micro = H["UNIV-V2-SEG-SMALL@v1"], H["UNIV-V2-SEG-MICRO@v1"]
    assert small["evidence"]["outcome_basis"] == "net_realizable_return" and small["evidence"]["n_observations"] == 120
    assert small["state"] == "FORWARD_VALIDATED" and small["influence_level"] == "NONE", small["reasons"]
    assert "menschlicher Freigabe" in str(small["recommendation"])
    assert micro["state"] == "REJECTED"                                    # signifikant negativ netto
    assert H["UNIV-V2-SEG-ULTRA_MICRO@v1"]["evidence"]["n_observations"] == 0
    for k, h in H.items():                                                 # V1-/Final-MC-Evidenz nie berührt
        if not k.startswith("UNIV-V2"):
            assert h["evidence"]["n_observations"] == 0 and h["influence_level"] == "NONE"
    assert small["multiple_testing"]["hypothesis_family"] == "universe_segment@V2_CANDIDATE"
    appr = {"UNIV-V2-SEG-SMALL@v1": {"approved_level": "TRADE_RECOMMENDATION_ENABLED", "pr": "test"}}
    s2 = _run(tmp_path, reg, led, outs, datetime(2027, 4, 30, tzinfo=timezone.utc), appr)
    assert s2["hypotheses"]["UNIV-V2-SEG-SMALL@v1"]["influence_level"] == "RERANK_ONLY"   # genau eine Stufe
    assert s2["hypotheses"]["UNIV-V2-SEG-MICRO@v1"]["influence_level"] == "NONE"


def test_approval_steps_one_level_and_only_with_current_evidence():
    c = _seg("SMALL")
    cur = {"new_state": "LIMITED_PRODUCTION", "influence_level": "WEIGHT_10", "timestamp": "2027-01-01T00:00:00+00:00"}
    d = {"decision": "ALLOW_10_PERCENT_WEIGHT", "new_state": "LIMITED_PRODUCTION", "influence_level": "WEIGHT_10",
         "reasons": [], "recommendation": None}
    weak = {"n_observations": 3, "n_independent_dates": 3, "calendar_span_days": 5, "n_fired": 3,
            "n_fired_independent_dates": 3, "regimes": [], "ci": [None, None], "delta_expectancy": None}
    r = pc.apply_approval(c, cur, dict(d), {"approved_level": "TRADE_RECOMMENDATION_ENABLED"}, weak,
                          hc.load_policy(), datetime(2027, 3, 1, tzinfo=timezone.utc))
    assert r["influence_level"] == "WEIGHT_10" and "nicht erfüllt" in r["reasons"][-1]
    r2 = pc.apply_approval(c, cur, dict(d), {"approved_level": "WEIGHT_10"}, weak, hc.load_policy(),
                           datetime(2027, 3, 1, tzinfo=timezone.utc))
    assert r2["influence_level"] == "WEIGHT_10"


def test_demotion_steps_down_ladder_to_v1_fallback():
    c = _seg("SMALL")
    post = {"n_observations": 40, "n_independent_dates": 30, "calendar_span_days": 60, "n_fired": 40,
            "n_fired_independent_dates": 30, "regimes": ["vix_high", "vix_normal"], "ci": [-0.2, 0.1],
            "delta_expectancy": -0.05, "fired": {"net_expectancy": -0.05, "avg_slippage": 0.3,
                                                 "execution_quality_ok_share": 0.4, "precision_at_k": 0.2},
            "policy": {}, "all": {}}
    cur = {"new_state": "LIMITED_PRODUCTION", "influence_level": "TRADE_RECOMMENDATION_ENABLED"}
    d = pc.decide(c, cur, post, hc.load_policy(), integrity_ok=True, looks=2, alpha_info={"planned_looks": 12},
                  now=datetime(2027, 6, 1, tzinfo=timezone.utc), post_ev=post)
    assert d["decision"] == "DEMOTE" and d["influence_level"] == "WEIGHT_10"
    assert any("Slippage" in r for r in d["reasons"]) and any("Execution" in r for r in d["reasons"])
    cur2 = {"new_state": "LIMITED_PRODUCTION", "influence_level": "RERANK_ONLY"}
    d2 = pc.decide(c, cur2, post, hc.load_policy(), integrity_ok=True, looks=2, alpha_info={"planned_looks": 12},
                   now=datetime(2027, 6, 1, tzinfo=timezone.utc), post_ev=post)
    assert d2["new_state"] == "PROSPECTIVE_CHALLENGER" and d2["influence_level"] == "NONE"       # zurück auf SHADOW


def test_recommendation_gate_only_for_enabled_segments_and_tradeable_rows():
    cs = [c for c in hc.load() if c.get("universe_version") == "V2"]
    rows = [{"ticker": "A", "tradeable_v2": True, "risk_flags": [], "contracts": {"UNIV-V2-SEG-SMALL@v1": {"fired": True}}},
            {"ticker": "B", "tradeable_v2": False, "risk_flags": [], "contracts": {"UNIV-V2-SEG-SMALL@v1": {"fired": True}}},
            {"ticker": "C", "tradeable_v2": True, "risk_flags": ["EXTREME_GAP"],
             "contracts": {"UNIV-V2-SEG-SMALL@v1": {"fired": True}}},
            {"ticker": "D", "tradeable_v2": True, "risk_flags": [], "contracts": {"UNIV-V2-SEG-MICRO@v1": {"fired": True}}}]
    assert v2l.recommendations(rows, {}, cs) == []                                   # Standard: nichts
    assert v2l.recommendations(rows, {"UNIV-V2-SEG-SMALL@v1": "WEIGHT_10"}, cs) == []
    got = v2l.recommendations(rows, {"UNIV-V2-SEG-SMALL@v1": "TRADE_RECOMMENDATION_ENABLED"}, cs)
    assert [r["ticker"] for r in got] == ["A"]


def test_trade_mail_shows_no_v2_without_enabled_segment(monkeypatch):
    from modules import email_reporter as er, production_intelligence_adapter as pia
    monkeypatch.setattr(pia, "v2_segment_levels", lambda *a, **k: {"UNIV-V2-SEG-SMALL@v1": "WEIGHT_10"})
    assert er._v2_recommendation_html("2026-10-13") == ""


# ── End-to-End ──────────────────────────────────────────────────────────────
def test_end_to_end_discovery_to_recommendation_gate(tmp_path, monkeypatch):
    # Discovery -> Optionability -> Liquidity -> Bucket
    js = {"fields": ["cik", "name", "ticker", "exchange"], "data": [[1, "S", "SMLL", "NYSE"], [2, "R", "REFV", "NYSE"],
                                                                  [3, "N", "NOOP", "Nasdaq"]]}
    caps = {"SMLL": 800e6, "REFV": 20e9, "NOOP": 900e6}
    recs, meta = uv.discover(TODAY, CFG, listings_js=js, quotes_fn=lambda ts: {t: _quote() for t in ts},
                             chain_fn=lambda t, d, c: ([] if t == "NOOP" else _chain(sym=t), "fake"),
                             cap_fn=lambda t: (caps[t], "fake"))
    snap = uv.save_snapshot(recs, TODAY, meta, snap_dir=tmp_path / "s", delistings=tmp_path / "d.jsonl")
    assert set(uv.membership(snap)) == {"SMLL", "REFV"}
    # V2 Candidate -> Shadow Ledger
    cands, _ = v2s.scan(TODAY, snap, vix=20, quotes_fn=lambda ts: {t: _quote() for t in ts},
                        news_fn=lambda t: ["n"], premc_fn=lambda t, p: 0.6,
                        chain_fn=lambda t, d, c: (_chain(sym=t), "fake"))
    reg = tmp_path / "r.jsonl"
    hc.register(hc.load(), reg, hc.load_policy())
    rows = v2l.record_candidates(cands, today=TODAY.isoformat(), vix=20, ctx=dict(CTX), registry=reg,
                                 ledger_dir=tmp_path / "led", v1_members={"REFV"},
                                 now=datetime(2026, 10, 12, 15, tzinfo=timezone.utc))
    assert {r["ticker"]: r["market_cap_bucket"] for r in rows} == {"SMLL": "SMALL", "REFV": "LARGE"}
    # Delayed Outcome -> Net Outcome
    exit_mid = {r["reference_option"]["symbol"]: 2 * r["reference_option"]["ask"] for r in rows}
    n = v2l.resolve_outcomes(today=date(2027, 1, 1), ledger_dir=tmp_path / "led", path=tmp_path / "o.jsonl",
                             bars_fn=_bars({"SMLL": 0.1, "REFV": 0.02}), option_close_fn=lambda s, d: exit_mid[s])
    oc = v2l.read_outcomes(tmp_path / "o.jsonl")
    o = oc[(rows[0]["observation_id"], 45)]
    assert n == 6 and o["net_realizable_return"] < o["option_theoretical_return"] and o["raw_underlying_return"] is not None
    # Forward Evidence -> PromotionController (zu wenig Daten) -> Adapter -> keine Empfehlung
    state = _run(tmp_path, reg, tmp_path / "led", tmp_path / "o.jsonl", datetime(2027, 1, 1, tzinfo=timezone.utc))
    h = state["hypotheses"]["UNIV-V2-SEG-SMALL@v1"]
    assert h["evidence"]["n_observations"] == 1 and "NEED_MORE_DATA" in h["reasons"][0]
    from modules import production_intelligence_adapter as pia
    levels = pia.v2_segment_levels(state_path=tmp_path / "state.json", registry=reg, transitions=tmp_path / "tr.jsonl")
    assert set(levels.values()) == {"NONE"}
    assert v2l.recommendations(rows, levels, v2l.v2_contracts()) == []                # optional: nur nach Freigabe
    assert v2l.recommendations(rows, {"UNIV-V2-SEG-SMALL@v1": "TRADE_RECOMMENDATION_ENABLED"},
                               v2l.v2_contracts())[0]["ticker"] == "SMLL"
