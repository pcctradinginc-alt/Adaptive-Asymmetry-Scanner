"""Final-MC-Population: eigener Ledger, eigene Verträge (v2), strikt getrennte Forward-Evidenz.

End-to-End: Final-MC-Survivor -> Vertragsauswertung -> Shadow-Ledger -> verzögertes Outcome ->
Forward-Evidenz -> PromotionController. Beweist: nichts vor forward_start zählt, v1 (Champion)
und v2 (Final-MC) werden nie vermischt, v2 erhält nie automatischen Produktionseinfluss."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from modules import final_mc_ledger as fml
from modules import hypothesis_contract as hc
from modules import promotion_controller as pc

CTX = {"safe_mode_active": 0, "blind_spot_sectors": [], "ml_cards": {}, "system_state_version": 7,
       "drift_level": "NORMAL"}


def _repo(key):
    return next(c for c in hc.load() if hc.key(c) == key)


# ── Bausteine ───────────────────────────────────────────────────────────────
def test_v2_contracts_store_population_and_do_not_touch_v1():
    cs = hc.load()
    v1 = [c for c in cs if c["version"] == 1]
    v2 = [c for c in cs if c["version"] == 2]
    assert len(v1) == 5 and len(v2) == 5
    assert all("eligible_stage" not in c for c in v1)              # v1-Spezifikation unverändert
    for c in v2:
        for f in ("eligible_stage", "population_definition", "forward_start", "horizon_days", "baseline",
                  "baseline_population", "promotion_criteria", "spec_hash", "minimum_independent_event_clusters"):
            assert c.get(f), f
        assert c["eligible_stage"] == c["baseline_population"] == fml.STAGE
        assert c["spec_hash"] == hc.spec_hash(c) and c["maximum_initial_influence"] == "NONE"
        assert c["promotion_criteria"]["min_fired_event_clusters"] >= 15
    baseline = json.loads(open("outputs/state/baselines/post-pr91_2026-10-03.json").read())
    frozen = {x["key"]: x["spec_hash"] for x in baseline["contracts"]}
    assert {hc.key(c): hc.spec_hash(c) for c in v1} == frozen      # gegen eingefrorene Baseline


def test_stage_contract_validation():
    c = dict(_repo("PROM-ABST-005@v2"))
    assert hc.validate(c) == []
    assert any("baseline_population" in e for e in hc.validate({**c, "baseline_population": "CHAMPION_TRADE"}))
    assert any("spec_hash" in e for e in hc.validate({**c, "thresholds": {"op": ">", "value": 25}}))
    assert any("Ereignis" in e or "event_clusters" in e
               for e in hc.validate({k: v for k, v in c.items() if k != "minimum_independent_event_clusters"}))


def test_path_outcome_direction_mfe_mae_and_incomplete_window():
    d0 = date(2026, 10, 5)
    bars = [(d0, 101, 99, 100.0), (d0 + timedelta(days=1), 110, 95, 104.0), (d0 + timedelta(days=20), 104, 100, 102.0)]
    r = fml.path_outcome(bars, d0, 20, "BULLISH")
    assert r["outcome"] == pytest.approx(0.02) and r["mfe"] == pytest.approx(0.10) and r["mae"] == pytest.approx(-0.05)
    b = fml.path_outcome(bars, d0, 20, "BEARISH")
    assert b["outcome"] == pytest.approx(-0.02) and b["mfe"] == pytest.approx(0.05) and b["mae"] == pytest.approx(-0.10)
    assert fml.path_outcome(bars, d0, 45, "BULLISH") is None        # Horizont noch nicht erreicht


def test_clusters_merge_repeated_ticker_signals():
    obs = [{"ticker": "AMD", "date": "2026-10-05"}, {"ticker": "AMD", "date": "2026-10-09"},
           {"ticker": "AMD", "date": "2026-10-30"}, {"ticker": "AVGO", "date": "2026-10-05"}]
    fml.assign_clusters(obs)
    assert len({o["cluster"] for o in obs}) == 3
    assert obs[0]["cluster"] == obs[1]["cluster"] != obs[2]["cluster"]


def test_insufficiency_counts_clusters_not_just_n():
    c = _repo("PROM-ABST-005@v2")
    ev = {"n_observations": 500, "n_independent_dates": 100, "calendar_span_days": 200, "n_fired": 100,
          "n_fired_independent_dates": 50, "regimes": ["vix_high", "vix_normal"], "n_event_clusters": 12,
          "n_fired_event_clusters": 3}
    need = pc.insufficiency(c, ev, hc.load_policy())
    assert any("Ereignis-Cluster 12/80" in n for n in need) and any("Treffer-Cluster 3/15" in n for n in need)


def test_family_alpha_separate_per_population():
    reg = [{"key": hc.key(c), "hypothesis_id": c["hypothesis_id"], "contract": c} for c in hc.load()]
    pol = hc.load_policy()
    a1 = pc.family_alpha(_repo("PROM-ABST-001@v1"), reg, pol)
    a2 = pc.family_alpha(_repo("PROM-ABST-001@v2"), reg, pol)
    assert a1["hypothesis_family"] == "abstention" and a1["family_size"] == 5      # v1 unverändert
    assert a2["hypothesis_family"] == "abstention@FINAL_MC_SURVIVOR" and a2["family_size"] == 5


def test_adapter_never_evaluates_final_mc_contracts_on_champion_trades(tmp_path):
    from modules import production_intelligence_adapter as pia
    cs = [_repo("PROM-ABST-005@v1"), _repo("PROM-ABST-005@v2")]
    reg = tmp_path / "reg.jsonl"
    hc.register(cs, reg, hc.load_policy())
    props = [{"ticker": "AAA", "features": {}, "trade_score": {"total": 60}, "simulation": {"hit_rate": 0.7}}]
    _, blocked, recs = pia.apply_to_proposals(props, vix=40, today="2026-10-06", context=dict(CTX), contracts=cs,
                                              state_path=tmp_path / "s.json", registry=reg,
                                              transitions=tmp_path / "t.jsonl", ledger_dir=tmp_path / "led",
                                              policy={})
    assert set(recs[0]["intelligence"]) == {"PROM-ABST-005@v1"} and not blocked


def test_survivor_never_becomes_trade_and_downstream_is_descriptive(tmp_path):
    m = fml.downstream_map(final_tickers={"A"}, roi_rejects=[{"ticker": "B", "fail_gates": {"Long-Term": "edge"}}],
                           reject_stats={"options_design_bear_case": {"tickers": ["C"]}},
                           other_reasons={"D": "correlation"})
    assert m["A"]["champion_decision"] == "TRADE" and m["B"]["fail_gates"] == {"Long-Term": "edge"}
    assert m["C"]["reason"] == "options_design_bear_case" and m["D"]["reason"] == "correlation"
    src = open("modules/final_mc_ledger.py").read()
    for forbidden in ("active_trades", "send_email", "trade_proposals", "config.yaml"):
        assert forbidden not in src


# ── End-to-End ──────────────────────────────────────────────────────────────
def _bars_factory(effect):
    """Kurs je Ticker: ab Signaltag linear auf (1+effect[ticker]) bis Tag 60."""
    def bars(ticker, start, end):
        r = effect[ticker]
        out, d = [], start
        while d <= end:
            k = min(60, (d - start).days)
            px = 100.0 * (1 + r * k / 60)
            out.append((d, px * 1.01, px * 0.99, px))
            d += timedelta(days=1)
        return out
    return bars


def test_end_to_end_final_mc_forward_evidence_separate_from_champion(tmp_path):
    v1, v2 = _repo("PROM-ABST-005@v1"), _repo("PROM-ABST-005@v2")     # Regel: VIX > 30
    reg, led, outs = tmp_path / "reg.jsonl", tmp_path / "fm", tmp_path / "fm_out.jsonl"
    assert all(s["status"] == "VALID" for s in hc.register([v1, v2], reg, hc.load_policy()).values())
    effect: dict[str, float] = {}

    def run_day(d: date, i: int, pre: bool = False):
        now = datetime(d.year, d.month, d.day, 15, tzinfo=timezone.utc)
        vix = 35.0 if i % 3 == 0 else 18.0
        sv = [{"ticker": f"T{i}_{j}", "features": {}, "simulation": {"hit_rate": 0.7, "current_price": 100.0},
               "info": {"sector": ("Technology", "Energy", "Health Care")[(i + j) % 3]},
               "deep_analysis": {"direction": "BULLISH"}} for j in range(2)]
        rows = fml.record_survivors(sv, today=d.isoformat(), vix=vix, ctx=dict(CTX), contracts=[v1, v2],
                                    registry=reg, ledger_dir=led, now=now)
        for s in sv:     # vor forward_start: umgekehrter Effekt (würde das Ergebnis drehen, wenn er zählte)
            fired = vix > 30
            effect[s["ticker"]] = (0.08 if fired else -0.08) if pre else (-0.06 if fired else 0.04)
        fml.record_downstream(rows, {"T%d_0" % i: {"champion_decision": "NO_TRADE", "reason": "roi_gate",
                                                    "fail_gates": {"Long-Term": "mc_pnl"}}},
                              path=tmp_path / "ds.jsonl", now=now)
        return rows

    for i, d in enumerate([date(2026, 10, 1) + timedelta(days=k) for k in range(4)]):
        rows = run_day(d, 1000 + i, pre=True)
        assert set(rows[0]["contracts"]) == {"PROM-ABST-005@v2"}    # nur Verträge der eigenen Population
    start = date(2026, 10, 5)
    for i in range(110):
        run_day(start + timedelta(days=i), i)
    n = fml.resolve_outcomes(today=date(2027, 4, 1), ledger_dir=led, path=outs, bars_fn=_bars_factory(effect))
    assert n == 3 * 2 * 114                                          # 3 Horizonte je Beobachtung

    rows, oc = fml.read_rows(led), fml.read_outcomes(outs)
    ev = fml.evidence(v2, hc.spec_hash(v2), rows, oc, hc.load_policy(), 0.05 / 60)
    assert ev["n_observations"] == 220 and ev["first_observation"] == "2026-10-05"   # nichts vor forward_start
    assert ev["n_event_clusters"] == 220 and ev["n_fired"] > 0
    assert ev["fired"]["expectancy"] < 0 < ev["not_fired"]["expectancy"]
    assert ev["triggered_minus_non_triggered"] < 0 and ev["delta_expectancy"] > 0
    assert ev["abstention"]["avoided_loss_potential"] > 0 and ev["fired"]["mae"] is not None
    assert ev["fired"]["downside_tail_10pct"] is not None and set(ev["secondary_horizons"]) == {"20", "60"}

    state = pc.run(contracts=[v1, v2], policy=hc.load_policy(), now=datetime(2027, 4, 1, tzinfo=timezone.utc),
                   registry=reg, transitions=tmp_path / "tr.jsonl", ledger_dir=tmp_path / "champion_ledger",
                   outcomes_path=tmp_path / "champ_out.jsonl", looks_path=tmp_path / "looks.jsonl",
                   state_path=tmp_path / "state.json", history={}, approvals={}, safe_mode={"active": False},
                   final_mc_dir=led, final_mc_outcomes=outs)
    h1, h2 = state["hypotheses"]["PROM-ABST-005@v1"], state["hypotheses"]["PROM-ABST-005@v2"]
    assert h1["evidence"]["n_observations"] == 0 and h1["eligible_stage"] == "CHAMPION_TRADE"   # nie vermischt
    assert h2["evidence"]["n_observations"] == 220 and h2["eligible_stage"] == fml.STAGE
    assert h2["state"] == "FORWARD_VALIDATED" and h2["influence_level"] == "NONE"   # nie automatische Wirkung
    assert h2["multiple_testing"]["hypothesis_family"] == "abstention@FINAL_MC_SURVIVOR"
    summ = fml.population_summary(rows, fml.read_jsonl(tmp_path / "ds.jsonl"))
    assert summ["roi_fail_gates"] == {"mc_pnl": 114} and summ["champion_trades"] == 0


def test_correlated_repeats_cannot_reach_promotion(tmp_path):
    """Gleicher Ticker täglich -> viele N, wenige Cluster -> NEED_MORE_DATA."""
    v2 = _repo("PROM-ABST-005@v2")
    reg, led, outs = tmp_path / "reg.jsonl", tmp_path / "fm", tmp_path / "o.jsonl"
    hc.register([v2], reg, hc.load_policy())
    for k in range(120):
        d = date(2026, 10, 5) + timedelta(days=k)
        fml.record_survivors([{"ticker": t, "features": {}, "simulation": {}} for t in ("AMD", "AVGO")],
                             today=d.isoformat(), vix=35.0 if k % 2 else 18.0, ctx=dict(CTX), contracts=[v2],
                             registry=reg, ledger_dir=led, now=datetime(d.year, d.month, d.day, 15,
                                                                         tzinfo=timezone.utc))
    fml.resolve_outcomes(today=date(2027, 4, 1), ledger_dir=led, path=outs,
                         bars_fn=_bars_factory({"AMD": -0.05, "AVGO": 0.05}))
    ev = fml.evidence(v2, hc.spec_hash(v2), fml.read_rows(led), fml.read_outcomes(outs), hc.load_policy(), 0.001)
    assert ev["n_observations"] == 240 and ev["n_event_clusters"] == 2
    assert any("Ereignis-Cluster" in n for n in pc.insufficiency(v2, ev, hc.load_policy()))
