"""Expectation Alpha – Regressionstests der Audit-Fixes (2026-10-10):
* technischer Fehler (Rechenfehler, ausgefallene Quelle) -> ERROR, nie ABSTAIN;
* späterer Lauf am selben Tag -> keine zweite Beobachtung je Ticker (keine Pseudo-Stichprobe);
* veränderter WAIT-Trigger -> DATA_BAD (nie attribuiert);
* Fehlerklassen nur für Shadow-TRADE (ABSTAIN-Verlust ist kein Thesen-Fehler)."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

import tests.ea_fixtures as fx
from modules import expectation_alpha as ea
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import evaluation as ev
from modules.expectation_alpha import expectation_gap as eg
from modules.expectation_alpha import ledger as eal
from modules.expectation_alpha import timing as tm
from modules.expectation_alpha.schemas import ABSTAIN, ERROR, TRADE, WAIT

DT = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
CFG = eacfg.load()


@pytest.fixture(scope="module")
def data():
    px = fx.make_px(end="2027-06-30", years=7)
    for t, etf, k in (("AAPL", "XLK", 1.3), ("XOM", "XLE", 0.8)):
        px[t] = px[etf] * k
    return {"px": px, "ar": fx.make_archive(), "cm": fx.make_commodity()}


def _run(tmp_path, data, analyses, **loader_over):
    ld = {**fx.loaders(data["px"], data["ar"], data["cm"]), **loader_over}
    return ea.enrich_candidates(analyses, decision_time=DT, cfg=CFG, loaders=ld, root=tmp_path, contracts=[],
                                registry=tmp_path / "r.jsonl")


def test_computation_error_is_error_never_abstain(tmp_path, data, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Bug in der Gap-Berechnung")
    monkeypatch.setattr(eg, "domain_gaps", boom)
    s = _run(tmp_path, data, [fx.make_analysis("AAPL"), fx.make_analysis("XOM", sector="Energy", impact=2,
                                                                        surprise=1)])
    assert s["status_counts"][ERROR] == 2 and s["status_counts"][ABSTAIN] == 0
    assert any(e.startswith("BERECHNUNGSFEHLER") and "expectation_gap" in e for e in s["run_errors"])


def test_commodity_outage_errors_only_oil_sensitive_theses(tmp_path, data):
    def boom(cfg, t):
        raise ConnectionError("EIA/Commodity-Build aus")
    s = _run(tmp_path, data, [fx.make_analysis("AAPL"), fx.make_analysis("XOM", sector="Energy")], commodity=boom)
    rows = {r["ticker"]: r for r in eal.read_rows(tmp_path)}
    assert rows["XOM"]["status"] == ERROR and rows["XOM"]["group"] == "X"  # Öl-Domäne hängt an der Quelle
    assert any("oil" in x for x in rows["XOM"]["status_reasons"])
    assert rows["AAPL"]["status"] != ERROR                                  # Technologie: keine Öl-Sensitivität
    assert any(e["where"] == "commodity" for e in s["errors"])


def test_rerun_same_day_no_second_observation_per_ticker(tmp_path, data):
    _run(tmp_path, data, [fx.make_analysis("AAPL", catalyst="Produktstart iPhone")])
    s2 = _run(tmp_path, data, [fx.make_analysis("AAPL", catalyst="Produkteinführung des iPhone")])  # LLM formuliert anders
    assert s2["recorded"] == 0 and len(eal.read_rows(tmp_path)) == 1
    # verschiedene Ereignisse desselben Tickers im SELBEN Lauf bleiben getrennt (Population je Tag/Ticker/Ereignis)
    other = tmp_path / "b"
    s3 = _run(other, data, [fx.make_analysis("AAPL", catalyst="Event A"), fx.make_analysis("AAPL", catalyst="Event B")])
    assert s3["recorded"] == 2


def test_tampered_wait_trigger_is_data_bad(tmp_path, data):
    _run(tmp_path, data, [fx.make_analysis("AAPL")])
    r = eal.read_rows(tmp_path)[0]
    good = tm.wait_trigger(CFG)
    assert tm.verify_trigger(good)
    bad = json.loads(json.dumps(good))
    bad["any_of"][0]["value"] = 0.01                                        # nachträglich gelockert
    assert not tm.verify_trigger(bad)
    w = dict(r, observation_id="waittampered0001", status=WAIT, wait_trigger=bad)
    eal.append_jsonl(eal.paths(tmp_path)["candidates"] / f"{r['date']}.jsonl", [w], sort_keys=True, default=str)
    st = eal.resolve_outcomes(today=date(2027, 1, 15), root=tmp_path, bars_fn=fx.bars_fn_from(data["px"]), cfg=CFG)
    assert st.get("trigger_hash_mismatch", 0) >= 1
    trig = [e for k, e in eal.read_outcomes(tmp_path).items() if k[0] == "waittampered0001" and k[2] == "triggered"]
    assert trig and all(e["entry"] == "DATA_BAD" and e["outcome_net"] is None for e in trig)
    imm = [e for k, e in eal.read_outcomes(tmp_path).items() if k[0] == "waittampered0001" and k[2] == "immediate"]
    assert imm and all(e["outcome_net"] is not None for e in imm)          # Sofort-Einstieg bleibt bewertbar


def test_failure_classes_count_only_shadow_trades(tmp_path):
    def row(oid, status):
        return {"observation_id": oid, "stage": "EA_NEWS_CANDIDATE", "status": status, "date": "2026-10-09",
                "ticker": oid.upper(), "group": "A", "timestamp": "2026-10-09T15:00:00+00:00",
                "selected_expression": {"selected_expression": "UNDERLYING"},
                "expressions": [{"expression": "UNDERLYING"}], "confirmation_ratio": 0.5,
                "effective_confirmation_ratio": 0.5}
    eal.append_jsonl(eal.paths(tmp_path)["candidates"] / "2026-10-09.jsonl",
                     [row("t1", TRADE), row("a1", ABSTAIN), row("a2", ABSTAIN)], sort_keys=True)
    outs = [{"observation_id": o, "expression": "UNDERLYING", "variant": "immediate", "horizon": 60,
             "outcome_net": -0.04, "outcome": -0.038, "mfe": 0.0, "mae": -0.06, "exit_date": "2027-01-06",
             "outcome_method": eal.OUTCOME_METHOD} for o in ("t1", "a1", "a2")]
    eal.append_jsonl(eal.paths(tmp_path)["outcomes"], outs, sort_keys=True)
    rep = ev.run(root=tmp_path, cfg=CFG, promotion_state={}, write=False)
    assert rep["failure_classes"] == {"THESIS_WRONG": 1}                    # nur die Shadow-TRADE-These
    assert rep["failure_classes_by_status"] == {TRADE: {"THESIS_WRONG": 1}, ABSTAIN: {"THESIS_WRONG": 2}}
    assert [c["bucket"] for c in rep["calibration_effective_confirmation"]] == \
        [c["bucket"] for c in rep["calibration_confirmation"]]


def test_unregistered_ea_contracts_valid_after_filter_change():
    """Vor der Registrierung (erster Feedback-Lauf ab forward_start) geänderte Filter bleiben gültige Verträge."""
    from modules import hypothesis_contract as hc
    cs = [c for c in hc.load() if c.get("eligible_stage") == "EA_NEWS_CANDIDATE"]
    assert len(cs) == 7
    for c in cs:
        assert "ea_status_valid" in c["population_filter"] or c["population_filter"] == "ea_is_wait == 1"
        assert hc.signal_names(c["population_filter"]) <= set(c["features"]), c["hypothesis_id"]
        assert not hc.validate(c), (c["hypothesis_id"], hc.validate(c))
    assert pd.Timestamp(cs[0]["forward_start"]) > pd.Timestamp(cs[0]["registered_at"])
