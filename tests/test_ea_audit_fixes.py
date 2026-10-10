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


def test_e2e_report_and_mail_values_equal_ledger(tmp_path, data):
    """E2E: Kandidaten -> These/Status -> Ledger -> Outcomes -> Evaluation -> Montagsbericht/Mail.
    Bericht und Mail zeigen exakt die Ledger-/Entscheidungswerte, nur als SHADOW, nie Ticker in der Mail."""
    from modules import email_reporter as er
    from reports import weekly
    root = tmp_path / "outputs" / "expectation_alpha"
    an = [fx.make_analysis("AAPL"), fx.make_analysis("XOM", sector="Energy", impact=2, surprise=1),
          fx.make_analysis("ZZZ", sector=None, impact=None)]
    s = ea.enrich_candidates(an, decision_time=DT, cfg=CFG, loaders=fx.loaders(data["px"], data["ar"], data["cm"]),
                             root=root, contracts=[], registry=tmp_path / "r.jsonl")
    rows = eal.read_rows(root)
    by = {k: sum(1 for r in rows if r["status"] == k) for k in (TRADE, WAIT, ABSTAIN, ERROR)}
    assert s["status_counts"] == by and s["recorded"] == len(rows) == 3
    eal.resolve_outcomes(today=date(2027, 1, 15), root=root, bars_fn=fx.bars_fn_from(data["px"]), cfg=CFG)
    rep = ev.run(root=root, cfg=CFG, promotion_state={}, write=True)
    assert rep["population"]["n"] == 3 and rep["population"]["by_status"] == {k: v for k, v in by.items() if v}
    wk = weekly.collect(tmp_path, "2027-01-15", state_path=tmp_path / "st.json")
    txt = repr(weekly.expectation_alpha_section(wk))
    assert "SHADOW" in txt and "keine Handelsempfehlung" in txt
    assert "('Population n', '3')" in txt
    sc = wk["ea_runs"][-1]["status_counts"]
    assert sc == by                                                       # Bericht liest den Lauf aus runs.jsonl
    html = er._ea_shadow_html(s)
    assert "SHADOW" in html and "keine Handelsempfehlung" in html
    assert f"Shadow-TRADE {by[TRADE]}" in html and f"ABSTAIN {by[ABSTAIN]}" in html and f"ERROR {by[ERROR]}" in html
    assert not any(t in html for t in ("AAPL", "XOM", "ZZZ"))            # nie Ticker, nie Einzelempfehlung


def test_calibration_buckets_values_exclude_error_and_missing(tmp_path):
    """Re-Audit 2026-10-10: Kalibrierung rechnet je Bucket den Netto-Mittelwert der eingefrorenen Quote;
    fehlende Quote zählt in keinen Bucket, ERROR-Zeilen nie, Obergrenze 1,0 gehört in den letzten Bucket."""
    ph = int((CFG.get("outcomes") or {}).get("primary_horizon", 60))
    rows, outs = [], []
    spec = [("lo", 0.1, -0.02, TRADE, 30), ("hi", 0.7, 0.03, ABSTAIN, 30), ("top", 1.0, 0.05, WAIT, 30),
            ("none", None, 0.5, TRADE, 5), ("err", 0.7, 1.0, ERROR, 30)]
    for tag, ratio, net, status, n in spec:
        for i in range(n):
            oid = f"{tag}{i:03d}"
            rows.append({"observation_id": oid, "stage": "EA_NEWS_CANDIDATE", "status": status,
                         "date": f"2026-10-{1 + i % 28:02d}", "ticker": oid.upper(), "group": "A",
                         "timestamp": "2026-10-01T15:00:00+00:00", "confirmation_ratio": ratio,
                         "effective_confirmation_ratio": ratio,
                         "selected_expression": {"selected_expression": "UNDERLYING"},
                         "expressions": [{"expression": "UNDERLYING"}]})
            outs.append({"observation_id": oid, "expression": "UNDERLYING", "variant": "immediate", "horizon": ph,
                         "outcome_net": net, "outcome": net + 0.002, "mfe": 0.0, "mae": 0.0,
                         "exit_date": "2027-01-06", "outcome_method": eal.OUTCOME_METHOD})
    eal.append_jsonl(eal.paths(tmp_path)["candidates"] / "2026-10.jsonl", rows, sort_keys=True)
    eal.append_jsonl(eal.paths(tmp_path)["outcomes"], outs, sort_keys=True)
    rep = ev.run(root=tmp_path, cfg=CFG, promotion_state={}, write=False)
    for field in ("calibration_confirmation", "calibration_effective_confirmation"):
        cal = {c["bucket"]: c for c in rep[field]}
        assert cal["[0.0,0.3]"]["n"] == 30 and cal["[0.0,0.3]"]["mean"] == pytest.approx(-0.02)
        assert cal["[0.3,0.6]"]["n"] == 0 and cal["[0.3,0.6]"]["status"] == "NEED_MORE_DATA"
        assert cal["[0.6,0.8]"]["n"] == 30 and cal["[0.6,0.8]"]["mean"] == pytest.approx(0.03)    # ERROR nie dabei
        assert cal["[0.8,1.0]"]["n"] == 30 and cal["[0.8,1.0]"]["mean"] == pytest.approx(0.05)    # 1,0 inklusive
        assert sum(c["n"] for c in cal.values()) == 90                                          # fehlend: kein Bucket


def test_tampered_counterfactual_expressions_are_data_bad(tmp_path, data):
    """Re-Audit 2026-10-10: Counterfactuals nur mit den vorab eingefrorenen Expressions. Wird die Liste nachträglich
    verändert (Expression ergänzt, Kosten gesenkt), passt registry_hash nicht mehr -> DATA_BAD, nie attribuiert."""
    _run(tmp_path, data, [fx.make_analysis("AAPL")])
    r = eal.read_rows(tmp_path)[0]
    assert tm.verify_expressions(r)                                         # unverändert: gültig
    added = json.loads(json.dumps(r))
    added["observation_id"] = "exprtampered0001"
    added["expressions"].append({"expression": "XLE", "symbol": "XLE", "instrument": "etf",
                                 "cost_per_side": 0.0, "available": True})   # Hindsight-Alternative
    cheaper = json.loads(json.dumps(r))
    cheaper["observation_id"] = "exprtampered0002"
    cheaper["expressions"][0]["cost_per_side"] = 0.0                         # Kosten nachträglich gesenkt
    assert not tm.verify_expressions(added) and not tm.verify_expressions(cheaper)
    eal.append_jsonl(eal.paths(tmp_path)["candidates"] / f"{r['date']}.jsonl", [added, cheaper], sort_keys=True,
                     default=str)
    st = eal.resolve_outcomes(today=date(2027, 6, 30), root=tmp_path, bars_fn=fx.bars_fn_from(data["px"]), cfg=CFG)
    assert st.get("expression_hash_mismatch", 0) >= 2
    outs = eal.read_outcomes(tmp_path)
    for oid in ("exprtampered0001", "exprtampered0002"):
        mine = [e for k, e in outs.items() if k[0] == oid]
        assert mine and all(e["entry"] == "DATA_BAD" and e["outcome_net"] is None for e in mine)
    good = [e for k, e in outs.items() if k[0] == r["observation_id"]]
    assert good and all(e["entry"] != "DATA_BAD" for e in good)               # Original bleibt bewertbar


def _learning_fixture(tmp_path, *, means_by_ratio, n=40, sel="SECTOR_ETF"):
    """Je Quote (Bucket) n Zeilen; TRADE-Zeilen mit Netto `m`, ABSTAIN-Zeilen mit −0,03. Zwei Expressions je Zeile:
    UNDERLYING = Netto, SECTOR_ETF = Netto + 0,01 (gewählte Expression immer um 1 pp besser)."""
    ph = int((CFG.get("outcomes") or {}).get("primary_horizon", 60))
    rows, outs = [], []
    for b, (ratio, m) in enumerate(means_by_ratio):
        for i in range(n):
            st = TRADE if i % 2 == 0 else ABSTAIN
            net = m if st == TRADE else -0.03
            oid = f"L{b}{i:03d}"
            rows.append({"observation_id": oid, "stage": "EA_NEWS_CANDIDATE", "status": st,
                         "date": str(date(2026, 1, 1) + pd.Timedelta(days=(b * n + i) % 150))[:10],
                         "ticker": oid, "group": "A", "timestamp": "2026-01-01T15:00:00+00:00",
                         "confirmation_ratio": ratio, "effective_confirmation_ratio": ratio,
                         "selected_expression": {"selected_expression": sel},
                         "expressions": [{"expression": "UNDERLYING"}, {"expression": "SECTOR_ETF"}]})
            for ex, v in (("UNDERLYING", net), ("SECTOR_ETF", net + 0.01)):
                outs.append({"observation_id": oid, "expression": ex, "variant": "immediate", "horizon": ph,
                             "outcome_net": v, "outcome": v, "mfe": 0.0, "mae": 0.0, "exit_date": "2026-09-01",
                             "outcome_method": eal.OUTCOME_METHOD})
    eal.append_jsonl(eal.paths(tmp_path)["candidates"] / "2026-01.jsonl", rows, sort_keys=True)
    eal.append_jsonl(eal.paths(tmp_path)["outcomes"], outs, sort_keys=True)
    return ev.run(root=tmp_path, cfg=CFG, promotion_state={}, write=True)


def test_ea_learning_values_and_proposals_never_applied(tmp_path):
    """Re-Audit 2026-10-10 (Punkt 9): Abstinenz-, Thesen- und Expression-Lernen rechnen korrekt; Vorschläge
    entstehen nur als PROPOSAL_ONLY (auto_applied=false), Challenger erst ab Mindestdaten."""
    rep = _learning_fixture(tmp_path / "a", means_by_ratio=[(0.1, 0.05), (0.4, 0.01), (0.7, 0.03)])
    ab = rep["values"]["abstention"]
    # alle: Hälfte TRADE (Mittel der drei Buckets 0,03), Hälfte ABSTAIN (−0,03) -> 0,0; kept = nur TRADE = 0,03
    assert ab["all"]["mean"] == pytest.approx(0.0, abs=1e-9) and ab["kept"]["mean"] == pytest.approx(0.03)
    assert ab["delta"] == pytest.approx(0.03)                                  # Abstinenz hat hier Wert
    exq = rep["values"]["expression"]
    assert exq["expression_quality"]["mean"] == pytest.approx(0.005)           # gewählt − Median(2) = +0,01/2
    assert exq["thesis_quality"]["mean"] == pytest.approx(0.005)               # Median aus beiden Expressions
    props = json.loads((eal.paths(tmp_path / "a")["proposals"]).read_text())
    assert props["auto_applied"] is False and props["status"] == "PROPOSAL_ONLY"
    assert [i["type"] for i in props["items"]] == ["CALIBRATION_REVIEW"]       # 0,05 > 0,01: nicht monoton
    assert props["challenger"]["status"] == "NEED_MORE_DATA"                   # 120 < 300 Outcomes
    assert {"Produktionseinfluss", "Promotion-Kriterien", "PIT-Regeln"} <= set(props["never_automatic"])
    mono = _learning_fixture(tmp_path / "b", means_by_ratio=[(0.1, 0.01), (0.4, 0.02), (0.7, 0.03)], n=110)
    assert mono["proposals"]["items"] == []                                    # monoton -> kein Prüfauftrag
    assert mono["proposals"]["challenger"]["status"] == "READY_FOR_HUMAN_REVIEW"   # 330 Outcomes, 150 Tage
    assert mono["proposals"]["auto_applied"] is False                          # auch bereit: nie automatisch
