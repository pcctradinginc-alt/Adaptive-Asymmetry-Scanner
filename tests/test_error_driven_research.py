"""Fehlergetriebene Forschung (config/error_research_protocol.yaml, v2): Forward-Prognosefehler -> Muster ->
Hypothese -> Walk-Forward-Screen mit Embargo -> eingefrorene Forward-Prüfungen (Bonferroni-Slot-Budget,
O'Brien-Fleming) -> Options-Prüfung ohne Auswahlverzerrung -> Research Memory -> nächste Entscheidung.
Synthetische Daten, kein Netz – kein Alpha-Nachweis, keine reale Performance.

Echter Lauf 2026-11-07 (docs/ERROR_DRIVEN_RESEARCH_2026-11-07.md): 8.590 von 8.976 Ledger-Prognosen gegen das
ml_predictions-Archiv geprüft; mit unabhängigen (alternierenden, gepurgten) Fenstern noch keine Entdeckung möglich;
ERR-d6695229 (v1-Registrierung) unverändert, Slot 1, eine Prüfung bei 4 Fenstern mit Alpha 0,05/24.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from modules import option_learner as ol
from modules import research_director as rd
from modules import research_lab as rl
from modules import research_memory as rm

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_SHA = "4050314a5ab5cbf95be67c734cf193a9e6ea7840af4aaa9c947cc127c4d64591"
MODELS = ("ml_a", "ml_b", "ml_c")
FEATURES = ("sentiment_score", "short_mom", "news_count", "rel_volume", "sentiment_drift")
START = "2026-08-03"


def test_protocol_is_pinned_and_bounded():
    raw = (ROOT / "config" / "error_research_protocol.yaml").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA, "Protokoll geändert – nur strenger oder bei belegtem Fehler"
    p = rd.error_protocol()
    assert p["version"] == "error-research-v2" and p["seen_until"] == "2026-11-06"
    assert p["budget"] == {"max_new_per_run": 2, "max_lifetime": 24}
    assert p["forward"] == {"test_budget": {"total_alpha": 0.05, "max_tests": 24}, "looks": [4, 8, 12]}
    assert len(rd.error_candidates(p)) == 23                       # fester Suchraum
    assert abs(sum(ol.alpha_spending(0.05 / 24)) - 0.05 / 24) < 1e-12   # Teil-Alphas summieren sich zum Slot


def _proto(**over):
    p = json.loads(json.dumps(rd.error_protocol()))
    p["forecast"]["models"] = list(MODELS)
    p["forecast"]["horizons"] = [5]
    p["search_space"] = {"ledger": list(FEATURES), "external": {}, "pairs": []}
    p["families"] = {"sentiment_score": "sentiment", "short_mom": "positioning", "news_count": "news",
                     "rel_volume": "trading_activity", "sentiment_drift": "attention"}
    for k, v in over.items():
        p[k] = {**p[k], **v} if isinstance(v, dict) else v
    return p


def _bdays(a: str, b: str) -> list[str]:
    d, out = date.fromisoformat(a), []
    while d <= date.fromisoformat(b):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _rank(v):
    return (np.argsort(np.argsort(v)) + 1) / len(v)


def _world(end="2027-03-12", n=40, seed=3):
    """Ledger, Prognose-Archiv und Optionskandidaten. sentiment_score erklärt den Fehler dauerhaft; rel_volume kehrt
    ab 21.09. (vor der Validierung) um; sentiment_drift kehrt ab 09.11. (nach der Registrierung) um; short_mom ist
    nur ein Größen-Stellvertreter; news_count ist Rauschen."""
    rnd = np.random.default_rng(seed)
    rows, preds, cands = [], [], []
    tickers = [f"T{i:02d}" for i in range(n)]
    for d in _bdays(START, end):
        p = rnd.uniform(0.01, 1.0, n).round(4)
        x, size, rv, dr = (rnd.normal(size=n) for _ in range(4))
        noise = rnd.poisson(3, n)
        ret = (0.02 * (p - 0.5) + 0.05 * (_rank(x) - 0.5) + 0.04 * (_rank(size) - 0.5)
               + (0.05 if d < "2026-09-21" else -0.05) * (_rank(rv) - 0.5)
               + (0.04 if d < "2026-11-09" else -0.04) * (_rank(dr) - 0.5) + rnd.normal(0, 0.02, n))
        y_call = 0.6 * (_rank(x) - 0.5) - 0.05 + rnd.normal(0, 0.15, n)
        y_spread = 0.3 * (_rank(x) - 0.5) - 0.02 + rnd.normal(0, 0.08, n)
        for m in MODELS:
            preds.append({"model_id": m[3:], "prediction_date": d, "created_at": f"{d}T08:00:00+00:00",
                          "rank_pct": dict(zip(tickers, map(float, p)))})
        lday = date.fromisoformat(d) + timedelta(days=7)
        for i, t in enumerate(tickers):
            f = {m: float(p[i]) for m in MODELS}
            f.update(sentiment_score=float(x[i]), log_market_cap=float(size[i]),
                     short_mom=float(size[i] + rnd.normal(0, 0.05)), news_count=int(noise[i]),
                     rel_volume=float(rv[i]), sentiment_drift=float(dr[i]),
                     sector="Technology" if i % 2 else "Energy")
            rows.append({"date": d, "ticker": t, "signal_timestamp": f"{d}T15:00:00+00:00", "features": f,
                         "gap_at_entry": 0.0, "outcomes": {"ret_5d": float(ret[i])}})
            ex = [{"expression": "single", "strategy": "LONG_CALL", "status": "VERIFIED", "y": float(y_call[i]),
                   "label_day": lday}]
            if i % 2 == 0:                                   # Spread nur bei der Hälfte zum Signalzeitpunkt erfasst
                ex.append({"expression": "spread", "strategy": "BULL_CALL_SPREAD", "status": "VERIFIED",
                           "y": float(y_spread[i]), "label_day": lday})
            cands.append({"date": d, "ticker": t, "signal_id": f"{d}-{t}", "exprs": ex})
    return rows, preds, cands


@pytest.fixture(scope="module")
def world():
    return _world()


def _run(world, as_of, tmp, *, proto=None, memory=None, persist=True, rows=None, preds=None, **kw):
    w_rows, w_preds, cands = world
    return rd.error_research([r for r in (rows or w_rows) if r["date"] < str(as_of)], preds or w_preds, as_of,
                             memory=[] if memory is None else memory,
                             proto=proto or _proto(budget={"max_new_per_run": 3, "max_lifetime": 24}),
                             registry=tmp / "reg.jsonl", journal=tmp / "journal.jsonl", report=tmp / "rep.json",
                             option_cands={5: cands}, persist=persist, **kw)


# ── Prognose-Herkunft ────────────────────────────────────────────────────────

def test_forecast_provenance_against_prediction_archive(world):
    rows, preds, _ = world
    r0 = rows[0]
    fake = {**r0, "features": {**r0["features"], "ml_a": 0.123}}
    late = [{**p, "created_at": f"{START}T16:00:00+00:00"} if p["prediction_date"] == START else p for p in preds]
    out, counts = rd.verified_forecasts([fake], preds, _proto()["forecast"])
    assert out[0]["status"] == "FORECAST_MISMATCH" and counts == {"FORECAST_MISMATCH": 1}
    assert rd.verified_forecasts([r0], late, _proto()["forecast"])[0][0]["status"] == "FORECAST_MISMATCH"
    ok, c = rd.verified_forecasts(rows[:40], preds, _proto()["forecast"])
    assert c == {"VERIFIED": 40} and ok[0]["n_models"] == 3
    few = {**r0, "features": {"ml_a": r0["features"]["ml_a"]}}
    assert rd.verified_forecasts([few], preds, _proto()["forecast"])[0][0]["status"] == "TOO_FEW_MODELS"


# ── P1: unabhängige Fenster ──────────────────────────────────────────────────

def test_adjacent_windows_overlap_so_only_every_second_window_counts():
    # Beleg der Überlappung: Freitags-Signal, 5-T-Label erst am Freitag der FOLGEWOCHE (nächstes Fenster)
    assert rl.label_due("2026-10-02", 5) == date(2026, 10, 9)
    ic = {d: 0.1 for d in _bdays("2026-09-29", "2026-11-27")}
    b = rl.period_blocks(ic, 5, "2026-09-29", date(2026, 12, 7))
    assert [x["start"] for x in b] == ["2026-09-28", "2026-10-12", "2026-10-26", "2026-11-09"]
    assert b[0]["n_dates"] == 4 and b[0]["dates"][0] == "2026-09-29"            # 28.09. vor dem Start
    for x, nxt in zip(b, b[1:]):                                                  # disjunkte Renditezeiträume
        assert all(rl.label_due(d, 5) < date.fromisoformat(nxt["start"]) for d in x["dates"])
    overl = rl.period_blocks(ic, 5, "2026-09-29", date(2026, 12, 7), purge=False)
    assert len(overl) == 2 * len(b) - 1 or len(overl) == 2 * len(b)              # v1 zählte jedes Fenster
    # 20 T: 28-Tage-Fenster, ebenfalls alternierend; Karenz: Fenster erst nach Fälligkeit + 3 Tagen fertig
    ic20 = {d: 0.1 for d in _bdays("2026-01-05", "2026-12-31")}
    b20 = rl.period_blocks(ic20, 20, "2026-01-05", date(2027, 1, 30))
    assert all((date.fromisoformat(y["start"]) - date.fromisoformat(x["start"])).days == 56
               for x, y in zip(b20, b20[1:]))
    assert all(rl.label_due(d, 20) < date.fromisoformat(y["start"]) for x, y in zip(b20, b20[1:]) for d in x["dates"])
    assert len(rl.period_blocks(ic, 5, "2026-09-29", date(2026, 11, 12))) == 3    # 30.10.: fällig 06.11. + 3 T
    assert len(rl.period_blocks(ic, 5, "2026-09-29", date(2026, 11, 9))) == 2


def test_walk_forward_screen_falsifies_and_respects_embargo():
    cfg = _proto()["validation"]
    ic = {d: (-0.05 if d >= "2026-10-26" else 0.5) for d in _bdays("2026-09-28", "2026-12-31")}
    wf = rl.ledger_walk_forward(ic, 5, "2026-10-18", "2026-12-14", cfg)
    assert wf["window_start"] == "2026-10-26" and wf["window_method"] == rl.WINDOW_METHOD   # Embargo 19.–25.10.
    assert [b["start"] for b in wf["blocks"]] == ["2026-10-26", "2026-11-09", "2026-11-23"]
    assert wf["status"] == "REJECTED_WF"
    ic2 = {d: 0.04 + (0.01 if int(d[-2:]) % 2 else -0.01) for d in ic}
    assert rl.ledger_walk_forward(ic2, 5, "2026-10-18", "2026-12-14", cfg)["status"] == "PASSED_WF"
    assert rl.ledger_walk_forward(ic2, 5, "2026-10-18", "2026-11-14", cfg)["status"] == "INCONCLUSIVE_WF"


def test_look_uses_only_its_first_windows():
    ic = {d: 0.05 + 0.01 * (i % 3) for i, d in enumerate(_bdays("2026-11-09", "2027-04-30"))}
    b1 = rl.forward_blocks(ic, 5, "2026-11-07", date(2027, 1, 9))
    later = {**ic, **{d: -0.9 for d in _bdays("2027-01-11", "2027-04-30")}}
    b2 = rl.forward_blocks(later, 5, "2026-11-07", date(2027, 5, 10))
    assert len(b1) == 4 and b1 == b2[:4]                                        # spätere Daten ändern Fenster 1–4 nicht
    assert rl.look_outcome(b1, 4, 0.05, final=False)["outcome"] in ("CONFIRMED", "CONTINUE")
    assert rl.look_outcome(b2, 8, 0.05, final=True)["outcome"] == "REJECTED"    # Mittel <= 0 = falsifiziert
    assert rl.forward_blocks(ic, 5, "2026-11-07", date(2026, 11, 20)) == []


# ── P0: Mehrfachtest-Kontrolle über die Testfamilie ──────────────────────────

def test_family_budget_is_strict_frozen_and_never_retroactive():
    budget = {"total_alpha": 0.05, "max_tests": 24}
    legacy = {"spec": {"forward": {"start_after": "2026-11-07", "min_periods": 4, "alpha": 0.05, "fdr_q": 0.05,
                                   "max_days": 150}}}
    plan = rd.forward_plan(legacy, 1, budget)
    assert plan["looks"] == [4] and plan["look_alpha"] == [0.05 / 24] and plan["supersedes"]   # nie lockerer
    v2 = {"spec": {"forward": {"start_after": "2026-11-07", "looks": [4, 8, 12], "max_days": 200}}}
    p2 = rd.forward_plan(v2, 2, budget)
    assert len(p2["look_alpha"]) == 3 and abs(sum(p2["look_alpha"]) - 0.05 / 24) < 1e-12
    assert p2["look_alpha"][0] < p2["look_alpha"][1] < p2["look_alpha"][2]           # O'Brien-Fleming
    # p ≈ 0,01 hätte im alten Verfahren (alpha 0,05) bestätigt – mit Slot-Alpha nicht
    blocks = [{"mean_ic": v} for v in (0.03, 0.05, 0.02, 0.06)]
    res = rl.look_outcome(blocks, 4, plan["look_alpha"][0], final=True)
    assert 0.0021 < res["p_one_sided"] < 0.05 and res["outcome"] == "NOT_CONFIRMED"


def test_sequential_forward_tests_keep_earlier_decisions(world, tmp_path):
    """Entscheidungen werden einmal getroffen und eingefroren; neue Hypothesen, neue Daten und spätere Läufe ändern
    sie nicht; jede Prüfung trägt Zeitpunkt, Stichprobe, p, Teil-Alpha, Verfahren und Datenstand."""
    r1 = _run(world, date(2026, 11, 7), tmp_path)
    reg1 = {h["spec"]["feature"]: h for h in r1["hypotheses"]}
    assert set(reg1) == {"sentiment_score", "rel_volume", "sentiment_drift"}
    assert reg1["rel_volume"]["status"] == "REJECTED_WF"                        # kippt vor der Validierung
    assert reg1["sentiment_score"]["walk_forward"]["status"] == "PASSED_WF"
    assert reg1["sentiment_drift"]["walk_forward"]["status"] in ("PASSED_WF", "INCONCLUSIVE_WF")   # nicht verworfen
    assert [h["forward_plan"]["slot"] for h in r1["hypotheses"]] == [1, 2, 3]
    r2 = _run(world, date(2027, 1, 12), tmp_path)                               # Prüfung 1 (4 Fenster)
    st2 = {h["spec"]["feature"]: h for h in r2["hypotheses"]}
    assert st2["sentiment_drift"]["status"] == "FORWARD_REJECTED"               # kippt nach Registrierung
    assert st2["sentiment_score"]["status"] == "PENDING_FORWARD"                # Teil-Alpha 1 sehr klein
    look1 = st2["sentiment_drift"]["forward"]["looks"][0]
    assert look1["final"] and look1["decided_at"] == "2027-01-12" and look1["stats"]["n_periods"] == 4
    assert look1["data_ref"]["n_rows"] > 0 and len(look1["data_ref"]["inputs_sha256"]) == 64
    r3 = _run(world, date(2027, 3, 15), tmp_path)                               # Prüfung 2 (8 Fenster)
    st3 = {h["spec"]["feature"]: h for h in r3["hypotheses"]}
    assert st3["sentiment_score"]["status"] == "FORWARD_CONFIRMED"
    assert st3["sentiment_score"]["forward"]["looks"][-1]["look"] == 2
    assert st3["sentiment_drift"]["forward"]["looks"] == st2["sentiment_drift"]["forward"]["looks"]   # unverändert
    events, problems = rd.read_journal(tmp_path / "journal.jsonl")
    assert not problems and sum(e["event"] == "test_budget" for e in events) == 1
    assert sum(e["event"] == "look" and e["final"] for e in events) == 2
    n = len(events)
    _run(world, date(2027, 3, 15), tmp_path)                                    # wiederholter Lauf: nichts Neues
    assert len(rd.read_journal(tmp_path / "journal.jsonl")[0]) == n
    assert len((tmp_path / "reg.jsonl").read_text().splitlines()) == 3


def test_data_change_after_decision_is_audited_not_redecided(world, tmp_path):
    _run(world, date(2026, 11, 7), tmp_path)
    r = _run(world, date(2027, 1, 12), tmp_path)
    drift = next(h for h in r["hypotheses"] if h["spec"]["feature"] == "sentiment_drift")
    d0 = drift["forward"]["looks"][0]["data_ref"]["first"]
    changed = [{**x, "outcomes": {"ret_5d": -x["outcomes"]["ret_5d"]}} if x["date"] == "2026-11-10" else x
               for x in world[0]]                                              # Kurs-/Ergebniskorrektur
    assert d0 <= "2026-11-10"
    r2 = _run(world, date(2027, 1, 12), tmp_path, rows=changed)
    d2 = next(h for h in r2["hypotheses"] if h["spec"]["feature"] == "sentiment_drift")
    assert d2["status"] == "FORWARD_REJECTED" and d2["forward"]["looks"] == drift["forward"]["looks"]
    assert len(d2["audits"]) == 1 and d2["audits"][0]["frozen_outcome"] == "REJECTED"
    assert d2["audits"][0]["frozen_sha256"] != d2["audits"][0]["current_sha256"]
    assert any(e["kind"] == "error_audit" for e in rm.error_entries(r2))
    r3 = _run(world, date(2027, 1, 12), tmp_path, rows=changed)                 # kein doppeltes Audit
    assert len(next(h for h in r3["hypotheses"] if h["spec"]["feature"] == "sentiment_drift")["audits"]) == 1


def test_tampering_and_forecast_changes_are_critical(world, tmp_path):
    _run(world, date(2026, 11, 7), tmp_path)
    j = tmp_path / "journal.jsonl"
    lines = j.read_text().splitlines()
    ev = json.loads(lines[1])
    ev["slot_alpha"] = 0.05                                                     # nachträglich gelockert
    j.write_text("\n".join([lines[0], json.dumps(ev)] + lines[2:]) + "\n")
    before = j.read_text()
    r = _run(world, date(2027, 1, 12), tmp_path)
    assert r["health"]["status"] == "CRITICAL" and "Hash-Kette" in r["health"]["problems"][0]
    assert j.read_text() == before                                               # nichts angehängt
    rows = [{**x, "features": {**x["features"], "ml_a": 0.5}} if x["date"] == "2026-10-01" else x for x in world[0]]
    r2 = _run(world, date(2026, 11, 7), tmp_path / "x", rows=rows, persist=False)
    assert r2["health"]["status"] == "CRITICAL" and "weichen" in r2["health"]["problems"][0]


def test_budget_is_lifetime_and_modified_registration_is_invalid(world, tmp_path):
    proto = _proto(budget={"max_new_per_run": 3, "max_lifetime": 1})
    r = _run(world, date(2026, 11, 7), tmp_path, proto=proto)
    assert [d["decision"] for d in r["decisions"]].count("register") == 1
    assert "defer" in [d["decision"] for d in r["decisions"]]
    reg = tmp_path / "reg.jsonl"
    e = json.loads(reg.read_text().splitlines()[0])
    e["spec"]["direction"] = -e["spec"]["direction"]                           # nachträglich geändert
    reg.write_text(json.dumps(e) + "\n")
    r2 = _run(world, date(2026, 11, 7), tmp_path, proto=proto, persist=False)
    assert [h["status"] for h in r2["hypotheses"]] == ["INVALID_MODIFIED"] and r2["health"]["status"] == "CRITICAL"
    assert rm.error_entries(r2)[0]["status"] == "REJECTED"
    bad = tmp_path / "proto.yaml"
    bad.write_text((ROOT / "config" / "error_research_protocol.yaml").read_text().replace(
        "max_lifetime: 24", "max_lifetime: 30"))
    with pytest.raises(ValueError, match="Slots"):                              # Familie größer als Budget
        rd.error_protocol(bad)


# ── P0: Options-Prüfung ohne Auswahlverzerrung ───────────────────────────────

def _opt_world(missing=(), spread_every=2):
    days = _bdays("2026-11-09", "2027-02-26")
    vals = {"sig": {}}
    cands = []
    for d in days:
        for i in range(9):
            t = f"S{i}"
            vals["sig"][(d, t)] = float(i)                                          # S8, S7, S6 = oberes Drittel
            st = "EXIT_QUOTE_MISSING" if t in missing else "VERIFIED"
            y = 0.2 if i >= 6 else -0.1
            ex = [{"expression": "single", "strategy": "LONG_CALL", "status": st, "y": y if st == "VERIFIED" else None,
                   "label_day": date.fromisoformat(d) + timedelta(days=7) if st == "VERIFIED" else None}]
            if i % spread_every == 0:
                ex.append({"expression": "spread", "strategy": "BULL_CALL_SPREAD", "status": "VERIFIED",
                           "y": 0.05 if i >= 6 else -0.02, "label_day": date.fromisoformat(d) + timedelta(days=7)})
            cands.append({"date": d, "ticker": t, "exprs": ex})
    spec = {"kind": "ledger", "feature": "sig", "parts": ["sig"], "horizon": 5, "direction": 1}
    return spec, vals, cands


def test_option_selection_is_fixed_at_signal_time_missing_quotes_are_gaps():
    proto = _proto()
    spec, vals, cands = _opt_world(missing=("S8",))                              # bester Titel ohne Exit-Quote
    oc = rd.option_check(spec, vals, cands, proto, ("2026-11-09", "2027-02-26"), "confirmatory", date(2027, 3, 20))
    lc = oc["structures"]["LONG_CALL"]
    assert lc["selected"] == 3 * lc["eligible"] // 9                            # S6–S8 gewählt, nicht ersetzt
    assert lc["coverage_gaps"] == {"EXIT_QUOTE_MISSING": lc["selected"] // 3}
    assert lc["coverage_selected"] == round(2 / 3, 3) and lc["status"] == "INSUFFICIENT_DATA"   # < 0,8
    full = rd.option_check(spec, vals, _opt_world()[2], proto, ("2026-11-09", "2027-02-26"), "confirmatory",
                           date(2027, 3, 20))["structures"]["LONG_CALL"]
    assert full["status"] == "OPTION_EDGE" and full["mean_net_selected"] == pytest.approx(0.2)
    assert full["vs_trade_all"]["mean_diff"] > 0 and full["independent_periods"] >= 5
    assert full["holding_days_median"] == 7


def test_long_calls_and_spreads_are_evaluated_separately_with_the_same_rule():
    spec, vals, cands = _opt_world(spread_every=2)                               # Spread nur S0, S2, S4, S6, S8
    oc = rd.option_check(spec, vals, cands, _proto(), ("2026-11-09", "2027-02-26"), "confirmatory",
                         date(2027, 3, 20))
    lc, sp = oc["structures"]["LONG_CALL"], oc["structures"]["BULL_CALL_SPREAD"]
    days = lc["eligible"] // 9
    assert sp["eligible"] == 5 * days and lc["eligible"] == 9 * days            # eigene Population je Struktur
    assert sp["selected"] == 2 * days                                            # oberes Drittel unter S0..S8 gerade
    assert sp["mean_net_selected"] == pytest.approx(0.05) and lc["mean_net_selected"] == pytest.approx(0.2)


def test_exploratory_or_thin_option_evidence_is_never_a_positive_proof():
    spec, vals, cands = _opt_world()
    expl = rd.option_check(spec, vals, cands, _proto(), ("2026-11-09", "2027-02-26"), "exploratory",
                           date(2027, 3, 20))
    assert expl["status"] == "NOT_PROVEN" and expl["structures"]["LONG_CALL"]["status"] == "NOT_PROVEN"
    thin = rd.option_check(spec, vals, cands, _proto(), ("2026-11-09", "2026-11-27"), "confirmatory",
                           date(2027, 3, 20))
    assert thin["status"] == "INSUFFICIENT_DATA"                                 # < 5 unabhängige Fenster
    pend = rd.option_check(spec, vals, cands, _proto(), ("2026-11-09", "2027-02-26"), "confirmatory",
                           date(2026, 11, 25))
    assert pend["structures"]["LONG_CALL"]["pending_labels"] > 0                 # nicht fällig != fehlend
    assert pend["structures"]["LONG_CALL"]["coverage_gaps"] == {}


# ── Kreislauf ────────────────────────────────────────────────────────────────

def test_patterns_recurring_explained_by_size_and_noise(world):
    rows, preds, _ = world
    proto = _proto()
    fc, _ = rd.verified_forecasts([r for r in rows if r["date"] <= "2026-11-06"], preds, proto["forecast"])
    errs = rd.forecast_errors(fc, 5, date(2026, 11, 7), proto)["by_date"]
    pat = rd.error_patterns(errs, rd.error_values(fc, proto), proto, 5, date(2026, 11, 7))
    v = {c["id"]: c["verdict"] for c in pat["candidates"]}
    assert pat["discovery"] == {"start": "2026-08-03", "end": "2026-09-20", "periods": 4}
    assert v["sentiment_score"] == "RECURRING" and v["short_mom"] == "EXPLAINED_BY_KNOWN_FACTOR"
    nc = next(c for c in pat["candidates"] if c["id"] == "news_count")["discovery"]["mean"]
    assert abs(nc) < 0.2 * next(c for c in pat["candidates"] if c["id"] == "sentiment_score")["discovery"]["mean"]
    # Rauschen kann den Screen (nominal p < 0,10) zufällig bestehen – Schutz ist allein der Forward-Test (Slot-Alpha)
    late = {d: [{**r, "resid": -r["resid"]} for r in rs] if d > "2026-09-20" else rs for d, rs in errs.items()}
    pat2 = rd.error_patterns(late, rd.error_values(fc, proto), proto, 5, date(2026, 11, 7))
    assert [c["discovery"] for c in pat2["candidates"]] == [c["discovery"] for c in pat["candidates"]]  # nur Entdeckung


def test_external_values_are_point_in_time():
    class O:
        def __init__(self, day, avail, value):
            self.observation_time = datetime.fromisoformat(f"{day}T00:00:00+00:00")
            self.available_at = datetime.fromisoformat(avail)
            self.metric, self.value, self.entity_id = "m", value, "AAA"
    series = rd._ext_obs(type("A", (), {"load": lambda self, s: [O("2026-10-05", "2026-10-06T10:00:00+00:00", 1.0),
                                                                  O("2026-10-08", "2026-10-09T20:00:00+00:00", 9.0)]})(),
                         {"source": "x", "metric": "m"})["AAA"]
    spec = {"transform": "level", "lag_days": 1}
    assert rd.external_value(series, "2026-10-09", "2026-10-09T15:00:00+00:00", spec) == 1.0   # 9.0 noch nicht da
    assert rd.external_value(series, "2026-10-12", "2026-10-12T15:00:00+00:00", spec) == 9.0
    assert rd.external_value(series, "2026-10-30", "2026-10-30T15:00:00+00:00", spec) is None  # zu alt


def test_end_to_end_error_to_decision(world, tmp_path):
    """Fehler -> Hypothese -> Test -> eingefrorenes Ergebnis -> Gedächtnis -> neue Entscheidung."""
    mem, src = tmp_path / "memory.jsonl", {"error_research": tmp_path / "rep.json"}
    r1 = _run(world, date(2026, 11, 7), tmp_path)
    assert r1["health"]["status"] == "OK" and r1["summary"]["registered_this_run"] == 3
    sent = next(h for h in r1["hypotheses"] if h["spec"]["feature"] == "sentiment_score")
    assert sent["option_check"]["evidence"] == "exploratory" and sent["option_check"]["status"] != "OPTION_EDGE"
    rm.sync(mem, src, now="2026-11-07T12:00:00+00:00")
    m1 = rm.latest(rm.load(mem))
    assert m1["error_pattern:ERRPAT-short_mom-5d"]["status"] == "NOT_ROBUST"
    _run(world, date(2027, 1, 12), tmp_path, memory=rm.load(mem))
    rm.sync(mem, src, now="2027-01-12T12:00:00+00:00")
    r3 = _run(world, date(2027, 3, 15), tmp_path, memory=rm.load(mem))
    st = {h["spec"]["feature"]: h for h in r3["hypotheses"]}
    assert st["sentiment_score"]["status"] == "FORWARD_CONFIRMED"
    oc = st["sentiment_score"]["option_check"]
    assert oc["evidence"] == "confirmatory" and oc["window"][0] >= "2026-11-09"   # nur Forward-Signale
    assert oc["structures"]["LONG_CALL"]["status"] == "OPTION_EDGE"
    assert st["rel_volume"]["option_check"]["status"] == "NOT_REACHED"
    assert any(d["decision"] == "skip" and "ERR-" in d["because"] for d in r3["decisions"])
    rm.sync(mem, src, now="2027-03-15T12:00:00+00:00")
    m3 = rm.latest(rm.load(mem))
    by_feat = {e["spec"]["signal"]: e for k, e in m3.items() if k.startswith("error_research:")}
    assert by_feat["sentiment_score"]["status"] == "FORWARD_VALIDATED" and by_feat["sentiment_score"]["prospective"]
    assert by_feat["sentiment_score"]["evidence"]["forward_outcome"] == "CONFIRMED"
    assert by_feat["rel_volume"]["status"] == "REJECTED" and by_feat["sentiment_drift"]["status"] == "REJECTED"
    assert rd.memory_prior(["trading_activity"], m3)["factor"] < 1 < rd.memory_prior(["sentiment"], m3)["factor"]
    events = rd.read_journal(tmp_path / "journal.jsonl")[0]
    assert sum(e["event"] == "option_check" for e in events) == 1               # bestätigende Optionsprüfung eingefroren


def test_memory_prior_changes_the_selection():
    c = {"id": "x", "kind": "ledger", "families": ["news"], "sources": ["candidate_ledger"], "coverage": 1.0,
         "discovery": {"mean": 0.05, "t": 3.0, "n_periods": 3}}
    d = {**c, "id": "y", "families": ["attention"]}
    neutral = [rd.error_priority(z, {}, {}, 5)["priority"] for z in (c, d)]
    assert neutral[0] == neutral[1]
    rejected = {f"error_research:H{i}": {"hypothesis_id": f"H{i}", "kind": "error_research", "status": "REJECTED",
                                         "spec": {"family": "error:news"}} for i in range(3)}
    pc, pd_ = (rd.error_priority(z, rejected, {}, 5) for z in (c, d))
    assert pc["priority"] < pd_["priority"] and pc["memory_prior"]["rejected"] == 3
    assert rd.error_priority(c, {}, {"x|5|1": "ERR-1"}, 5)["memory_hit"] == "ERR-1"


def test_error_gaps_drive_active_learning_only_for_unexplained_segments():
    cat = [{"id": "pos", "status": "candidate", "dimension": ["positioning"], "license": "CC0", "pit": "vintages",
            "history_from": 2010}]
    seg = {"gap_id": "error:segment:high_short_interest:5d", "kind": "unexplained_segment", "severity": 0.9,
           "dimensions": ["positioning"]}
    hist = {**seg, "gap_id": "error:history:x:5d", "kind": "history_too_short"}
    assert rd.active_learning({}, [], cat)[0]["expected_information_gain"] == 0.0
    assert rd.active_learning({}, [], cat, [hist])[0]["expected_information_gain"] == 0.0
    a = rd.active_learning({}, [], cat, [seg])[0]
    assert a["current_uncertainty"] == 0.9 and a["driven_by_error_gaps"] == [seg["gap_id"]]


def test_director_hypotheses_never_confirmed_on_contaminated_locked(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(rl, "add_exposures", lambda df: df)
    monkeypatch.setattr(rl, "memory_fields", lambda *a, **k: {})
    monkeypatch.setattr(rl, "load_hypotheses", lambda *a, **k: [])
    monkeypatch.setattr(rl, "load_director", lambda *a, **k: [{"id": "RD-1", "signal": "rev_1m", "direction": 1,
                                                              "source": "director"}])
    monkeypatch.setattr(rl, "load_factory", lambda *a, **k: [])
    monkeypatch.setattr(rl, "load_autonomous", lambda *a, **k: [])
    monkeypatch.setattr(rl, "evaluate_hypothesis", lambda panel, hyp, seen, pol: {
        "id": hyp["id"], "source": hyp["source"], "signal": hyp["signal"], "status": "passed_pre_fdr",
        "p_value": 1e-6, "reasons": [], "stages": {}})
    monkeypatch.setattr(rl, "locked_check", lambda *a, **k: (_ for _ in ()).throw(AssertionError("kontaminiert")))
    from modules.alt_data import contracts as ac
    monkeypatch.setattr(ac, "load", lambda: [])
    monkeypatch.setattr(ac, "register", lambda _: {})
    monkeypatch.setattr(ac, "lab_hypotheses", lambda *a: [])
    pol = {"periods": {"locked_status": "CONTAMINATED"}, "multiple_testing": {"fdr_q": 0.1},
           "hypothesis_acceptance": {"locked_must_be_positive": True}}
    db = rl.run(pd.DataFrame(), protocol=pol, db_path=tmp_path / "db.json", with_discovery=False)
    assert db["hypotheses"]["RD-1"]["status"] == "historical_exploratory_pending_forward"
    assert db["hypotheses"]["RD-1"]["canonical_status"] == "INCONCLUSIVE"


def test_workflow_runs_error_research_weekly_before_director_and_critical():
    import yaml
    steps = yaml.safe_load((ROOT / ".github/workflows/ml_research.yml").read_text())["jobs"]["ml"]["steps"]
    ids = [s.get("id") for s in steps]
    er = steps[ids.index("error_research")]
    assert "if" not in er and er["run"] == "python -m modules.research_director --errors"     # wöchentlich + full
    assert ids.index("error_research") < ids.index("director")
    check = steps[-1]["env"]
    assert "error_research=${{ steps.error_research.outcome }}" in check["CRITICAL"]
    assert "error_research=" not in check["DEGRADED"]


def test_cli_exit_code_is_two_on_critical(monkeypatch):
    monkeypatch.setattr(rd, "run_errors", lambda: {"summary": {}, "decisions": [], "hypotheses": [],
                                                   "health": {"status": "CRITICAL", "problems": ["x"]}})
    assert rd.main(["--errors"]) == 2
    monkeypatch.setattr(rd, "run_errors", lambda: {"summary": {}, "decisions": [], "hypotheses": [],
                                                   "health": {"status": "OK", "problems": []}})
    assert rd.main(["--errors"]) == 0
