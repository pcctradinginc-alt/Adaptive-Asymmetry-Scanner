"""Fehlergetriebene Forschung (config/error_research_protocol.yaml): Forward-Prognosefehler -> Muster -> Hypothese ->
Walk-Forward-Screen mit Embargo -> einmaliger Forward-Test mit BH -> Options-Prüfung -> Research Memory -> nächste
Entscheidung. Synthetische Daten, kein Netz – kein Alpha-Nachweis.

Echter Lauf 2026-11-07 (docs/ERROR_DRIVEN_RESEARCH_2026-11-07.md): 8.590 von 8.976 Ledger-Prognosen gegen das
ml_predictions-Archiv geprüft (386 mit < 3 Modellen), 7.342 fällige 5-Tage-Fehler; 1 Hypothese registriert
(sentiment_score, Forward ab 08.11.), 1 Muster durch Größe erklärt (news_count*sentiment_score), 13 Datenlücken.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from modules import research_director as rd
from modules import research_lab as rl
from modules import research_memory as rm

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_SHA = "ef2914e42f346c7a51f9122051faa4425bc409d665755a14f868e7095f8030f2"
MODELS = ("ml_a", "ml_b", "ml_c")
FEATURES = ("sentiment_score", "short_mom", "news_count", "rel_volume")


def test_protocol_is_pinned_and_bounded():
    raw = (ROOT / "config" / "error_research_protocol.yaml").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA, "Protokoll geändert – nur strenger oder bei belegtem Fehler"
    p = rd.error_protocol()
    assert p["seen_until"] == "2026-11-06" and p["budget"] == {"max_new_per_run": 2, "max_lifetime": 24}
    assert len(rd.error_candidates(p)) == 23                       # fester Suchraum = Mehrfachtest-Nenner
    assert p["forward"]["alpha"] == 0.05 and p["forward"]["fdr_q"] == 0.05 and p["forward"]["min_periods"] == 4


def _proto(**over):
    p = rd.error_protocol()
    p = json.loads(json.dumps(p))
    p["forecast"]["models"] = list(MODELS)
    p["forecast"]["horizons"] = [5]
    p["search_space"] = {"ledger": list(FEATURES), "external": {}, "pairs": []}
    p["families"] = {"sentiment_score": "sentiment", "short_mom": "positioning", "news_count": "news",
                     "rel_volume": "trading_activity"}
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


def _world(end="2026-12-31", n=40, seed=3, regime_switch="2026-10-19"):
    """Ledger + Prognose-Archiv: sentiment_score erklärt den Fehler dauerhaft, rel_volume nur bis regime_switch,
    short_mom ist nur ein Größen-Stellvertreter, news_count ist Rauschen."""
    rnd = np.random.default_rng(seed)
    rows, preds, opt = [], [], {}
    tickers = [f"T{i:02d}" for i in range(n)]
    for d in _bdays("2026-09-28", end):
        p = rnd.uniform(0.01, 1.0, n).round(4)
        x, size, rv, noise = rnd.normal(size=n), rnd.normal(size=n), rnd.normal(size=n), rnd.poisson(3, n)
        ret = (0.02 * (p - 0.5) + 0.05 * (_rank(x) - 0.5) + 0.04 * (_rank(size) - 0.5)
               + (0.05 * (_rank(rv) - 0.5) if d < regime_switch else 0.0) + rnd.normal(0, 0.03, n))
        y_opt = 0.6 * (_rank(x) - 0.5) - 0.05 + rnd.normal(0, 0.2, n)
        for m in MODELS:
            preds.append({"model_id": m[3:], "prediction_date": d, "created_at": f"{d}T08:00:00+00:00",
                          "rank_pct": dict(zip(tickers, map(float, p)))})
        for i, t in enumerate(tickers):
            f = {m: float(p[i]) for m in MODELS}
            f.update(sentiment_score=float(x[i]), log_market_cap=float(size[i]),
                     short_mom=float(size[i] + rnd.normal(0, 0.05)), news_count=int(noise[i]),
                     rel_volume=float(rv[i]), sector="Technology" if i % 2 else "Energy")
            rows.append({"date": d, "ticker": t, "signal_timestamp": f"{d}T15:00:00+00:00", "features": f,
                         "gap_at_entry": 0.0, "outcomes": {"ret_5d": float(ret[i])}})
            opt[(d, t)] = float(y_opt[i])
    return rows, preds, opt


@pytest.fixture(scope="module")
def world():
    return _world()


def test_forecast_provenance_against_prediction_archive(world):
    rows, preds, _ = world
    r0 = rows[0]
    fake = {**r0, "ticker": r0["ticker"], "date": r0["date"], "features": {**r0["features"], "ml_a": 0.123}}
    late = [{**p, "created_at": "2026-09-28T16:00:00+00:00"} if p["prediction_date"] == "2026-09-28" else p
            for p in preds]                                                  # nach dem Signal erstellt
    out, counts = rd.verified_forecasts([fake], preds, _proto()["forecast"])
    assert out[0]["status"] == "FORECAST_MISMATCH" and counts == {"FORECAST_MISMATCH": 1}
    assert rd.verified_forecasts([r0], late, _proto()["forecast"])[0][0]["status"] == "FORECAST_MISMATCH"
    ok, c = rd.verified_forecasts(rows[:40], preds, _proto()["forecast"])
    assert c == {"VERIFIED": 40} and ok[0]["n_models"] == 3
    few = {**r0, "features": {"ml_a": r0["features"]["ml_a"]}}
    assert rd.verified_forecasts([few], preds, _proto()["forecast"])[0][0]["status"] == "TOO_FEW_MODELS"


def test_period_blocks_are_weekly_complete_and_start_bounded():
    ic = {d: 0.1 for d in _bdays("2026-09-29", "2026-11-06")}
    b = rl.period_blocks(ic, 5, "2026-09-29", date(2026, 11, 7))
    assert [x["start"] for x in b][:2] == ["2026-09-28", "2026-10-05"] and len(b) == 5   # Montag verankert
    assert b[0]["n_dates"] == 4 and b[-1]["end"] == "2026-11-01"         # 28.09. vor dem Start; 02.–08.11. offen
    assert len(rl.period_blocks(ic, 5, "2026-09-29", date(2026, 11, 6))) == 4        # 30.10.-Label erst am 06.11.
    assert rl.period_blocks(ic, 20, "2026-09-29", date(2026, 11, 7)) == []           # 20 T: kein Fenster fertig


def test_walk_forward_screen_falsifies_and_respects_embargo():
    cfg = _proto()["validation"]
    ic = {d: (-0.05 if d >= "2026-10-26" else 0.5) for d in _bdays("2026-09-28", "2026-11-20")}
    wf = rl.ledger_walk_forward(ic, 5, "2026-10-18", "2026-11-23", cfg)
    assert wf["window_start"] == "2026-10-26"                           # 1 Fenster Embargo (19.–25.10.)
    assert wf["status"] == "REJECTED_WF" and wf["n_periods"] == 3            # 16.–22.11. noch offen
    ic2 = {d: 0.04 + (0.01 if int(d[-2:]) % 2 else -0.01) for d in ic}
    assert rl.ledger_walk_forward(ic2, 5, "2026-10-18", "2026-11-23", cfg)["status"] == "PASSED_WF"
    assert rl.ledger_walk_forward(ic2, 5, "2026-10-18", "2026-11-07", cfg)["status"] == "INCONCLUSIVE_WF"


def test_forward_test_is_a_single_look_on_frozen_windows():
    cfg = _proto()["forward"]
    ic = {d: 0.05 + 0.01 * (i % 3) for i, d in enumerate(_bdays("2026-11-09", "2027-01-29"))}
    first = rl.ledger_forward_test(ic, 5, "2026-11-07", date(2026, 12, 15), cfg)
    later = rl.ledger_forward_test({**ic, **{d: -0.9 for d in _bdays("2026-12-14", "2027-01-29")}}, 5,
                                   "2026-11-07", date(2027, 2, 5), cfg)
    assert first["status"] == later["status"] == "TESTED"
    assert first["blocks"] == later["blocks"] and first["p_one_sided"] == later["p_one_sided"]   # spätere Daten egal
    assert rl.ledger_forward_test(ic, 5, "2026-11-07", date(2026, 11, 20), cfg)["status"] == "PENDING_FORWARD"
    assert rl.ledger_forward_test({}, 5, "2026-11-07", date(2027, 6, 1), cfg)["status"] == "INCONCLUSIVE"


def test_patterns_recurring_explained_by_size_and_noise(world):
    rows, preds, _ = world
    proto = _proto()
    fc, _ = rd.verified_forecasts([r for r in rows if r["date"] <= "2026-11-06"], preds, proto["forecast"])
    errs = rd.forecast_errors(fc, 5, date(2026, 11, 7), proto)["by_date"]
    pat = rd.error_patterns(errs, rd.error_values(fc, proto), proto, 5, date(2026, 11, 7))
    v = {c["id"]: c["verdict"] for c in pat["candidates"]}
    assert pat["discovery"] == {"start": "2026-09-28", "end": "2026-10-18", "periods": 3}
    assert v["sentiment_score"] == "RECURRING" and v["short_mom"] == "EXPLAINED_BY_KNOWN_FACTOR"
    assert v["news_count"] == "NOT_RECURRING"
    late = {d: [{**r, "resid": -r["resid"]} for r in rs] if d > "2026-10-18" else rs for d, rs in errs.items()}
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
    """Fehler -> Hypothese -> Test -> Ergebnis -> Gedächtnis -> neue Entscheidung (zwei Läufe)."""
    rows, preds, opt = world
    proto = _proto()
    reg, rep_p = tmp_path / "error_hypotheses.jsonl", tmp_path / "error_research.json"
    src = {"error_research": rep_p}
    r1 = rd.error_research([r for r in rows if r["date"] <= "2026-11-06"], preds, date(2026, 11, 7), memory=[],
                           proto=proto, registry=reg, report=rep_p, option_labels=opt)
    regd = {d["subject"]: d for d in r1["decisions"] if d["decision"] == "register"}
    assert set(regd) == {"sentiment_score (5 T)", "rel_volume (5 T)"}             # Budget 2 je Lauf
    assert {h["status"] for h in r1["hypotheses"]} == {"PENDING_FORWARD"}
    assert all(h["walk_forward"]["status"] == "INCONCLUSIVE_WF" for h in r1["hypotheses"])
    assert all(h["spec"]["forward"]["start_after"] == "2026-11-07" for h in r1["hypotheses"])
    mem = tmp_path / "memory.jsonl"
    rm.sync(mem, src, now="2026-11-07T12:00:00+00:00")
    m1 = rm.latest(rm.load(mem))
    assert m1["error_pattern:ERRPAT-short_mom-5d"]["status"] == "NOT_ROBUST"
    assert {e["status"] for k, e in m1.items() if k.startswith("error_research:")} == {"PENDING_FORWARD"}
    # Forward-Daten bis Jahresende: sentiment bestätigt, rel_volume (Regimewechsel) verworfen
    r2 = rd.error_research(rows, preds, date(2026, 12, 31), memory=rm.load(mem), proto=proto, registry=reg,
                           report=rep_p, option_labels=opt)
    st = {h["spec"]["feature"]: h for h in r2["hypotheses"]}
    assert st["sentiment_score"]["status"] == "FORWARD_CONFIRMED"
    assert st["rel_volume"]["status"] == "FORWARD_REJECTED"
    oc = st["sentiment_score"]["option_check"]
    assert oc["status"] == "OPTION_EDGE" and oc["evidence"] == "confirmatory"
    assert oc["window"][0] == "2026-11-08"                                 # nur Forward-Signale
    assert st["rel_volume"]["option_check"]["status"] == "NOT_REACHED"
    assert all(d["decision"] != "register" or d["subject"] not in regd for d in r2["decisions"])
    assert any(d["decision"] == "skip" and "ERR-" in d["because"] for d in r2["decisions"])
    rm.sync(mem, src, now="2026-12-31T12:00:00+00:00")
    m2 = rm.latest(rm.load(mem))
    by_id = {e["hypothesis_id"]: e for k, e in m2.items() if k.startswith("error_research:")}
    assert by_id[st["sentiment_score"]["id"]]["status"] == "FORWARD_VALIDATED"
    assert by_id[st["sentiment_score"]["id"]]["prospective"] is True
    assert by_id[st["rel_volume"]["id"]]["status"] == "REJECTED"
    assert rd.memory_prior(["trading_activity"], m2)["factor"] < 1 < rd.memory_prior(["sentiment"], m2)["factor"]


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


def test_modified_registration_is_invalid_and_budget_is_lifetime(world, tmp_path):
    rows, preds, opt = world
    proto = _proto(budget={"max_new_per_run": 2, "max_lifetime": 1})
    reg = tmp_path / "r.jsonl"
    r = rd.error_research([x for x in rows if x["date"] <= "2026-11-06"], preds, date(2026, 11, 7), memory=[],
                          proto=proto, registry=reg, report=tmp_path / "e.json", option_labels=opt)
    assert [d["decision"] for d in r["decisions"]].count("register") == 1
    assert "defer" in [d["decision"] for d in r["decisions"]]
    e = json.loads(reg.read_text().splitlines()[0])
    e["spec"]["direction"] = -e["spec"]["direction"]                      # nachträglich geändert
    reg.write_text(json.dumps(e) + "\n")
    r2 = rd.error_research([x for x in rows if x["date"] <= "2026-11-06"], preds, date(2026, 11, 7), memory=[],
                           proto=proto, registry=reg, report=tmp_path / "e.json", option_labels=opt, persist=False)
    assert [h["status"] for h in r2["hypotheses"]] == ["INVALID_MODIFIED"]
    assert rm.error_entries(r2)[0]["status"] == "REJECTED"


def test_error_gaps_drive_active_learning_only_for_unexplained_segments():
    cat = [{"id": "pos", "status": "candidate", "dimension": ["positioning"], "license": "CC0", "pit": "vintages",
            "history_from": 2010}]
    seg = {"gap_id": "error:segment:high_short_interest:5d", "kind": "unexplained_segment", "severity": 0.9,
           "dimensions": ["positioning"]}
    hist = {**seg, "gap_id": "error:history:x:5d", "kind": "history_too_short"}
    base = rd.active_learning({}, [], cat)[0]
    assert base["expected_information_gain"] == 0.0
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
