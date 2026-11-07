"""Autonomie-Upgrade 2026-11-01: Abnahme A4 (Hypothesen), B5 (Options-Lernen), C5 (Discovery) und End-to-End.

Alle Daten sind SYNTHETISCH und vollständig kontrolliert. Die Tests belegen technische Funktionsfähigkeit
(Erkennung, Registrierung, Prüfkette, Zeitpunkte, Duplikate, Fail-closed) – NIE wirtschaftlichen Alpha-Erfolg.
"""
from __future__ import annotations

import ast
import copy
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from modules import hypothesis_factory as hf
from modules import ml_research as ml
from modules import option_learner as ol
from modules import research_director as rd
from modules import research_lab as rl
from modules import research_memory as rm
from modules.atomic_io import read_jsonl
from modules.autonomous_alpha import option_link as al

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 30, 6, tzinfo=timezone.utc)
PLANTED = "rank(ted_awards_z) * step(-mom_3m)"          # statische Fabrik-Divergenz (bereits bekannt)
SIBLING = "rank(ted_awards_z) * step(-rev_1m)"           # gleiche Information, vom Scan selbst gefunden
NOISE = "rank(vol_ratio) * step(-ret_5d)"


def _proto():
    p = copy.deepcopy(ml.PROTOCOL)
    p["periods"].update(train_start="2015-01-01", discovery_end="2017-01-01", first_test_year=2017,
                        locked_from="2020-01-01")
    return p


def _synthetic_panel(effect: str | None = PLANTED, size: float = 0.05, seed: int = 11):
    from tests.test_ml_research import _panel
    p = _panel(n_days=1700, n_stocks=60, signal=False)
    rnd = np.random.default_rng(seed)
    p["sector"] = np.where(p["ticker"].str[1:].astype(int) % 3 == 0, "Technology", "Energy")
    p["ted_awards_z"] = rnd.normal(0, 1, len(p))
    p["fwd_xs_20"] = rnd.normal(0, 0.02, len(p))
    if effect:
        p["fwd_xs_20"] += size * rl.eval_signal(rl.add_exposures(p), effect).fillna(0)
    p.loc[p["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    return p


@pytest.fixture(scope="module")
def env():
    """Geschütztes Protokoll unverändert; nur für den synthetischen Zeitraum verschobene Fenster (Kopie)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ml, "MIN_CROSS_SECTION", 20)
        proto = _proto()
        mp.setattr(ml, "PROTOCOL", proto)
        yield {"proto": proto, "panel": _synthetic_panel()}


def _plan(panel, memory=None):
    return hf.plan(panel, memory=memory or [], machine_state=None, now=NOW, conditions={"sources": {}})


@pytest.fixture(scope="module")
def lab_run(env, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lab")
    res = _plan(env["panel"])
    hyps = [{**h, "source": "factory"} for h in hf.lab_hypotheses(res)]
    hyps.append({"id": "FAC-NOISE", "signal": NOISE, "direction": 1, "source": "factory", "family": "noise",
                 "spec_hash": "n1", "title": "Rauschen"})
    (tmp / "h.yaml").write_text(yaml.safe_dump({"hypotheses": hyps}, allow_unicode=True))
    (tmp / "h.json").write_text(json.dumps({"hypotheses": hyps}))
    db = rl.run(env["panel"], protocol=env["proto"], hyp_path=tmp / "h.yaml", db_path=tmp / "db.json",
                with_discovery=False)
    return {"plan": res, "db": db, "tmp": tmp, "hyps": hyps}


def _anomalies(res):
    return [h for h in res["ideas"] if h["idea_source"] == "anomaly" and h["signal"]]


# ── A4: anomaliegetriebene Hypothesengenerierung ─────────────────────────────

def test_a4_1_new_anomaly_creates_a_valid_documented_hypothesis(lab_run):
    an = _anomalies(lab_run["plan"])
    assert [h["signal"] for h in an] == [SIBLING]                         # gemessen, nicht aus einer Vorlage
    h = an[0]
    rl.validate_expr(h["signal"])
    assert h["id"].startswith("FAC-") and h["spec_hash"] == hf.spec_hash(h) and h["plan_status"] == "SELECTED"
    assert h["anomaly"]["version"] == hf.ANOMALY_VERSION and h["anomaly"]["window"] == "2015-01-01..2017-01-01"
    assert h["anomaly"]["p_two_sided"] <= h["anomaly"]["fdr_q"] and h["direction"] == 1
    d = h["dossier"]
    for k in ("research_question", "mechanism", "data", "population", "direction", "lag", "horizon_trading_days",
              "comparison_model", "primary_metric", "falsification", "data_quality", "data_gaps", "effort_cost",
              "novelty"):
        assert d.get(k) is not None, k
    assert d["data"]["features"] == ["ted_awards_z", "rev_1m"] and d["effort_cost"]["llm_calls"] == 0
    assert "kein Beleg" in d["selection_evidence"]                         # Auswahlfenster nie als Evidenz
    assert all(isinstance(x.get("dossier"), dict) for x in lab_run["plan"]["ideas"])   # Dossier für jede Idee


def test_a4_2_known_hypothesis_is_not_registered_again(env):
    h = _anomalies(_plan(env["panel"]))[0]
    mem = [rm._entry("factory", h["id"], "REJECTED", {"signal": h["signal"], "family": h["family"], "direction": 1,
                                                      "idea_source": "anomaly"}, source="factory", evidence=None)]
    again = _plan(env["panel"], memory=mem)
    assert h["signal"] not in [x["signal"] for x in _anomalies(again)]    # bekanntes Signal nicht neu gescannt
    assert h["id"] not in again["selected"]


def test_a4_3_missing_data_is_deferred_as_data_gap(env):
    res = _plan(env["panel"].drop(columns=["ted_awards_z"]))
    gaps = {h["anomaly_type"]: h for h in res["ideas"] if h["idea_source"] == "anomaly" and not h["signal"]}
    assert {"option_vs_stock", "contradiction"} <= set(gaps)
    for h in gaps.values():
        assert h["plan_status"] == "DATA_GAP" and h["priority"] == 0.0 and h["id"] not in res["selected"]
        assert "nicht testbar" in h["dossier"]["research_question"] and h["dossier"]["data_gaps"]
    assert not any(h["plan_status"] == "VALIDATED" for h in res["ideas"])


def test_a4_4_same_data_same_research_decision(env):
    a, b = _plan(env["panel"]), _plan(env["panel"])
    def key(r):
        return [(h["id"], h["plan_status"], h["priority"]) for h in r["ideas"]]
    assert key(a) == key(b) and a["selected"] == b["selected"] and a["anomaly_scan"] == b["anomaly_scan"]


def test_a4_5_anomaly_hypotheses_pass_through_the_existing_lab_chain(lab_run):
    an = _anomalies(lab_run["plan"])[0]
    rec = lab_run["db"]["hypotheses"][an["id"]]
    assert rec["stages"] == {"leakage": "pass", "duplicate": "pass", "coverage": "pass", "walk_forward": "pass"}
    assert rec["status"] == "accepted" and rec["family"] == an["family"] and rec["spec_hash"] == an["spec_hash"]
    assert rec["locked"]["base"]["n_cohorts"] >= rl.MIN_LOCKED_COHORTS


def test_a4_6_no_confirmation_without_evidence(env, lab_run, tmp_path, monkeypatch):
    noise = lab_run["db"]["hypotheses"]["FAC-NOISE"]
    assert noise["canonical_status"] == "REJECTED" and noise["reasons"]
    for name in ("RESULTS_OUT", "CHALLENGERS", "FORWARD_LEDGER"):
        monkeypatch.setattr(hf, name, tmp_path / getattr(hf, name).name)
    proto = hf.load_protocol()
    proto["robustness"] = {**proto["robustness"], "placebo_permutations": 20}
    res = hf.evaluate(env["panel"], proto, NOW, db_path=lab_run["tmp"] / "db.json",
                      hyp_path=lab_run["tmp"] / "h.json")["results"]
    assert res["FAC-NOISE"]["status"] == "REJECTED"
    assert all(r["evidence_class"] == "EXPLORATORY_HISTORICAL" and "CONTAMINATED" in r["locked_holdout"]
               for r in res.values())
    ids = [json.loads(x)["hypothesis_id"] for x in hf.CHALLENGERS.read_text().splitlines()]
    assert "FAC-NOISE" not in ids
    pure_noise = hf.scan_anomalies(_synthetic_panel(effect=None, seed=5))
    assert pure_noise["status"] == "OK" and pure_noise["n_scanned"] > 50 and pure_noise["survivors"] == []


def test_lab_text_similarity_no_longer_drops_untested_factory_specs(env, tmp_path):
    """Defekt 2026-10-03: Vorlagentext ähnlich -> 4 von 6 Fabrik-Tests ungeprüft "REJECTED". Jetzt geprüft;
    Text-Hypothesen ohne maschinelle Spezifikation bleiben gesperrt (unverändert)."""
    tmpl = "{f} wirkt bedingt (Sektor Healthcare)"
    stmt = "Gemessen in den Auswahljahren: Quelle zeigt nur bedingt Information. Mechanismus offen – Test entscheidet."
    db = {"hypotheses": {"FAC-OLD": {"id": "FAC-OLD", "title": tmpl.format(f="vol_20"), "statement": stmt,
                                     "signal": "rank(vol_20)", "status": "rejected", "canonical_status": "REJECTED",
                                     "source": "factory"}}}
    (tmp_path / "db.json").write_text(json.dumps(db))
    hyps = [{"id": "FAC-NEW", "title": tmpl.format(f="vol_60"), "statement": stmt, "signal": "rank(vol_60)",
             "direction": 1, "source": "factory", "family": "cond_vol_60", "spec_hash": "s1"},
            {"id": "DIR-NEW", "title": tmpl.format(f="vol_60"), "statement": stmt, "signal": "rank(beta_126)",
             "direction": 1, "source": "director"}]
    (tmp_path / "h.yaml").write_text(yaml.safe_dump({"hypotheses": hyps}, allow_unicode=True))
    out = rl.run(env["panel"], protocol=env["proto"], hyp_path=tmp_path / "h.yaml", db_path=tmp_path / "db.json",
                 with_discovery=False)["hypotheses"]
    assert out["FAC-NEW"]["status"] != "blocked_similar_to_rejected" and out["FAC-NEW"]["stages"]["leakage"] == "pass"
    # Audit 2026-11-06: auch Director-Ideen mit auswertbarem Signal werden geprüft; Textähnlichkeit nur vermerkt
    assert out["DIR-NEW"]["status"] != "blocked_similar_to_rejected" and out["DIR-NEW"]["stages"]["leakage"] == "pass"
    assert out["DIR-NEW"]["text_similar_to"] in ("FAC-OLD", "FAC-NEW")      # gleicher Vorlagentext
    assert out["FAC-OLD"]["canonical_status"] == "REJECTED"                  # Historie unverändert


def test_priority_extensions_only_reorder_and_are_transparent(env):
    proto = hf.load_protocol()
    base = {"family": "f", "relevance": 1.0, "readiness": {"status": "READY", "data_quality": 1.0, "cost": 1.0}}
    p0 = hf.priority(dict(base), {}, 0.0, proto)
    assert hf.priority({**base, "mechanism_source": "measured_condition"}, {}, 0.0, proto) == pytest.approx(0.75 * p0, rel=1e-3)
    thin = {**base, "readiness": {**base["readiness"], "coverage_2016plus": 0.4}}
    assert hf.priority(thin, {}, 0.0, proto) == pytest.approx(0.8 * p0, rel=1e-3)
    contra = {**base, "independent_evidence": {"prospective_negative": 1, "prospective_positive": 0}}
    assert hf.priority(contra, {}, 0.0, proto) == pytest.approx(0.5 * p0, rel=1e-3)
    res = _plan(env["panel"])
    assert len(res["selected"]) <= proto["budget"]["tests_per_run"]          # Budget unverändert
    sel = [h for h in res["ideas"] if h["plan_status"] == "SELECTED"]
    assert all(set(h["priority_components"]) >= {"eig", "plausibility", "testability", "independent_evidence",
                                                 "role"} for h in sel)
    assert all("kein unabhängiger Alpha-Nachweis" in h["priority_components"]["role"] for h in sel)


# ── B5: Lernkreislauf des Options-Lernmodells (synthetisches Ledger) ─────────

def _ledger(monkeypatch, n_days=40, drop_exit_for=None):
    from tests.test_option_learner import H, _history, _trading_days
    days = _trading_days(date(2026, 1, 5), n_days)
    rows, quotes = _history(days)
    if drop_exit_for:
        sym = next(r for r in rows if r["signal_id"] == drop_exit_for)["real_option"]["symbol"]
        mark = next(r for r in rows if r["signal_id"] == drop_exit_for)["outcomes"][f"real_strat_mark_date_{H}d"]
        quotes = [q for q in quotes if not (q["occ_symbol"] == sym and q["quote_time"].startswith(mark))]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: [r for r in rows if r["date"] <= as_of.isoformat()])
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    return days, rows, quotes, H


def _at(d):
    return datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).replace(hour=17)


def test_b5_closed_loop_labels_instances_freeze_idempotence_and_reference(monkeypatch, tmp_path):
    from tests.test_option_learner import _row
    days, rows, quotes, h = _ledger(monkeypatch)
    d1, d2 = days[20], days[38]
    rows += [_row(d1.isoformat(), 1, 0.8), _row(d2.isoformat(), 1, 0.8)]
    r1 = ol.recommend(d1, horizon=h, out=tmp_path, now=_at(d1))
    ref_path = tmp_path / ol.REFERENCE_DIR / f"{h}d_{ol.experiment_id()}.json"
    frozen = ref_path.read_text()
    r2 = ol.recommend(d2, horizon=h, out=tmp_path, now=_at(d2))
    assert ol.recommend(d2, horizon=h, out=tmp_path, now=_at(d2))["journaled"] == 0       # (4) keine Dubletten
    inst = read_jsonl(tmp_path / ol.INSTANCES)
    assert [i["model_id"] for i in inst] == [r1["model_id"], r2["model_id"]]              # (2) registriert
    # (1) ein Label fließt erst nach seinem Mark-Tag ein: Instanz 2 kennt mehr Labels, alle vor ihrem Stichtag
    assert inst[0]["max_label_day"] < d1.isoformat() and inst[1]["max_label_day"] < d2.isoformat()
    assert inst[1]["n_train_labels"] > inst[0]["n_train_labels"]
    late = [d for d in ol.dataset(rows, quotes, d2, h) if d1 <= d["label_day"] < d2]
    assert late and inst[0]["n_train_labels"] == len([d for d in ol.dataset(rows, quotes, d1, h)
                                                       if d["label_day"] < d1])
    # (3) Prognose/Entscheidung vor dem Ergebnis festgehalten
    j = [x for x in read_jsonl(tmp_path / ol.JOURNAL) if x["date"] == d2.isoformat()]
    assert j and all(x["created_at"][:10] == d2.isoformat() for x in j)
    assert all(x["model_id"] in {r2["model_id"], r1["model_id"]} and x["train_cutoff"] <= x["date"] for x in j)
    # (8) Referenz unverändert, entscheidet über dieselben Kandidaten
    assert ref_path.read_text() == frozen and r2["reference_model_id"] == r1["model_id"]
    learn = {(x["date"], x["ticker"]) for x in j if x["kind"] == "decision" and x["role"] == "learning"}
    refs = {(x["date"], x["ticker"]) for x in j if x["kind"] == "decision" and x["role"] == "reference"}
    assert learn == refs


def test_b5_5_foreign_experiment_versions_are_excluded(monkeypatch, tmp_path):
    days, rows, quotes, h = _ledger(monkeypatch)
    cands = ol.candidates(rows, quotes, days[-1], h)
    j = [{"date": days[5].isoformat(), "signal_id": "x", "ticker": "T1", "kind": "decision", "role": "learning",
          "experiment_id": "OTHER", "model_id": "m", "train_cutoff": days[5].isoformat(), "horizon": h,
          "created_at": _at(days[5]).isoformat(), "action": "TRADE", "contract": ["T1270319C00100000"]}]
    fw = ol.forward(j, cands, h, [], days[-1])
    assert fw["excluded"] == {"OTHER_EXPERIMENT": 1} and fw["valid_decisions"] == 0


def test_b5_6_missing_quotes_never_create_labels(monkeypatch):
    _days, rows, _quotes, h = _ledger(monkeypatch, drop_exit_for=None)
    sid = rows[0]["signal_id"]
    days, rows, quotes, h = _ledger(monkeypatch, drop_exit_for=sid)
    cands = ol.candidates(rows, quotes, days[-1], h)
    c = next(x for x in cands if x["signal_id"] == sid)
    assert c["exprs"][0]["status"] == "EXIT_NOT_ARCHIVED" and c["exprs"][0]["y"] is None
    data = ol.dataset(rows, quotes, days[-1], h)
    assert sid not in {d["signal_id"] for d in data}
    lp = ol.label_pipeline(cands, days[-1], h, start=days[0].isoformat())
    assert lp["verified"] == len(data) and lp["technically_missing"].get("EXIT_NOT_ARCHIVED", 0) >= 1
    assert lp["technical_missing_share"] == round(sum(lp["technically_missing"].values())
                                                  / lp["due_after_archive_start"], 4)


def test_b5_7_negative_outcomes_are_valued_with_costs():
    from tests.test_option_learner import H, _row
    quotes = []
    r = _row("2026-01-05", 1, -0.5, ret=-0.3, quotes=quotes)
    y = ol.net_label(r, H)
    assert y == pytest.approx(r["outcomes"][f"real_strat_ret_{H}d"] - 2 * 0.65 / (10.0 * 100)) and y < -0.29
    total = ol.net_label({**r, "outcomes": {f"real_strat_ret_{H}d": -1.0}}, H)
    assert total < -1.0                                                          # Totalverlust plus Gebühren
    picks = [{"cand": {"date": "2026-01-05", "ticker": "T1", "signal_id": "a", "sample": "directional",
                       "exprs": [{"expression": "single", "contract": ["S"], "strategy": "LONG_CALL",
                                  "status": "VERIFIED", "y": y, "ask": 10.0, "mid": 9.9}]},
              "dec": {"action": "TRADE", "contract": ["S"]}, "preds": {("S",): 0.1}}]
    s = ol.assess(picks, H)
    assert s["mean_selected"] == pytest.approx(y, abs=1e-6) and s["mean_selected_adverse_entry"] < y


def test_b5_9_learning_status_stays_need_more_data_and_maps_existing_verdicts():
    lp = {"verified": 10, "independent_signal_days": 2, "not_yet_due": 5, "due_after_archive_start": 10,
          "technical_missing_share": 0.0, "technically_missing": {}}
    nm = {"verdict": "NEED_MORE_DATA"}
    assert ol.learning_status(lp, nm, nm)["state"] == "NEED_MORE_DATA"
    assert ol.learning_status({**lp, "verified": 0, "not_yet_due": 0}, nm, nm)["state"] == "NO_DATA"
    bad = {**lp, "due_after_archive_start": 40, "technical_missing_share": 0.5, "technically_missing": {"X": 20}}
    assert ol.learning_status(bad, nm, nm)["state"] == "TECHNICAL_DEFECT"
    assert ol.learning_status({**bad, "due_after_archive_start": 5}, nm, nm)["state"] == "NEED_MORE_DATA"
    assert ol.learning_status(lp, {"verdict": "NOT_VALIDATED"}, nm)["state"] == "NOT_VALIDATED"
    pos = ol.learning_status(lp, {"verdict": "VALIDATED"}, {"verdict": "FORWARD_POSITIVE"})
    assert pos["state"] == "FORWARD_POSITIVE" and "release_gate" in pos["alpha_claim"]
    assert ol.learning_status(lp, nm, nm)["alpha_claim"] == "keiner"


def test_b5_10_learning_loop_has_no_production_influence(monkeypatch, tmp_path):
    from tests.test_option_learner import _row
    days, rows, _quotes, h = _ledger(monkeypatch, n_days=30)
    rows.append(_row(days[25].isoformat(), 1, 0.8))
    res = ol.recommend(days[25], horizon=h, out=tmp_path, now=_at(days[25]))
    st = ol.evaluate(days[-1], horizon=h, out=tmp_path)
    assert res["production_influence"] == st["production_influence"] == "NONE"
    assert res["orders_created"] == st["orders_created"] == 0
    assert {p.name for p in tmp_path.iterdir()} <= {"predictions.jsonl", "model_instances.jsonl", "reference",
                                                     "recommendations", "forward_plans", "forward_looks",
                                                     f"status_{h}d.json", f"status_history_{h}d.jsonl",
                                                     "report_card.json", "report_card.md", "look_reports",
                                                     "variants.jsonl", "variants"}
    src = (ROOT / "modules" / "option_learner.py").read_text(encoding="utf-8")
    for forbidden in ("send_order", "place_order", "champion", "risk_limits", "position_size"):
        assert forbidden not in src


def test_b2_b3_segments_adverse_and_director_proposals_need_approval():
    picks = []
    for i in range(30):
        picks.append({"date": f"2026-01-{(i % 10) + 1:02d}", "y": -0.2, "sample": "directional",
                      "direction": "bullish", "expression": "spread", "gate": "g", "sector": "Energy",
                      "regime": "vol_high|risk_low", "y_adverse": -0.25})
        picks.append({"date": f"2026-01-{(i % 10) + 1:02d}", "y": 0.3, "sample": "directional",
                      "direction": "bullish", "expression": "single", "gate": "g", "sector": "Technology",
                      "regime": "vol_low|risk_high", "y_adverse": 0.25})
    seg = ol.segments(picks)
    weak = {(w["dimension"], w["level"]) for w in seg["weak"]}
    assert {("expression", "spread"), ("sector", "Energy"), ("regime", "vol_high|risk_low")} <= weak
    assert ("gate", "g") not in weak                                       # nur eine Ausprägung: kein Vergleich
    status = {5: {"verified_labels": 50, "walk_forward": {"segments": seg}, "forward": {}, "model": {},
                  "learning_status": {"state": "NEED_MORE_DATA"}}}
    gaps = rd.option_gaps(status, [{"horizon": 5}])
    seg_gaps = [g for g in gaps if g["gap_id"].split(":")[1] == "weak_segment"]
    assert len(seg_gaps) == len(seg["weak"])
    props = rd.option_proposals(gaps, {"approved_methods": {}})
    review = [p for p in props if p["kind"] == "segment_review"]
    assert review and all(p["approval_required"] and not p["auto_experiment"] for p in review)
    assert ol.regime_of("2026-01-05", Path("/nonexistent/world_state.jsonl")) == "unknown"


def test_regime_is_point_in_time(tmp_path):
    ws = tmp_path / "w.jsonl"
    ws.write_text("\n".join(json.dumps(x) for x in (
        {"date": "2026-01-02", "volatility_state": "low", "risk_appetite_state": "high"},
        {"date": "2026-01-09", "volatility_state": "high", "risk_appetite_state": "low"})))
    assert ol.regime_of("2026-01-09", ws) == "vol_low|risk_high"               # nie der Zustand desselben Tages
    assert ol.regime_of("2026-01-10", ws) == "vol_high|risk_low"
    assert ol.regime_of("2026-01-01", ws) == "unknown"


# ── C5: neue Zusammenhänge ───────────────────────────────────────────────────

def test_c5_1_planted_leading_macro_relation_is_detected(env):
    sig = "exp_energy * sign(spy_mom_63 - 0)"
    scan = hf.scan_anomalies(_synthetic_panel(effect=sig, size=0.03, seed=7))
    found = {a["signal"]: a for a in scan["survivors"]}
    assert sig in found and found[sig]["type"] == "macro_reaction" and found[sig]["direction"] == 1


def test_c5_2_and_7_meaningless_relations_rejected_under_full_multiple_testing(env):
    clean = hf.scan_anomalies(_synthetic_panel(effect=None, seed=5))
    assert clean["survivors"] == [] and clean["n_significant"] == 0
    assert clean["n_scanned"] == sum(clean["by_type"].values()) >= 50       # jeder gescannte Ausdruck zählt im BH
    # BH begrenzt die Falschentdeckungsrate (nicht jede Zufallsauffälligkeit): eine solche wird danach von der
    # unveränderten Lab-Prüfung auf den späteren Jahren verworfen – das Auswahlfenster ist nie Evidenz.
    noise = _synthetic_panel(effect=None, seed=21)
    lucky = hf.scan_anomalies(noise)["survivors"]
    assert lucky, "Testvoraussetzung: Seed 21 erzeugt eine Zufallsauffälligkeit im Auswahlfenster"
    for a in lucky:
        rec = rl.evaluate_hypothesis(rl.add_exposures(noise), {"id": "X", "signal": a["signal"],
                                                               "direction": a["direction"]}, {}, env["proto"])
        assert rec["status"] == "rejected" and rec["reasons"]
    many = hf.scan_anomalies(_synthetic_panel(effect="rank(ted_awards_z) + rank(mom_3m) + rank(rev_1m)", size=0.1))
    assert len(many["survivors"]) <= hf.MAX_ANOMALY_IDEAS < many["n_significant"]


def test_c5_3_known_signal_is_not_reconfirmed_under_a_new_name(env, tmp_path):
    keys = hf.known_signal_keys([{"signal": SIBLING}], [])
    assert rl.signal_key(SIBLING) in keys and rl.signal_key(SIBLING) not in {
        rl.signal_key(a["signal"]) for a in hf.scan_anomalies(env["panel"], keys)["survivors"]}
    hyps = [{"id": "FAC-A", "signal": PLANTED, "direction": 1, "source": "factory", "family": "a", "spec_hash": "a"},
            {"id": "FAC-RENAMED", "signal": f"({PLANTED}) * 2", "direction": 1, "source": "factory",
             "family": "b", "spec_hash": "b", "title": "neuer Name"}]
    (tmp_path / "h.yaml").write_text(yaml.safe_dump({"hypotheses": hyps}))
    out = rl.run(env["panel"], protocol=env["proto"], hyp_path=tmp_path / "h.yaml", db_path=tmp_path / "db.json",
                 with_discovery=False)["hypotheses"]
    assert out["FAC-RENAMED"]["status"] == "duplicate" and out["FAC-RENAMED"]["duplicate_of"] == "FAC-A"


def test_c5_4_effect_explained_by_known_factor_gets_no_incremental_credit(env):
    p = _synthetic_panel(effect="rank(mom_3m)", size=0.06, seed=3)
    proto = hf.load_protocol()
    proto["robustness"] = {**proto["robustness"], "placebo_permutations": 10}
    bt = hf.battery(p, {"signal": "rank(mom_3m) * step(spy_trend_200 + 1000)", "direction": 1}, proto)
    assert not bt["robust"] and any("inkrementell" in r for r in bt["reasons"])
    assert bt["tests"]["explained_by"]["groups"]["momentum"]["explains"] is True
    assert bt["tests"]["ablation_controls"]["sector_dummies"] == 2               # Sektor zusätzlich kontrolliert


def test_c5_5_missing_point_in_time_data_is_data_gap(env):
    scan = hf.scan_anomalies(env["panel"].drop(columns=["ted_awards_z"]))
    assert "ted_awards_z" in scan["data_gaps"]["divergence"]
    proto = hf.load_protocol()
    r = hf.readiness({"signal": "rank(sec_insider_net_value_90d)", "domain_kind": "alt_feature"},
                     env["panel"], proto)
    assert r["status"] == "DATA_GAP"
    assert hf.scan_anomalies(None)["status"] == "NO_PANEL"


def test_c5_6_hypotheses_are_versioned_and_traceable(lab_run):
    h = _anomalies(lab_run["plan"])[0]
    lab = next(x for x in hf.lab_hypotheses(lab_run["plan"]) if x["id"] == h["id"])
    assert lab["idea_source"] == "anomaly" and lab["anomaly"]["version"] == hf.ANOMALY_VERSION
    assert lab["spec_hash"] == h["spec_hash"] and h["id"] == hf._hid("anomaly", h["signal"], h["direction"])
    assert lab_run["plan"]["anomaly_scan"]["survivor_ids"] == [h["id"]]


def _factory_root(root: Path, challengers: list[dict], rows: list[dict]):
    for path, items in ((root / al.FACTORY_CHALLENGERS, challengers), (root / al.FACTORY_FORWARD, rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x) + "\n" for x in items))


def test_c5_8_prospective_validation_only_after_registration(tmp_path):
    from collections import Counter
    tickers = [f"T{i:02d}" for i in range(12)]

    def row(d, rec, sh="abc"):
        return {"hypothesis_id": "FAC-1", "spec_hash": sh, "date": d, "top_decile": tickers,
                "n_cross_section": 120, "recorded_at": rec}

    def batches(ch, rows, now):
        _factory_root(tmp_path, [ch], rows)
        stats = Counter()
        return al._factory_batches(tmp_path, {"top_k": 3}, date(2026, 11, 2), timedelta(days=4), now, stats), stats
    ch = {"hypothesis_id": "FAC-1", "spec_hash": "abc", "registered_at": "2026-11-13T08:00:00+00:00",
          "forward_start": "2026-11-16", "signal": "x", "direction": 1}
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    out, stats = batches(ch, [row("2026-11-20", "2026-11-21T09:00:00+00:00"),            # gültig
                              row("2026-11-13", "2026-11-18T09:00:00+00:00"),            # vor forward_start
                              row("2026-11-19", "2026-11-20T09:00:00+00:00", sh="zzz"),  # andere Spezifikation
                              row("2026-11-16", "2026-11-21T10:00:00+00:00"),            # Nachtrag (5 Tage alt)
                              row("2026-11-20", "2026-11-14T09:00:00+00:00")], now)      # zu alt erfasst
    assert [b["pit_data_date"] for b in out] == ["2026-11-20"]
    assert stats == Counter(fac_pre_forward_start=1, fac_spec_hash_mismatch=1, fac_backfilled_stale=1,
                            fac_late_or_future=1)
    b = out[0]
    assert len(b["top"]) == 3 and {t for t, _ in b["top"]} <= set(tickers) and b["bottom"] == []
    assert batches(ch, [row("2026-11-20", "2026-11-21T09:00:00+00:00")], now)[0][0]["top"] == b["top"]
    assert b["hypothesis_source"] == "factory_challenger" and b["signal_time"] > ch["registered_at"]
    early = {**ch, "registered_at": "2026-11-20T08:00:00+00:00"}                      # Prognose vor Registrierung
    out, stats = batches(early, [row("2026-11-19", "2026-11-20T07:00:00+00:00")], now)
    assert out == [] and stats == Counter(fac_pre_registration=1)


def test_c5_9_and_config_no_production_influence():
    cfg = al.config(ROOT)
    assert cfg["sources"]["factory_challenger"] == {"enabled": True, "top_k": 3, "bottom_k": 0}
    assert cfg["production_influence"] == "NONE" and cfg["allow_llm"] is False
    for mod in ("hypothesis_factory.py", "research_lab.py"):
        src = (ROOT / "modules" / mod).read_text(encoding="utf-8")
        for forbidden in ("send_order", "promotion_hypotheses.yaml", "model_registry.yaml\"", "challengers.yaml"):
            assert f'"{forbidden}' not in src
    tree = ast.parse((ROOT / "modules" / "hypothesis_factory.py").read_text(encoding="utf-8"))
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m and ("broker" in m or "execution" in m or "pipeline" == m) for m in imports)


# ── End-to-End: Anomalie -> Lab -> Challenger -> Optionsentscheidung -> verifiziertes Ergebnis ──

def test_e2e_anomaly_to_quote_verified_option_outcome(env, lab_run, tmp_path, monkeypatch):
    from modules.alt_data import contracts as ac
    from tests.test_alpha_option_link import (
        DECIDE,
        NOW_REG,
        _exit_quote,
        _setup,
        install_market,
    )
    root = tmp_path / "repo"
    _setup(root)
    for name, path in (("CHALLENGERS", al.FACTORY_CHALLENGERS), ("FORWARD_LEDGER", al.FACTORY_FORWARD),
                       ("RESULTS_OUT", Path("outputs/research/factory_results.json"))):
        monkeypatch.setattr(hf, name, root / path)
    proto = hf.load_protocol()
    proto["robustness"] = {**proto["robustness"], "placebo_permutations": 20}
    h = _anomalies(lab_run["plan"])[0]
    res = hf.evaluate(env["panel"], proto, NOW, db_path=lab_run["tmp"] / "db.json",
                      hyp_path=lab_run["tmp"] / "h.json")["results"]
    assert res[h["id"]]["status"] == "PROSPECTIVE_CHALLENGER", res[h["id"]]["reasons"]
    ch = next(json.loads(x) for x in hf.CHALLENGERS.read_text().splitlines() if h["id"] in x)
    assert ch["forward_start"] == "2026-11-02" and ch["registered_at"] == NOW.isoformat(timespec="seconds")
    # prospektiver Stichtag nach der Registrierung (neue Querschnittsdaten), Erfassung am Folgetag
    rnd = np.random.default_rng(1)
    fresh = pd.DataFrame({"date": pd.Timestamp("2026-11-06"), "ticker": [f"T{i:02d}" for i in range(60)],
                          "ted_awards_z": rnd.normal(0, 1, 60), "rev_1m": rnd.normal(0, 0.05, 60),
                          "mom_3m": rnd.normal(0, 0.1, 60),
                          "sector": np.where(np.arange(60) % 3 == 0, "Technology", "Energy")})

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 7, 14, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(ac, "datetime", _Clock)
    assert hf.record_forward(pd.concat([env["panel"], fresh], ignore_index=True)) >= 1
    rows = [json.loads(x) for x in hf.FORWARD_LEDGER.read_text().splitlines()]
    assert {r["date"] for r in rows} == {"2026-11-06"}                          # nichts vor forward_start
    # Optionsentscheidung: eingefroren nach der Registrierung, vor jedem Ergebnis
    reg = al.register(root, panel=None, now=NOW_REG)
    allsig = [s for s in al._read_kind(root, "signals")[0] if s["hypothesis_source"] == "factory_challenger"]
    sigs = [s for s in allsig if s["hypothesis_id"] == h["id"]]
    assert reg["n_new_signals"] == len(allsig) and len(sigs) == 3                 # je Challenger top_k = 3
    assert all(s["registered_at"] > s["signal_time"] > ch["registered_at"] and not s["order_placed"] for s in sigs)
    install_market(monkeypatch)
    al.decide(root, now=DECIDE)
    ds = [d for d in al._read_kind(root, "decisions")[0] if d["hypothesis_id"] == h["id"]]
    assert ds and all(d["decided_at"] >= d["signal_registered_at"] and d["order_placed"] is False for d in ds)
    d = next(x for x in ds if x["structure"] == "LONG_CALL" and x["decision"] == "TRADE")
    at = datetime(2026, 12, 8, 16, 0, tzinfo=timezone.utc)
    q = _exit_quote(root, d["legs"][0]["occ_symbol"], at, 1.0, 1.2)            # Verlust: realistisch negativ
    al.settle(root, now=datetime(2026, 12, 12, 15, 0, tzinfo=timezone.utc))
    o = next(x for x in al._read_kind(root, "outcomes")[0] if x["decision_id"] == d["decision_id"])
    assert o["status"] == "EXIT_OBSERVED" and o["evidence_class"] == "QUOTE_VERIFIED_PAPER"
    assert o["exit_legs"][0]["quote_record_id"] == q["record_id"]
    sig = next(s for s in sigs if s["signal_id"] == d["signal_id"])
    assert sig["hypothesis_id"] == h["id"] and sig["hypothesis_version"] == ch["spec_hash"]
    # wirtschaftliche Bewertung + prospektive Validierung derselben Kette: nur Research, ohne Mindestdaten kein Urteil
    import shutil

    from modules.autonomous_alpha import option_economics as oe
    from modules.autonomous_alpha import option_validation as ov
    shutil.copy(ROOT / ov.CONFIG, root / ov.CONFIG)
    later = datetime(2026, 12, 14, 15, 0, tzinfo=timezone.utc)
    econ = oe.evaluate(root, now=later, persist=False)
    mine = econ["hypotheses"][h["id"]]
    assert mine["LONG_CALL"]["coverage"]["settled_verified"] >= 1 and mine["LONG_CALL"]["comparison_group"] == "TREATMENT"
    assert mine["BULL_CALL_SPREAD"]["coverage"]["settled_verified"] == 0         # ohne Exit-Quotes kein Ergebnis
    assert econ["order_placed"] is False and econ["production_influence"] == "NONE"
    rep = ov.run(root, econ=econ, now=later, persist=True)
    assert rep["order_placed"] is False and rep["production_influence"] == "NONE" and rep["llm_calls"] == 0
    assert rep["status"] == "NEED_MORE_DATA" and rep["plans"] == []              # ein Einzelfall begründet nichts


def test_e2e_research_memory_feeds_the_next_research_decision(env, tmp_path):
    """Ergebnis -> Research Memory -> Meta-Learning -> nächste Planung: getestete Idee wird nicht wiederholt,
    die Richtung (Ideentyp) wird nach Misserfolgen nachrangig – nie als Alpha-Nachweis."""
    first = _plan(env["panel"])
    h = _anomalies(first)[0]
    results = {"results": {h["id"]: {"status": "REJECTED", "spec": {k: h.get(k) for k in (
        "signal", "direction", "family", "domain", "idea_source", "anomaly_type")}, "reasons": ["BH"],
        "data_kind": "historical_walk_forward"}}}
    (tmp_path / "fr.json").write_text(json.dumps(results))
    mem = rm.collect({"factory_results": tmp_path / "fr.json"})
    mem += [rm._entry("factory", f"FAC-X{i}", "REJECTED", {"signal": f"rank(vol_20) * {i}", "family": f"x{i}",
                                                          "idea_source": "anomaly"}, source="factory",
                      evidence=None) for i in range(4)]
    dirs = rm.directions(mem)
    assert dirs["idea_source:anomaly"]["tested"] == 5 and dirs["idea_source:anomaly"]["success"] == 0
    nxt = _plan(env["panel"], memory=mem)
    assert h["id"] not in nxt["selected"] and h["signal"] not in [x["signal"] for x in _anomalies(nxt)]


# ── Workflows: kritische Forschungsschritte dürfen nicht formal grün enden ───

def test_workflows_mark_prospective_research_steps_critical():
    ml_wf = yaml.safe_load((ROOT / ".github/workflows/ml_research.yml").read_text())
    steps = ml_wf["jobs"]["ml"]["steps"]
    ids = [s.get("id") for s in steps]
    assert ids.index("factory_forward") < ids.index("alpha_option_link")
    fwd = steps[ids.index("factory_forward")]
    assert fwd["if"] == "env.MODE == 'weekly'" and "hypothesis_factory forward" in fwd["run"]
    check = steps[-1]["env"]
    for name in ("factory_forward", "alpha_option_link", "option_learner"):
        assert f"{name}=${{{{ steps." in check["CRITICAL"] and f" {name}=" not in " " + check["DEGRADED"]
    scan = yaml.safe_load((ROOT / ".github/workflows/scanner.yml").read_text())["jobs"]["scan"]["steps"]
    assert "option_learner=${{ steps.option_learner.outcome }}" in scan[-1]["env"]["CRITICAL"]
    assert scan[-1]["if"] == "always()" and "ci_stage_check" in scan[-1]["run"]


def test_factory_forward_cli_needs_panel(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_PANEL_CACHE", raising=False)
    assert hf.main(["forward"]) == 1


def test_anomaly_scan_failure_is_visible_but_keeps_other_ideas(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("defekt")
    monkeypatch.setattr(hf, "scan_anomalies", boom)
    res = _plan(env["panel"])
    assert res["anomaly_scan"]["status"] == "ERROR" and "defekt" in res["anomaly_scan"]["error"]
    assert res["selected"] and not any(h["idea_source"] == "anomaly" for h in res["ideas"])
