"""Scientific Hypothesis Factory + Research Memory: gepinntes Protokoll, falsifizierbare
Hypothesen, DATA_GAP statt Simulation, Ähnlichkeitssperre, Varianten-Deckel, Budget mit
Explorationsanteil, LLM nur Mechanismus, Robustheits-Batterie, Prospective Challenger nur
nach Lab + Batterie, Meta-Learning über Richtungen, Reviewer nur prospektiv und unabhängig."""
from __future__ import annotations

import ast
import hashlib
import json
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from modules import hypothesis_factory as hf
from modules import research_memory as rm

ROOT = Path(__file__).resolve().parents[1]
PINNED = "d1d9d16412d7656687755130f612ba4019bd9eaab6ef57f79b53f63091e3ede2"
NOW = datetime(2026, 10, 3, 6, tzinfo=timezone.utc)


def test_protocol_pinned():
    h = hashlib.sha256((ROOT / "config" / "factory_protocol.yaml").read_bytes()).hexdigest()
    assert h == PINNED, "config/factory_protocol.yaml geändert – nur strenger zulässig (CODEOWNERS)"


def _plan(memory=None, panel=None, ms=None, conditions=None):
    """Isoliert: keine Repo-Ausgaben (z.B. outputs/research/source_conditions.json aus CI-Läufen)."""
    return hf.plan(panel, memory=memory or [], machine_state=ms, now=NOW,
                   conditions=conditions if conditions is not None else {"sources": {}})


def test_ideas_are_falsifiable_contracts():
    res = _plan(ms={"which_features_are_decaying": ["vol_20: Strukturbruch", "vix: x"]})
    req = ("population", "exposure", "signal", "direction", "lag", "horizon", "control_group", "primary_metric",
           "mechanism", "failure_condition", "H0", "H_alt", "spec_hash")
    srcs = {h["idea_source"] for h in res["ideas"]}
    assert srcs == {"cross_domain", "cross_source_divergence", "drift"}
    for h in res["ideas"]:
        assert all(h.get(k) not in (None, "") for k in req if not (k == "signal" and h["plan_status"] == "DATA_GAP")), h["id"]
        assert h["id"].startswith("FAC-") and "REJECTED" in h["H_alt"]
    drift = [h for h in res["ideas"] if h["idea_source"] == "drift"]
    assert [h["family"] for h in drift] == ["drift_vol_20"]                         # Makro-Merkmale ausgenommen
    assert len({h["id"] for h in res["ideas"]}) == len(res["ideas"])


def test_selected_signals_pass_the_lab_whitelist():
    from modules import research_lab as rl
    for h in _plan()["ideas"]:
        if h["plan_status"] != "DATA_GAP":
            rl.validate_expr(h["signal"])


def test_missing_domains_are_data_gaps_with_free_sources_never_simulated():
    res = _plan()
    gaps = {h["family"]: h for h in res["ideas"] if h["plan_status"] == "DATA_GAP"}
    assert {"weather_x_consumer", "river_x_chemicals", "power_x_industrials", "downloads_x_technology"} <= set(gaps)
    for h in gaps.values():
        assert h["signal"] is None and h["readiness"]["free_sources"] and h["priority"] == 0.0
    assert not any(h["id"] in res["selected"] for h in gaps.values())


def test_readiness_flags_mapping_and_thin_coverage():
    proto = hf.load_protocol()
    import pandas as pd
    h = {"signal": "exp_technology * sign(tnx - 3.0)", "domain_kind": "pit_panel"}
    r = hf.readiness(h, None, proto)
    assert r["status"] == "READY" and r["data_quality"] == 0.6 and "non_pit_mapping" in r["flags"][0]
    dates = pd.to_datetime(["2017-01-06"] * 10 + ["2018-01-05"] * 10)
    panel = pd.DataFrame({"date": dates, "tnx": [np.nan] * 18 + [3.0, 3.1], "sector": "Technology"})
    r2 = hf.readiness(h, panel, proto)
    assert r2["status"] == "DATA_GAP" and "Abdeckung" in r2["reason"]
    r3 = hf.readiness({"signal": "rank(sec_insider_net_value_90d)", "domain_kind": "alt_feature"}, panel, proto)
    assert r3["status"] == "DATA_GAP" and "nicht im Panel" in r3["reason"]


def test_budget_exploratory_share_and_cap():
    res = _plan()
    proto = hf.load_protocol()["budget"]
    sel = [h for h in res["ideas"] if h["plan_status"] == "SELECTED"]
    assert len(sel) == proto["tests_per_run"] == len(res["selected"])
    assert sum(h["exploratory"] for h in sel) >= res["budget"]["exploratory_slots"] == 2
    assert any(h["plan_status"] == "NOT_SELECTED_BUDGET" for h in res["ideas"])


def _mem(hid, status, spec, kind="research"):
    return rm._entry(kind, hid, status, spec, source="t", evidence=None)


def test_memory_blocks_similar_already_tested_and_exhausted_families():
    base = {h["family"]: h for h in _plan()["ideas"]}
    fin = base["rates_x_financials"]
    re_ = base["rates_x_realestate"]
    oil = base["oil_x_energy"]
    mem = [_mem(fin["id"], "REJECTED", {"signal": fin["signal"], "family": fin["family"], "direction": 1}),
           _mem("OTHER-1", "REJECTED", {"signal": re_["signal"], "direction": -1, "family": "manual"})]
    mem += [_mem(f"OIL-{i}", "REJECTED", {"signal": f"rank(wti_63d_chg) * {i}", "family": "oil_x_energy"}) for i in range(3)]
    res = {h["family"]: h for h in _plan(memory=mem)["ideas"]}
    assert res["rates_x_financials"]["plan_status"] == "ALREADY_TESTED"           # nie still wiederholt
    assert res["rates_x_realestate"]["plan_status"] == "SIMILAR_TO_TESTED"
    assert res["rates_x_realestate"]["similar_to"][0][0] == "research:OTHER-1"
    assert res["oil_x_energy"]["plan_status"] == "FAMILY_EXHAUSTED"
    retest = [_mem(fin["id"], "RETEST_LATER", {"signal": fin["signal"], "family": fin["family"]})]
    assert {h["family"]: h for h in _plan(memory=retest)["ideas"]}["rates_x_financials"]["plan_status"] in (
        "SELECTED", "NOT_SELECTED_BUDGET")


def test_priority_uses_direction_posterior_and_novelty():
    proto = hf.load_protocol()
    h = {"family": "f", "relevance": 1.0, "readiness": {"status": "READY", "data_quality": 1.0, "cost": 1.0}}
    p0 = hf.priority(h, {}, 0.0, proto)
    bad = rm.directions([_mem(f"X{i}", "REJECTED", {"family": "f", "signal": f"rank(vol_20)*{i}"}) for i in range(6)])
    good = rm.directions([_mem(f"X{i}", "PROSPECTIVE_CHALLENGER", {"family": "f", "signal": f"rank(vol_20)*{i}"})
                          for i in range(3)])
    assert hf.priority(h, bad, 0.0, proto) < p0 < hf.priority(h, good, 0.0, proto)
    assert hf.priority(h, {}, 0.5, proto) == pytest.approx(p0 * 0.5, rel=1e-3)
    assert hf.priority({**h, "readiness": {"status": "DATA_GAP"}}, {}, 0.0, proto) == 0.0


class _FakeClient:
    def __init__(self, text=None, stop="end_turn", exc=None):
        self.text, self.stop, self.exc, self.kwargs = text, stop, exc, None
        self.messages = self

    def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return types.SimpleNamespace(stop_reason=self.stop,
                                     content=[types.SimpleNamespace(type="text", text=self.text)])


def test_llm_only_mechanism_never_numbers_or_evidence():
    h = _plan()["ideas"][0]
    ok = _FakeClient(json.dumps({"mechanism": "Steilere Kurve erhöht die Zinsmarge der Banken."}))
    assert hf.llm_mechanism(h, ok) == "Steilere Kurve erhöht die Zinsmarge der Banken."
    assert ok.kwargs["model"] == hf.LLM_MODEL and "Zahlen" in ok.kwargs["system"]
    sent = json.loads(ok.kwargs["messages"][0]["content"])
    assert set(sent) == {"title", "population", "exposure", "signal", "direction", "horizon"}   # keine Ergebnisse
    assert hf.llm_mechanism(h, _FakeClient(json.dumps({"mechanism": "Studien zeigen 3 % Mehrrendite."}))) is None
    assert hf.llm_mechanism(h, _FakeClient("{}", stop="refusal")) is None
    assert hf.llm_mechanism(h, _FakeClient(exc=RuntimeError("offline"))) is None


@pytest.fixture(scope="module")
def synth():
    from modules import ml_research as ml
    from tests.test_ml_research import _panel
    old = ml.MIN_CROSS_SECTION
    ml.MIN_CROSS_SECTION = 20
    p = _panel(n_days=1700, n_stocks=60, signal=False)
    rnd = np.random.default_rng(3)
    p["sector"] = np.where(p["ticker"].str[1:].astype(int) % 3 == 0, "Technology", "Energy")
    med = float(p["spy_mom_63"].median())
    tech = (p["sector"] == "Technology").astype(float)
    p["fwd_xs_20"] = 0.03 * tech * np.sign(p["spy_mom_63"] - med) + rnd.normal(0, 0.02, len(p))
    p.loc[p["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    proto = hf.load_protocol()
    proto["robustness"] = {**proto["robustness"], "placebo_permutations": 20}
    yield p, med, proto
    ml.MIN_CROSS_SECTION = old


def test_battery_accepts_real_effect_and_rejects_noise_and_redundancy(synth):
    p, med, proto = synth
    real = hf.battery(p, {"signal": f"exp_technology * sign(spy_mom_63 - {med})", "direction": 1,
                          "exposure_sector": "Technology"}, proto)
    assert real["robust"], real["reasons"]
    assert real["tests"]["placebo"]["p"] <= 0.05 and real["tests"]["lag_profile"]["0"] > 0
    noise = hf.battery(p, {"signal": "rank(vol_20)", "direction": 1}, proto)
    assert not noise["robust"] and any("Placebo" in r for r in noise["reasons"])
    assert any("inkrementell" in r for r in noise["reasons"])                       # nur bestehendes Merkmal
    wrong = hf.battery(p, {"signal": f"exp_technology * sign(spy_mom_63 - {med})", "direction": -1,
                           "exposure_sector": "Technology"}, proto)
    assert not wrong["robust"]                                                      # kein Vorzeichen-Retten


def test_evaluate_registers_challenger_only_if_lab_accepted_and_robust(synth, tmp_path, monkeypatch):
    p, med, proto = synth
    for name in ("RESULTS_OUT", "CHALLENGERS", "FORWARD_LEDGER"):
        monkeypatch.setattr(hf, name, tmp_path / getattr(hf, name).name)
    sig = f"exp_technology * sign(spy_mom_63 - {med})"
    hyps = [{"id": "FAC-A", "signal": sig, "direction": 1, "exposure_sector": "Technology", "family": "x", "spec_hash": "h1"},
            {"id": "FAC-B", "signal": sig, "direction": 1, "exposure_sector": "Technology", "family": "x", "spec_hash": "h2"},
            {"id": "FAC-C", "signal": "rank(vol_20)", "direction": 1, "family": "y", "spec_hash": "h3"},
            {"id": "FAC-D", "signal": "rank(vol_60)", "direction": 1, "family": "z", "spec_hash": "h4"}]
    (tmp_path / "h.json").write_text(json.dumps({"hypotheses": hyps}))
    (tmp_path / "db.json").write_text(json.dumps({"hypotheses": {
        "FAC-A": {"canonical_status": "ACCEPTED"}, "FAC-B": {"canonical_status": "REJECTED", "reasons": ["BH"]},
        "FAC-C": {"canonical_status": "ACCEPTED"}}}))
    res = hf.evaluate(p, proto, NOW, db_path=tmp_path / "db.json", hyp_path=tmp_path / "h.json")["results"]
    assert res["FAC-A"]["status"] == "PROSPECTIVE_CHALLENGER"
    assert res["FAC-B"]["status"] == "REJECTED" and res["FAC-B"]["reasons"] == ["BH"]
    assert res["FAC-C"]["status"] == "NOT_ROBUST"
    assert res["FAC-D"]["status"] == "NOT_TESTED"
    cs = [json.loads(x) for x in hf.CHALLENGERS.read_text().splitlines()]
    assert [c["hypothesis_id"] for c in cs] == ["FAC-A"] and cs[0]["forward_start"] == "2026-10-05"
    hf.evaluate(p, proto, NOW, db_path=tmp_path / "db.json", hyp_path=tmp_path / "h.json")
    assert len(hf.CHALLENGERS.read_text().splitlines()) == 1                        # append-only, keine Dublette


def test_forward_ledger_only_after_forward_start(synth, tmp_path, monkeypatch):
    p, med, _ = synth
    monkeypatch.setattr(hf, "CHALLENGERS", tmp_path / "c.jsonl")
    monkeypatch.setattr(hf, "FORWARD_LEDGER", tmp_path / "f.jsonl")
    start = str(p["date"].sort_values().unique()[-3].date())
    hf.CHALLENGERS.write_text(json.dumps({"hypothesis_id": "FAC-A", "spec_hash": "h", "signal": "rank(mom_3m)",
                                          "direction": 1, "forward_start": start}) + "\n")
    n = hf.record_forward(p)
    rows = [json.loads(x) for x in hf.FORWARD_LEDGER.read_text().splitlines()]
    assert n == len(rows) == 3 and all(r["date"] >= start for r in rows)
    assert hf.record_forward(p) == 0


# ── Research Memory ─────────────────────────────────────────────────────────
def _sources(tmp_path):
    (tmp_path / "db.json").write_text(json.dumps({"hypotheses": {
        "H1": {"canonical_status": "REJECTED", "signal": "rank(mom_3m)", "direction": 1, "source": "config",
               "reasons": ["t<2"], "walk_forward": {"base": {"mean": 0.001, "t_months": 0.5}}},
        "FAC-1": {"canonical_status": "ACCEPTED", "signal": "exp_energy * sign(wti_63d_chg - 0)", "direction": 1,
                  "source": "factory", "family": "oil_x_energy"}}}))
    (tmp_path / "plan.json").write_text(json.dumps({"ideas": [
        {"id": "FAC-G", "family": "weather_x_consumer", "domain": "weather", "plan_status": "DATA_GAP",
         "readiness": {"reason": "fehlt", "free_sources": ["Open-Meteo"]}},
        {"id": "FAC-S", "family": "x", "plan_status": "SELECTED", "readiness": {}}]}))
    (tmp_path / "promo.json").write_text(json.dumps({"hypotheses": {
        "P1": {"state": "FORWARD_VALIDATED", "description": "VIX > 30 abstain", "sector_scope": ["Technology"],
               "regime_scope": ["all"], "evidence": {"n_observations": 40}},
        "P2": {"state": "PROSPECTIVE_CHALLENGER", "description": "x"}}}))
    return {"hypothesis_db": tmp_path / "db.json", "factory_plan": tmp_path / "plan.json",
            "promotion_state": tmp_path / "promo.json", "alt_validation": tmp_path / "none.json"}


def test_memory_sync_is_append_only_and_records_data_gaps(tmp_path):
    src, path = _sources(tmp_path), tmp_path / "mem.jsonl"
    assert rm.sync(path, src, "t0") == 5
    assert rm.sync(path, src, "t1") == 0                                            # unverändert -> nichts neu
    e = rm.latest(rm.load(path))
    assert e["factory:FAC-G"]["status"] == "DATA_GAP" and e["factory:FAC-G"]["free_sources"] == ["Open-Meteo"]
    assert "factory:FAC-S" not in e                                                 # nur DATA_GAP aus dem Plan
    assert e["promotion:P1"]["prospective"] and not e["research:H1"]["prospective"]
    assert e["promotion:P2"]["status"] == "PENDING_FORWARD"                        # registriert ≠ Erfolg
    assert rm.directions(rm.load(path))["promotion:champion_trades"]["tested"] == 1
    db = json.loads((tmp_path / "db.json").read_text())
    db["hypotheses"]["H1"]["canonical_status"] = "RETEST_LATER"
    (tmp_path / "db.json").write_text(json.dumps(db))
    assert rm.sync(path, src, "t2") == 1 and len(rm.load(path)) == 6                # Verlauf bleibt erhalten
    assert rm.latest(rm.load(path))["research:H1"]["status"] == "RETEST_LATER"


def test_similarity_and_search():
    a = {"signal": "exp_energy * sign(wti_63d_chg - 0)", "direction": 1, "family": "oil_x_energy"}
    assert rm.similarity(a, dict(a)) == 1.0
    assert rm.similarity(a, {"signal": "exp_energy*sign(wti_63d_chg-0)"}) == 0.95     # identisches Signal
    assert rm.similarity(a, {"signal": "rank(vix)"}) == 0.0
    mem = [_mem("X", "REJECTED", a), _mem("Y", "ACCEPTED", {"signal": "rank(vix)"})]
    assert [k for k, _, _ in rm.search(a, mem, 0.5)] == ["research:X"]


def test_directions_meta_learning():
    mem = [_mem(f"R{i}", "REJECTED", {"family": "a", "signal": f"rank(vol_20)*{i}"}) for i in range(4)]
    mem += [_mem("S1", "PROSPECTIVE_CHALLENGER", {"family": "b"}), _mem("S2", "FORWARD_VALIDATED", {"family": "b"}),
            _mem("G", "DATA_GAP", {"family": "c"})]
    d = rm.directions(mem)
    assert d["family:a"]["assessment"] == "verschwendet Kapazität" and d["family:a"]["wasted_capacity"] == 4
    assert d["family:b"]["assessment"] == "liefert Forward-Mehrwert" and d["family:b"]["prospective"] == 1
    assert d["family:c"]["tested"] == 0 and d["family:c"]["data_gap"] == 1
    assert d["family:b"]["posterior_success"] > d["family:a"]["posterior_success"]


def test_reviewer_sees_only_prospective_findings_in_scope(tmp_path):
    src, path = _sources(tmp_path), tmp_path / "mem.jsonl"
    rm.sync(path, src, "t0")
    mem = rm.load(path)
    obs = rm.review_candidate({"sector": "Technology", "regime": "RISK_ON"}, mem)
    assert [o["hypothesis"] for o in obs] == ["P1"]                                 # nicht P2, nicht Lab-ACCEPTED
    assert rm.review_candidate({"sector": "Energy", "regime": "RISK_ON"}, mem) == []
    assert set(obs[0]) == {"hypothesis", "status", "evidence", "applies_because"}  # Beobachtung, kein Score


def test_reviewer_runs_after_independent_analysis():
    for mod in ("modules/deep_analysis.py",):
        tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        mods = {getattr(n, "module", None) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert "research_memory" not in names and "modules.research_memory" not in mods | names
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    i_da, i_rev = src.index("analyses = _da.run("), src.index("_rm.review_candidate(")
    assert i_da < i_rev < src.index("Stufe 5: Mismatch-Score")
    block = src[i_rev - 900: i_rev + 300]
    assert "memory_review=" in block and "score" not in block.split("_rm.review_candidate(")[1].split("except")[0]


def test_factory_never_touches_production_paths():
    src = (ROOT / "modules" / "hypothesis_factory.py").read_text(encoding="utf-8")
    for forbidden in ("config.yaml", "promotion_hypotheses.yaml", "model_registry", "challengers.yaml", "send_order"):
        assert f'"{forbidden}' not in src and f"'{forbidden}" not in src
    tree = ast.parse(src)
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("write_text", "open")]
    assert writes                                                                  # schreibt nur eigene outputs/research
    assert all(str(p).startswith("outputs/research") for p in (hf.HYP_OUT, hf.PLAN_OUT, hf.RESULTS_OUT,
                                                                hf.CHALLENGERS, hf.FORWARD_LEDGER))


def test_lab_loads_factory_hypotheses_and_keeps_spec(tmp_path):
    from modules import research_lab as rl
    (tmp_path / "f.json").write_text(json.dumps({"hypotheses": [{"id": "FAC-1", "signal": "rank(vol_20)",
                                                                 "family": "fam", "spec_hash": "abc"}]}))
    hs = rl.load_factory(tmp_path / "f.json")
    assert hs[0]["source"] == "factory" and rl.load_factory(tmp_path / "missing.json") == []
    import pandas as pd
    p = rl.add_exposures(pd.DataFrame({"sector": ["Technology", "Energy", None]}))
    assert p["exp_technology"].tolist()[:2] == [1.0, 0.0] and np.isnan(p["exp_technology"].iloc[2])
