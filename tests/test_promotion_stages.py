"""Promotion-Stufen (Master-Auftrag 2026-10-10): Sizing (nur Verkleinerung), globaler Kill-Switch, Meta-Label-Signal.
Rein technische Tests mit synthetischen Ledgern; sie belegen die Mechanik, kein wirtschaftliches Alpha."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modules import hypothesis_contract as hc
from modules import meta_label as ml
from modules import position_sizing as ps
from modules import production_intelligence_adapter as pia
from modules import promotion_controller as pc
from tests.test_promotion import POLICY, ROOT, T0, Env, _active, contract, synth

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "_safe_mode_state", lambda: {"active": False, "reasons": []})
    monkeypatch.setattr(pia, "KILL_SWITCH", tmp_path / "kill_switch_missing.yaml")
    monkeypatch.setattr(ml, "ROOT", tmp_path / "no_meta_model")


def sizing_contract(**kw) -> dict:
    return contract(hypothesis_id="TEST-SIZE-1", production_class="sizing", maximum_initial_influence="SIZE_DOWN_ONLY",
                    size_multiplier=0.5, title="Test-Sizing", **kw)


# ── Vertrag ─────────────────────────────────────────────────────────────────
def test_sizing_contract_only_reduces():
    assert hc.validate(sizing_contract(), POLICY) == []
    for bad in (1.0, 1.5, 0.1, None, "0.5"):
        c = sizing_contract()
        c["size_multiplier"] = bad
        assert any("size_multiplier" in e for e in hc.validate(c, POLICY)), bad
    too_high = sizing_contract()
    too_high["maximum_initial_influence"] = "RERANK_ONLY"
    assert any("Klasse" in e for e in hc.validate(too_high, POLICY))
    order = hc.INFLUENCE_LEVELS                                        # liegt über der automatischen Obergrenze
    assert order.index("ABSTENTION_ONLY") < order.index("SIZE_DOWN_ONLY") < order.index("RERANK_ONLY")


def test_meta_label_contract_registered_as_abstention():
    cs = {hc.key(c): c for c in hc.load(ROOT / "config" / "promotion_hypotheses.yaml")}
    c = cs["PROM-META-001@v1"]
    assert hc.validate(c, POLICY) == []
    assert c["production_class"] == "abstention" and c["maximum_initial_influence"] == "ABSTENTION_ONLY"
    assert c["signal_definition"] == "meta_p_take" and c.get("eligible_stage") is None


# ── Controller: Sizing nie automatisch, nur per menschlicher Freigabe ──────
def test_sizing_forward_validated_but_only_human_approval_applies(tmp_path):
    e = Env(tmp_path)
    c = sizing_contract()
    synth(e, c, T0, 40)
    s1 = e.run([c], datetime(2027, 2, 1, tzinfo=UTC))["hypotheses"][hc.key(c)]
    assert s1["state"] == "FORWARD_VALIDATED" and s1["influence_level"] == "NONE"     # Policy: automatisch nur Abstinenz
    assert s1["recommendation"] == "SIZE_DOWN_ONLY"
    assert e.verified([c])[0][hc.key(c)]["level"] == "NONE"
    appr = {hc.key(c): {"approved_level": "WEIGHT_25", "pr": "test"}}                 # wird auf SIZE_DOWN_ONLY gekappt
    s2 = e.run([c], datetime(2027, 3, 5, tzinfo=UTC), approvals=appr)["hypotheses"][hc.key(c)]
    assert s2["decision"] == "ALLOW_SIZE_DOWN" and s2["influence_level"] == "SIZE_DOWN_ONLY"
    assert s2["state"] == "LIMITED_PRODUCTION"
    active, problems = e.verified([c])
    assert not problems and active[hc.key(c)]["level"] == "SIZE_DOWN_ONLY"
    s3 = e.run([c], datetime(2027, 4, 5, tzinfo=UTC), approvals=appr)["hypotheses"][hc.key(c)]
    assert s3["influence_level"] == "SIZE_DOWN_ONLY" and s3["decision"] == "ALLOW_SIZE_DOWN"   # nie höher
    # Adapter: getroffener Trade wird halbiert, nicht blockiert; anderer Trade unverändert
    props = [{"ticker": "AAA", "features": {"risk_flag": 1}, "trade_score": {"total": 70}, "sector": "Tech"},
             {"ticker": "BBB", "features": {"risk_flag": 0}, "trade_score": {"total": 60}, "sector": "Tech"}]
    kept, blocked, recs = pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0}, state_path=e.state,
                                                 contracts=[c], registry=e.reg, transitions=e.tr,
                                                 ledger_dir=tmp_path / "live", policy=POLICY)
    assert not blocked and [p["ticker"] for p in kept] == ["AAA", "BBB"]
    assert kept[0]["intelligence_size_multiplier"] == 0.5 and "intelligence_size_multiplier" not in kept[1]
    assert all(r["final_production_decision"] == "TRADE" for r in recs) and recs[0]["size_multiplier"] == 0.5


def test_sizing_decision_bounded_and_acts_in_safe_mode():
    c = sizing_contract()
    d = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "SIZE_DOWN_ONLY"), safe_mode=True, sector="Tech",
                             regime="vix_low", policy=POLICY)
    assert d["size_multiplier"] == 0.5 and d["final_production_decision"] == "TRADE"     # verkleinert, blockiert nie
    shadow = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "NONE", "FORWARD_VALIDATED"), safe_mode=False,
                                  sector="Tech", regime="vix_low", policy=POLICY)
    assert shadow["size_multiplier"] == 1.0 and shadow["size_applied_by"] == []        # ohne Stufe keine Wirkung
    tiny = {**c, "size_multiplier": 0.01}                                             # an Validierung vorbei
    assert pia.decide_for_trade({}, {"risk_flag": 1}, _active(tiny, "SIZE_DOWN_ONLY"), safe_mode=False,
                                sector="Tech", regime="vix_low", policy=POLICY)["size_multiplier"] == hc.MIN_SIZE_MULTIPLIER
    up = {**c, "size_multiplier": 3.0}
    assert pia.decide_for_trade({}, {"risk_flag": 1}, _active(up, "SIZE_DOWN_ONLY"), safe_mode=False,
                                sector="Tech", regime="vix_low", policy=POLICY)["size_multiplier"] == 1.0
    missing = pia.decide_for_trade({}, {}, _active(c, "SIZE_DOWN_ONLY"), safe_mode=False, sector="Tech",
                                   regime="vix_low", policy=POLICY)
    assert missing["size_multiplier"] == 1.0                                         # fehlendes Merkmal: nie Wirkung


def test_apply_size_multiplier_never_increases(monkeypatch):
    monkeypatch.setattr(ps, "_portfolio_cfg", lambda: {"assumed_loss_pct": 1.0})
    base = {"position_pct": 0.02, "position_usd": 2000.0, "contracts": 8, "cost_per_contract": 250.0,
            "max_risk_usd": 2000.0}
    half = ps.apply_size_multiplier(base, 0.5, ["K"])
    assert half["contracts"] == 4 and half["position_usd"] == 1000.0 and half["max_risk_usd"] == 1000.0
    assert half["size_adjustment"] == {"multiplier": 0.5, "applied_by": ["K"], "original_contracts": 8,
                                       "original_position_usd": 2000.0}
    one = ps.apply_size_multiplier({**base, "contracts": 1, "position_usd": 300.0}, 0.5)
    assert one["contracts"] == 0 and "unter 1 Kontrakt" in one["note"]                 # Abrundung sichtbar
    for m in (1.0, 1.5, 0, -1, None, "x"):
        assert ps.apply_size_multiplier(base, m) == base
    assert base["contracts"] == 8                                                     # Original unverändert


# ── Kill-Switch ─────────────────────────────────────────────────────────────
def test_kill_switch_file_states(tmp_path):
    ks = tmp_path / "ks.yaml"
    assert pia.kill_switch_active(ks) is False                                        # fehlt -> aus
    ks.write_text("active: false\nreason: null\n")
    assert pia.kill_switch_active(ks) is False
    ks.write_text("active: true\nreason: Test\n")
    assert pia.kill_switch_active(ks) is True
    ks.write_text("active: [unclosed\n")
    assert pia.kill_switch_active(ks) is True                                         # defekt -> fail-safe aktiv
    repo = ROOT / "config" / "production_kill_switch.yaml"
    assert pia.kill_switch_active(repo) is False                                      # ausgelieferter Zustand
    co = (ROOT / ".github" / "CODEOWNERS").read_text()
    assert "config/production_kill_switch.yaml" in co


def test_kill_switch_returns_champion_but_keeps_logging(tmp_path, monkeypatch):
    ab = contract()
    sz = sizing_contract()
    act = {**_active(ab, "ABSTENTION_ONLY", "GUARDED_PRODUCTION"), **_active(sz, "SIZE_DOWN_ONLY")}
    monkeypatch.setattr(pia, "load_verified_state", lambda *a, **k: (act, []))

    def run(sub):
        props = [{"ticker": "AAA", "features": {"risk_flag": 1}, "trade_score": {"total": 70}, "sector": "Tech"},
                 {"ticker": "BBB", "features": {"risk_flag": 0}, "trade_score": {"total": 60}, "sector": "Tech"}]
        return pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0}, contracts=[ab, sz],
                                      ledger_dir=tmp_path / sub, policy=POLICY)

    kept, blocked, _ = run("on")
    assert [p["ticker"] for p, _ in blocked] == ["AAA"]                               # ohne Kill-Switch wirkt Abstinenz
    ks = tmp_path / "ks.yaml"
    ks.write_text("active: true\nreason: Test\n")
    monkeypatch.setattr(pia, "KILL_SWITCH", ks)
    kept, blocked, recs = run("off")
    assert not blocked and [p["ticker"] for p in kept] == ["AAA", "BBB"]
    assert all("intelligence_size_multiplier" not in p for p in kept)
    assert all(r["final_production_decision"] == "TRADE" for r in recs)
    assert recs[0]["intelligence_decision"] == "ABSTAIN"                               # weiter ausgewertet + geloggt
    assert recs[0]["intelligence"][hc.key(ab)]["fired"] is True


# ── Meta-Label-Signal im Adapter ────────────────────────────────────────────
def test_meta_p_take_is_none_without_model():
    env = pia.candidate_env({"ticker": "AAA", "features": {"impact": 7}, "deep_analysis": {"direction": "BULLISH"}},
                            {"safe_mode_active": 0})
    assert "meta_p_take" in env and env["meta_p_take"] is None                        # kein Modell -> unbekannt, nie 0
    assert env["meta_label_version"] is None


def test_meta_p_take_flows_into_decision_ledger_with_version(tmp_path, monkeypatch):
    from tests.test_learning_engine import _recs
    t = datetime(2025, 4, 1, tzinfo=UTC)
    ml.train(recs=_recs(), root=ml.ROOT, as_of=t.date(), now=t)
    c = pia.candidate_env({"ticker": "AAA", "features": {"impact": 9, "surprise": 5},
                           "deep_analysis": {"direction": "BULLISH"}}, {"safe_mode_active": 0})
    assert 0 <= c["meta_p_take"] <= 1 and c["meta_label_version"] == ml._registry(ml.ROOT)["active"]
    meta = contract(hypothesis_id="TEST-META", signal_definition="meta_p_take", features=["meta_p_take"],
                    thresholds={"op": "<", "value": 0.4})
    monkeypatch.setattr(pia, "load_verified_state", lambda *a, **k: (_active(meta, "NONE", "PROSPECTIVE_CHALLENGER"), []))
    props = [{"ticker": "AAA", "features": {"impact": 1, "surprise": 1}, "deep_analysis": {"direction": "BULLISH"},
              "trade_score": {"total": 70}, "sector": "Tech"}]
    kept, blocked, recs = pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0}, contracts=[meta],
                                                 ledger_dir=tmp_path / "live", policy=POLICY)
    assert kept and not blocked                                                      # SHADOW: nie Wirkung
    assert recs[0]["meta_label"]["version"] == c["meta_label_version"]
    assert recs[0]["intelligence"][hc.key(meta)]["evaluable"] is True
