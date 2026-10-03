"""Research→Production-Brücke: unveränderliche Verträge, Zustandsmaschine, reine
Forward-Evidenz, Multiple Testing, Abstinenz/Rerank/Score/Gewicht mit harten Caps,
Safe-Mode-Vorrang, Demotion/Rollback, Decision-/Counterfactual-Ledger und
Failure-Injection (Auftrag „Adaptive Production“)."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modules import hypothesis_contract as hc
from modules import production_intelligence_adapter as pia
from modules import promotion_controller as pc


@pytest.fixture(autouse=True)
def _no_safe_mode(monkeypatch):
    """Tests steuern Safe Mode explizit; der echte Repo-Zustand darf sie nicht beeinflussen."""
    monkeypatch.setattr(pc, "_safe_mode_state", lambda: {"active": False, "reasons": []})

ROOT = Path(__file__).resolve().parent.parent
POLICY = hc.load_policy(ROOT / "config" / "promotion_policy.yaml")
PINNED_POLICY = "79d68a25766a6a1cda090d1b2d2ce4d08a995db02619fe0d14e0614d6d03c395"


def contract(**kw) -> dict:
    base = copy.deepcopy(hc.load(ROOT / "config" / "promotion_hypotheses.yaml")[0])
    base.update(hypothesis_id="TEST-ABST-1", signal_definition="risk_flag", features=["risk_flag"],
                thresholds={"op": ">=", "value": 1}, title="Test-Abstinenz")
    base.update(kw)
    return base


class Env:
    """Isolierte Pfade je Test."""
    def __init__(self, tmp: Path):
        self.reg, self.tr = tmp / "reg.jsonl", tmp / "tr.jsonl"
        self.ledger, self.out, self.looks = tmp / "ledger", tmp / "outcomes.jsonl", tmp / "looks.jsonl"
        self.state = tmp / "state.json"

    def run(self, contracts, now, policy=POLICY, approvals=None):
        return pc.run(contracts=contracts, policy=policy, now=now, registry=self.reg, transitions=self.tr,
                      ledger_dir=self.ledger, outcomes_path=self.out, looks_path=self.looks, state_path=self.state,
                      history={}, approvals=approvals or {})

    def verified(self, contracts):
        return pia.load_verified_state(self.state, contracts, self.reg, self.tr)


def synth(env: Env, c: dict, start: datetime, n_dates: int, *, fired_ret=-0.4, kept_ret=0.15, step_days=3,
          per_day=2, fired_every=4, regimes=("vix_low", "vix_high"), sectors=("Tech", "Energy", "Health", "Util"),
          seed=1, spec_hash=None, noise=0.05):
    """Decision-Ledger-Zeilen + Outcomes wie vom Adapter geschrieben (Regel-Auswertung eingefroren)."""
    rng = random.Random(seed)
    h = spec_hash or hc.spec_hash(c)
    k = hc.key(c)
    rows, outs, i = [], [], 0
    for d in range(n_dates):
        t = start + timedelta(days=d * step_days)
        for j in range(per_day):
            i += 1
            fired = i % fired_every == 0
            did = hashlib.sha256(f"{k}{seed}{t}{j}".encode()).hexdigest()[:16]
            rows.append({"decision_id": did, "timestamp": t.isoformat(), "date": t.date().isoformat(),
                         "ticker": f"T{i}", "sector": sectors[(i // fired_every) % len(sectors)],
                         "regime": regimes[(d * len(regimes)) // n_dates], "champion_decision": "TRADE",
                         "champion_probability": 0.55, "champion_rank": j + 1, "intelligence_rank": j + 1,
                         "intelligence_decision": "ABSTAIN" if fired else "TRADE",
                         "final_production_decision": "TRADE",
                         "intelligence": {k: {"spec_hash": h, "evaluable": True, "in_scope": True, "fired": fired,
                                              "applied": False}}})
            outs.append({"decision_id": did, "outcome": (fired_ret if fired else kept_ret) + rng.uniform(-noise, noise),
                         "outcome_method": "option_quote"})
    env.ledger.mkdir(parents=True, exist_ok=True)
    with open(env.ledger / f"{start:%Y-%m}-{seed}.jsonl", "a") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with open(env.out, "a") as fh:
        fh.writelines(json.dumps(o) + "\n" for o in outs)
    return rows


T0 = datetime(2026, 10, 6, 15, tzinfo=timezone.utc)


# ── 1. Vertrag ──────────────────────────────────────────────────────────────
def test_policy_pinned_and_codeowned():
    h = hashlib.sha256((ROOT / "config" / "promotion_policy.yaml").read_bytes()).hexdigest()
    assert h == PINNED_POLICY, "promotion_policy.yaml geändert – nur strenger zulässig (CODEOWNERS)"
    co = (ROOT / ".github" / "CODEOWNERS").read_text()
    for f in ("config/promotion_policy.yaml", "config/promotion_hypotheses.yaml", "config/promotion_approvals.yaml",
              "modules/promotion_controller.py", "modules/production_intelligence_adapter.py"):
        assert f in co
    assert POLICY["max_automatic_influence"] == "ABSTENTION_ONLY"          # Default laut Auftrag


def test_repo_contracts_valid_and_complete():
    cs = hc.load(ROOT / "config" / "promotion_hypotheses.yaml")
    assert len(cs) >= 5 and all(hc.validate(c, POLICY) == [] for c in cs)
    for c in cs:
        # Champion-Population: höchstens Abstinenz; andere Populationen (FINAL_MC_SURVIVOR): nie automatisch
        expected = "ABSTENTION_ONLY" if c.get("eligible_stage") is None else "NONE"
        # UNIVERSE_V2-Segmente: eigene Klasse, nie automatische Wirkung
        cls = "universe_segment" if c.get("universe_version") == "V2" else "abstention"
        assert c["production_class"] == cls and c["maximum_initial_influence"] == expected
        assert set(hc.REQUIRED) <= set(c)
        h0, halt = hc.hypotheses_pair(c)
        assert h0.startswith("H0") and "REJECTED" in halt


def test_immutable_contract_and_new_version(tmp_path):
    e = Env(tmp_path)
    c = contract()
    assert hc.register([c], e.reg, POLICY)[hc.key(c)]["status"] == "VALID"
    moved = {**c, "thresholds": {"op": ">=", "value": 2}}                   # Schwelle nachträglich ändern
    assert hc.register([moved], e.reg, POLICY)[hc.key(c)]["status"] == "INVALID_MODIFIED"
    flipped = {**c, "direction": 1}
    assert hc.register([flipped], e.reg, POLICY)[hc.key(c)]["status"] == "INVALID_MODIFIED"
    v2 = {**moved, "version": 2}
    assert hc.register([v2], e.reg, POLICY)[hc.key(v2)]["status"] == "VALID"   # neue Version ok
    assert hc.read_registry(e.reg)[1] == []


def test_contract_validation_rules():
    assert hc.validate(contract(signal_definition="__import__('os')"), POLICY)
    assert any("nicht deklarierte" in x for x in hc.validate(contract(signal_definition="risk_flag + other"), POLICY))
    assert any("Untergrenze" in x for x in hc.validate(contract(minimum_sample_size=10), POLICY))
    assert any("WEIGHT_10" in x or "Klasse" in x for x in hc.validate(contract(maximum_initial_influence="WEIGHT_25"), POLICY))
    assert any("forward_start" in x for x in hc.validate(contract(forward_start="2026-10-01T00:00:00Z"), POLICY))
    assert any("demotion" in x for x in hc.validate(contract(demotion_criteria={}), POLICY))
    w = contract(production_class="weight", maximum_initial_influence="WEIGHT_25")
    assert any("WEIGHT_10" in x for x in hc.validate(w, POLICY))


def test_registry_tamper_detected_fail_closed(tmp_path):
    e = Env(tmp_path)
    c = contract()
    hc.register([c], e.reg, POLICY)
    lines = e.reg.read_text().splitlines()
    ent = json.loads(lines[0])
    ent["contract"]["thresholds"]["value"] = 0                             # Registry/Hash manipuliert
    e.reg.write_text(json.dumps(ent) + "\n")
    _, problems = hc.read_registry(e.reg)
    assert problems
    assert hc.register([c], e.reg, POLICY)[hc.key(c)]["status"] == "REGISTRY_TAMPERED"


def test_similarity_blocks_recycling(tmp_path):
    e = Env(tmp_path)
    a = contract()
    hc.register([a], e.reg, POLICY)
    b = contract(hypothesis_id="TEST-ABST-2")                              # gleiche Regel, neue ID
    st = hc.register([b], e.reg, POLICY)[hc.key(b)]
    assert st["status"] == "DUPLICATE_SIMILAR" and st["similar_to"][0][0] == hc.key(a)
    b2 = contract(hypothesis_id="TEST-ABST-2", distinct_from={"ids": [hc.key(a)],
                                                              "justification": "anderer Wirkmechanismus"})
    assert hc.register([b2], e.reg, POLICY)[hc.key(b2)]["status"] == "VALID"
    other = contract(hypothesis_id="TEST-ABST-3", signal_definition="vix", features=["vix"],
                     thresholds={"op": ">", "value": 30})
    assert hc.register([other], e.reg, POLICY)[hc.key(other)]["status"] == "VALID"


# ── 2. Zustandsmaschine ─────────────────────────────────────────────────────
def test_state_machine_transitions_logged_and_guarded(tmp_path):
    p = tmp_path / "tr.jsonl"
    with pytest.raises(pc.TransitionError):
        pc.transition("X@v1", "GUARDED_PRODUCTION", reason="direkt", path=p)       # kein Sprung in Produktion
    pc.transition("X@v1", "HISTORICAL_RESEARCH", reason="r", path=p)
    pc.transition("X@v1", "HISTORICALLY_VALIDATED", reason="exzellenter Walk-Forward", path=p)
    with pytest.raises(pc.TransitionError):                                         # historisch -> max. HV
        pc.transition("X@v1", "FORWARD_VALIDATED", reason="backtest", path=p)
    pc.transition("X@v1", "PROSPECTIVE_CHALLENGER", reason="eingefroren", path=p)
    with pytest.raises(pc.TransitionError):
        pc.transition("X@v1", "FULL_PRODUCTION", reason="auto", path=p, decision="ALLOW_25_PERCENT_WEIGHT")
    entries, problems = pc.read_transitions(p)
    assert not problems and [x["new_state"] for x in entries] == ["HISTORICAL_RESEARCH", "HISTORICALLY_VALIDATED",
                                                                  "PROSPECTIVE_CHALLENGER"]
    for x in entries:
        assert {"previous_state", "new_state", "timestamp", "reason", "evidence_snapshot", "metrics", "code_version",
                "data_version"} <= set(x)
    raw = p.read_text().splitlines()
    tampered = json.loads(raw[1])
    tampered["new_state"] = "GUARDED_PRODUCTION"
    p.write_text("\n".join([raw[0], json.dumps(tampered), raw[2]]) + "\n")
    assert pc.read_transitions(p)[1]                                                # Kette erkennt Manipulation


# ── 3. Forward-Evidenz ──────────────────────────────────────────────────────
def test_future_only_and_hash_bound_observations(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, datetime(2026, 9, 1, tzinfo=timezone.utc), 10, seed=2)             # vor Registrierung (Historie)
    synth(e, c, T0, 10, seed=3)                                                     # nach forward_start
    synth(e, c, T0, 10, seed=4, spec_hash="deadbeef")                              # andere Spezifikation
    rows = pc.read_jsonl_dir(e.ledger)
    obs = pc.observations(c, hc.spec_hash(c), rows, pc.read_outcomes(e.out))
    assert len(obs) == 20 and all(o["ts"] >= T0 for o in obs)


def test_need_more_data_even_with_spectacular_returns(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 2, per_day=2, fired_every=2, fired_ret=-3.0, kept_ret=3.0)      # n=4, +300 %
    st = e.run([c], datetime(2026, 10, 20, tzinfo=timezone.utc))
    h = st["hypotheses"][hc.key(c)]
    assert h["state"] == "PROSPECTIVE_CHALLENGER" and h["influence_level"] == "NONE"
    assert h["decision"] == "KEEP_SHADOW" and "NEED_MORE_DATA" in h["reasons"][0]
    for need in ("n 4/60", "unabhängige Tage", "Kalenderspanne"):
        assert need in h["reasons"][0]


def test_single_outlier_explains_effect_no_promotion(tmp_path):
    e = Env(tmp_path)
    c = contract()
    rows = synth(e, c, T0, 40, fired_ret=0.10, kept_ret=0.10, noise=0.0)
    first_fired = next(r for r in rows if r["intelligence"][hc.key(c)]["fired"])
    with open(e.out) as fh:
        outs = [json.loads(x) for x in fh]
    for o in outs:
        if o["decision_id"] == first_fired["decision_id"]:
            o["outcome"] = -40.0                                                    # ein Trade trägt alles
    e.out.write_text("".join(json.dumps(o) + "\n" for o in outs))
    ev = pc.evidence(c, hc.spec_hash(c), pc.read_jsonl_dir(e.ledger), pc.read_outcomes(e.out), POLICY, 0.05)
    assert ev["delta_expectancy"] > 0 and ev["outlier_trims"]["top1"] <= 0
    ok, fails = pc.promotion_checks(c, ev, POLICY)
    assert not ok and any("ausreißer" in f for f in fails)


def test_single_sector_dominance_needs_scoped_hypothesis(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40, sectors=("Energy",))
    ev = pc.evidence(c, hc.spec_hash(c), pc.read_jsonl_dir(e.ledger), pc.read_outcomes(e.out), POLICY, 0.05)
    assert any("dominiert" in x for x in pc.insufficiency(c, ev, POLICY))
    scoped = contract(sector_scope=["Energy"])
    assert not any("dominiert" in x for x in pc.insufficiency(scoped, {**ev}, POLICY))


def test_single_regime_insufficient(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40, regimes=("vix_low",))
    ev = pc.evidence(c, hc.spec_hash(c), pc.read_jsonl_dir(e.ledger), pc.read_outcomes(e.out), POLICY, 0.05)
    assert any("Marktregime" in x for x in pc.insufficiency(c, ev, POLICY))


def test_multiple_testing_alpha_and_look_spending(tmp_path):
    e = Env(tmp_path)
    cs = [contract(hypothesis_id=f"F{i}", signal_definition=f"x{i}", features=[f"x{i}"]) for i in range(5)]
    hc.register(cs, e.reg, POLICY)
    reg, _ = hc.read_registry(e.reg)
    a1 = pc.family_alpha(cs[0], reg[:1], POLICY)["alpha_effective"]
    a5 = pc.family_alpha(cs[0], reg, POLICY)
    assert a5["family_size"] == 5 and a5["alpha_effective"] == pytest.approx(a1 / 5)
    assert a5["alpha_effective"] == pytest.approx(0.05 / (5 * 12))
    c = contract()
    synth(e, c, T0, 40)
    st1 = e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc))
    assert st1["hypotheses"][hc.key(c)]["looks_used"] == 1
    st2 = e.run([c], datetime(2027, 2, 5, tzinfo=timezone.utc))                    # < 28 Tage: kein neuer Look
    assert st2["hypotheses"][hc.key(c)]["looks_used"] == 1
    assert st1["multiple_testing"]["number_of_hypotheses_tested"] >= 6


def test_looks_exhausted_expires(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40, fired_ret=0.15, kept_ret=0.15)                              # kein Effekt
    pol = {**POLICY, "statistics": {**POLICY["statistics"], "planned_looks": 1}}
    st = e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc), policy=pol)
    assert st["hypotheses"][hc.key(c)]["state"] == "EXPIRED"


def test_h_alt_wins_reject_no_sign_flip(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40, fired_ret=0.6, kept_ret=0.0)                                # blockierte wären Gewinner
    st = e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc))
    h = st["hypotheses"][hc.key(c)]
    assert h["state"] == "REJECTED" and "H_alt" in h["reasons"][0] and h["influence_level"] == "NONE"
    assert hc.load(ROOT / "config" / "promotion_hypotheses.yaml")[0]["direction"] == -1   # nichts umgedreht


# ── 4. Ende-zu-Ende: Promotion -> Abstinenz -> Decay -> Demotion ────────────
def test_acceptance_promote_abstention_then_decay_demotes(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40)
    st = e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc))
    h = st["hypotheses"][hc.key(c)]
    assert h["decision"] == "ALLOW_ABSTENTION" and h["state"] == "GUARDED_PRODUCTION"
    assert h["influence_level"] == "ABSTENTION_ONLY"
    states = [x["new_state"] for x in pc.read_transitions(e.tr)[0]]
    assert states == ["PROSPECTIVE_CHALLENGER", "FORWARD_VALIDATED", "GUARDED_PRODUCTION"]
    assert st["notices"] and st["notices"][0]["proposed_level"] == "ABSTENTION_ONLY"
    active, problems = e.verified([c])
    assert not problems and active[hc.key(c)]["level"] == "ABSTENTION_ONLY"
    # Adapter wendet NUR Abstinenz an
    props = [{"ticker": "AAA", "features": {"risk_flag": 1}, "trade_score": {"total": 70}, "sector": "Tech"},
             {"ticker": "BBB", "features": {"risk_flag": 0}, "trade_score": {"total": 60}, "sector": "Tech"}]
    kept, blocked, recs = pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0}, state_path=e.state,
                                                 contracts=[c], registry=e.reg, transitions=e.tr,
                                                 ledger_dir=tmp_path / "live", policy=POLICY)
    assert [p["ticker"] for p in kept] == ["BBB"] and blocked[0][0]["ticker"] == "AAA"
    assert recs[0]["final_production_decision"] == "ABSTAIN" and recs[0]["champion_decision"] == "TRADE"
    # Alpha zerfällt nach Promotion: blockierte Trades wären jetzt Gewinner
    synth(e, c, datetime(2027, 2, 3, tzinfo=timezone.utc), 30, fired_ret=0.5, kept_ret=0.0, seed=9)
    st2 = e.run([c], datetime(2027, 5, 1, tzinfo=timezone.utc))
    h2 = st2["hypotheses"][hc.key(c)]
    assert h2["decision"] == "DEMOTE" and h2["state"] == "PROSPECTIVE_CHALLENGER" and h2["influence_level"] == "NONE"
    active2, _ = e.verified([c])
    assert active2[hc.key(c)]["level"] == "NONE"


def test_policy_max_auto_none_keeps_shadow(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40)
    st = e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc), policy={**POLICY, "max_automatic_influence": "NONE"})
    h = st["hypotheses"][hc.key(c)]
    assert h["state"] == "FORWARD_VALIDATED" and h["influence_level"] == "NONE" and h["recommendation"] == "ABSTENTION_ONLY"


def test_human_rollback(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40)
    e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc))
    st = e.run([c], datetime(2027, 2, 2, tzinfo=timezone.utc),
               approvals={hc.key(c): {"key": hc.key(c), "rollback": True, "reason": "manuell"}})
    h = st["hypotheses"][hc.key(c)]
    assert h["decision"] == "ROLLBACK" and h["state"] == "DEMOTED" and h["influence_level"] == "NONE"


def test_contract_changed_after_promotion_rolls_back(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40)
    e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc))
    hacked = {**c, "thresholds": {"op": ">=", "value": 0}}                          # eigene Schwelle ändern
    st = e.run([hacked], datetime(2027, 2, 2, tzinfo=timezone.utc))
    h = st["hypotheses"][hc.key(c)]
    assert h["registry_status"] == "INVALID_MODIFIED" and h["influence_level"] == "NONE"
    active, problems = e.verified([hacked])
    assert hc.key(c) not in active and problems


def test_state_file_manipulation_ignored(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 2)
    e.run([c], datetime(2026, 10, 20, tzinfo=timezone.utc))
    s = json.loads(e.state.read_text())
    s["hypotheses"][hc.key(c)]["influence_level"] = "WEIGHT_25"                     # State manipuliert
    e.state.write_text(json.dumps(s))
    active, problems = e.verified([c])
    assert active == {} and "manipuliert" in problems[0]
    s["state_hash"] = pc.state_digest(s)                                            # auch Hash nachgezogen
    e.state.write_text(json.dumps(s))
    active, problems = e.verified([c])
    assert hc.key(c) not in active and any("Transition-Kette" in p for p in problems)


# ── 5. Adapter: Caps, Safe Mode, Scoping, keine neuen Trades ────────────────
def _active(c, level, state="LIMITED_PRODUCTION"):
    return {hc.key(c): {"contract": c, "level": level, "state": state, "spec_hash": hc.spec_hash(c)}}


def test_score_adjustment_hard_cap_and_safe_mode():
    c = contract(production_class="score", score_points=8, direction=1, maximum_initial_influence="SCORE_LIMITED")
    d = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "SCORE_LIMITED"), safe_mode=False, sector="Tech",
                             regime="vix_low", policy=POLICY)
    assert d["score_adjustment_raw"] == 8 and d["score_adjustment"] == 3                # +8 -> +3
    d2 = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "SCORE_LIMITED"), safe_mode=True, sector="Tech",
                              regime="vix_low", policy=POLICY)
    assert d2["score_adjustment"] == 0                                                   # Safe Mode: kein Boost
    neg = contract(production_class="score", score_points=8, direction=-1, maximum_initial_influence="SCORE_LIMITED")
    d3 = pia.decide_for_trade({}, {"risk_flag": 1}, _active(neg, "SCORE_LIMITED"), safe_mode=True, sector="Tech",
                              regime="vix_low", policy=POLICY)
    assert d3["score_adjustment"] == -3


def test_meta_model_requests_100pct_weight_capped():
    c = contract(production_class="weight", signal_definition="p_meta", features=["p_meta"], direction=1,
                 maximum_initial_influence="WEIGHT_10")
    env = {"p_meta": 0.9, "champion_probability": 0.5}
    d = pia.decide_for_trade({}, env, _active(c, "WEIGHT_10"), safe_mode=False, sector="Tech", regime="vix_low",
                             requested_weights={hc.key(c): 1.0}, policy=POLICY)
    assert d["probability_adjustment"] == pytest.approx(0.10 * (0.9 - 0.5))           # w <= 0.10
    d25 = pia.decide_for_trade({}, env, _active(c, "WEIGHT_25"), safe_mode=False, sector="Tech", regime="vix_low",
                               requested_weights={hc.key(c): 1.0}, policy=POLICY)
    assert d25["probability_adjustment"] == pytest.approx(0.25 * 0.4)
    pia.HARD_CAPS  # Konstante im Code
    d_safe = pia.decide_for_trade({}, env, _active(c, "WEIGHT_25"), safe_mode=True, sector="Tech", regime="vix_low",
                                  requested_weights={hc.key(c): 1.0}, policy=POLICY)
    assert d_safe["probability_adjustment"] == 0.0                                      # Safe Mode: kein Gewicht


def test_policy_cannot_raise_hard_caps():
    loose = {**POLICY, "influence": {"score_adjustment_max_points": 50, "weight_level_1": 0.9, "weight_level_2": 1.0}}
    c = contract(production_class="score", score_points=40, direction=1, maximum_initial_influence="SCORE_LIMITED")
    d = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "SCORE_LIMITED"), safe_mode=False, sector="T",
                             regime="vix_low", policy=loose)
    assert d["score_adjustment"] == 3


def test_safe_mode_precedence_abstention_only_validated():
    c = contract()
    shadow = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "NONE", "PROSPECTIVE_CHALLENGER"), safe_mode=True,
                                  sector="Tech", regime="vix_low", policy=POLICY)
    assert shadow["intelligence_decision"] == "ABSTAIN" and shadow["final_production_decision"] == "TRADE"
    guarded = pia.decide_for_trade({}, {"risk_flag": 1}, _active(c, "ABSTENTION_ONLY", "GUARDED_PRODUCTION"),
                                   safe_mode=True, sector="Tech", regime="vix_low", policy=POLICY)
    assert guarded["final_production_decision"] == "ABSTAIN"                            # defensive validierte Regel


def test_missing_feature_never_fires():
    c = contract()
    d = pia.decide_for_trade({}, {}, _active(c, "ABSTENTION_ONLY", "GUARDED_PRODUCTION"), safe_mode=False,
                             sector="Tech", regime="vix_low", policy=POLICY)
    h = d["hypotheses"][hc.key(c)]
    assert not h["evaluable"] and not h["fired"] and d["final_production_decision"] == "TRADE"


def test_regime_and_sector_scoping():
    c = contract(regime_scope=["vix_high"], sector_scope=["Energy"])
    act = _active(c, "ABSTENTION_ONLY", "GUARDED_PRODUCTION")
    for sector, regime, blocked in (("Energy", "vix_high", True), ("Energy", "vix_low", False),
                                    ("Tech", "vix_high", False), (None, "vix_high", False), ("Energy", None, False)):
        d = pia.decide_for_trade({}, {"risk_flag": 1}, act, safe_mode=False, sector=sector, regime=regime, policy=POLICY)
        assert (d["final_production_decision"] == "ABSTAIN") is blocked, (sector, regime)


def test_rerank_only_reorders_never_adds(tmp_path, monkeypatch):
    c = contract(production_class="rerank", signal_definition="q", features=["q"], direction=1,
                 maximum_initial_influence="RERANK_ONLY")
    monkeypatch.setattr(pia, "load_verified_state", lambda *a, **k: (_active(c, "RERANK_ONLY"), []))
    props = [{"ticker": t, "features": {"q": q}, "trade_score": {"total": 60}} for t, q in (("A", 0.1), ("B", 0.9), ("C", 0.5))]
    kept, blocked, recs = pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0}, contracts=[c],
                                                 ledger_dir=tmp_path, policy=POLICY)
    assert [p["ticker"] for p in kept] == ["B", "C", "A"] and not blocked
    assert {p["ticker"] for p in kept} == {"A", "B", "C"}
    assert [r["champion_rank"] for r in recs] == [1, 2, 3] and recs[1]["intelligence_rank"] == 1
    props[2]["features"] = {}                                                        # fehlendes Signal
    kept2, _, _ = pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0}, contracts=[c],
                                         ledger_dir=tmp_path, policy=POLICY)
    assert [p["ticker"] for p in kept2] == ["A", "B", "C"]                           # Champion-Reihenfolge
    kept3, _, _ = pia.apply_to_proposals(props[:2], vix=14, context={"safe_mode_active": 1}, contracts=[c],
                                         ledger_dir=tmp_path, policy=POLICY)
    assert [p["ticker"] for p in kept3] == ["A", "B"]                                # Safe Mode: kein Rerank


def test_default_no_state_champion_unchanged_but_logged(tmp_path):
    e = Env(tmp_path)
    c = contract()
    hc.register([c], e.reg, POLICY)
    props = [{"ticker": "AAA", "features": {"risk_flag": 1}, "trade_score": {"total": 70},
              "simulation": {"hit_rate": 0.6}, "sector": "Tech"}]
    kept, blocked, recs = pia.apply_to_proposals(props, vix=20, context={"safe_mode_active": 1},
                                                 state_path=tmp_path / "none.json", contracts=[c], registry=e.reg,
                                                 transitions=e.tr, ledger_dir=e.ledger, policy=POLICY)
    assert kept == props and not blocked
    r = recs[0]
    assert r["intelligence_decision"] == "ABSTAIN" and r["final_production_decision"] == "TRADE"
    for f in ("champion_decision", "champion_score", "champion_probability", "intelligence_decision",
              "intelligence_adjustment", "final_production_decision", "active_hypotheses", "hypothesis_versions",
              "promotion_levels", "safe_mode_state", "meta_model_version", "world_model_version", "data_snapshot",
              "code_commit", "evidence_snapshot", "influence_level", "abstention_reason"):
        assert f in r
    on_disk = pc.read_jsonl_dir(e.ledger)
    assert on_disk[0]["decision_id"] == r["decision_id"]                             # Production-Ledger geschrieben


def test_counterfactual_outcome_of_blocked_trade_resolved(tmp_path):
    e = Env(tmp_path)
    row = {"decision_id": "d1", "timestamp": T0.isoformat(), "date": "2026-10-06", "ticker": "AAA",
           "champion_decision": "TRADE", "final_production_decision": "ABSTAIN", "intelligence": {}}
    row2 = {**row, "decision_id": "d2", "ticker": "BBB", "final_production_decision": "TRADE"}
    row3 = {**row, "decision_id": "d3", "ticker": "CCC", "final_production_decision": "TRADE"}
    e.ledger.mkdir()
    (e.ledger / "2026-10.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (row, row2, row3)))
    hist = {"counterfactual_closed": [{"ticker": "AAA", "entry_date": "2026-10-06", "outcome": -0.42,
                                       "close_date": "2026-11-20", "close_reason": "stop_loss",
                                       "outcome_method": "option_quote"}],
            "closed_trades": [{"ticker": "BBB", "entry_date": "2026-10-06", "outcome": 0.3,
                               "outcome_method": "spread_quote"},
                              {"ticker": "CCC", "entry_date": "2026-10-06", "outcome": 4.9,
                               "outcome_method": "delta_approx_clipped"}]}
    assert pc.resolve_outcomes(hist, e.ledger, e.out) == 3
    assert pc.resolve_outcomes(hist, e.ledger, e.out) == 0                          # append-only, einmalig
    o = pc.read_outcomes(e.out)
    assert o["d1"]["outcome"] == -0.42 and o["d1"]["source"] == "counterfactual_trade"
    assert o["d1"]["close_reason"] == "stop_loss" and o["d3"]["outcome_method"] == "delta_approx_clipped"
    arms = pc.evaluate_arms(pc.read_jsonl_dir(e.ledger), o)
    assert arms["n_excluded_approximate_outcomes"] == 1                              # Näherung zählt nicht
    assert arms["CHAMPION_ONLY"]["trade_count"] == 2 and arms["ADAPTIVE_ACTUAL"]["trade_count"] == 1
    assert arms["ADAPTIVE_ACTUAL"]["avoided_losers"] == 1 and arms["data_kind"] == "prospective_forward_only"


def test_approximate_outcomes_never_count_as_evidence(tmp_path):
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40)
    with open(e.out) as fh:
        outs = [json.loads(x) for x in fh]
    for o in outs:
        o["outcome_method"] = "delta_approx"
    e.out.write_text("".join(json.dumps(o) + "\n" for o in outs))
    ev = pc.evidence(c, hc.spec_hash(c), pc.read_jsonl_dir(e.ledger), pc.read_outcomes(e.out), POLICY, 0.05)
    assert ev["n_observations"] == 0


def test_counterfactual_trade_same_lifecycle_as_real_trade(monkeypatch):
    """Blockierter Trade: gleiche Exit-Regeln (TP/SL/Time), Outcome-Methode, keine Lern-Updates."""
    import feedback
    trade = {"ticker": "AAA", "entry_date": "2026-10-06", "strategy": "LONG_CALL", "entry_debit": 2.0,
             "option": {"ask": 2.0, "expiry": "2027-03-19"}, "simulation": {"current_price": 100.0}, "outcome": None}
    hist = {"counterfactual_trades": [dict(trade)], "closed_trades": [], "feature_stats": {}}
    monkeypatch.setattr(feedback, "get_current_option_price", lambda *a, **k: 0.8)    # -60 % -> Stop-Loss
    n = feedback.advance_counterfactual_trades(hist, datetime(2026, 10, 20), price_fn=lambda t: 95.0)
    assert n == 1 and not hist["counterfactual_trades"] and hist["closed_trades"] == []
    cf = hist["counterfactual_closed"][0]
    assert cf["close_reason"] == "stop_loss" and cf["outcome_method"] == "option_quote"
    assert cf["outcome"] == pytest.approx(-0.6) and hist["feature_stats"] == {}
    hist2 = {"counterfactual_trades": [dict(trade)]}
    monkeypatch.setattr(feedback, "get_current_option_price", lambda *a, **k: 2.2)    # +10 % -> offen
    assert feedback.advance_counterfactual_trades(hist2, datetime(2026, 10, 20), price_fn=lambda t: 101.0) == 0
    assert hist2["counterfactual_trades"][0]["current_return"] == pytest.approx(0.1)


def test_pipeline_blocked_trades_use_same_trade_record():
    src = (ROOT / "pipeline.py").read_text()
    assert src.count("build_trade_record(p, today)") == 2                          # echt + counterfactual
    assert 'history.setdefault("counterfactual_trades"' in src


# ── 6. Keine Umgehung der Brücke ────────────────────────────────────────────
RESEARCH_MODULES = ("research_director", "alpha_discovery", "causal_research", "meta_cognition", "research_lab",
                    "meta_learning", "world_model", "counterfactual", "blind_spots", "next_intelligence",
                    "knowledge_graph", "hc_scanner", "decision_intel", "factor_monitor", "trade_memory",
                    "promotion_controller", "hypothesis_contract", "production_intelligence_adapter")
PROTECTED = ("config.yaml", "pipeline.py", "challengers.yaml", "config/", "model_registry", "risk_gates")


def _writes_to_protected(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            seg = ast.get_source_segment(src, n) or ""
            is_write = (isinstance(n.func, ast.Attribute) and n.func.attr in ("write_text", "write_bytes")) or \
                       (isinstance(n.func, ast.Name) and n.func.id == "open" and any(
                           isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value[:1] in "wax"
                           for a in n.args[1:]))
            if is_write and any(p in seg for p in PROTECTED):
                hits.append(seg[:80])
    return hits


def test_research_components_cannot_write_production_config():
    for m in RESEARCH_MODULES:
        assert _writes_to_protected(ROOT / "modules" / f"{m}.py") == [], m
    # Konstanten, auf die Schreibzugriffe zielen könnten, sind keine Produktionsdateien
    for m in ("promotion_controller", "production_intelligence_adapter", "hypothesis_contract"):
        src = (ROOT / "modules" / f"{m}.py").read_text()
        assert "config.yaml" not in src.replace("config.yaml ändern", "").replace("NIE config.yaml", "")


def test_pipeline_uses_only_adapter_for_research_influence():
    tree = ast.parse((ROOT / "pipeline.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("modules."):
            imported.add(n.module.split(".")[1])
    research = {m for m in imported if m in RESEARCH_MODULES or m in ("ml_research", "promotion_controller")}
    # ml_research nur read-only als Ledger-Beobachtung (_ml_shadow_ranks), Wirkung nur über den Adapter
    assert research == {"production_intelligence_adapter", "ml_research"}
    src = (ROOT / "pipeline.py").read_text()
    assert src.count("apply_to_proposals(") == 1


def test_adapter_never_creates_trades_property(tmp_path, monkeypatch):
    rng = random.Random(5)
    cs = [contract(hypothesis_id=f"P{i}", signal_definition="risk_flag", features=["risk_flag"]) for i in range(3)]
    act = {}
    for c, lvl in zip(cs, ("ABSTENTION_ONLY", "NONE", "ABSTENTION_ONLY")):
        act.update(_active(c, lvl, "GUARDED_PRODUCTION"))
    monkeypatch.setattr(pia, "load_verified_state", lambda *a, **k: (act, []))
    for _ in range(50):
        props = [{"ticker": f"X{i}", "features": {"risk_flag": rng.choice([0, 1, None])},
                  "trade_score": {"total": rng.randint(55, 90)}} for i in range(rng.randint(0, 6))]
        kept, blocked, recs = pia.apply_to_proposals(props, vix=rng.choice([None, 12, 40]),
                                                     context={"safe_mode_active": rng.choice([0, 1])},
                                                     contracts=cs, ledger_dir=tmp_path, policy=POLICY)
        ids = [id(p) for p in props]
        assert all(id(p) in ids for p in kept) and len(kept) + len(blocked) == len(props)
        assert len(recs) == len(props)


def test_safe_mode_blocks_promotion_but_not_demotion(tmp_path, monkeypatch):
    """Source Health / Meta-Cognition Safe Mode: keine Promotion neuer Hypothesen."""
    e = Env(tmp_path)
    c = contract()
    synth(e, c, T0, 40)
    monkeypatch.setattr(pc, "_safe_mode_state", lambda: {"active": True, "reasons": ["DATA: kritische Pflichtdaten fehlen"]})
    st = e.run([c], datetime(2027, 2, 1, tzinfo=timezone.utc))
    h = st["hypotheses"][hc.key(c)]
    assert h["state"] == "PROSPECTIVE_CHALLENGER" and h["influence_level"] == "NONE" and st["safe_mode"]["active"]
    assert any("SAFE MODE" in r for r in h["reasons"])
    assert [x["new_state"] for x in pc.read_transitions(e.tr)[0]] == ["PROSPECTIVE_CHALLENGER"]
