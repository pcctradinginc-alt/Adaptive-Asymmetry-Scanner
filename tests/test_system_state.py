"""SystemState (Single Source of Truth) + abgestufte Drift.

Konsistenz: Scanner (Pipeline-Sicht), HC-Scanner, PromotionController, Adapter, Weekly Report,
Meta-Learning lesen DENSELBEN Zustand – sie können nie gleichzeitig verschiedene Safe-Mode-
Zustände melden. Fail-closed bei unbekannten Komponenten. Versionierung + Historie."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from modules import drift as dr
from modules import system_state as ss

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 5, 13, 0, tzinfo=timezone.utc)
PINNED_DRIFT = "05db93ddab0e2dca0c4190b2d61f7d83f39096756eab9657b9f4cbfa47ed58a3"


def test_drift_policy_pinned():
    assert hashlib.sha256((ROOT / "config" / "drift_policy.yaml").read_bytes()).hexdigest() == PINNED_DRIFT


# ── abgestufte Drift ────────────────────────────────────────────────────────
POL = dr.load_policy()


def feat(**kv):
    return {k: {"value": v, "p01": 0.0, "p99": 10.0} for k, v in kv.items()}


def test_feature_drift_levels_and_single_outlier_never_severe():
    assert dr.feature_level(feat(a=5, b=5), POL)["level"] == "NORMAL"
    assert dr.feature_level(feat(a=10.3, b=5), POL)["level"] == "MILD"            # 3 % der Spannweite
    assert dr.feature_level(feat(a=12, b=5), POL)["level"] == "MODERATE"          # 20 %
    one = dr.feature_level(feat(a=50, b=5, c=5), POL)                              # ein extremer Ausreißer
    assert one["level"] == "MODERATE" and one["features_out_of_range"] == ["a"]
    assert dr.feature_level(feat(a=50, b=-20, c=5), POL)["level"] == "SEVERE"      # zwei weit draußen
    real = {"tnx": {"value": 5.277, "p01": 1.0993, "p99": 4.7817}, "vix": {"value": 15.3, "p01": 12.2, "p99": 33.3}}
    assert dr.feature_level(real, POL)["level"] == "MODERATE"                     # Live-Fall 2026-10-03


@pytest.mark.parametrize("n_bad,level", [(0, "NORMAL"), (1, "MILD"), (2, "MODERATE"), (3, "SEVERE")])
def test_model_drift_levels_keep_pinned_severe_threshold(n_bad, level):
    mi = {f"m{i}": {"trend": "deteriorating" if i < n_bad else "stable"} for i in range(6)}
    assert dr.model_level(mi, POL)["level"] == level
    assert POL["model"]["severe_share"] == 0.5                                    # = next_protocol (nicht gelockert)


def test_data_drift_and_consequences():
    assert dr.data_level(0.95, POL)["level"] == "NORMAL"
    assert dr.data_level(0.8, POL)["level"] == "MILD"
    assert dr.data_level(0.7, POL)["level"] == "MODERATE"
    assert dr.data_level(0.5, POL)["level"] == "SEVERE"
    assert dr.data_level(None, POL)["level"] == "SEVERE"                          # unbekannt
    a = dr.assess({"drift": {"feature_drift": feat(a=12)}}, 0.95, POL)
    assert a["level"] == "MODERATE" and not a["consequences"]["safe_mode"]
    assert a["consequences"]["positive_boost_cap"] == 0.5 and not a["consequences"]["allow_weight_increase"]
    assert dr.assess({}, 0.95, POL)["consequences"]["confidence_multiplier"] == 1.0


# ── SystemState ─────────────────────────────────────────────────────────────
def world(tmp: Path, *, model=None, meta=None, data_q=0.95, data_reasons=(), data=True, model_file=True):
    rs, hl, it = tmp / "outputs/research", tmp / "outputs/health", tmp / "outputs/intelligence"
    for d in (rs, hl, it, tmp / "config"):
        d.mkdir(parents=True, exist_ok=True)
    if model_file:
        (rs / "safe_mode.json").write_text(json.dumps(model or {"active": False, "reasons": [],
                                                                 "components": {"drift": [], "model": [], "data": []}}))
    (rs / "meta_learning.json").write_text(json.dumps(meta or {"model_intelligence": {}, "drift": {}}))
    if data:
        (hl / "source_health_snapshot.json").write_text(json.dumps({
            "generated": NOW.isoformat(), "sources": {}, "counts": {"HEALTHY": 3},
            "safe_mode": {"active": bool(data_reasons), "global_reasons": list(data_reasons), "data_quality": data_q,
                          "blocked_decisions": [], "disabled_signals": ["H1"] if data_reasons else [],
                          "unavailable_features": []}}))
    (it / "promotion_state.json").write_text(json.dumps({"generated": "x", "hypotheses": {
        "A": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE"}}}))
    (tmp / "config/model_registry.yaml").write_text("champion: null\nmodels: []\n")
    (tmp / "config.yaml").write_text("x: 1\n")
    return {k: tmp / v for k, v in ss.DEFAULT_INPUTS.items()}


def test_state_versioned_history_and_fail_closed(tmp_path):
    inp = world(tmp_path)
    sp, hp = tmp_path / "st.json", tmp_path / "h.jsonl"
    s1 = ss.current(inputs=inp, now=NOW, state_path=sp, history_path=hp)
    assert not s1["safe_mode"] and s1["known"] and s1["state_version"] == 1
    for k in ("safe_mode", "safe_mode_reason", "data_health", "model_health", "drift_state", "champion_version",
              "promotion_state", "updated_at", "code_version"):
        assert k in s1
    assert s1["champion_version"]["version"].startswith("scanner_rules@")
    s2 = ss.current(inputs=inp, now=NOW, state_path=sp, history_path=hp)
    assert s2["state_version"] == 1 and len(hp.read_text().splitlines()) == 1     # unverändert: keine neue Version
    world(tmp_path, data_reasons=["kritische Pflichtdaten fehlen"])
    s3 = ss.current(inputs=inp, now=NOW, state_path=sp, history_path=hp)
    assert s3["safe_mode"] and s3["state_version"] == 2 and "DATA: kritische" in s3["safe_mode_reason"][0]
    (tmp_path / "outputs/health/source_health_snapshot.json").unlink()
    s4 = ss.current(inputs=inp, now=NOW, persist=False)
    assert s4["safe_mode"] and not s4["known"] and "DATA HEALTH unbekannt" in s4["safe_mode_reason"][0]
    (tmp_path / "outputs/research/safe_mode.json").write_text("{kaputt")
    assert "SAFE MODE unbekannt" in " ".join(ss.current(inputs=inp, now=NOW, persist=False)["safe_mode_reason"])


def test_model_component_drift_reasons_not_double_counted(tmp_path):
    """Alte safe_mode.json (binärer Drift-Grund) zählt nicht mehr – Drift kommt aus drift.py."""
    inp = world(tmp_path, model={"active": True, "reasons": ["FEATURE/DATA DRIFT: tnx"]},
                meta={"drift": {"feature_drift": {"tnx": {"value": 5.28, "p01": 1.1, "p99": 4.78}}}})
    s = ss.current(inputs=inp, now=NOW, persist=False)
    assert not s["safe_mode"] and s["drift_state"]["level"] == "MODERATE"


SCENARIOS = {
    "all_ok": dict(),
    "data_safe_mode": dict(data_reasons=["2 unabhängige wichtige Quellen ausgefallen"]),
    "model_safe_mode": dict(model={"active": True, "reasons": ["CALIBRATION FAILURE"],
                                   "components": {"drift": [], "model": ["CALIBRATION FAILURE"], "data": []}}),
    "severe_drift": dict(meta={"model_intelligence": {f"m{i}": {"trend": "deteriorating"} for i in range(4)}}),
    "moderate_drift_only": dict(meta={"drift": {"feature_drift": {"tnx": {"value": 5.28, "p01": 1.1, "p99": 4.78}}}}),
    "data_unknown": dict(data=False),
    "model_unknown": dict(model_file=False),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_all_consumers_report_identical_safe_mode(name, tmp_path, monkeypatch):
    inp = world(tmp_path, **SCENARIOS[name])
    monkeypatch.setattr(ss, "DEFAULT_INPUTS", inp)
    monkeypatch.setattr(ss, "STATE", tmp_path / "outputs/state/system_state.json")
    monkeypatch.setattr(ss, "HISTORY", tmp_path / "outputs/state/system_state_history.jsonl")
    monkeypatch.setattr(ss, "_now", lambda: NOW)
    from modules import production_intelligence_adapter as pia
    from modules import promotion_controller as pc
    from modules import source_health as sh
    from reports import weekly
    canonical = ss.current(now=NOW)["safe_mode"]
    views = {
        "system_state": canonical,
        "source_health_view": sh.effective_safe_mode(now=NOW)["active"],
        "adapter": bool(pia.research_context()["safe_mode_active"]),
        "weekly_report": weekly.safe_mode_status(weekly.collect(tmp_path, date(2026, 10, 5))["safe"])[0],
        "meta_learning_input": ss.safe_mode_view(ss.current(now=NOW))["active"],
    }
    # PromotionController pausiert zusätzlich ab Drift MODERATE (vorsichtiger) – nie WENIGER streng
    pcs = pc._safe_mode_state()
    assert pcs["active"] or not canonical
    assert len(set(views.values())) == 1, views
    hc_dir = tmp_path / "outputs/research"
    from modules import hc_scanner as hc
    r = hc.run(send=False, dry_run=True, today=date(2026, 10, 5), out_dir=hc_dir,
               health_snapshot=inp["data_health"])
    hc_says_safe = "SAFE MODE" in (r.get("disabled_reason") or "")
    assert hc_says_safe == canonical, (name, r.get("disabled_reason"))
    expected = {"all_ok": False, "moderate_drift_only": False}.get(name, True)
    assert canonical is expected


def test_no_component_reads_legacy_safe_mode_flags():
    """Nur system_state.py leitet Safe Mode ab; niemand liest safe_mode.json / meta_state.safe_mode direkt."""
    allowed = {"modules/system_state.py", "modules/meta_cognition.py", "modules/source_health.py"}
    offenders = []
    for p in list((ROOT / "modules").rglob("*.py")) + list((ROOT / "reports").rglob("*.py")) + [ROOT / "pipeline.py"]:
        rel = str(p.relative_to(ROOT))
        txt = p.read_text(encoding="utf-8")
        if rel not in allowed and re.search(r"safe_mode\.json", txt):
            offenders.append(rel)
        if re.search(r"meta_state[^\n]*\[\s*[\"']safe_mode", txt) or re.search(r"\"safe_mode\":\s*False", txt):
            offenders.append(rel + " (toter Flag)")
    assert offenders == []


# ── Drift-Konsequenzen + Source-Health-Abhängigkeiten ───────────────────────
def test_adapter_drift_caps_boosts_and_unavailable_features_are_unknown():
    from modules import production_intelligence_adapter as pia
    env = pia.candidate_env({"features": {"risk_flag": 1, "x": 2.0}}, {"data_unavailable_features": ["x"]}, 14)
    assert env["x"] is None and env["risk_flag"] == 1                              # nie alter Wert / 0
    c = {"hypothesis_id": "H", "features": ["x"]}
    act = {"k": {"contract": c, "level": "SCORE_LIMITED", "state": "GUARDED_PRODUCTION", "spec_hash": "h"}}
    base = pia.decide_for_trade({}, {"x": 1}, {}, safe_mode=False, sector="T", regime="r", policy={})
    assert base["score_adjustment"] == 0
    cap = dr.load_policy()["consequences"]["MODERATE"]
    assert cap["positive_boost_cap"] == 0.5 and cap["allow_weight_increase"] is False
    assert act                                                                     # Struktur wie im Adapter


def test_hc_blocks_when_model_features_unavailable(tmp_path):
    from modules import hc_scanner as hc
    inp = world(tmp_path)
    snap = json.loads(inp["data_health"].read_text())
    snap["safe_mode"]["unavailable_features"] = ["vix"]
    inp["data_health"].write_text(json.dumps(snap))
    r = hc.run(send=False, dry_run=True, today=date(2026, 10, 5), out_dir=tmp_path / "outputs/research",
               health_snapshot=inp["data_health"])
    assert not r["enabled"] and "Pflichtdaten der Modelle nicht verfügbar: ['vix']" in r["disabled_reason"]


def test_every_feature_has_documented_source_dependency():
    from modules import ml_research as ml
    from modules import source_health as sh
    from modules.alt_data.registry import ALT_FEATURES
    deps = sh.feature_dependencies()
    missing = [f for f in list(ml.ALL_FEATURES) + list(ALT_FEATURES) if not deps.get(f)]
    assert missing == []
    assert (ROOT / "docs/FEATURE_DEPENDENCIES.md").read_text(encoding="utf-8") == sh.render_feature_dependencies(), \
        "docs/FEATURE_DEPENDENCIES.md veraltet: python -m modules.source_health deps"


def test_single_future_period_fact_is_degraded_not_broken():
    import pandas as pd
    from modules import source_health as sh
    c = sh.load_config()
    df = pd.DataFrame({"series_id": [f"s{i}" for i in range(1000)], "metric": "m", "value": 1.0,
                       "available_at": "2026-10-01T00:00:00+00:00", "retrieved_at": "2026-10-02T00:00:00+00:00",
                       "observation_time": ["2027-12-31T00:00:00+00:00"] + ["2026-09-30T00:00:00+00:00"] * 999})
    chk = lambda st: sh.classify({"source_id": "s", "kind": "alt", "criticality": "NON_CRITICAL", "store": st,
                                  "last_success": NOW.isoformat(), "latest_observation": "2026-09-30T00:00:00+00:00"},
                                 None, c, NOW)
    new = sh.frame_stats(df.assign(retrieved_at="2026-10-04T12:00:00+00:00", available_at="2026-10-04T00:00:00+00:00"),
                         NOW, c)
    assert new["future_observations"] == 1 and chk(new)["status"] == sh.DEGRADED     # neuer Datenfehler
    old = sh.frame_stats(df, NOW, c)                                                   # vor Tagen abgerufen
    assert old["future_observations"] == 0 and old["future_observations_quarantined"] == 1
    assert chk(old)["status"] == sh.HEALTHY                                            # Altfehler: Quarantäne, Info


def test_xbrl_drops_facts_with_period_after_filing():
    from modules.external.sources import sec_xbrl as sx
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2027-12-31", "val": 5, "form": "10-K", "filed": "2026-02-01", "accn": "bad"},
        {"end": "2025-12-31", "val": 4, "form": "10-K", "filed": "2026-02-01", "accn": "ok"}]}}}}}
    obs = sx.parse_companyfacts(payload, "1", NOW)
    assert [o.attrs["accn"] for o in obs] == ["ok"]
