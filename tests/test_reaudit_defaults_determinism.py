"""Re-Audit 2026-10-10 (Punkte 1/15): fail-safe Defaults bei fehlender/defekter Konfiguration und Determinismus der
Lern-/Evaluationspfade (gleiche Eingaben -> identische Ergebnisse). Rein technische Tests mit synthetischen Daten."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modules import meta_label as ml
from modules import production_intelligence_adapter as pia
from modules import promotion_controller as pc
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import evaluation as ev
from tests.test_learning_engine import _recs

UTC = timezone.utc


# ── Defaults ────────────────────────────────────────────────────────────────
def test_ea_is_off_when_config_missing_broken_or_unknown_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(eacfg, "PATH", tmp_path / "fehlt.yaml")
    assert eacfg.enabled() is False                                            # Datei fehlt -> aus
    for body in ("", "mode: production\n", "mode: [kaputt\n", "decision: {}\n"):
        p = tmp_path / "c.yaml"
        p.write_text(body)
        monkeypatch.setattr(eacfg, "PATH", p)
        assert eacfg.enabled() is False, body                                  # nie stillschweigend aktiv


def test_promotion_cap_defaults_to_abstention_and_unknown_to_none():
    assert pc._cap_level("WEIGHT_25", {}, {}) == ("ABSTENTION_ONLY", True)     # Policy ohne Obergrenze
    assert pc._cap_level("WEIGHT_25", {}, {"max_automatic_influence": "ALLES"}) == ("NONE", True)
    assert pc._cap_level("SIZE_DOWN_ONLY", {}, {}) == ("ABSTENTION_ONLY", True)   # Sizing nie automatisch


def test_learning_defaults_without_artifacts(tmp_path):
    assert pia.kill_switch_active(tmp_path / "fehlt.yaml") is False
    assert ml.status(tmp_path) == {"status": "NEVER_RUN"}
    assert ml.active_model(tmp_path) == (None, None)
    assert ml.predict_one({"impact": 5}, "BULLISH", root=tmp_path) is None
    assert ml.predict_one({"impact": 5}, "NEUTRAL", root=tmp_path) is None     # ohne Richtung nie eine Prognose


# ── Determinismus ───────────────────────────────────────────────────────────
def test_meta_label_training_is_deterministic(tmp_path):
    recs = _recs()
    t = datetime(2025, 5, 15, tzinfo=UTC)
    a = ml.train(recs=recs, root=tmp_path / "a", as_of=t.date(), now=t)
    b = ml.train(recs=recs, root=tmp_path / "b", as_of=t.date(), now=t)
    assert a["evaluation"] == b["evaluation"] and a["active_model"] == b["active_model"]
    va, vb = ml._registry(tmp_path / "a")["versions"][0], ml._registry(tmp_path / "b")["versions"][0]
    assert va["train_ids_hash"] == vb["train_ids_hash"]
    f = {"impact": 8, "surprise": 3, "scan_day_ret": 0.01}
    pa = ml.predict_one(f, "BEARISH", as_of=t + timedelta(days=1), root=tmp_path / "a")
    pb = ml.predict_one(f, "BEARISH", as_of=t + timedelta(days=1), root=tmp_path / "b")
    assert pa["p"] == pb["p"]


def test_ea_evaluation_is_deterministic(tmp_path):
    from tests.test_ea_audit_fixes import _learning_fixture
    r1 = _learning_fixture(tmp_path / "x", means_by_ratio=[(0.1, 0.02), (0.7, 0.04)])
    r2 = ev.run(root=tmp_path / "x", cfg=eacfg.load(), promotion_state={}, write=False)
    def strip(r):
        return {k: v for k, v in r.items() if k not in ("generated", "proposals")}
    assert strip(r1) == strip(r2)
    assert r1["values"]["abstention"]["all"]["ci95"] == r2["values"]["abstention"]["all"]["ci95"]   # Bootstrap-Seed


# ── Risk-Isolation (Punkt 11) ───────────────────────────────────────────────
def test_pipeline_order_risk_gates_then_adapter_then_sizing():
    """Intelligence (inkl. Sizing-Stufe) wirkt nur auf bereits durch alle Risk Gates gelaufene Trades und vor der
    Positionsgröße; kein Pfad vorbei an den Gates."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "pipeline.py").read_text(encoding="utf-8")
    pos = [src.index(m) for m in ("STUFE 0: Risk Gates", "STUFE 10: Options Design + ROI-Gate",
                                  "STUFE 10b: Korrelations-Check", "apply_to_proposals(",
                                  "trade_proposals = enrich_with_sizing(trade_proposals)")]
    assert pos == sorted(pos)
    assert src.count("enrich_with_sizing(") == 1 and src.count("apply_size_multiplier") == 0   # nur über Sizing-Modul


def test_size_multiplier_never_increases_any_risk_property(monkeypatch):
    import random

    from modules import position_sizing as ps
    monkeypatch.setattr(ps, "_portfolio_cfg", lambda: {"assumed_loss_pct": 0.6})
    rng = random.Random(11)
    for _ in range(500):
        cost = rng.choice([0.0, 35.0, 120.0, 480.0])
        base = {"position_pct": rng.uniform(0, 0.05), "position_usd": rng.uniform(0, 5000), "cost_per_contract": cost}
        base["contracts"] = int(base["position_usd"] // cost) if cost else 0
        base["max_risk_usd"] = round(base["contracts"] * cost * 0.6, 2)
        m = rng.choice([rng.uniform(-1, 3), 0.25, 0.5, 0.999, 1.0, None, "x"])
        out = ps.apply_size_multiplier(dict(base), m)
        for k in ("contracts", "position_usd", "position_pct", "max_risk_usd"):
            assert out[k] <= base[k] + 1e-9, (k, m, base, out)
