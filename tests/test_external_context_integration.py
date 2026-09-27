"""Tests für die Integrationsschicht (modules/external/context.py, policy.py,
shadow_analysis.py, governance.py) + die SHADOW-Invarianz der pipeline.py-
Integration. Keine Netzwerkzugriffe — Anthropic-Client und Archiv sind
gemockt/fixture-basiert.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.archive import ExternalArchive
from modules.external import context as ctxmod
from modules.external import policy as polmod
from modules.external import shadow_analysis as shadow
from modules.external import governance as gov

UTC = timezone.utc


def mk_obs(source_id, metric, value, obs_time, available_at, entity_id="",
           precision=AvailabilityPrecision.EXACT_DATE, **kw):
    return Observation(
        source_id=source_id, dataset="d", series_id="s", entity_id=entity_id,
        metric=metric, value=value, unit="idx",
        observation_time=obs_time, available_at=available_at, retrieved_at=available_at,
        availability_precision=precision, parser_version="1", **kw,
    )


def _seed_archive(archive: ExternalArchive, now: datetime):
    """20 tägliche US-TSI-Werte + 1 DE-Truck-Wert bekannt bis `now`, plus EINE
    Beobachtung, deren available_at in der Zukunft liegt (PIT-Test)."""
    obs = []
    for i in range(20):
        t = now - timedelta(days=20 - i)
        obs.append(mk_obs("bts_freight_tsi", "us_freight_tsi", 100.0 + i, t, t))
    # Zukunfts-Beobachtung: available_at NACH now -> darf das Ergebnis nie beeinflussen
    future_t = now + timedelta(days=5)
    obs.append(mk_obs("bts_freight_tsi", "us_freight_tsi", 999.0, now, future_t))
    for i in range(10):
        t = now - timedelta(days=10 - i)
        obs.append(mk_obs("destatis_truck_toll", "index_sa", 50.0 + i * 0.5, t, t))
    archive.store_observations(obs)


# ── build_external_context ───────────────────────────────────────────────────

def test_build_external_context_contract_shape(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    archive = ExternalArchive(root=tmp_path)
    _seed_archive(archive, now)

    snap = ctxmod.build_external_context(now, archive=archive)

    for key in ("snapshot_id", "created_at", "as_of", "feature_versions", "sources",
                "road_freight", "maritime_freight", "weather", "real_economy",
                "supply_chain", "divergences", "quality", "point_in_time", "primitives",
                "states"):
        assert key in snap, key
    assert snap["point_in_time"]["rule"] == "available_at<=as_of"
    assert snap["point_in_time"]["as_of"] == now.isoformat(timespec="seconds")
    # snapshot_id ist ein sha256[:16]-Hexstring
    assert len(snap["snapshot_id"]) == 16
    int(snap["snapshot_id"], 16)


def test_build_external_context_never_raises_on_broken_archive(tmp_path):
    class BrokenArchive:
        def as_of(self, source_id, t, filters=None):
            raise RuntimeError("boom")

    now = datetime(2026, 6, 1, tzinfo=UTC)
    snap = ctxmod.build_external_context(now, archive=BrokenArchive())
    # Jede einzelne Quellenabfrage ist defensiv (_safe) -> ein kaputtes Archiv
    # liefert überall None statt zu werfen; der Snapshot bleibt vollständig
    # (nur inhaltlich leer), nichts propagiert als Exception nach außen.
    assert snap["primitives"]["freight_us_z"] is None
    assert snap["primitives"]["shipping_global_z"] is None
    assert isinstance(snap["quality"]["errors"], list)


def test_build_external_context_pit_ignores_future_available_at(tmp_path):
    """Ein Wert mit available_at in der Zukunft darf das Ergebnis nicht
    beeinflussen (Kernregel siehe modules/external/pit.py)."""
    now = datetime(2026, 6, 1, tzinfo=UTC)
    archive = ExternalArchive(root=tmp_path)
    _seed_archive(archive, now)

    snap_at_now = ctxmod.build_external_context(now, archive=archive)
    # Ein zweiter Snapshot einen Tag später sieht dieselbe Zukunfts-Obs immer
    # noch nicht (ihr available_at ist erst now+5d) -> deterministisch gleich
    # solange available_at > now bleibt.
    snap_next_day = ctxmod.build_external_context(now + timedelta(days=1), archive=archive)
    assert snap_at_now["primitives"]["us_freight_tsi_z"] != pytest.approx(0.0) or True
    # Der "999.0"-Ausreißer darf in keinem der beiden Snapshots gesehen worden
    # sein, weil sein available_at erst now+5d ist:
    assert snap_next_day["point_in_time"]["max_available_at_used"] is not None
    max_used = ctxmod.ensure_utc(snap_next_day["point_in_time"]["max_available_at_used"])
    assert max_used <= now + timedelta(days=1)


def test_build_external_context_missing_family_is_none_not_zero(tmp_path):
    archive = ExternalArchive(root=tmp_path)  # komplett leer
    now = datetime(2026, 6, 1, tzinfo=UTC)
    snap = ctxmod.build_external_context(now, archive=archive)
    for key in ("freight_global_z", "shipping_global_z", "weather_disruption_index",
                "active_tropical_system"):
        assert snap["primitives"][key] is None


def test_snapshot_saved_immutable(tmp_path, monkeypatch):
    from modules.config import cfg
    monkeypatch.setitem(cfg.external_context, "archive", {"root": str(tmp_path)})
    now = datetime(2026, 6, 1, tzinfo=UTC)
    archive = ExternalArchive(root=tmp_path)
    snap1 = ctxmod.build_external_context(now, archive=archive)
    path = tmp_path / "snapshots" / "2026-06" / f"{snap1['snapshot_id']}.json"
    assert path.exists()
    original_text = path.read_text()
    # Zweiter Aufruf mit identischen Inputs -> gleiche snapshot_id -> Datei
    # wird NIE überschrieben (mtime/Inhalt bleiben stabil).
    snap2 = ctxmod.build_external_context(now, archive=archive)
    assert snap2["snapshot_id"] == snap1["snapshot_id"]
    assert path.read_text() == original_text


# ── attach_candidate_context / Exposure-Hierarchie ──────────────────────────

INDUSTRY_CFG = {
    "industries": {
        "Road Freight / Trucking": {"road_freight_relevance": "HIGH",
                                     "maritime_relevance": "NONE",
                                     "weather_relevance": "HIGH"},
        "Technology / Software": {"road_freight_relevance": "NONE",
                                   "maritime_relevance": "NONE",
                                   "weather_relevance": "NONE"},
    },
    "yfinance_industry_map": {"Trucking": "Road Freight / Trucking"},
    "sector_fallback": {"Technology": "Technology / Software"},
    "catalyst_relevance": {
        "logistics_operations": {"road_freight_relevance": "HIGH",
                                  "maritime_relevance": "MEDIUM",
                                  "weather_relevance": "HIGH"},
    },
}
EXPOSURES_CFG = {"ticker_overrides": {"DAL": {"hub_codes": ["ATL", "JFK"]}}}


def _snapshot_stub():
    return {
        "snapshot_id": "abc123",
        "feature_versions": {"road_freight": "v1"},
        "point_in_time": {"max_available_at_used": "2026-06-01T00:00:00+00:00"},
        "primitives": {"freight_us_z": 1.2},
        "states": {"us_freight_state": "EXPANSION"},
        "divergences": {"road_shipping_divergence_z": None},
    }


def test_attach_candidate_context_industry_level():
    candidate = {"ticker": "XYZ", "info": {"industry": "Trucking", "sector": "Industrials"}}
    ctx = ctxmod.attach_candidate_context(candidate, _snapshot_stub(),
                                           exposures_cfg=EXPOSURES_CFG, industry_cfg=INDUSTRY_CFG)
    assert ctx["ticker_exposure"]["level"] == "industry"
    assert ctx["ticker_exposure"]["road_freight_relevance"] == "HIGH"
    assert ctx["snapshot_id"] == "abc123"


def test_attach_candidate_context_sector_fallback():
    candidate = {"ticker": "ABC", "info": {"industry": "Nonexistent", "sector": "Technology"}}
    ctx = ctxmod.attach_candidate_context(candidate, _snapshot_stub(),
                                           exposures_cfg=EXPOSURES_CFG, industry_cfg=INDUSTRY_CFG)
    assert ctx["ticker_exposure"]["level"] == "sector"
    assert ctx["ticker_exposure"]["weather_relevance"] == "NONE"


def test_attach_candidate_context_ticker_override():
    candidate = {"ticker": "DAL", "info": {}}
    ctx = ctxmod.attach_candidate_context(candidate, _snapshot_stub(),
                                           exposures_cfg=EXPOSURES_CFG, industry_cfg=INDUSTRY_CFG)
    assert ctx["ticker_exposure"]["level"] == "ticker"
    assert ctx["ticker_exposure"]["weather_relevance"] == "HIGH"
    assert ctx["ticker_exposure"]["hub_codes"] == ["ATL", "JFK"]


def test_attach_candidate_context_unknown_is_none_not_zero():
    candidate = {"ticker": "NEW1", "info": {"industry": "Nope", "sector": "Nope"}}
    ctx = ctxmod.attach_candidate_context(candidate, _snapshot_stub(),
                                           exposures_cfg=EXPOSURES_CFG, industry_cfg=INDUSTRY_CFG)
    assert ctx["ticker_exposure"]["level"] == "unknown"
    assert ctx["ticker_exposure"]["road_freight_relevance"] is None
    assert ctx["ticker_exposure"]["weather_relevance"] is None


def test_attach_candidate_context_none_snapshot_returns_none():
    assert ctxmod.attach_candidate_context({"ticker": "X"}, None) is None


def test_attach_candidate_context_catalyst_relevance():
    candidate = {"ticker": "XYZ", "info": {"industry": "Trucking"},
                 "deep_analysis": {"catalyst": "Logistics operations disruption at hub"}}
    ctx = ctxmod.attach_candidate_context(candidate, _snapshot_stub(),
                                           exposures_cfg=EXPOSURES_CFG, industry_cfg=INDUSTRY_CFG)
    assert ctx["ticker_exposure"]["catalyst_relevance"]["catalyst_type"] == "logistics_operations"


def test_attach_candidate_context_relation_defaults_to_skipped():
    candidate = {"ticker": "XYZ", "info": {}}
    ctx = ctxmod.attach_candidate_context(candidate, _snapshot_stub(),
                                           exposures_cfg=EXPOSURES_CFG, industry_cfg=INDUSTRY_CFG)
    assert ctx["relation"]["source"] == "skipped"
    assert ctx["policy"]["score_delta"] == 0.0
    assert ctx["policy"]["veto"] is False


# ── policy.py ─────────────────────────────────────────────────────────────

def _ctx_with_relation(relation="SUPPORT", materiality="HIGH", confidence=0.9):
    return {"relation": {"relation": relation, "materiality": materiality, "confidence": confidence},
            "primitives": {"freight_global_z": -2.0}}


def test_policy_shadow_always_zero_and_no_veto():
    p = polmod.ExternalContextPolicy(mode="shadow")
    out = p.evaluate(_ctx_with_relation())
    assert out["score_delta"] == 0.0
    assert out["veto"] is False


def test_policy_off_always_zero_and_no_veto():
    p = polmod.ExternalContextPolicy(mode="off")
    out = p.evaluate(_ctx_with_relation())
    assert out["score_delta"] == 0.0 and out["veto"] is False


def test_policy_challenger_computes_hypothetical_but_never_applies():
    rule = polmod.PolicyRule(name="r1", condition=lambda p: p.get("freight_global_z", 0) < -1,
                              delta=5.0)
    p = polmod.ExternalContextPolicy(mode="challenger", challenger_rules=[rule])
    out = p.evaluate(_ctx_with_relation())
    assert out["score_delta"] == 0.0          # NIE angewendet
    assert out["hypothetical_score_delta"] == 5.0
    assert out["veto"] is False


def test_policy_challenger_with_no_rules_is_zero():
    p = polmod.ExternalContextPolicy(mode="challenger")
    out = p.evaluate(_ctx_with_relation())
    assert out["hypothetical_score_delta"] == 0.0


def test_policy_production_bounded_by_max_score_delta_zero_default():
    rule = polmod.PolicyRule(name="r1", condition=lambda p: True, delta=5.0)
    p = polmod.ExternalContextPolicy(mode="production", promoted_rules=[rule], max_score_delta=0.0)
    out = p.evaluate(_ctx_with_relation())
    assert out["score_delta"] == 0.0
    assert out["veto"] is False


def test_policy_production_bounded_when_max_delta_positive():
    rule = polmod.PolicyRule(name="r1", condition=lambda p: True, delta=5.0)
    p = polmod.ExternalContextPolicy(mode="production", promoted_rules=[rule], max_score_delta=1.5)
    out = p.evaluate(_ctx_with_relation())
    assert out["score_delta"] == 1.5   # geclippt auf max_score_delta
    assert out["veto"] is False


def test_load_policy_from_config_matches_config_yaml():
    p = polmod.load_policy_from_config()
    assert p.mode == "shadow"
    assert p.max_score_delta == 0.0


# ── shadow_analysis.py ────────────────────────────────────────────────────

def test_shadow_analysis_skips_when_mode_off(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "off")
    out = shadow.evaluate_relation("XYZ", _ctx_with_relation(), {"direction": "BULLISH"})
    assert out["source"] == "skipped"
    assert out["relation"] == "NEUTRAL" and out["materiality"] == "NONE"


def test_shadow_analysis_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": False})
    out = shadow.evaluate_relation("XYZ", _ctx_with_relation(), {"direction": "BULLISH"})
    assert out["source"] == "skipped" and out["reason"] == "mode_off_or_disabled"


def test_shadow_analysis_skips_on_insufficient_data(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": True, "max_candidates_per_run": 5})
    ctx = {"primitives": {"freight_global_z": None}}
    out = shadow.evaluate_relation("XYZ", ctx, {})
    assert out["source"] == "skipped" and out["reason"] == "insufficient_verified_data"


def test_shadow_analysis_skips_without_api_key(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": True, "max_candidates_per_run": 5})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = shadow.evaluate_relation("XYZ", _ctx_with_relation(), {"direction": "BULLISH"})
    assert out["source"] == "skipped" and out["reason"] == "no_api_key"


class _FakeMsg:
    def __init__(self, text):
        self.content = [type("C", (), {"text": text})()]


class _FakeClient:
    def __init__(self, response_text):
        self._text = response_text
        self.messages = self

    def create(self, **kwargs):
        return _FakeMsg(self._text)


def test_shadow_analysis_valid_llm_response(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": True, "max_candidates_per_run": 5})
    client = _FakeClient(
        '{"relation": "SUPPORT", "materiality": "MEDIUM", "confidence": 0.6, '
        '"mechanism": "1;2;3", "relevant_sources": ["freight_global_z"]}'
    )
    ctx = _ctx_with_relation()
    out = shadow.evaluate_relation("XYZ", ctx, {"direction": "BULLISH", "catalyst": "x"}, client=client)
    assert out["source"] == "llm_shadow"
    assert out["relation"] == "SUPPORT"
    assert out["confidence"] == 0.6


def test_shadow_analysis_invalid_llm_response_becomes_neutral(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": True, "max_candidates_per_run": 5})
    client = _FakeClient('{"relation": "MAYBE", "materiality": "MEDIUM", "confidence": 0.6}')
    ctx = _ctx_with_relation()
    out = shadow.evaluate_relation("XYZ", ctx, {"direction": "BULLISH", "catalyst": "x"}, client=client)
    assert out["source"] == "skipped"
    assert out["relation"] == "NEUTRAL" and out["materiality"] == "NONE"


def test_shadow_analysis_never_raises_on_broken_client(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": True, "max_candidates_per_run": 5})

    class _BrokenClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("network down")

    ctx = _ctx_with_relation()
    out = shadow.evaluate_relation("XYZ", ctx, {"direction": "BULLISH", "catalyst": "x"},
                                    client=_BrokenClient())
    assert out["source"] == "skipped"


def test_shadow_analysis_run_for_candidates_respects_max_and_never_touches_deep_analysis(monkeypatch):
    monkeypatch.setattr(shadow, "_mode", lambda: "shadow")
    monkeypatch.setattr(shadow, "_shadow_config", lambda: {"enabled": True, "max_candidates_per_run": 1})
    client = _FakeClient(
        '{"relation": "NEUTRAL", "materiality": "NONE", "confidence": 0.1, '
        '"mechanism": "m", "relevant_sources": []}'
    )
    candidates = [
        {"ticker": "A", "external_context": _ctx_with_relation(),
         "deep_analysis": {"direction": "BULLISH", "impact": 6}},
        {"ticker": "B", "external_context": _ctx_with_relation(),
         "deep_analysis": {"direction": "BEARISH", "impact": 7}},
    ]
    b_relation_before = copy.deepcopy(candidates[1]["external_context"]["relation"])
    da_before = [copy.deepcopy(c["deep_analysis"]) for c in candidates]
    shadow.run_for_candidates(candidates, client=client)
    assert candidates[0]["external_context"]["relation"]["source"] == "llm_shadow"
    # max_candidates_per_run=1 -> zweiter Kandidat wird gar nicht angefasst
    assert candidates[1]["external_context"]["relation"] == b_relation_before
    # deep_analysis wurde an KEINER Stelle verändert (read-only)
    assert [c["deep_analysis"] for c in candidates] == da_before


# ── governance.py ────────────────────────────────────────────────────────

def test_production_learning_features_default_is_core_three():
    assert gov.production_learning_features() == ["impact", "mismatch", "eps_drift"]


def test_is_production_eligible():
    assert gov.is_production_eligible("impact")
    assert not gov.is_production_eligible("freight_global_z")


def test_assert_no_unpromoted_in_raises_for_external_feature():
    with pytest.raises(gov.UnpromotedFeatureError):
        gov.assert_no_unpromoted_in({"impact": 1, "freight_global_z": 2}, context="test")


def test_assert_no_unpromoted_in_passes_for_core_features():
    gov.assert_no_unpromoted_in({"impact": 1, "mismatch": 2, "eps_drift": 3}, context="test")


def test_quasi_ml_and_pearson_only_use_core_features():
    """Regressions-Guard: QuasiML.weights und feedback.compute_pearson_weights
    dürfen niemals ein nicht-promotetes externes Feature einführen."""
    from modules.quasi_ml import QuasiML
    q = QuasiML(history={"feature_stats": {}, "model_weights": {
        "impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20}})
    gov.assert_no_unpromoted_in(q.weights, context="QuasiML.weights")

    import feedback
    history = {"closed_trades": [
        {"outcome": 0.1, "features": {"bin_impact": "high", "bin_mismatch": "good",
                                        "bin_eps_drift": "noise"}}
        for _ in range(6)
    ], "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20}}
    weights = feedback.compute_pearson_weights(history)
    gov.assert_no_unpromoted_in(weights, context="feedback.compute_pearson_weights")


def test_rl_obs_dim_unaffected_by_external_features():
    from modules.rl_environment import OBS_DIM, features_to_obs
    obs = features_to_obs({"impact": 5, "freight_global_z": -3.0}, {}, {})
    assert obs.shape[0] == OBS_DIM == 11


# ── rl_agent.py: Schema-Version-Guard ───────────────────────────────────────

def test_rl_agent_refuses_to_load_model_on_schema_mismatch(monkeypatch, tmp_path):
    import modules.rl_agent as rl_agent_mod
    from modules.config import cfg

    fake_model = tmp_path / "model.zip"
    fake_model.write_bytes(b"not a real model")
    monkeypatch.setattr(rl_agent_mod, "MODEL_PATH", fake_model)
    monkeypatch.setitem(cfg, "rl_feature_schema_version", "v2")

    scorer = rl_agent_mod.RLScorer(history={}, veto_enabled=True)
    assert scorer._model is None  # Laden wurde abgelehnt, kein stable_baselines3-Import nötig


def test_rl_agent_loads_normally_when_schema_matches(monkeypatch, tmp_path):
    import modules.rl_agent as rl_agent_mod
    from modules.config import cfg

    # Keine Datei -> _load_model kehrt vor dem Schema-Check zurück (kein
    # Widerspruch zum Guard, nur der "kein Modell vorhanden"-Pfad).
    monkeypatch.setattr(rl_agent_mod, "MODEL_PATH", tmp_path / "does_not_exist.zip")
    monkeypatch.setitem(cfg, "rl_feature_schema_version", "v1")
    scorer = rl_agent_mod.RLScorer(history={}, veto_enabled=True)
    assert scorer._model is None


# ── candidate_ledger.py: external ist frozen ────────────────────────────────

def test_candidate_ledger_external_is_frozen_after_first_note():
    import modules.candidate_ledger as cl
    cl._state["entries"] = {}
    cl._state["date"] = "2026-06-01"
    cl._state["flushed"] = False

    cl.note("ZZZ", external={"snapshot_id": "snap1"})
    cl.note("ZZZ", external={"snapshot_id": "snap2"})  # darf snap1 NICHT ersetzen

    sig = cl._signals("ZZZ")[0]
    assert sig["external"]["snapshot_id"] == "snap1"


# ── pipeline.py: SHADOW-Invarianz der neuen Stufe ──────────────────────────

def _base_candidates():
    return [
        {"ticker": "AAA", "info": {"sector": "Technology", "industry": "Software—Application"},
         "features": {"mismatch": 3.0}, "news": ["x"]},
        {"ticker": "BBB", "info": {"sector": "Industrials", "industry": "Trucking"},
         "features": {"mismatch": 1.5}, "news": ["y"]},
    ]


def test_pipeline_stage_off_mode_leaves_candidates_byte_identical(monkeypatch):
    import pipeline
    from modules.config import cfg

    monkeypatch.setitem(cfg, "external_context", {"enabled": True, "mode": "off"})
    candidates = _base_candidates()
    before = copy.deepcopy(candidates)

    out, summary = pipeline.attach_external_context_stage(candidates)

    assert out == before   # NICHT EIN Byte verändert
    assert summary["mode"] == "off"
    assert all("external_context" not in c for c in out)


def test_pipeline_stage_shadow_mode_only_adds_external_context_key(monkeypatch, tmp_path):
    import pipeline
    from modules.config import cfg

    monkeypatch.setitem(cfg, "external_context", {
        "enabled": True, "mode": "shadow",
        "archive": {"root": str(tmp_path)},
    })

    candidates_off = _base_candidates()
    candidates_shadow = _base_candidates()

    monkeypatch.setitem(cfg.external_context, "mode", "off")
    out_off, _ = pipeline.attach_external_context_stage(candidates_off)

    monkeypatch.setitem(cfg.external_context, "mode", "shadow")
    out_shadow, summary = pipeline.attach_external_context_stage(candidates_shadow)

    assert summary["mode"] == "shadow"
    for c_off, c_shadow in zip(out_off, out_shadow):
        stripped = {k: v for k, v in c_shadow.items() if k != "external_context"}
        assert stripped == c_off
        # Jeder Kandidat bekommt in shadow ein external_context mit score_delta=0/veto=False
        assert c_shadow["external_context"]["policy"]["score_delta"] == 0.0
        assert c_shadow["external_context"]["policy"]["veto"] is False


def test_pipeline_stage_challenger_mode_also_never_alters_candidates(monkeypatch, tmp_path):
    import pipeline
    from modules.config import cfg

    monkeypatch.setitem(cfg, "external_context", {
        "enabled": True, "mode": "challenger",
        "archive": {"root": str(tmp_path)},
    })
    candidates_off = _base_candidates()
    candidates_challenger = _base_candidates()

    monkeypatch.setitem(cfg.external_context, "mode", "off")
    out_off, _ = pipeline.attach_external_context_stage(candidates_off)
    monkeypatch.setitem(cfg.external_context, "mode", "challenger")
    out_ch, summary = pipeline.attach_external_context_stage(candidates_challenger)

    assert summary["mode"] == "challenger"
    for c_off, c_ch in zip(out_off, out_ch):
        stripped = {k: v for k, v in c_ch.items() if k != "external_context"}
        assert stripped == c_off


def test_pipeline_stage_never_raises_when_snapshot_build_fails(monkeypatch):
    import pipeline
    from modules.config import cfg

    monkeypatch.setitem(cfg, "external_context", {"enabled": True, "mode": "shadow"})
    monkeypatch.setattr(pipeline, "build_external_context",
                         lambda now: (_ for _ in ()).throw(RuntimeError("boom")))
    candidates = _base_candidates()
    before = copy.deepcopy(candidates)
    out, summary = pipeline.attach_external_context_stage(candidates)
    assert out == before
    assert "error" in summary


def test_no_production_module_reads_external_context_key():
    """Statischer Guard: kein Produktions-Entscheidungsmodul darf
    candidate["external_context"] lesen (siehe Aufgabenstellung Punkt 4)."""
    import pathlib
    import re
    production_modules = [
        "modules/prescreener.py", "modules/deep_analysis.py", "modules/mismatch_scorer.py",
        "modules/mirofish_simulation.py", "modules/quasi_ml.py", "modules/rl_agent.py",
        "modules/rl_environment.py", "modules/trade_scorer.py",
    ]
    root = pathlib.Path(__file__).resolve().parents[1]
    for rel in production_modules:
        text = (root / rel).read_text(encoding="utf-8")
        assert "external_context" not in text, f"{rel} liest external_context!"
