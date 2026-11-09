"""Reproduzierbarkeit der LLM-Schritte: temperature=0, Audit-Hashes, Cache-Version."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules import analysis_cache, deep_analysis
from modules import prescreener as ps
from modules.deep_analysis import DeepAnalysis

RAW = json.dumps({"impact": 7, "surprise": 6, "direction": "BULLISH", "catalyst_confidence": 6})
CAND = {"ticker": "AAA", "news": ["x"], "info": {}, "features": {}}


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _resp(text=RAW):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason="end_turn")


def _da():
    da = DeepAnalysis.__new__(DeepAnalysis)
    da._macro = {"data_available": False, "macro_regime": "unknown"}
    da.client = MagicMock()
    da._get_48h_move = lambda ticker: 0.0
    return da


def test_deep_analysis_params_temperature_zero():
    assert _da()._prepare(CAND)["params"]["temperature"] == 0


def test_deep_analysis_batch_and_ab_requests_temperature_zero():
    da = _da()
    captured = {}

    def fake_run_batch(client, **kw):
        captured.update(kw["requests"])
        return {}

    shortlist = [dict(CAND, ticker=t) for t in ("AAA", "BBB")]
    with patch.object(deep_analysis.cost_telemetry, "batch_settings",
                      return_value={"min_requests": 1, "max_wait_s": 1, "poll_s": 1}), \
         patch.object(deep_analysis.cost_telemetry, "run_batch", side_effect=fake_run_batch), \
         patch.object(deep_analysis.analysis_cache, "lookup", return_value=None), \
         patch.object(deep_analysis.model_routing, "sample", side_effect=lambda ids, n: ids):
        da._prefetch_batch(shortlist, None, {"other": "claude-x", "n": 2})
    assert any(k.startswith("da-") for k in captured) and any(k.startswith("ab-") for k in captured)
    assert all(p["temperature"] == 0 for p in captured.values())


def test_prompt_version_depends_on_temperature():
    base = analysis_cache.prompt_version(deep_analysis.SYSTEM_PROMPT, deep_analysis.ANALYSIS_TEMPLATE)
    assert deep_analysis.PROMPT_VERSION != base
    assert deep_analysis.PROMPT_VERSION == analysis_cache.prompt_version(
        deep_analysis.SYSTEM_PROMPT, deep_analysis.ANALYSIS_TEMPLATE, "temperature=0")
    assert deep_analysis.PROMPT_VERSION != analysis_cache.prompt_version(
        deep_analysis.SYSTEM_PROMPT, deep_analysis.ANALYSIS_TEMPLATE, "temperature=1")


def test_deep_analysis_llm_audit_hashes():
    da = _da()
    ctx = da._prepare(CAND)
    with patch.object(deep_analysis.analysis_cache, "store"), \
         patch.object(deep_analysis.analysis_cache, "compare_observed"):
        res = da._complete(ctx, _resp())
    p = ctx["params"]
    system, user = p["system"][0]["text"], p["messages"][0]["content"]
    assert res["llm_audit"] == {
        "model": p["model"], "temperature": 0, "prompt_version": deep_analysis.PROMPT_VERSION,
        "prompt_sha256": _sha(system + user), "response_sha256": _sha(RAW)}


def test_deep_analysis_cache_hit_keeps_original_hashes():
    da = _da()
    ctx = da._prepare(CAND)
    stored = {"impact": 7, "surprise": 6, "direction": "BULLISH",
              "llm_audit": {"model": "m", "temperature": 0, "prompt_version": "v",
                            "prompt_sha256": "a", "response_sha256": "b"}}
    with patch.object(deep_analysis.analysis_cache, "lookup", return_value=stored):
        res = da._lookup_cache(ctx)
    assert res["llm_audit"] == {**stored["llm_audit"], "cached": True}


def test_audit_deterministic():
    p = {"model": "m", "temperature": 0, "system": "S", "messages": [{"role": "user", "content": "U"}]}
    a = analysis_cache.llm_audit(p, _resp("abc"), version="v1")
    assert a == analysis_cache.llm_audit(p, _resp("abc"), version="v1")
    assert a["prompt_sha256"] == _sha("SU") and a["response_sha256"] == _sha("abc")


def test_prescreener_params_temperature_zero():
    batch = [{"ticker": "AAA", "news": ["n"]}]
    assert ps.Prescreener._params(batch)["temperature"] == 0
    assert ps.Prescreener._params(batch, ps.USER_TEMPLATE_COMPACT)["temperature"] == 0


def test_prescreener_batch_requests_temperature_and_audit():
    with patch.object(ps.anthropic, "Anthropic"):
        p = ps.Prescreener()
    batches = [[{"ticker": "AAA", "news": ["n"]}], [{"ticker": "BBB", "news": ["m"]}]]
    seen = {}
    text = json.dumps({"results": [{"ticker": "AAA", "decision": "[YES]", "category": "x"}]})

    def fake_run_batch(client, **kw):
        seen.update(kw["requests"])
        return {"ps-0": _resp(text)}

    with patch.object(ps.cost_telemetry, "batch_settings",
                      return_value={"min_requests": 1, "max_wait_s": 1, "poll_s": 1}), \
         patch.object(ps.cost_telemetry, "run_batch", side_effect=fake_run_batch):
        out = p._prefetch_batch(batches)
    assert seen and all(r["temperature"] == 0 for r in seen.values())
    audit = out[0][0]["llm_audit"]
    system, user = analysis_cache.request_texts(seen["ps-0"])
    assert audit["prompt_sha256"] == _sha(system + user)
    assert audit["response_sha256"] == _sha(text)
    assert audit["prompt_version"] == ps.PROMPT_VERSION and audit["temperature"] == 0
    assert ps.PROMPT_VERSION == analysis_cache.prompt_version(
        ps.SYSTEM_PROMPT, ps.USER_TEMPLATE, ps.USER_TEMPLATE_COMPACT, "temperature=0")


# ── Regime-Trennung der A/B-Paare (Temperatur/Vorlagen) ─────────────────────
def test_routing_decisions_count_only_pairs_of_the_current_prompt_regime():
    from modules import model_routing as mr
    pol = {"model_routing": {"deep_analysis": {"challenger": "H", "mode": "auto", "min_pairs": 4, "min_days": 2,
                                               "min_gate_agreement": 0.9, "min_pass_recall": 0.9,
                                               "min_direction_agreement": 0.9}},
           "bearish_prefilter": {"mode": "auto", "min_pass_n": 4, "min_days": 2, "max_pass_loss": 0.05}}
    same = {"impact": 7, "surprise": 6, "direction": "BULLISH"}
    rows = [{**mr.pair_row("deep_analysis", f"T{i}", "S", "H", same, same, prompt_version=v),
             "ts": f"2026-11-{10 + i % 2}"} for i, v in enumerate(["old"] * 6 + ["new"] * 2)]
    assert mr.decision("deep_analysis", "S", rows, pol)["status"] == "VALIDATED"           # ohne Filter: gemischt
    d = mr.decision("deep_analysis", "S", rows, pol, prompt_version="new")
    assert d["status"] == "AB_RUNNING" and d["ab"]["pairs"] == 2 and d["pairs_other_regimes"] == 6
    assert d["active"] == "S"                                                               # Champion bleibt
    pf = [{**mr.prefilter_pair(f"T{i}", "BULLISH", same, prompt_version="old"), "ts": f"2026-11-{10 + i % 2}"}
          for i in range(6)]
    assert mr.prefilter_decision(pf, pol)["active"] is True
    assert mr.prefilter_decision(pf, pol, prompt_version="new")["active"] is False
    pp = mr.prescreen_pair([{"ticker": "A", "decision": "[YES]"}], [{"ticker": "A", "decision": "[YES]"}],
                           False, prompt_version="x")
    assert pp["prompt_version"] == "x"
    assert mr.current_version("route_pair") == deep_analysis.PROMPT_VERSION
    assert mr.current_version("prefilter_pair") == f"{ps.PROMPT_VERSION}+{deep_analysis.PROMPT_VERSION}"
    assert mr.current_version("unknown_kind") == "unavailable"                             # fail closed
