"""Reproduzierbarkeit der LLM-Schritte (Aktionsplan P8, Owner-Entscheidung 2026-11-10): temperature=0 über die
API-Version 2023-06-01 mit Fallback, Audit-Hashes, versionsreiner Analyse-Cache, A/B-Paare je Regime.

Hintergrund: anthropic>=2.0 sendet die API-Version 2026-10-21, die temperature/top_p/top_k auf jedem Modell mit 400
ablehnt. Prescreening und Deep Analysis senden deshalb je Aufruf anthropic-version 2023-06-01 + temperature=0; lehnt
die API das ab und gelingt der Aufruf ohne Override, läuft der Prozess mit API-Default weiter (protokolliert).
"""
import hashlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from modules import analysis_cache, deep_analysis
from modules import cost_telemetry as ct
from modules import prescreener as ps
from modules.deep_analysis import DeepAnalysis

RAW = json.dumps({"impact": 7, "surprise": 6, "direction": "BULLISH", "catalyst_confidence": 6})
CAND = {"ticker": "AAA", "news": ["x"], "info": {}, "features": {}}
SAMPLING = {"temperature", "top_p", "top_k"}


@pytest.fixture(autouse=True)
def _fresh_sampling(tmp_path, monkeypatch):
    ct.reset_sampling()
    monkeypatch.setattr(ct, "record", lambda *a, **k: None)       # keine Telemetrie-Dateien im Repo
    yield
    ct.reset_sampling()


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _resp(text=RAW):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason="end_turn", usage=None, model="m")


def _da():
    da = DeepAnalysis.__new__(DeepAnalysis)
    da._macro = {"data_available": False, "macro_regime": "unknown"}
    da.client = MagicMock()
    da._get_48h_move = lambda ticker: 0.0
    return da


class _ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code


# ── Override + Fallback (cost_telemetry) ─────────────────────────────────────
def test_installed_sdk_needs_the_old_api_version_for_temperature():
    """Begründung des Overrides: das installierte SDK (API 2026-10-21) kennt temperature nicht mehr."""
    from anthropic.resources.messages import Messages
    if int(anthropic.__version__.split(".")[0]) >= 2:
        assert "temperature" not in inspect.signature(Messages.create).parameters
        assert {"extra_headers", "extra_body"} <= set(inspect.signature(Messages.create).parameters)


def test_sync_call_sends_temperature_zero_under_api_version_2023_06_01():
    client = MagicMock()
    client.messages.create.return_value = _resp()
    out = {}
    ct.tracked_create(client, workflow="deep_analysis", sampling=True, sampling_out=out, model="m", max_tokens=5,
                      messages=[{"role": "user", "content": "u"}])
    kw = client.messages.create.call_args.kwargs
    assert kw["extra_headers"] == {"anthropic-version": "2023-06-01"} and kw["extra_body"] == {"temperature": 0}
    assert not SAMPLING & set(kw) and out == {"used": True} and ct.sampling_active()


def test_rejected_override_falls_back_to_api_default_for_the_rest_of_the_run():
    client = MagicMock()
    client.messages.create.side_effect = [_ApiError(400), _resp(), _resp()]
    out = {}
    r = ct.tracked_create(client, workflow="prescreening", sampling=True, sampling_out=out, model="m", max_tokens=5,
                          messages=[])
    assert r is not None and out == {"used": False} and not ct.sampling_active()
    second = client.messages.create.call_args_list[1].kwargs
    assert "extra_headers" not in second and "extra_body" not in second               # Gegenprobe ohne Override
    ct.tracked_create(client, workflow="prescreening", sampling=True, sampling_out=out, model="m", max_tokens=5,
                      messages=[])
    assert "extra_body" not in client.messages.create.call_args_list[2].kwargs       # bleibt aus


def test_unrelated_client_error_keeps_sampling_and_is_raised():
    client = MagicMock()
    client.messages.create.side_effect = [_ApiError(400), _ApiError(400)]               # scheitert auch ohne Override
    with pytest.raises(_ApiError):
        ct.tracked_create(client, workflow="deep_analysis", sampling=True, model="m", max_tokens=5, messages=[])
    assert ct.sampling_active()
    client.messages.create.side_effect = [_ApiError(500)]                               # Serverfehler: kein Fallback
    with pytest.raises(_ApiError):
        ct.tracked_create(client, workflow="deep_analysis", sampling=True, model="m", max_tokens=5, messages=[])
    assert ct.sampling_active() and client.messages.create.call_count == 3


def test_batch_sends_temperature_per_request_and_old_version_with_fallback():
    client = MagicMock()
    client.messages.batches.create.return_value = SimpleNamespace(id="b1", processing_status="ended")
    client.messages.batches.results.return_value = []
    out = {}
    ct.run_batch(client, workflow="deep_analysis", requests={"da-0": {"model": "m", "max_tokens": 5}},
                 sampling=True, sampling_out=out, sleep=lambda s: None)
    kw = client.messages.batches.create.call_args.kwargs
    assert kw["extra_headers"] == {"anthropic-version": "2023-06-01"} and out == {"used": True}
    assert kw["requests"] == [{"custom_id": "da-0", "params": {"model": "m", "max_tokens": 5, "temperature": 0}}]
    client.messages.batches.create.side_effect = [_ApiError(400), SimpleNamespace(id="b2", processing_status="ended")]
    ct.run_batch(client, workflow="deep_analysis", requests={"da-0": {"model": "m"}}, sampling=True,
                 sampling_out=out, sleep=lambda s: None)
    last = client.messages.batches.create.call_args.kwargs
    assert "extra_headers" not in last and last["requests"][0]["params"] == {"model": "m"}
    assert out == {"used": False} and not ct.sampling_active()


def test_calls_without_sampling_flag_are_unchanged():
    client = MagicMock()
    client.messages.create.return_value = _resp()
    ct.tracked_create(client, workflow="hypothesis_factory", model="m", max_tokens=5, messages=[])
    assert "extra_body" not in client.messages.create.call_args.kwargs


# ── Deep Analysis / Prescreening ─────────────────────────────────────────────
def test_prompt_versions_separate_the_sampling_regimes():
    assert deep_analysis.PROMPT_VERSION_DEFAULT == analysis_cache.prompt_version(
        deep_analysis.SYSTEM_PROMPT, deep_analysis.ANALYSIS_TEMPLATE)                    # Alt-Cache unverändert
    assert deep_analysis.PROMPT_VERSION == analysis_cache.prompt_version(
        deep_analysis.SYSTEM_PROMPT, deep_analysis.ANALYSIS_TEMPLATE, ct.SAMPLING_TAG)
    assert deep_analysis.prompt_version_now() == deep_analysis.PROMPT_VERSION
    ct._SAMPLING["active"] = False
    assert deep_analysis.prompt_version_now() == deep_analysis.PROMPT_VERSION_DEFAULT
    assert ps.prompt_version_now() == ps.PROMPT_VERSION_DEFAULT


def test_deep_analysis_sync_call_uses_sampling_and_audits_it():
    da = _da()
    with patch.object(deep_analysis.analysis_cache, "lookup", return_value=None), \
         patch.object(deep_analysis.analysis_cache, "store") as store, \
         patch.object(deep_analysis.analysis_cache, "compare_observed"), \
         patch.object(deep_analysis.cost_telemetry, "tracked_create",
                      side_effect=lambda *a, sampling_out=None, **k: (sampling_out.update(used=k["sampling"]),
                                                                      _resp())[1]):
        res = da._analyze(CAND)
    audit = res["llm_audit"]
    assert audit["sampling"] == ct.SAMPLING_TAG and audit["prompt_version"] == deep_analysis.PROMPT_VERSION
    assert store.called


def test_deep_analysis_does_not_cache_answers_under_a_foreign_version():
    da = _da()
    ctx = da._prepare(CAND)                                       # Schlüssel unter temperature=0
    with patch.object(deep_analysis.analysis_cache, "store") as store, \
         patch.object(deep_analysis.analysis_cache, "compare_observed"):
        res = da._complete({**ctx, "sampled": False}, _resp())   # Antwort kam nach Fallback ohne Override
    p = ctx["params"]
    system, user = p["system"][0]["text"], p["messages"][0]["content"]
    assert res["llm_audit"] == {
        "model": p["model"], "sampling": "api-default", "prompt_version": deep_analysis.PROMPT_VERSION_DEFAULT,
        "prompt_sha256": _sha(system + user), "response_sha256": _sha(RAW)}
    assert not store.called


def test_deep_analysis_batch_requests_go_out_with_sampling():
    da = _da()
    seen = {}

    def fake_run_batch(client, **kw):
        seen.update(kw)
        kw["sampling_out"]["used"] = True
        return {}

    shortlist = [dict(CAND, ticker=t) for t in ("AAA", "BBB")]
    with patch.object(deep_analysis.cost_telemetry, "batch_settings",
                      return_value={"min_requests": 1, "max_wait_s": 1, "poll_s": 1}), \
         patch.object(deep_analysis.cost_telemetry, "run_batch", side_effect=fake_run_batch), \
         patch.object(deep_analysis.analysis_cache, "lookup", return_value=None), \
         patch.object(deep_analysis.model_routing, "sample", side_effect=lambda ids, n: ids):
        da._prefetch_batch(shortlist, None, {"other": "claude-x", "n": 2})
    assert seen["sampling"] is True
    assert any(k.startswith("da-") for k in seen["requests"]) and any(k.startswith("ab-") for k in seen["requests"])
    assert all(not SAMPLING & set(p) for p in seen["requests"].values())    # Temperatur setzt erst run_batch


def test_deep_analysis_cache_hit_keeps_original_hashes():
    da = _da()
    ctx = da._prepare(CAND)
    stored = {"impact": 7, "surprise": 6, "direction": "BULLISH",
              "llm_audit": {"model": "m", "sampling": ct.SAMPLING_TAG, "prompt_version": "v",
                            "prompt_sha256": "a", "response_sha256": "b"}}
    with patch.object(deep_analysis.analysis_cache, "lookup", return_value=stored):
        res = da._lookup_cache(ctx)
    assert res["llm_audit"] == {**stored["llm_audit"], "cached": True}


def test_audit_deterministic():
    p = {"model": "m", "system": "S", "messages": [{"role": "user", "content": "U"}]}
    a = analysis_cache.llm_audit(p, _resp("abc"), version="v1", sampled=True)
    assert a == analysis_cache.llm_audit(p, _resp("abc"), version="v1", sampled=True)
    assert a["prompt_sha256"] == _sha("SU") and a["response_sha256"] == _sha("abc")
    assert a["sampling"] == ct.SAMPLING_TAG
    assert analysis_cache.llm_audit(p, _resp("abc"), version="v1")["sampling"] == "api-default"


def test_prescreener_batch_uses_sampling_and_audits_regime():
    with patch.object(ps.anthropic, "Anthropic"):
        p = ps.Prescreener()
    batches = [[{"ticker": "AAA", "news": ["n"]}], [{"ticker": "BBB", "news": ["m"]}]]
    seen = {}
    text = json.dumps({"results": [{"ticker": "AAA", "decision": "[YES]", "category": "x"}]})

    def fake_run_batch(client, **kw):
        seen.update(kw)
        kw["sampling_out"]["used"] = True
        return {"ps-0": _resp(text)}

    with patch.object(ps.cost_telemetry, "batch_settings",
                      return_value={"min_requests": 1, "max_wait_s": 1, "poll_s": 1}), \
         patch.object(ps.cost_telemetry, "run_batch", side_effect=fake_run_batch):
        out = p._prefetch_batch(batches)
    assert seen["sampling"] is True and all(not SAMPLING & set(r) for r in seen["requests"].values())
    audit = out[0][0]["llm_audit"]
    system, user = analysis_cache.request_texts(seen["requests"]["ps-0"])
    assert audit["prompt_sha256"] == _sha(system + user) and audit["response_sha256"] == _sha(text)
    assert audit["prompt_version"] == ps.PROMPT_VERSION and audit["sampling"] == ct.SAMPLING_TAG


# ── Regime-Trennung der A/B-Paare (Sampling/Vorlagen) ────────────────────────
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
    ct._SAMPLING["active"] = False                                                          # nach Fallback
    assert mr.current_version("route_pair") == deep_analysis.PROMPT_VERSION_DEFAULT
    assert mr.current_version("unknown_kind") == "unavailable"                             # fail closed
