"""Tests modules/expectation_alpha/claim_extraction.py – offline, ohne Netzwerk und ohne API-Key.

Geprüft wird nur die EXTRAKTION (separater LLM-Call, Prompt, Parser, Skip-/Fehlerpfade, Limits).
Die Prüfung der Claims gegen Evidenz (claims.verify) ist nicht Gegenstand dieser Tests."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modules import cost_telemetry
from modules.expectation_alpha import claim_extraction as ce
from modules.expectation_alpha import claims

TEXT = {
    "catalyst": "Der Umsatz stieg im Q3 um 14 % gegenüber dem Vorjahr, das Management hob die Prognose an.",
    "asymmetry_reasoning": "Die Aktienanzahl sank durch Rückkäufe.",
    "bear_case": "Die Nettomarge könnte unter Druck geraten.",
    "macro_assessment": "Zinsen bleiben hoch.",
}
ANALYSIS = {
    "deep_analysis": {**TEXT, "direction": "BULLISH", "impact": 9, "surprise": 7,
                      "time_to_materialization": "SECRET-TTM-123", "trade_score": 77},
}
CFG_ON = {"claims": {"enabled": True, "model": "test-model-x", "max_candidates_per_run": 12, "max_tokens": 321}}

VALID = {"claims": [
    {"claim_type": "revenue_up", "entity": "ACME", "metric": "revenue", "direction": "up",
     "value_if_known": 14, "source_reference": "catalyst", "confidence": 0.9},
    {"claim_type": "guidance_raised", "entity": "ACME", "metric": "guidance", "direction": "up",
     "value_if_known": None, "source_reference": "catalyst", "confidence": 0.8},
]}
VALID_JSON = json.dumps(VALID)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Kein Test darf je einen echten Anthropic-Client bauen oder einen API-Key sehen."""
    import anthropic

    def _boom(*a, **k):
        raise AssertionError("anthropic.Anthropic darf in diesem Test nicht erzeugt werden")
    monkeypatch.setattr(anthropic, "Anthropic", _boom)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


class FakeClient:
    """Minimaler Anthropic-Ersatz: .messages.create(**kw) -> Objekt mit .content[0].text und .usage."""

    def __init__(self, text=VALID_JSON, exc=None):
        self.calls: list[dict] = []
        self._text, self._exc = text, exc
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50,
                                                     cache_creation_input_tokens=0, cache_read_input_tokens=0),
                               model=kw.get("model"), stop_reason="end_turn")


# ── Prompts ─────────────────────────────────────────────────────────────────
def test_system_prompt_contains_every_claim_type_and_description():
    for name, spec in claims.CLAIM_TYPES.items():
        assert name in ce.SYSTEM_PROMPT
        assert spec["description"] in ce.SYSTEM_PROMPT


def test_system_prompt_names_source_fields_schema_and_limit():
    for f in claims.SOURCE_FIELDS:
        assert f in ce.SYSTEM_PROMPT
    for key in ("claim_type", "entity", "metric", "direction", "value_if_known", "source_reference",
                "confidence", '"claims"'):
        assert key in ce.SYSTEM_PROMPT
    assert str(ce.MAX_CLAIMS) in ce.SYSTEM_PROMPT
    assert "other" in ce.SYSTEM_PROMPT


def test_user_prompt_contains_only_source_fields():
    texts = {**TEXT, "direction": "BULLISH", "impact": "NICHT-ERLAUBT-IMPACT"}
    up = ce.build_user_prompt("ACME", texts)
    assert "ACME" in up
    for f, t in TEXT.items():
        assert f"[{f}]" in up and t in up
    assert "NICHT-ERLAUBT-IMPACT" not in up and "BULLISH" not in up


def test_user_prompt_from_analysis_has_no_other_candidate_data():
    up = ce.build_user_prompt("ACME", claims.source_texts(ANALYSIS))
    for leak in ("SECRET-TTM-123", "BULLISH", "trade_score", "77"):
        assert leak not in up


def test_user_prompt_skips_missing_and_empty_fields():
    up = ce.build_user_prompt("ACME", {"catalyst": "Umsatz stieg.", "bear_case": "  "})
    assert "[catalyst]" in up
    assert "[bear_case]" not in up and "[asymmetry_reasoning]" not in up and "[macro_assessment]" not in up


# ── parse_response ──────────────────────────────────────────────────────────
def test_parse_plain_json():
    assert ce.parse_response(VALID_JSON) == VALID["claims"]


def test_parse_fenced_json():
    assert ce.parse_response(f"```json\n{VALID_JSON}\n```") == VALID["claims"]
    assert ce.parse_response(f"```\n{VALID_JSON}\n```") == VALID["claims"]


def test_parse_json_with_leading_prose():
    assert ce.parse_response(f"Hier ist das Ergebnis:\n{VALID_JSON}\nViel Erfolg!") == VALID["claims"]


def test_parse_json_with_braces_in_leading_prose():
    assert ce.parse_response(f"Format {{claims}} beachtet.\n{VALID_JSON}") == VALID["claims"]


def test_parse_empty_claims_list_is_a_list_not_none():
    assert ce.parse_response('{"claims": []}') == []


@pytest.mark.parametrize("text", [
    "das ist kein json", "", "   ", "{kaputt", '{"claims": "nicht-liste"}', '{"andere": []}', "[1, 2, 3]",
    '{"claims": [', None, 123, b'{"claims": []}',
])
def test_parse_garbage_returns_none(text):
    assert ce.parse_response(text) is None


# ── extract: llm_fn ─────────────────────────────────────────────────────────
def test_extract_with_llm_fn_ok_and_prompts_passed():
    seen = {}

    def fn(system, user):
        seen["system"], seen["user"] = system, user
        return VALID_JSON
    r = ce.extract("ACME", ANALYSIS, {}, llm_fn=fn)       # kein claims-Abschnitt, enabled=False: llm_fn gewinnt
    assert r["status"] == "OK" and r["reason"] is None
    assert r["raw_claims"] == VALID["claims"]
    assert seen["system"] == ce.SYSTEM_PROMPT
    assert "ACME" in seen["user"] and TEXT["catalyst"] in seen["user"]
    assert "SECRET-TTM-123" not in seen["user"]


def test_extract_without_text_is_skipped_and_llm_fn_not_called():
    called = []
    r = ce.extract("ACME", {"deep_analysis": {"direction": "BULLISH"}}, CFG_ON,
                   llm_fn=lambda s, u: called.append(1) or VALID_JSON)
    assert r["status"] == "SKIPPED" and "Text" in r["reason"] and r["raw_claims"] is None
    assert not called
    assert ce.extract("ACME", {}, CFG_ON)["status"] == "SKIPPED"
    assert ce.extract("ACME", None, CFG_ON)["status"] == "SKIPPED"


def test_extract_garbage_output_is_error():
    r = ce.extract("ACME", ANALYSIS, CFG_ON, llm_fn=lambda s, u: "ich kann das nicht")
    assert r["status"] == "ERROR" and r["reason"] == "Antwort nicht parsebar" and r["raw_claims"] is None


def test_extract_llm_fn_raising_is_error_without_propagation():
    def fn(s, u):
        raise RuntimeError("x" * 500)
    r = ce.extract("ACME", ANALYSIS, CFG_ON, llm_fn=fn)
    assert r["status"] == "ERROR" and r["raw_claims"] is None
    assert r["reason"].startswith("RuntimeError: ")
    assert len(r["reason"]) <= 200


def test_extract_caps_claims_at_max():
    many = json.dumps({"claims": [{"claim_type": "other"} for _ in range(ce.MAX_CLAIMS + 5)]})
    r = ce.extract("ACME", ANALYSIS, CFG_ON, llm_fn=lambda s, u: many)
    assert r["status"] == "OK" and len(r["raw_claims"]) == ce.MAX_CLAIMS


# ── extract: Skip-Pfade (kein Netzwerk) ─────────────────────────────────────
def test_extract_enabled_without_api_key_is_skipped_and_never_builds_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    allow_calls = []
    monkeypatch.setattr(cost_telemetry, "allow", lambda wf, *a, **k: allow_calls.append(wf) or True)
    r = ce.extract("ACME", ANALYSIS, CFG_ON)              # _no_network: Anthropic() würde AssertionError werfen
    assert r == {"status": "SKIPPED", "reason": "kein ANTHROPIC_API_KEY", "raw_claims": None, "model": None}
    assert allow_calls == []


@pytest.mark.parametrize("cfg", [{}, {"claims": None}, {"claims": {}}, {"claims": {"enabled": False}}])
def test_extract_disabled_is_skipped_even_with_api_key_and_client(monkeypatch, cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = FakeClient()
    for kw in ({}, {"client": client}):
        r = ce.extract("ACME", ANALYSIS, cfg, **kw)
        assert r["status"] == "SKIPPED" and r["reason"] == "claims.enabled=false"
    assert client.calls == []


def test_extract_cost_guard_blocks_without_building_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    seen = []
    monkeypatch.setattr(cost_telemetry, "allow", lambda wf, *a, **k: seen.append(wf) or False)
    r = ce.extract("ACME", ANALYSIS, CFG_ON)
    assert r["status"] == "SKIPPED" and r["reason"] == "Kosten-Guard"
    assert seen == ["ea_claims"]


# ── extract: injizierter Client ─────────────────────────────────────────────
def test_extract_with_fake_client_ok_uses_workflow_ea_claims(monkeypatch):
    events = []
    monkeypatch.setattr(cost_telemetry, "record", lambda ev, *a, **k: events.append(ev))
    client = FakeClient()
    r = ce.extract("ACME", ANALYSIS, CFG_ON, client=client)   # kein API-Key nötig, kein allow()-Aufruf
    assert r["status"] == "OK" and r["raw_claims"] == VALID["claims"] and r["model"] == "test-model-x"
    assert len(client.calls) == 1
    kw = client.calls[0]
    assert kw["model"] == "test-model-x" and kw["max_tokens"] == 321
    assert kw["system"] == ce.SYSTEM_PROMPT
    assert kw["messages"] == [{"role": "user", "content": ce.build_user_prompt("ACME", claims.source_texts(ANALYSIS))}]
    llm = [e for e in events if e.get("kind") == "llm"]
    assert len(llm) == 1
    assert llm[0]["workflow"] == "ea_claims" and llm[0]["ticker"] == "ACME" and llm[0]["success"] is True


def test_extract_model_and_limits_default_when_keys_missing():
    from modules.config import cfg as main_cfg
    client = FakeClient()
    r = ce.extract("ACME", ANALYSIS, {"claims": {"enabled": True}}, client=client)
    assert r["status"] == "OK"
    assert client.calls[0]["model"] == main_cfg.models.prescreener == r["model"]
    assert client.calls[0]["max_tokens"] == 700


def test_extract_fake_client_garbage_and_exception_are_errors():
    r = ce.extract("ACME", ANALYSIS, CFG_ON, client=FakeClient(text="kein json"))
    assert r["status"] == "ERROR" and r["reason"] == "Antwort nicht parsebar"
    r = ce.extract("ACME", ANALYSIS, CFG_ON, client=FakeClient(exc=TimeoutError("zu langsam")))
    assert r["status"] == "ERROR" and r["reason"] == "TimeoutError: zu langsam" and r["raw_claims"] is None


def test_extract_empty_content_is_error():
    client = FakeClient()
    client.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[], usage=None))
    r = ce.extract("ACME", ANALYSIS, CFG_ON, client=client)
    assert r["status"] == "ERROR" and r["reason"].startswith("IndexError")


# ── extract_many ────────────────────────────────────────────────────────────
def _items(n):
    return [(f"T{i}", ANALYSIS) for i in range(n)]


def test_extract_many_respects_max_candidates_per_run():
    calls = []
    cfg = {"claims": {"enabled": True, "max_candidates_per_run": 2}}
    out = ce.extract_many(_items(4), cfg, llm_fn=lambda s, u: calls.append(1) or VALID_JSON)
    assert list(out) == ["T0", "T1", "T2", "T3"]
    assert [out[t]["status"] for t in out] == ["OK", "OK", "SKIPPED", "SKIPPED"]
    assert out["T2"]["reason"] == out["T3"]["reason"] == "Limit je Lauf"
    assert len(calls) == 2


def test_extract_many_default_cap_is_12():
    calls = []
    out = ce.extract_many(_items(15), {}, llm_fn=lambda s, u: calls.append(1) or VALID_JSON)
    assert len(calls) == 12
    assert sum(r["status"] == "OK" for r in out.values()) == 12
    assert all(out[f"T{i}"]["reason"] == "Limit je Lauf" for i in (12, 13, 14))


def test_extract_many_deadline_already_passed():
    import time
    calls = []
    out = ce.extract_many(_items(3), CFG_ON, llm_fn=lambda s, u: calls.append(1) or VALID_JSON,
                          deadline=time.monotonic() - 1)
    assert calls == []
    assert all(r["status"] == "SKIPPED" and r["reason"] == "Laufzeitbudget" for r in out.values())


def test_extract_many_deadline_expires_mid_run(monkeypatch):
    clock = iter([0.0, 100.0, 100.0])
    monkeypatch.setattr(ce, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    out = ce.extract_many(_items(3), CFG_ON, llm_fn=lambda s, u: VALID_JSON, deadline=50.0)
    assert out["T0"]["status"] == "OK"
    assert out["T1"]["reason"] == out["T2"]["reason"] == "Laufzeitbudget"


def test_extract_many_future_deadline_and_no_deadline_process_all():
    import time
    for dl in (None, time.monotonic() + 3600):
        out = ce.extract_many(_items(3), CFG_ON, llm_fn=lambda s, u: VALID_JSON, deadline=dl)
        assert [r["status"] for r in out.values()] == ["OK"] * 3


def test_extract_many_never_raises_on_errors():
    def fn(s, u):
        raise ValueError("boom")
    out = ce.extract_many(_items(2), CFG_ON, llm_fn=fn)
    assert all(r["status"] == "ERROR" for r in out.values())


def test_extract_many_with_fake_client_uses_one_call_per_item():
    client = FakeClient()
    out = ce.extract_many(_items(3), CFG_ON, client=client)
    assert len(client.calls) == 3 and all(r["status"] == "OK" for r in out.values())
    assert [c["messages"][0]["content"].split("\n")[0] for c in client.calls] == ["TICKER: T0", "TICKER: T1", "TICKER: T2"]


# ── Ende-zu-Ende mit claims.normalize (nur Extraktion + Schema, keine Verifikation) ───────────
def test_invented_value_does_not_pass_through_normalize():
    invented = json.dumps({"claims": [
        {"claim_type": "revenue_up", "entity": "ACME", "metric": "revenue", "direction": "up",
         "value_if_known": 12.5, "source_reference": "catalyst", "confidence": 0.9},      # 12.5 steht nicht im Text
        {"claim_type": "revenue_up", "entity": "ACME", "metric": "revenue", "direction": "up",
         "value_if_known": 14, "source_reference": "catalyst", "confidence": 0.7},        # 14 steht im Text
    ]})
    texts = claims.source_texts(ANALYSIS)
    r = ce.extract("ACME", ANALYSIS, CFG_ON, llm_fn=lambda s, u: invented)
    assert r["status"] == "OK"
    ok, dropped = claims.normalize(r["raw_claims"], texts, "ACME")
    assert len(ok) == 2
    assert ok[0].value_if_known is None
    assert ok[1].value_if_known == 14.0
    assert any("value_not_in_source" in d["reason"] for d in dropped)


def test_unknown_claim_type_and_direction_conflict_handled_by_normalize():
    raw = ce.parse_response(json.dumps({"claims": [
        {"claim_type": "erfundener_typ", "direction": "up", "source_reference": "catalyst"},
        {"claim_type": "revenue_up", "direction": "down", "source_reference": "catalyst"},
        "kein-objekt",
    ]}))
    ok, dropped = claims.normalize(raw, claims.source_texts(ANALYSIS), "ACME")
    assert [c.claim_type for c in ok] == ["other"]
    assert len(dropped) == 2
