"""
modules/external/shadow_analysis.py – optionale LLM-Einschätzung, ob der
externe Kontext zur (bereits produktiv erzeugten) Deep-Analysis-These passt.

HARTE REGELN:
  - Eigener, SEPARATER Anthropic-Call — modifiziert NIE den Prompt/Ergebnis
    der Produktions-Deep-Analysis (modules/deep_analysis.py wird von hier
    nie importiert/aufgerufen, nur READ-ONLY dessen bereits vorhandenes
    Ergebnis-Dict gelesen).
  - Input NUR verifizierte, bereits normalisierte Werte (primitives/states/
    exposure aus dem Kontext + die vier Deep-Analysis-Felder direction/
    impact/catalyst/time_to_materialization) — kein Rohtext, keine
    Kandidaten-internals.
  - Strenges JSON-Ausgabeschema; jede Abweichung -> NEUTRAL/NONE mit reason.
  - Wird komplett übersprungen (relation.source == "skipped"), wenn:
      * enabled=false in config,
      * mode == "off",
      * kein ANTHROPIC_API_KEY,
      * zu wenig verifizierte Eingabedaten (keine einzige primitive != None
        UND keine deep_analysis-Felder).
  - max_candidates_per_run begrenzt die Anzahl LLM-Calls pro Lauf hart.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

VALID_RELATIONS = {"SUPPORT", "NEUTRAL", "CONTRADICT"}
VALID_MATERIALITY = {"NONE", "LOW", "MEDIUM", "HIGH"}

SYSTEM_PROMPT = """Du bewertest AUSSCHLIESSLICH, ob bereits berechnete, normalisierte
externe Kontext-Daten (Fracht/Schifffahrt/Wetter-Primitive und -Zustände) zur
bereits vorliegenden fundamentalen These (Richtung, Katalysator) eines Tickers
PASSEN, WIDERSPRECHEN oder NEUTRAL sind.

PFLICHT-BEGRÜNDUNGSKETTE (mechanism MUSS diese 3 Schritte explizit nennen):
  1. MECHANISMUS: Über welchen konkreten Kanal könnte das externe Primitiv
     das Geschäft dieses Tickers beeinflussen?
  2. EXPOSURE: Ist der Ticker laut ticker_exposure überhaupt in diesem Kanal
     exponiert (relevance-Stufe)? Ohne Exposure ist relation immer NEUTRAL.
  3. KATALYSATOR-BEZUG: Passt der Zeithorizont/Inhalt des Katalysators zum
     externen Signal?

VERBOTEN:
  - Allgemeine Makro-Kommentare ohne Bezug zu den gegebenen Primitiven.
  - Erfinden von Daten, die nicht im Input stehen.
  - Eine andere relation als SUPPORT/NEUTRAL/CONTRADICT.

Antworte AUSSCHLIESSLICH mit validem JSON:
{"relation": "SUPPORT|NEUTRAL|CONTRADICT", "materiality": "NONE|LOW|MEDIUM|HIGH",
 "confidence": <0.0-1.0>, "mechanism": "<Schritt1;Schritt2;Schritt3>",
 "relevant_sources": ["<primitive_name>", ...]}"""

USER_TEMPLATE = """TICKER: {ticker}

DEEP-ANALYSIS-THESE (bereits produktiv erzeugt, nur lesend übernommen):
  direction: {direction}
  catalyst: {catalyst}
  time_to_materialization: {ttm}
  impact: {impact}

TICKER-EXPOSURE (Relevanz-Kategorie, KEINE Richtung):
  road_freight_relevance: {road_rel}
  maritime_relevance: {maritime_rel}
  weather_relevance: {weather_rel}
  exposure_source: {exposure_source}

EXTERNE PRIMITIVE (normalisiert, None = unbekannt -- NIE als 0 lesen):
{primitives_block}

REGIONALE ZUSTÄNDE:
{states_block}

Antworte NUR mit dem JSON-Schema aus der System-Anweisung."""


def _neutral_result(reason: str) -> dict:
    return {"relation": "NEUTRAL", "materiality": "NONE", "confidence": 0.0,
            "mechanism": "", "relevant_sources": [], "source": "skipped", "reason": reason}


def _llm_result(relation: str, materiality: str, confidence: float, mechanism: str,
                 relevant_sources: list) -> dict:
    return {"relation": relation, "materiality": materiality, "confidence": confidence,
            "mechanism": mechanism, "relevant_sources": relevant_sources,
            "source": "llm_shadow", "reason": ""}


def _shadow_config() -> dict:
    try:
        from modules.config import cfg
        ext = getattr(cfg, "external_context", None)
        sa = getattr(ext, "shadow_analysis", None)
        if sa is None:
            return {"enabled": False, "max_candidates_per_run": 0}
        return dict(sa)
    except Exception:
        return {"enabled": False, "max_candidates_per_run": 0}


def _mode() -> str:
    try:
        from modules.config import cfg
        return str(getattr(getattr(cfg, "external_context", None), "mode", "off") or "off")
    except Exception:
        return "off"


def _model_name(cfg_shadow: dict) -> str:
    model = cfg_shadow.get("model")
    if model:
        return model
    try:
        from modules.config import cfg
        return cfg.models.prescreener
    except Exception:
        return "claude-haiku-4-5-20251001"


def _has_verifiable_input(candidate_ctx: dict, deep_analysis: dict) -> bool:
    primitives = (candidate_ctx or {}).get("primitives") or {}
    has_primitive = any(v is not None for v in primitives.values())
    has_da = bool((deep_analysis or {}).get("direction") or (deep_analysis or {}).get("catalyst"))
    return has_primitive and has_da


def _coerce(raw: dict) -> dict:
    relation = raw.get("relation")
    if relation not in VALID_RELATIONS:
        return _neutral_result("invalid_relation_field")
    materiality = raw.get("materiality")
    if materiality not in VALID_MATERIALITY:
        return _neutral_result("invalid_materiality_field")
    try:
        confidence = float(raw.get("confidence"))
        if not (0.0 <= confidence <= 1.0):
            raise ValueError
    except Exception:
        return _neutral_result("invalid_confidence_field")
    mechanism = raw.get("mechanism")
    if not isinstance(mechanism, str) or not mechanism.strip():
        return _neutral_result("missing_mechanism")
    relevant_sources = raw.get("relevant_sources")
    if not isinstance(relevant_sources, list):
        relevant_sources = []
    return _llm_result(relation, materiality, confidence, mechanism, relevant_sources)


def _build_prompt(ticker: str, candidate_ctx: dict, deep_analysis: dict) -> str:
    exposure = candidate_ctx.get("ticker_exposure") or {}
    primitives = candidate_ctx.get("primitives") or {}
    states = candidate_ctx.get("states") or {}
    primitives_block = "\n".join(f"  {k}: {v}" for k, v in sorted(primitives.items()))
    states_block = "\n".join(f"  {k}: {v}" for k, v in sorted(states.items()))
    return USER_TEMPLATE.format(
        ticker=ticker,
        direction=deep_analysis.get("direction"),
        catalyst=deep_analysis.get("catalyst"),
        ttm=deep_analysis.get("time_to_materialization"),
        impact=deep_analysis.get("impact"),
        road_rel=exposure.get("road_freight_relevance"),
        maritime_rel=exposure.get("maritime_relevance"),
        weather_rel=exposure.get("weather_relevance"),
        exposure_source=exposure.get("exposure_source"),
        primitives_block=primitives_block or "  (keine)",
        states_block=states_block or "  (keine)",
    )


def evaluate_relation(ticker: str, candidate_ctx: dict, deep_analysis: dict,
                       client=None) -> dict:
    """Gibt IMMER ein relation-Dict zurück (nie None, wirft nie).
    `client` ist injizierbar für Tests (muss .messages.create(...) bieten,
    identisch zum anthropic.Anthropic()-Client)."""
    shadow_cfg = _shadow_config()
    mode = _mode()

    if mode == "off" or not shadow_cfg.get("enabled", False):
        return _neutral_result("mode_off_or_disabled")
    if not _has_verifiable_input(candidate_ctx, deep_analysis):
        return _neutral_result("insufficient_verified_data")
    if client is None and not os.getenv("ANTHROPIC_API_KEY"):
        return _neutral_result("no_api_key")

    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        prompt = _build_prompt(ticker, candidate_ctx, deep_analysis)
        response = client.messages.create(
            model=_model_name(shadow_cfg),
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        if "```" in raw_text:
            parts = raw_text.split("```")
            raw_text = parts[1].lstrip("json").strip() if len(parts) > 1 else raw_text
        if not raw_text.startswith("{"):
            idx = raw_text.find("{")
            if idx != -1:
                raw_text = raw_text[idx:]
        parsed = json.loads(raw_text)
        return _coerce(parsed)
    except Exception as e:  # noqa: BLE001 - darf die Pipeline nie stören
        log.debug(f"shadow_analysis Fehler ({ticker}): {e} -> NEUTRAL/NONE")
        return _neutral_result(f"exception:{type(e).__name__}")


def run_for_candidates(candidates: list[dict], client=None) -> None:
    """Setzt candidate['external_context']['relation'] für bis zu
    max_candidates_per_run Kandidaten mit vorhandenem external_context +
    deep_analysis. Mutiert NUR das relation-Feld -- alles andere (inklusive
    des Deep-Analysis-Ergebnisses selbst) bleibt unangetastet."""
    shadow_cfg = _shadow_config()
    if _mode() == "off" or not shadow_cfg.get("enabled", False):
        return
    max_n = int(shadow_cfg.get("max_candidates_per_run", 25) or 0)
    n_done = 0
    for c in candidates:
        if n_done >= max_n:
            break
        ctx = c.get("external_context")
        da = c.get("deep_analysis")
        if not ctx or not da:
            continue
        ctx["relation"] = evaluate_relation(c.get("ticker", ""), ctx, da, client=client)
        n_done += 1
