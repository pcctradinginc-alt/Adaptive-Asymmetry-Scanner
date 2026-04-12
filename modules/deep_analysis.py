"""
Stufe 3: Deep Analysis – "Asymmetry Reasoning"
- Claude 3.5 Sonnet für Second-Order-Thinking
- Warum ist der Markt hier ineffizient?
- JSON-Output: impact, surprise, mispricing_logic, catalyst, time_to_materialization
"""

import json
import logging
import os
import anthropic

log = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """Du bist ein quantitativer Hedge-Fund-Analyst mit Spezialisierung auf Markt-Ineffizienzen.
Du suchst nach Informations-Asymmetrien: Situationen, in denen der Markt eine fundamentale
Änderung noch nicht vollständig eingepreist hat (Underreaction innerhalb 48 Stunden).

Dein Ziel: Second-Order-Thinking. Nicht "was ist passiert", sondern "warum hat der Markt 
falsch reagiert" und "welcher Katalysator erzwingt die Einpreisung".

Sei präzise, skeptisch und datengetrieben. Ignoriere Hype. Fokussiere auf Fundamentals.
Antworte ausschließlich mit validem JSON."""

ANALYSIS_TEMPLATE = """Analysiere diesen Ticker auf Informations-Asymmetrien:

Ticker: {ticker}
Vorselektion-Grund: {prescreen_reason}
Headlines: {headlines}
EPS-Drift: {eps_drift}
Forward EPS (aktuell): {current_eps}
Analyst Konsens: {rec_mean}
48h-Preisbewegung: {price_move_48h}%

Führe eine Asymmetry-Analyse durch:

1. Warum ist der Markt hier ineffizient (Underreaction)?
2. Was ist der konkrete Katalysator, der die Einpreisung erzwingt?
3. Welche regulatorischen oder makroökonomischen Risiken könnten die These sofort entwerten?

Antworte mit diesem exakten JSON:
{{
  "ticker": "{ticker}",
  "impact": <0-10, fundamentale Stärke der News>,
  "surprise": <0-10, wie unerwartet war die News>,
  "mispricing_logic": "<Herleitung warum der Markt falsch reagiert hat, max 100 Wörter>",
  "catalyst": "<Konkretes Event/Datum das die Einpreisung erzwingt>",
  "time_to_materialization": "<4-8 Wochen | 2-3 Monate | 6 Monate>",
  "bear_case": "<Stärkster Gegenargument, max 50 Wörter>",
  "bear_case_severity": <0-10, wie stark ist der Bear Case>,
  "direction": "<BULLISH | BEARISH>"
}}"""


class DeepAnalysis:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, shortlist: list[dict]) -> list[dict]:
        analyses = []
        for candidate in shortlist:
            analysis = self._analyze(candidate)
            if analysis:
                analyses.append({**candidate, **analysis})
        return analyses

    def _analyze(self, candidate: dict) -> dict | None:
        ticker  = candidate["ticker"]
        info    = candidate["info"]
        drift   = candidate["eps_drift"]
        news    = candidate["news"]

        # 48h-Preisbewegung berechnen
        price_move = self._get_48h_move(ticker)

        prompt = ANALYSIS_TEMPLATE.format(
            ticker=ticker,
            prescreen_reason=candidate.get("prescreen_reason", ""),
            headlines=" | ".join(news[:5]),
            eps_drift=drift.get("drift", 0),
            current_eps=drift.get("current_eps", "N/A"),
            rec_mean=drift.get("rec_mean", "N/A"),
            price_move_48h=round(price_move * 100, 2),
        )

        try:
            response = self.client.messages.create(
                model=SONNET_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # JSON aus möglichem Markdown-Block extrahieren
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            log.info(
                f"  [{ticker}] Impact={parsed.get('impact')} "
                f"Surprise={parsed.get('surprise')} "
                f"Direction={parsed.get('direction')}"
            )
            return {"deep_analysis": parsed, "price_move_48h": price_move}
        except Exception as e:
            log.error(f"Deep Analysis Fehler für {ticker}: {e}")
            return None

    def _get_48h_move(self, ticker: str) -> float:
        """Berechnet die 2-Tages-Rendite."""
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 2:
                return 0.0
            close = hist["Close"]
            return float((close.iloc[-1] - close.iloc[-3]) / close.iloc[-3])
        except Exception:
            return 0.0
