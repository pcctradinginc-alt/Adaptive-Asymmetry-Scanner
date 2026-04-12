"""
Stufe 2: Vorselektion – "Der Türsteher"
- Batch-Analyse aller Headlines via Claude 3 Haiku (kostengünstig)
- Unterscheidet temporäres Rauschen von strukturellen Änderungen
- Output: Nur Ticker mit [YES] kommen weiter
"""

import json
import logging
import os
import anthropic

log = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """Du bist ein erfahrener Finanzanalyst mit Fokus auf strukturelle Marktveränderungen.
Deine Aufgabe: Unterscheide zwischen temporärem Rauschen und echten strukturellen Änderungen.

Temporäres Rauschen (→ [NO]):
- Aktienrückkäufe ohne strategischen Kontext
- Analysten-Upgrades/-Downgrades ohne fundamentale Begründung
- Quartalsergebnisse im Rahmen der Erwartungen
- Dividendenankündigungen
- CEO-Statements ohne konkrete Ankündigung

Strukturelle Änderungen (→ [YES]):
- Neue Produktkategorien oder Märkte
- Technologische Durchbrüche (neue IP, Patente)
- Management-Turnarounds mit konkretem Plan
- Regulatorische Entscheidungen mit langfristiger Wirkung
- M&A mit strategischer Logik
- Verlust/Gewinn eines Großkunden (>10% Umsatz)
- Fundamentale Geschäftsmodelländerungen

Antworte ausschließlich mit validem JSON."""

USER_TEMPLATE = """Analysiere diese News-Headlines pro Ticker.
Für jeden Ticker: Entscheide ob die News eine strukturelle Änderung darstellt.

Ticker und Headlines:
{ticker_news}

Antworte mit folgendem JSON-Format:
{{
  "results": [
    {{
      "ticker": "AAPL",
      "decision": "[YES]" oder "[NO]",
      "reason": "Kurze Begründung (max 20 Wörter)"
    }}
  ]
}}"""


class Prescreener:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        # Batch-Prompt aufbauen
        ticker_news_str = "\n".join([
            f"[{c['ticker']}]: {' | '.join(c['news'][:5])}"
            for c in candidates
        ])

        prompt = USER_TEMPLATE.format(ticker_news=ticker_news_str)

        response = None
        try:
            response = self.client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # JSON aus Markdown-Block extrahieren
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            # Geschweifte Klammer suchen
            if not raw.startswith("{"):
                idx = raw.find("{")
                if idx != -1:
                    raw = raw[idx:]
            parsed = json.loads(raw)
            results = parsed.get("results", [])
        except Exception as e:
            preview = response.content[0].text[:300] if response else "keine Antwort"
            log.error(f"Haiku Prescreening Fehler: {e}")
            log.error(f"Haiku raw output: {preview}")
            return []

        # Nur [YES]-Ticker weiterleiten
        yes_tickers = {
            r["ticker"]: r["reason"]
            for r in results
            if r.get("decision") == "[YES]"
        }

        shortlist = []
        for c in candidates:
            if c["ticker"] in yes_tickers:
                c["prescreen_reason"] = yes_tickers[c["ticker"]]
                shortlist.append(c)
                log.info(f"  [YES] {c['ticker']}: {yes_tickers[c['ticker']]}")

        return shortlist
