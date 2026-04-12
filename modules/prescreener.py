"""
Stufe 2: Vorselektion – "Der Türsteher"

Fix: Bei 91 Kandidaten wurde max_tokens=2048 überschritten → JSON abgeschnitten.
     Lösung:
       1. Batching: max 20 Ticker pro API-Aufruf (statt alle 91 auf einmal)
       2. max_tokens: 4096 statt 2048
       3. Nur 3 Headlines pro Ticker statt 5 (kürzer = mehr Ticker pro Batch)
"""

import json
import logging
import os
import time
import anthropic

from modules.config import cfg

log = logging.getLogger(__name__)

MAX_RETRIES   = 3
BACKOFF_BASE  = 2
BATCH_SIZE    = 20    # Fix: max Ticker pro API-Call (verhindert Token-Overflow)
MAX_HEADLINES = 3     # Fix: 3 statt 5 Headlines (kürzer = mehr Ticker pro Batch)

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
Fuer jeden Ticker: Entscheide ob die News eine strukturelle Aenderung darstellt.

Ticker und Headlines:
{ticker_news}

Antworte mit folgendem JSON-Format:
{{
  "results": [
    {{
      "ticker": "AAPL",
      "decision": "[YES]" oder "[NO]",
      "reason": "Kurze Begruendung (max 20 Woerter)"
    }}
  ]
}}"""


class Prescreener:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        # Fix: Kandidaten in Batches aufteilen
        all_yes: dict[str, str] = {}

        batches = [
            candidates[i:i + BATCH_SIZE]
            for i in range(0, len(candidates), BATCH_SIZE)
        ]
        log.info(
            f"Prescreening: {len(candidates)} Kandidaten in "
            f"{len(batches)} Batch(es) a max {BATCH_SIZE}"
        )

        for batch_idx, batch in enumerate(batches, 1):
            log.info(f"  Batch {batch_idx}/{len(batches)}: {len(batch)} Ticker")
            results = self._call_with_retry(batch)

            if results is None:
                log.warning(f"  Batch {batch_idx} fehlgeschlagen -> uebersprungen")
                continue

            for r in results:
                if r.get("decision") == "[YES]":
                    all_yes[r["ticker"]] = r.get("reason", "")

        shortlist = []
        for c in candidates:
            if c["ticker"] in all_yes:
                c["prescreen_reason"] = all_yes[c["ticker"]]
                shortlist.append(c)
                log.info(f"  [YES] {c['ticker']}: {all_yes[c['ticker']]}")

        return shortlist

    def _call_with_retry(self, batch: list[dict]) -> list | None:
        ticker_news_str = "\n".join([
            f"[{c['ticker']}]: {' | '.join(c['news'][:MAX_HEADLINES])}"
            for c in batch
        ])
        prompt = USER_TEMPLATE.format(ticker_news=ticker_news_str)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=cfg.models.prescreener,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()

                if "```" in raw:
                    parts = raw.split("```")
                    raw   = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                if not raw.startswith("{"):
                    idx = raw.find("{")
                    if idx != -1:
                        raw = raw[idx:]

                parsed  = json.loads(raw)
                results = parsed.get("results", [])
                log.info(f"    Batch OK (Versuch {attempt}): {len(results)} Ticker bewertet")
                return results

            except (json.JSONDecodeError, KeyError) as e:
                log.error(f"    Parsing-Fehler (Versuch {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(BACKOFF_BASE ** attempt)

            except Exception as e:
                wait = BACKOFF_BASE ** attempt
                log.warning(f"    API-Fehler (Versuch {attempt}/{MAX_RETRIES}): {e} -> Warte {wait}s")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(wait)

        return None
