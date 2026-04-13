"""
modules/prescreener.py v7.0 – Verschärfter Haiku-Prefilter

Problem vorher: Haiku liess 17+ Ticker durch → Deep Analysis lief für jeden
                einzeln → teuer (~$0.25 nur für Sonnet) + langsam (4 Min).

Fix: Strengerer System-Prompt + härtere Entscheidungsregeln.
     Ziel: max. 3-5 YES pro Batch (statt 17+).

Verschärfte Regeln:
  - Im Zweifel IMMER [NO]
  - Kein [YES] für Routine-Earnings, Upgrades, Dividenden
  - [YES] nur für eindeutige, strukturelle, unerwartete Änderungen
  - Explizite Kategorie-Abfrage damit Haiku begründen muss
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
BATCH_SIZE    = 20
MAX_HEADLINES = 3

SYSTEM_PROMPT = """Du bist ein extrem selektiver Quantitativ-Analyst.
Deine Aufgabe: Filtere AGGRESSIV. Nur echte strukturelle Signale kommen durch.

HARTE REGELN — kein Ausnahme:
1. Im Zweifel IMMER [NO]. Lieber ein echtes Signal verpassen als 10 falsche durchlassen.
2. [NO] für: Earnings in Rahmen, Analyst-Upgrades/-Downgrades, Dividenden, Aktienrückkäufe, CEO-Statements ohne Substanz, allgemeine Markt-News.
3. [YES] NUR wenn ALLE drei Bedingungen erfüllt:
   a) STRUKTURELL: Neue Produktkategorie, Akquisition, FDA-Zulassung, Verlust/Gewinn Grosskunde, Technologie-Durchbruch, Regulierungs-Entscheid.
   b) UNERWARTET: Nicht im Konsens erwartet, überrascht den Markt.
   c) FUNDAMENTAL: Verändert die langfristigen Ertragserwartungen (2-6 Monate).

Ziel: Maximal 10-20% der Ticker bekommen [YES].
Antworte ausschliesslich mit validem JSON."""

USER_TEMPLATE = """Bewerte diese {n} Ticker. Sei SEHR restriktiv.

{ticker_news}

Antworte NUR mit diesem JSON:
{{
  "results": [
    {{
      "ticker": "AAPL",
      "decision": "[YES]" oder "[NO]",
      "category": "structural_change|routine_news|analyst_opinion|earnings|other",
      "reason": "Max 15 Woerter Begruendung"
    }}
  ]
}}

Erinnerung: Ziel ist max. 10-20% YES. Im Zweifel [NO]."""


class Prescreener:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        all_yes: dict[str, str] = {}

        batches = [
            candidates[i:i + BATCH_SIZE]
            for i in range(0, len(candidates), BATCH_SIZE)
        ]
        log.info(
            f"Prescreening: {len(candidates)} Kandidaten in "
            f"{len(batches)} Batch(es) à max {BATCH_SIZE}"
        )

        for batch_idx, batch in enumerate(batches, 1):
            log.info(f"  Batch {batch_idx}/{len(batches)}: {len(batch)} Ticker")
            results = self._call_with_retry(batch)

            if results is None:
                log.warning(f"  Batch {batch_idx} fehlgeschlagen → übersprungen")
                continue

            yes_count = 0
            no_count  = 0
            for r in results:
                decision = r.get("decision", "[NO]")
                category = r.get("category", "other")
                ticker   = r.get("ticker", "")

                if decision == "[YES]":
                    # Zusatz-Check: Routine-Kategorien nie durchlassen
                    if category in ("routine_news", "analyst_opinion", "earnings"):
                        log.info(
                            f"  [{ticker}] Override: Kategorie='{category}' "
                            f"→ trotz [YES] auf [NO] gesetzt"
                        )
                        no_count += 1
                        continue
                    all_yes[ticker] = r.get("reason", "")
                    yes_count += 1
                else:
                    no_count += 1

            log.info(
                f"  Batch {batch_idx}: {yes_count} YES, {no_count} NO "
                f"({yes_count/(yes_count+no_count)*100:.0f}% YES-Rate)"
            )

        shortlist = []
        for c in candidates:
            if c["ticker"] in all_yes:
                c["prescreen_reason"] = all_yes[c["ticker"]]
                shortlist.append(c)
                log.info(f"  [YES] {c['ticker']}: {all_yes[c['ticker']]}")

        log.info(
            f"Prescreening gesamt: {len(shortlist)}/{len(candidates)} YES "
            f"({len(shortlist)/len(candidates)*100:.0f}% YES-Rate)"
        )
        return shortlist

    def _call_with_retry(self, batch: list[dict]) -> list | None:
        ticker_news_str = "\n".join([
            f"[{c['ticker']}]: {' | '.join(c['news'][:MAX_HEADLINES])}"
            for c in batch
        ])
        prompt = USER_TEMPLATE.format(
            n           = len(batch),
            ticker_news = ticker_news_str,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model      = cfg.models.prescreener,
                    max_tokens = 4096,
                    system     = SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": prompt}],
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
                log.info(
                    f"    Batch OK (Versuch {attempt}): "
                    f"{len(results)} Ticker bewertet"
                )
                return results

            except (json.JSONDecodeError, KeyError) as e:
                log.error(f"    Parsing-Fehler (Versuch {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(BACKOFF_BASE ** attempt)

            except Exception as e:
                wait = BACKOFF_BASE ** attempt
                log.warning(
                    f"    API-Fehler (Versuch {attempt}/{MAX_RETRIES}): "
                    f"{e} → Warte {wait}s"
                )
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(wait)

        return None
