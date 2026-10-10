# API-Kosten-Optimierung (2026-10-09)

Ziel: ≤ 10 USD/Monat bei unveränderter Entscheidungsqualität. Qualität und Production Safety haben Vorrang.
Scope: ausschließlich Datenzugriff, Cache und Kosten-Telemetrie. Unverändert bleiben Alpha, Champion,
Promotion, Drift, Forward-Starts, ROI-/Stop-Regeln, Universe, Hypothesis Contracts, ML/RL und
V2-/Commodity-Promotion.

## 1. Forensische Kostenmatrix
Datenbasis: Ledger `outputs/costs/ledger-2026-10.jsonl`, 3 vollständige Scanner-Tage (06.–08.10.).
Erzeugt mit `python scripts/cost_matrix.py`.

| Provider | Endpoint | Consumer | Req./Lauf | Ticker | Ø In/Out-Tokens | $/Request | $/Lauf | Anteil | Cache | Freshness |
|---|---|---|---|---|---|---|---|---|---|---|
| Anthropic | messages/batch | deep_analysis (Sonnet 4.6) | 92 | 193 | 1402 / 1125 | 0,0127 | 1,169 | **88,7 %** | Analyse-Cache 0 % (276 Lookups, observe); Prompt-Cache-Read 48 % | REALTIME (Kandidat je Tag) |
| Anthropic | messages/batch | prescreening (Haiku 4.5) | 19,3 | – | 2041 / 1745 | 0,0054 | 0,104 | 7,9 % | – | REALTIME |
| Anthropic | messages/batch | A/B-Challenger (Haiku 4.5) | 6 | 17 | 2441 / 1532 | 0,0051 | 0,030 | 2,3 % | – | DAILY |
| Anthropic | messages/batch | shadow_relation (Haiku 4.5) | 8 | 23 | 1564 / 407 | 0,0018 | 0,014 | 1,1 % | – | DAILY |
| Tradier | REST (Quotes/Chains/Expirations) | Scanner | 1132 | – | – | Plan unbekannt (Brokerage, kein Preis je Request) | n/v | n/v | neu: Expirations-Dedup | REALTIME / DAILY |
| Finnhub | REST (News, Earnings) | Scanner | 584 (13–60 × 429) | – | – | 0 (Free) | 0 | 0 % | – | REALTIME |
| NewsAPI | everything | Scanner | 15 | – | – | 0 (Free, 100/Tag) | 0 | 0 % | – | REALTIME |
| Alpha Vantage | query | Scanner | 5 | – | – | 0 (Free, 25/Tag) | 0 | 0 % | – | DAILY |

Ergebnis:
- Kosten je Scanner-Tag: 1,32 $ (1,23 / 1,25 / 1,48 $).
- Monat: ≈ 29 $ bei 22 Handelstagen. Die Oktober-Projektion ab Telemetriestart am 06.10. liegt bei 25,04 $.
- **100 % der gemessenen Kosten sind Anthropic-Tokens.** Die Daten-APIs laufen im Free-Tier; Tradier hat keinen Preis je Request.
- Top-Treiber:
  1. Deep Analysis (Sonnet), davon ~75 % Output-Tokens;
  2. Prescreening;
  3. A/B-Stichprobe;
  4. shadow_relation.

## 2. Freshness-Klassen und Tiers
In `config/data_freshness.yaml` sind Klasse, `max_staleness_s` und Budget-Tier je Endpoint bzw. LLM-Workflow
festgelegt.
- Quotes, Chains, News und Earnings sind `REALTIME_OR_DAILY_CRITICAL` mit `max_staleness_s: 0`, werden also nie wiederverwendet.
- Expirations sind `DAILY` (6 h, ein Handelstag).
- Externe Registry-Quellen behalten ihre eigene Frequenz-, PIT- und `is_due`-Logik:
  - EIA/CFTC fragen schon jetzt nur nach einer neuen Veröffentlichung ab (Delta).
  - FRED/ALFRED wird bewusst nicht auf Delta umgestellt (Revisions-Vintages, PIT).

## 3./8. Zentraler Request-Cache und Deduplizierung
`modules/request_cache.py` ist ein prozess-lokaler Cache.
- **Schlüssel:** Provider + Endpoint + sortierte Parameter. Header und Keys fließen nie in den Schlüssel ein.
- **Gecacht** werden nur Endpoints mit `cache_in_run: true` und `max_staleness_s > 0`, heute nur Tradier-Expirations. Diese werden im selben Lauf von `alpha_sources` (Skew), `options_designer` (2×) und `market_snapshot` identisch abgefragt.
- **Invalidierung** erfolgt bei:
  - Ablauf von `max_staleness`;
  - neuem UTC-Tag;
  - gezielt je Symbol (Corporate Action);
  - beschädigtem Eintrag (dann Miss).
- Fehler werden nie gecacht. Werte werden als Kopie ausgegeben.
- Die Hit-Rate je Lauf steht im Ledger (`kind=request_cache`).
- Wirkung: weniger Tradier-Requests, in Dollar 0 $, da kein Preis je Request.

## 4.–7. Delta, Cheap-First, Universe V2, Source Health
Der Ist-Zustand ist geprüft; es gibt keinen Kosteneffekt in Dollar und deshalb keine Änderung.
- **Cheap-First:** Die Pipeline lädt Chains bereits erst für Kandidaten nach Prescreen und Deep Analysis.
- **Universe V2:** Die Discovery läuft wöchentlich (Sonntag); der tägliche V2-Scan nutzt den Snapshot.
- **Source Health:** Es gibt kleine Probes; Registry-Quellen liest Source Health aus der Health-Datei des Orchestrators und lädt keinen Datensatz neu.
- **Beobachtung ohne Dollar-Wirkung:** `imf_portwatch` lädt täglich ~650 k Rohzeilen, davon ~208 k Duplikate je Lauf. Das kostet Bandbreite, kein Geld; ein Kandidat für einen späteren `is_due`-Gate.

## 10./11. Budget und Projektion
- **Konfiguration** in `config/cost_policy.yaml`, Abschnitt `api_budget`:
  - `monthly_api_budget_usd = 10`
  - `monthly_soft_budget_usd = 10`
  - `daily_soft_budget_usd = 0,45`
- **Projektion:** MTD + Ø je Scanner-Tag × verbleibende Handelstage. Sie liefert `cost_today`, `cost_month_to_date` und `projected_month_cost`, jeweils auch nach Provider, Modul und Endpoint.
- **Wo die Projektion erscheint:**
  - im Montagsreport (`Projected monthly API cost`);
  - in `stats.cost_budget` des Tagesreports;
  - in `python -m modules.cost_telemetry budget`.
- **Verhalten bei Budgetdruck:**
  - Projektion über dem Ziel: Report und Warnung; zurückgestellt wird nur `TIER4_OPTIONAL`.
  - `TIER3_RESEARCH` wird erst an den bestehenden harten Grenzen (Woche 6 $ / Monat 25 $) zurückgestellt. Diese Grenzen sind unverändert.
  - `TIER1_PRODUCTION_REQUIRED` und `TIER2_IMPORTANT` werden nie budgetgedrosselt.
- **Warum die 10 $ nicht hart durchgesetzt werden:** Research und Shadow kosten zusammen ≈ 0,3 $/Monat. Sie ab ~Tag 8 jedes Monats zu stoppen, würde das Ziel nicht erreichbar machen, aber Evidenz verlieren.

## 12./13. Qualitätsschutz
Der Produktionspfad (LLM-Calls, Prompts, Modelle, Token-Limits, Gates) ist unverändert. Damit sind Kandidaten,
Final-MC, ROI und Trade-Entscheidungen trivial identisch. Für die einzige geänderte Abrufstelle (Expirations)
belegen Tests dasselbe Ergebnis mit und ohne Cache: gleiche Expiries, gleicher Kontrakt, gleiche Bid/Ask.
Chains und Quotes bleiben immer frisch.

## Was das 10-$-Ziel nur mit Qualitätsverlust erreichen würde
| Hebel | Effekt/Monat | Warum nicht umgesetzt |
|---|---|---|
| Deep Analysis auf Haiku | ≈ −15 $ | Auf fairer Basis (10 gültige Paare): Gate-Agreement 0,50, Pass-Recall 0,17. Haiku verwirft 5 von 6 Sonnet-Pässen |
| Bearish-Vorfilter (bestehende Policy, `auto`) | ≈ −6 $ (−23 % DA-Calls) | Verliert 6 von 167 Gate-Pässen (3,6 %); aktiviert sich nur nach den vorab registrierten Kriterien (min_days 10) – hier unverändert |
| Kürzere DA-Ausgabe | bis ≈ −8 $ | Output = ~75 % der DA-Kosten. Auswirkung auf Entscheidungen unbewiesen, bräuchte gepaarten A/B wie `prescreen_compact` |
| Prescreen kompakt (bestehende Policy) | ≈ −1 $ | Agreement 17/20 = 0,85 < 0,95 |
| Prompt-Cache 1 h TTL im Batch | −1,3 $ bis +1 $ | Read-Ratio heute 48 %; ohne Messung ungewiss. Ohne Qualitätseffekt, als gemessener Versuch möglich |

Minimal sicher erreichbar ohne Qualitätsverlust: ≈ 29 $/Monat (22 Handelstage), das ist der Ist-Stand.
Verbleibende Treiber: Sonnet Deep Analysis 89 %, Haiku Prescreen 8 %.
