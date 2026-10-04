# Finaler End-to-End-Audit – 2026-10-04 (nach UNIVERSE_V2 und Commodity Intelligence)

Grundlage:
- Codepfade (Producer → Artefakt → Consumer) und alle 15 Workflows samt **tatsächlicher GitHub-Laufhistorie**.
- Echte Artefakte im Repo (`outputs/`).
- Neue Integrationstests `tests/test_final_audit.py` (38) und `tests/test_commodity_intelligence.py` (47).
- Volle Suite (siehe §20).

Der Status bewertet nur technische und wissenschaftliche Funktionsfähigkeit, kein Alpha.

**STATUS: `AUTONOMOUS_READY_WITH_LIMITATIONS`** (Begründung in §1 und §22/§23)

---

## 1. SYSTEM STATUS

| Bereich | Befund (Stand 2026-10-04 ~08:30 UTC) |
|---|---|
| Kanonischer State | `outputs/state/system_state.json` (Schema **v2**). Felder: safe_mode, data_health, drift, model_health, champion, **universe_version, promotion_state, allowed_influence, commodity_data_health, learning_health**. Diese Ableitung lesen Scanner, Adapter, PromotionController, HC-Scanner und Wochenbericht. |
| Safe Mode | aus |
| Drift | **MODERATE**: `tnx` 5,28 liegt über dem p99 des Trainingsbereichs (4,78). Folge: Promotion pausiert, Demotion erlaubt. |
| Data Health | 30 HEALTHY, 9 UNVALIDATED, Data Quality 1,0 |
| Champion | `scanner_rules@<config-hash>`, kein ML-Champion |
| Produktionseinfluss Research | **keiner**: 5 Verträge PROSPECTIVE_CHALLENGER, Einfluss NONE |
| UNIVERSE | V1 produktiv (eingefroren, Hash `c9ac84a4…`, unverändert); V2 SHADOW, **erster Discovery-Lauf ausstehend** |
| Commodity | RESEARCH. WTI live (FRED, HEALTHY). EIA, Brent/Henry Hub/Metalle/Agrar (FRED) und CFTC sind seit dem Merge noch nicht live gelaufen; der manuell angestoßene Ingestion-Lauf ist in §7 dokumentiert. |
| LEARNING_HEALTH | siehe §10 – OK insgesamt; Gate learning / Universe V2 / Commodity UNVALIDATED (noch keine Live-Daten) |

**Warum nicht `AUTONOMOUS_READY`:**
- **V2-Discovery** ist noch nie live gelaufen. Der erste Cron war heute 07:13 UTC; GitHub verzögert Crons in diesem Repo um 4–6 h.
- **Final-MC-Survivor-Ledger, V2-Shadow-Scan und Kosten-Ledger** wurden nach ihren Merges (03.10.) noch von keinem Scanner-Lauf befüllt. Der erste Scanner-Lauf ist Montag, 05.10.
- **Branch-Schutz mit Code-Owner-Review** ist nicht nachweisbar aktiv: Die Rulesets-API liefert `[]`, und Merges liefen ohne Review.
- Die übrigen Einschränkungen sind nicht kritisch (§23).

## 2. AUTOMATION STATUS

Belegt durch die GitHub-Laufhistorie: Workflow, Cron, letzte Läufe, Ergebnis.

| Workflow | Cron (UTC) | real gestartet (Verzögerung) | letzte Läufe | Pflicht-Stufe | Concurrency |
|---|---|---|---|---|---|
| external_data | täglich 06:17 | ~11:40–13:20 (+5–7 h) | 29 Läufe, zuletzt grün | Ingestion | external-data-write |
| source_health | täglich 12:41 | ~16:39 (+4 h) | grün | Health Check | source-health-write |
| scanner | Mo–Fr 13:30 | ~18:30 (+5 h, vor US-Schluss 20:00) | 177 Läufe, 5/5 grün | pipeline.py, Feedback, Commit | history-write |
| feedback | Mo–Fr 15:30 + 19:30 | ~20:00 + 23:06 | 163 Läufe, grün | feedback.py, PromotionController | history-write |
| universe_v2 | So 07:13 | **noch kein Lauf** (Datei seit 03.10. auf main) | – | Discovery | universe-v2 |
| ml_research | Sa (weekly) + 3. des Monats (full) | grün; 2 rote Läufe am 02./03.10. (MergeError, behoben) | 15 Läufe | ML + Fabrik + Lab | research-write |
| research | 2. des Monats | grün | 3 Läufe | Event-Studie, Surprise, Memory | research-write |
| weekly_report | Mo 05:47 | erster Cron-Lauf am 05.10. (bisher 1 manueller, grün) | 1 Lauf | Bericht, Kalibrierung | research-write |
| monthly_report | 1. des Monats | grün (4×) | 4 Läufe | Bericht, Challenger | – |
| world_model | So 09:13 | bisher 1 manueller Lauf | 1 Lauf | World Model | research-write |
| alt_data | Fr 05:37 | grün; 1 roter Lauf (rückdatierte Entity-Änderung, im Folgelauf behoben) | 5 Läufe | SEC/TED | research-write |
| tests | push/PR | grün | 108 Läufe | Suite | je Ref |
| external_preflight, faf_build, repro | manuell | faf_build 3× rot (manuell, nicht im Lernpfad) | – | – | – |

Regelmäßig ohne manuelles Eingreifen läuft:
- V1-Prüfung: über SystemState (Hash-Vergleich je Ableitung).
- V2-Discovery: wöchentlich; noch nicht belegt.
- Optionsverfügbarkeit und Tradeability: V2 wöchentlich und im V2-Shadow-Scan täglich, V1 täglich im Scanner.
- EIA, FRED/ALFRED, CFTC: im täglichen External-Data-Lauf, abgerufen nur bei fälligem Release (`is_due`).
- Source Health und Scanner: täglich.
- Shadow und Counterfactual: täglich über feedback.py.
- Outcomes: 2× werktäglich.
- Feedback, PromotionController (Promotion und Demotion im selben Lauf): 2× werktäglich.
- Research: wöchentlich und monatlich.
- Hypothesen erzeugen und evaluieren: monatlich (full).
- Berichte: wöchentlich und monatlich.
- Kosten-Telemetrie: im Scanner bzw. bei LLM-Calls, seit #89 noch kein Scanner-Lauf.

**In diesem Audit behobene Automationsdefekte:**

| # | Defekt | Fehlerklasse | Fix |
|---|---|---|---|
| A1 | **P0** – Job-Limit Scanner 75 min. Real dauert der Scanner 53–55 min, danach liefen V2-Scan (bis 25 min) und Feedback VOR dem Commit. Ein Timeout hätte die history.json des Tages gekostet. | DEGRADED → Datenverlust | Neue Reihenfolge: Scanner → Feedback → Produktions-Commit → V2-Scan → V2-Commit → Stufen-Bilanz. Limit 110 min. |
| A2 | Workflows meldeten „grün“, obwohl zentrale Stufen (PromotionController, Fabrik, Research-Lab, Meta-Cognition, Kalibrierung, Memory-Sync, Challenger-Registrierung, V2-Scan) per `continue-on-error`/`\|\| true` ausfielen. | sonst still | `scripts/ci_stage_check.sh`: Nach dem Commit wird der Lauf ROT bei zentralen Stufen, `::warning` bei Nebenstufen. Alle `\|\| true` entfernt. |
| A3 | Parallele Workflows (6 Concurrency-Gruppen) schreiben auf main. Ein Rebase-Konflikt in `outputs/state/*` oder in einem gemeinsamen Ledger ließ den Push scheitern, und die Daten des Laufs gingen verloren. | RETRY fehlte | `scripts/ci_push.sh` (bis 5 Push-Versuche mit Backoff) und `.gitattributes` mit Union-Merge für Append-only-Ledger. Abgeleitete State-Dateien übernehmen die Version des Laufs. Jeder andere Konflikt führt zum Abbruch mit rotem Lauf (FAIL_CLOSED). Lokal mit echtem Konfliktszenario verifiziert. |
| A4 | Scanner-Workflow ohne `FRED_KEY`-Fallback | DEGRADED | `secrets.FRED_API_KEY \|\| secrets.FRED_KEY` |

**Verbleibend:**
- GitHub-Crons sind 4–6 h verspätet. Die Reihenfolge External → Health → Scanner ist bisher stabil, aber nicht garantiert. Abgesichert wird das durch das Snapshot-Höchstalter von 30 h und durch Fail-closed bei unbekannter Health.
- Es gibt keinen Watchdog, der einen ausgebliebenen Cron sofort meldet. Sichtbar wird er erst als **STALLED** in LEARNING_HEALTH (Wochenbericht §1).

## 3. DATA HEALTH

- Source Health prüft täglich alle Registry-Quellen und die Champion-Pflichtdaten: Erreichbarkeit, Auth, Frische, Schema, Missingness, Duplikate, Plausibilität und Downstream-Nutzung.
- Status: HEALTHY / DEGRADED / STALE / BROKEN / UNVALIDATED.
- **Nur abhängige Komponenten** werden UNAVAILABLE (`feature_contracts` je Feature).
- Commodity- und Research-Quellen tragen `research_only: true` und zählen **nie** für den globalen Safe Mode. Belegt durch Tests: alle vier Commodity-Quellen BROKEN → kein globaler Safe Mode.
- Ein CRITICAL-Marktdatenausfall blockiert `scanner_candidates` (Test).
- **Fix:** Model Health (Meta-Cognition), die älter als 14 Tage ist, gilt jetzt als **unbekannt → Safe Mode (fail-closed)**. Vorher galt eine beliebig alte `safe_mode.json` als aktueller Befund.

## 4. UNIVERSE V1

- Eingefroren in `outputs/universe/universe_v1_frozen.json`.
- Definitions-Hash `c9ac84a41d6a…`; `v1_unchanged() = True` wird bei jeder SystemState-Ableitung geprüft. Eine Abweichung ergibt einen Safe-Mode-Grund (neu).
- V1-Verträge: PROM-ABST-001…005@v1 unverändert gegenüber dem Baseline-Snapshot (Test `test_segment_contracts_valid_and_v1_untouched`).
- Der Candidate-Ledger trägt `universe_version=V1`. Neu: Der Decision-Record trägt ebenfalls `universe_version`.

## 5. UNIVERSE V2

- Dynamische Discovery: SEC-Listing → Tradier-Quotes und Chains → yfinance-Market-Cap.
- Wöchentlicher PIT-Snapshot, rollierend.
- Delisting und Suspendierung: STALE bei mehr als 5 Handelstagen ohne Trade → Ausschluss plus `v2_delisting_events.jsonl`.
- Outcomes: `DELISTED_WORST_CASE` mit −100 %. `CORPORATE_ACTION` (Split) wird gezählt, aber nicht als Evidenz.
- Fehlende Market Cap → `UNKNOWN`, nie 0. Research hat keine Cap-Untergrenze.
- **Ungeprüft live:** Der erste Lauf steht aus.
- Segmentweise Promotion (SMALL validiert, MICRO REJECTED) ist getestet, mit menschlicher Freigabe je Stufe.
- Eine globale V2-Promotion ist nicht möglich.

## 6. OPTIONABILITY / TRADEABILITY

- `optionable` heißt: mindestens 1 Verfall.
- `research_ok` sind großzügige Gates.
- `tradeable_ok` sind Produktions-Gates: OI, Volumen, Spread absolut und relativ, Strike-Dichte, Ausführungskosten ≤ 8 %, Risikoflags.
- **optionable ≠ tradeable**: Phase-6-Test (Micro-Cap, Research ok, nicht handelbar).
- Buckets: Market Cap, Liquidität, Options-Liquidität und Execution Quality sind getestet.

## 7. COMMODITY SOURCES

Erlaubt sind ausschließlich EIA v2, FRED/ALFRED und CFTC Public Reporting. Im Test geprüft: keine USDA-, Yahoo-, CME- oder ICE-Quellen in Konfiguration oder Code.

| SOURCE | SERIES | FREQUENCY | RELEASE LAG (Regel) | PIT SAFE | HEALTH (vor Live-Lauf) | CONSUMER | PRODUCTION IMPACT |
|---|---|---|---|---|---|---|---|
| eia_petroleum_weekly | WCESTUS1, WCRFPUS2, WCRIMUS2, WCREXUS2, WPULEUS3, WRPUPUS2, WGTSTUS1, WDISTUS1 | wöchentlich | Fr + 6 T 16:00 UTC | ja (Regel ≥ Release, ≤ Abruf; Revisionen ab Abruf) | UNVALIDATED | Fundamental-/Divergenz-Features, Fabrik | keiner |
| eia_petroleum_weekly | RWTC, RBRTE | täglich | + 9 T | ja | UNVALIDATED | nur Gegenprobe gegen FRED | keiner |
| eia_natural_gas | NW2_EPG0_SWO_R48_BCF | wöchentlich | Fr + 7 T | ja | UNVALIDATED | Gas-Features | keiner |
| eia_natural_gas | RNGWHHD | täglich | + 9 T | ja | UNVALIDATED | Gegenprobe | keiner |
| eia_natural_gas | N9070US2, N9133US2 | monatlich | + 100 T | ja; Erstimport als `backfill_latest_vintage` markiert | UNVALIDATED | Gas-Features | keiner |
| fred_regime_macro | DCOILWTICO (WTI) | täglich | ALFRED `realtime_start` | ja (Vintages) | **HEALTHY** | Regime, Commodity-Preis | keiner neu |
| fred_commodities | DCOILBRENTEU, DHHNGSP, PCOPPUSDM, PWHEAMTUSDM, PMAIZMTUSDM, PSOYBUSDM | täglich/monatlich | ALFRED | ja | UNVALIDATED | Preis-Features | keiner |
| cftc_cot | 8 Märkte (Code + Name) | wöchentlich (Di) | Fr 21:00 UTC; Feiertagswoche Mo | ja (Report-Datum ≠ Veröffentlichung) | UNVALIDATED | Positionierung | keiner |

- **Gold/Silber-Preis:** UNAVAILABLE (LBMA wurde aus FRED entfernt). Gold/Silber nur über COT.
- **Je Beobachtung gespeichert:**
  - `observation_time`, `available_at` (= RELEASE_TIME-Regel, gekappt auf den Abruf), `retrieved_at`, `unit`, `parser_version`.
  - Neu: `attrs.schema_version` und `attrs.release_time_rule` (EIA) bzw. `release_rule_time` (COT).
  - `availability_precision=CONSERVATIVE_DATE`. FRED: `EXACT_DATE` aus ALFRED.

**Live-Lauf 37188700178 (manuell angestoßen, gleicher Job wie der Cron):** siehe Nachtrag am Ende.

## 8. COMMODITY FEATURE ENGINE

- PIT je Stichtag (D 21:00 UTC). Je Periode gilt die jüngste Vintage mit `available_at ≤ Stichtag`.
- `age_days` je Reihe. Oberhalb der Frische-Grenze (Frequenz + Release-Lag) ist das Feature **UNAVAILABLE (NaN)**.
- Nie 0, nie neutral, nie unbegrenzt fortgeschrieben (Tests).

**RAW / STANDARDIZED / SURPRISE:**
- RAW: `chg_1w`, `chg_4w`.
- STANDARDIZED: `chg_z_52w`, `z_60d`, `vs_5y`.
- **ACTUAL SURPRISE: keine** – es gibt keine Konsens-Erwartung (`EXPECTATION_UNKNOWN`, Test: keine Surprise-Spalte).

**Feature-Gruppen:**
- PRICE: `ret_1/5/20/60d`, `mom_60_5`, `vol_20d`, `z_60d`, `dd_252d`; monatlich `ret_1m/3m`, `z_36m`.
- FUNDAMENTAL: Bestände (chg, z, vs. 5J), Flüsse (`chg_4w`, `z_52w`), Monat (`yoy`, `chg_3m`).
- POSITIONING: `mm_net`, `comm_net`, in % des OI, Δ 1/4/13 W, Perzentile 1J/3J, OI-Δ, `mm_extreme`.
- COT > OI wird verworfen.

**Leakage:** Phase 5 belegt es. Ein Signal, das mit `observation_time` perfekt korreliert (ρ < −0,99), verschwindet unter PIT (|ρ| < 0,25).

**Neu – Entscheidungszeitpunkt:**
- `commodity_intelligence.decision_snapshot/decision_features` liefert dieselben PIT-Merkmale in `candidate_env` (Adapter, Final-MC-Ledger) und im V2-Ledger.
- Ohne diese Verbindung hätte ein registrierter Commodity-Vertrag **nie prospektiv ausgewertet** werden können. Die Kette war vorher **nicht geschlossen** (P1).
- Gelesen wird es nur; ohne promoteten Vertrag sind die Entscheidungen identisch (Test mit Merkmalen = 1e6).

## 9. COMMODITY-EQUITY MAPPING

- `exposure-v1`, **NON_PIT** (`non_pit_mapping=true`).
- Gespeicherte Felder: commodity, ticker, exposure_type, expected_direction, mapping_source, mapping_version, confidence, point_in_time_status.
- Die Richtung ist explizit: Öl → Produzenten +1, Airlines −1, Raffinerien 0.
- **Neu:**
  - Commodity-Verträge müssen `mapping_version` fixieren (Teil des Spec-Hash).
  - Eine abweichende Mapping-Version macht die Regel **nicht auswertbar** (`fires → None`); eine neue Vertragsversion ist nötig.
- NON_PIT kann Promotion nie allein tragen: Promotion zählt **nur Forward-Beobachtungen**. Für diese war das Mapping zum Entscheidungszeitpunkt fixiert und bekannt.
- **Lücke (Daten, kein Defekt):**
  - Einzeltitel-Regeln decken nur wenige S&P-500-Titel ab; Branchenregeln greifen nur mit einer `industry`-Spalte.
  - Gruppen *Mining Equipment, Seeds* haben keine Regel.

## 10. LEARNING LOOPS

`modules/learning_health.py` ist neu und kanonisch in SystemState eingebunden, im Wochenbericht §1. Echter Stand:

| Pfad | Status | Detail |
|---|---|---|
| Outcome ingestion | OK | 119 geschlossen, 79 RELIABLE (Definition s. §17); letzte Schließung 30.09. |
| Shadow lifecycle | ACTIVE | 300 Shadow-Kandidaten, 66 mit Outcome |
| Gate learning (Final-MC-Survivor) | UNVALIDATED | Ledger leer (seit 03.10., erster Scanner-Lauf am 05.10.) |
| Research generation | OK | 24 Ideen |
| Hypothesis testing | OK | 35 getestet |
| Forward validation | NEED_MORE_DATA | 5 Challenger, **0 Decision-Ledger-Zeilen** (kein Champion-Trade seit Adapter-Start) |
| Promotion pipeline | OK | 5 × PROSPECTIVE_CHALLENGER |
| Demotion pipeline | OK | im selben Controller-Lauf (`_monitor`) |
| Calibration | NEED_MORE_DATA | prequentiell OOS n = 8 → UNCALIBRATED |
| Universe V1 | OK | |
| Universe V2 | UNVALIDATED | noch kein Snapshot |
| Commodity Intelligence | UNVALIDATED → RESEARCH | Feature-Store entsteht im ml_research-Lauf |
| Commodity Sources | UNVALIDATED | bis zum ersten Live-Lauf |
| RL | SHADOW | Veto aus |
| Research Memory | ACTIVE | 110 Einträge |

Lernende Komponenten (13 Fragen kompakt):

| Komponente | lernt was / woraus | Outcomes, Zeitraum, Population, V1/V2 | Leakage-Schutz | Output → Consumer | Prod.? | Promotion / Demotion | Commodity? |
|---|---|---|---|---|---|---|---|
| ML Research | 20-T-Querschnitt aus PIT-Panel | Labels fwd_xs_20, 2016–, PIT-S&P-500, V1 | präregistriertes Protokoll, Walk-Forward, Locked (kontaminiert markiert) | `ml_*` → Adapter-Karten (read-only), Bericht | nein | nur über Vertrag | ja, als Kandidat (`cmdx_*`) |
| Meta-Learning | wann Champion/Modelle funktionieren | Paper- und Panel-Outcomes, V1 | Cross-Fit, Gate | `meta_learning.json` → Bericht | nein | Gate (kein bestandenes) | nein |
| PPO / Robust RL | SKIP/NORMAL/BOOST | RELIABLE-Closed-Trades, V1 | Walk-Forward | Shadow-Meta → rl_promotion → Bericht | **nein** (Veto aus, Default jetzt fail-safe) | rl_promotion meldet nur | nein |
| Alpha Discovery | Gate-/Ledger-Muster | Candidate Ledger, V1 | BH, exploratorisch | Vorschläge → Challenger-Registrar | nein | Challenger + PR | nein |
| Research Director | Forschungsrichtungen | Memory, Ergebnisse | – | Kandidaten → Fabrik | nein | – | indirekt |
| Hypothesis Factory | falsifizierbare Hypothesen | Panel 2016–, V1 | Signal-Whitelist, BH, Ähnlichkeitssperre | Plan/Ergebnisse → Lab → Challenger | nein | Challenger → Vertrag (PR) | **ja** (9 Familien + 2 Divergenzen) |
| Surprise Engine | Fundamental vs. Reaktion | Earnings-Events | präregistriert, S5-Holdout prospektiv | Studie → Memory | nein | – | nur DATA_GAP |
| Causal / Unknown Unknowns | Strukturhypothesen | Panel | BH OOS | Bericht | nein | – | nein |
| Research Memory | was getestet/verworfen wurde | alle Ergebnisse | Fingerprint-Sperre | Fabrik-Priorität, Sperre | nein | – | **ja** (`commodity_entries`) |
| Commodity Intelligence | BASE vs. BASE+COMMODITY | Panel, gemappte Titel, V1 | PIT-Engine, NON_PIT-Flag, Bonferroni | `commodity_evaluation.json` → Memory, Bericht | nein | Vertrag (max. SCORE_LIMITED) | – |
| PromotionController | Forward-Evidenz je Vertrag | Decision-, Final-MC-, V2-Ledger; RELIABLE only | nur ≥ forward_start, Spec-Hash, eingefroren | `promotion_state.json` → Adapter | **einzige Stelle** | Leiter + Demotion | ja (Verträge) |
| feedback (Bins, Pearson) | QuasiML-Gewichte | RELIABLE-Closed | Fisher-KI, MIN_EFF_N | `model_weights` → **kein Entscheidungs-Abnehmer** | nein | – | nein |

## 11. SHADOW / COUNTERFACTUAL

Pfad: Final-MC-Survivor → `final_mc_downstream` (fail_gates je Gate) → Outcome (20/45/60 T) → Gate-Effektivität je Gate.
- Gates: ROI, Edge, MC-P&L, Liquidity, Spread, OI, DTE, Risk, Intelligence.
- Code und Tests vorhanden (`test_final_mc_ledger`).
- **Live noch leer** (s. §1).
- Shadow-Trades (300, Archiv append-only) und Counterfactual-Trades mit gleichem Lebenszyklus wie echte Trades.
- Es werden keine Gate-Schwellen auf kleinem N geändert: Die Promotion-Floors verlangen N ≥ 60–100.

## 12. HYPOTHESIS ENGINE

Problem → Idee → falsifizierbare Hypothese (H0/H_alt, Population, Horizont, Baseline, OOS, BH) → Lab-Walk-Forward → Robustheit → PROSPECTIVE_CHALLENGER (eingefroren) → Forward-Ledger → Vertrag per PR → PromotionController.
- Die LLM darf nur den Mechanismus-Text liefern.
- Ähnlichkeitssperre und Memory verhindern Recycling, auch für Commodity-Ideen (Test).

## 13. PROMOTION

Einziger Pfad: Evidence → Contract → OOS → Prospective Forward → PromotionController → ProductionIntelligenceAdapter.

**Bypass-Suche** (Config-Writes, Score-Mutation, Gate-Änderung, Veto, Modellgewichte, Universe-Promotion, Commodity-Boost):

| Fundstelle | Bewertung |
|---|---|
| `feedback.py` → `history["model_weights"]` | wirkt nur auf QuasiML-`final_score` → **kein Entscheidungs-Abnehmer** (SHADOW) |
| `pipeline.py` RL-Veto, Default `True` bei fehlendem Config-Schlüssel | **behoben** (Default `False`) |
| Research-Module → `config*.yaml` | keine Schreibzugriffe (Test `test_research_components_cannot_write_production_config`) |
| Commodity → Produktion | nur Lesen in `candidate_env`; wirkt nur über promoteten Vertrag (Test identischer Entscheidungen) |

**Controller-Defekte behoben:**
1. Lag die Policy-Kappung (ABSTENTION_ONLY) unter der ersten Stufe einer Rerank- oder Score-Leiter, setzte der Controller GUARDED_PRODUCTION mit einer Stufe, die der Adapter für diese Klasse nie anwendet. Das war ein irreführender State; jetzt bleibt der Vertrag FORWARD_VALIDATED mit Empfehlung.
2. Für Rerank-, Score- und Weight-Verträge, also alle Commodity-Verträge, gab es **keinen** menschlichen Freigabeweg. Jetzt gilt `promotion_approvals.yaml`:
   - je Look eine Stufe, nur bei aktuell erfüllter Evidenz;
   - Commodity höchstens SCORE_LIMITED, sonst höchstens WEIGHT_10.
   - Phase-3-Test: Freigabe WEIGHT_25 wird auf die Leiter gekappt → RERANK_ONLY.

## 14. DEMOTION

- Vorab im Vertrag fixiert: rollierende Expectancy, ECE, Abstinenz-Nettowert, Segment-Kriterien.
- Je Verstoß eine Stufe nach unten; bei Effektumkehr (H_alt) DEMOTED/REJECTED; bei erschöpften Looks EXPIRED.
- Integritätsfehler → ROLLBACK.
- Test `test_full_demotion_ladder…`:
  - WEIGHT_10 → RERANK_ONLY → NONE (SHADOW) → terminal.
  - Kalibrierungskollaps (ECE 0,45) demotiert real.
- Für Equity (Phase 2/4) und Commodity (Phase 3/4) belegt.

## 15. CALIBRATION

- **RAW** (MC-Hit-Rate) ≠ **CALIBRATED** (prequentiell, nur vorher geschlossene Trades) ≠ **EMPIRICAL** (Band-Win-Rate).
- OOS n = 8 → in Mails „UNCALIBRATED“/„nicht verfügbar“, nie die rohe Zahl als Wahrscheinlichkeit.
- Vertrags-Demotion bei ECE > 0,10 (Test).
- V2- und Bucket-Kalibrierung über Brier/ECE je Segment im V2-Ledger. Bisher keine Daten.

## 16. LEAKAGE / PIT

| Risiko | Status |
|---|---|
| Look-ahead, Release-Time (EIA, COT) | PIT-Engine; Tests Phase 5, COT Fr/Mo-Feiertag, EIA Mi/Do |
| Revision (FRED/ALFRED, EIA) | ALFRED-Vintages; EIA-Revisionen ab Abruf; Monats-Backfill markiert |
| Survivorship / Delisting | V2: Snapshot-Diff + Worst-Case-Outcome. ML-Panel: 135 von 255 entfernten Titeln ohne Kurse (offen, ausgewiesen) |
| Current Sector / Market Cap / Mapping | NON_PIT gekennzeichnet; Market Cap ab erstem V2-Snapshot PIT |
| Holdout-Reuse | ML-Locked **CONTAMINATED** markiert; S5-Holdout prospektiv |
| Multiple Testing | BH (Lab, Fabrik), Bonferroni (Controller-Familien, Alt-/Commodity-Ablation) |
| Post-hoc-Schwellen / Vertragsänderung | Spec-Hash, Registry-Kette, Änderung = neue Version (Test) |
| Research-Memory-Kontamination | append-only, Fingerprint-Sperre |

## 17. EXECUTION REALITY

- THEORETICAL (Mid→Mid) ≠ REALIZABLE NET: Ask + Slippage beim Kauf, Bid − Slippage beim Verkauf, Kommission.
- **Phase-6-Test:**
  - Rohrendite +30 %, theoretisch +20 %, netto negativ.
  - MICRO wird **nicht** promotet, Basis `net_realizable_return`.
- **Outcome-Klassen** (`outcomes.outcome_class`, neu): 119 Closed = 79 RELIABLE (davon nur **2** mit dokumentierter Quote-Methode, **77 Altbestand ohne Methode**) + 40 RECONSTRUCTED.
- Strikt wären die 77 **UNKNOWN** → OWNER DECISION (§22). Shadow: 66 bewertete Altfälle ohne Methode = UNKNOWN (strikt); neue Shadow-Outcomes tragen `outcome_method`.

## 18. REPRODUCIBILITY

Der Decision-Record speichert jetzt zusätzlich:
- `universe_version`, `config_version`, `contract_hashes`, `prompt_version`;
- `commodity_features_used`, `commodity_data_version`;
- `non_reproducible_inputs` = [`live_option_chain`, `llm_response`, `live_quotes`].

Bereits vorher gespeichert wurden code_commit, system_state_version, Hypothesen-Spec-Hashes, env_hash und Gate-Ergebnisse. Der Repro-Lauf (`repro.yml`) ergab früher 604 Kernwerte mit 0 Abweichungen.

## 19. COSTS / SCALE

| Messgröße | Wert |
|---|---|
| Scanner | 53–55 min (MEASURED, GitHub). V1-Universum 344–405 Kandidaten/Tag nach Hard-Filter. |
| V2 | Discovery-Budget 50 min/Woche (rollierend), Shadow-Scan ≤ 150 Chain-Abfragen/Tag, ≤ 25 min, LLM-frei. Ultra-/Micro-Caps können den Produktionslauf nicht verlangsamen (eigener Schritt nach dem Produktions-Commit). **Laufzeit noch nicht gemessen.** |
| Commodity | `is_due` → typischerweise 1 CFTC-Call/Woche, 14 EIA-Serienabrufe/Woche, 6 FRED-Calls/Tag. Entscheidungs-Snapshot ~0,6 s. |
| Kosten-Telemetrie | MEASURED vs. ESTIMATED getrennt (#91). **Noch kein Ledger**: seit #89 kein Scanner-Lauf → UNVALIDATED. |
| Actions-Minuten | Scanner ~60 min/Werktag + V2 ≤ 25 min; ml_research 20–40 min/Woche |

## 20. TESTS

- `tests/test_final_audit.py` (38):
  - Phasen 1–6 über den echten Controller, Adapter und die Ledger;
  - volle Demotion-Leiter, Vertragsunveränderlichkeit, V1/V2/Commodity-Trennung;
  - abgeschnittene Ledger-Zeile, atomares Schreiben, korrupte history.json fail-closed, abgeschnittenes Transition-Log fail-closed, Neustart-Idempotenz;
  - Ausfall-Matrix, NaN/extreme/fehlende Signale, RL-Veto-Default, absurde COT-/EIA-Werte;
  - kanonische State-Felder, veraltete Model Health, STALLED-Erkennung, Outcome-Klassen;
  - Workflow-Invarianten: Cron, keine stillen Fehler, sicherer Push, Concurrency.
- `tests/test_commodity_intelligence.py` (47), inklusive Verhaltenstest „keine Produktionswirkung“.
- Volle Suite: siehe Nachtrag.

Abdeckung der 30 geforderten Szenarien:
- 1: test_pipeline_orchestration
- 2: Stop-Reasons in test_pipeline
- 3: test_universe_v2 discovery
- 4: Phase 6
- 5, 6: test_universe_v2
- 7, 8: test_universe_v2 corporate action / delisting
- 9–16: test_commodity_intelligence
- 17: Mapping-Tests
- 18: Phase 3
- 19: Ausfall-Matrix
- 20, 21: test_system_state drift
- 22: test_promotion Safe Mode
- 23, 24: Phasen 2–4
- 25: Locked CONTAMINATED (test_ml_research/protocol)
- 26: ECE-Demotion
- 27: RL-Veto
- 28: test_promotion Counterfactual
- 29: Shadow-Archiv
- 30: Neustart-Idempotenz

## 21. DEAD / UNUSED COMPONENTS

| Komponente | Status |
|---|---|
| QuasiML / `model_weights` / feature_stats | SHADOW, lernt ohne Entscheidungs-Abnehmer |
| PPO-Veto | UNUSED (Veto aus) |
| `feature_stats_external` | SHADOW (nur Engine-Monitor-Warnungen) |
| `faf_build.yml` | manuell, 3× rot, nicht im Lernpfad → UNUSED/BROKEN (Owner) |
| Modul-Importgraph | alle Module referenziert (kein DEAD-Code-Modul) |
| `datetime.utcnow()` (32 Stellen) | naive UTC, konsistent, deprecated – nicht geändert |
| `except Exception` ohne Begründung | 267 Stellen (geloggt) – kein stiller Datenverlust gefunden, aber Risiko |

## 22. OWNER ACTIONS / OWNER DECISIONS

1. **Branch-Schutz „Require review from Code Owners“ auf main aktivieren.** Merges laufen derzeit ohne Review; CODEOWNERS wirkt nicht.
2. **Secrets prüfen:** `EIA_KEY` (neu), `FRED_API_KEY` oder `FRED_KEY`. Ohne EIA-Key bleiben die EIA-Features UNAVAILABLE.
3. **OWNER DECISION Outcomes:** Die 77 Alt-Outcomes ohne dokumentierte Methode zählen derzeit als RELIABLE (Entscheidung vom 03.10.); strikt wären sie UNKNOWN. Eine Änderung verkleinert die Lernbasis von 79 auf 2.
4. **OWNER DECISION Drift:** Drift MODERATE (nur `tnx` außerhalb des Trainingsbereichs) pausiert **jede** Promotion. Bleibt `tnx` hoch, findet monatelang keine Promotion statt.
5. **OWNER DECISION Freigaben:** Rerank-, Score- und Commodity-Verträge über NONE nur per `config/promotion_approvals.yaml`. `max_automatic_influence` bleibt ABSTENTION_ONLY. Policy, Schwellen, Forward-Starts und Tradeability-Gates wurden **nicht** verändert.
6. **ci_push-Konfliktregel bestätigen:** Abgeleitete State-Dateien nehmen die Version des Laufs; alles andere wird rot.
7. `faf_build.yml` reparieren oder entfernen.
8. **Erste Live-Läufe kontrollieren:**
   - V2-Discovery heute;
   - Montag 05.10.: Scanner mit Final-MC-Ledger, V2-Scan, Kosten-Ledger und Decision-Ledger.

## 23. REMAINING RISKS

- **Keine Forward-Evidenz:** Decision-Ledger 0 Zeilen, Final-MC- und V2-Ledger leer. Promotion ist frühestens nach N ≥ 60–100 und 90 Kalendertagen möglich.
- **Champion ohne belegten Vorteil:** n = 79, PF 1,12. Die MC-Wahrscheinlichkeit ist fehlkalibriert.
- **EIA/CFTC:** Feldnamen und Serien-IDs nicht live verifiziert (siehe Nachtrag). Bei Abweichung meldet der Konnektor laut SCHEMA_CHANGED/UNAVAILABLE.
- **Exposure-Mapping:** NON_PIT und dünn → Commodity-Ablation im geschützten Alt-Protokoll voraussichtlich REJECT (Abdeckung).
- **GitHub-Crons:** 4–6 h Verzögerung, kein sofortiger Watchdog.
- **Push-Konflikte** in nicht abgeleiteten Dateien führen weiterhin zu einem roten Lauf, ohne Datenverlust, aber mit nötigem Re-Run.
- **ML-Panel:** Survivorship-Rest; Sektor nur teilweise PIT.
- **V2-Laufzeit und Rate-Limits** sind unbekannt bis zum ersten Lauf.

---

## Abschlusstabellen

### CAPABILITY | WORKING | AUTOMATED | TESTED | FORWARD VALIDATED | PRODUCTION IMPACT | COMMENT
| Capability | Working | Automated | Tested | Forward validated | Production impact | Comment |
|---|---|---|---|---|---|---|
| V1-Scanner (Gates, LLM, MC, Options, ROI) | ja | ja (täglich) | ja | **nein** (PF 1,12) | ACTIVE | Champion |
| Source Health / SystemState | ja | ja | ja | – | ACTIVE (Steuerung) | jetzt inkl. Universe/Commodity/Influence/Learning |
| Outcome-Rückführung | ja | ja (2× werktäglich) | ja | – | Evidenzbasis | nur RELIABLE lernt |
| Shadow/Counterfactual | ja | ja | ja | ACCUMULATING | keine | Final-MC-Ledger startet 05.10. |
| PromotionController / Adapter | ja | ja | ja (E2E) | 0 Verträge | einzige Stelle, Einfluss NONE | Leiter + Demotion belegt |
| Hypothesis Factory / Lab / Memory | ja | ja (monatlich) | ja | 0 bestätigt | keine | Commodity-Domäne aktiv |
| ML / Meta / World / Causal | ja | ja | ja | nein | keine | SHADOW |
| RL (PPO, robust) | ja | ja | ja | degeneriert | keine | Veto aus (fail-safe) |
| UNIVERSE_V2 Discovery | Code ja | Cron ja | ja | – | keine | **erster Live-Lauf ausstehend** |
| V2 Shadow-Scan / Segment-Verträge | Code ja | ja (täglich) | ja | 0 | NONE (je Segment Mensch) | erster Lauf 05.10. |
| Commodity-Konnektoren | ja | ja (täglich, `is_due`) | ja | – | keine | Live-Lauf s. Nachtrag |
| Commodity-Features / Mapping | ja | ja (wöchentlich) | ja | – | keine | NON_PIT |
| Commodity → Vertrag → Forward | **jetzt** ja | ja | ja (Phase 3) | 0 | max. SCORE_LIMITED per Freigabe | vorher nicht geschlossen |
| Kosten-Telemetrie | Code ja | ja | ja | – | Betrieb | noch kein Ledger |
| Crash-Recovery / Idempotenz | ja | – | ja | – | – | atomare Writes neu |

### SOURCE | HEALTH | PIT SAFE | USED BY | PROD IMPACT
| Source | Health | PIT safe | Used by | Prod impact |
|---|---|---|---|---|
| Tradier / yfinance (Kurse, Chains) | HEALTHY | live | Scanner (CRITICAL), V2 | ACTIVE |
| VIX / Risk Gates | HEALTHY | live | Scanner | ACTIVE |
| fred_regime_macro (inkl. WTI) | HEALTHY | ja (ALFRED) | Regime, ML-Panel, Commodity-Preis | indirekt (Drift/Regime) |
| fred_us/world_macro, Eurostat, ENTSO-E, NOAA … | überwiegend HEALTHY | ja/konservativ | External Context (SHADOW) | keine |
| SEC (Form 345, Submissions, XBRL) | HEALTHY | ja (Filing-Datum) | Alt-Data (REJECTED) | keine |
| eia_petroleum_weekly / eia_natural_gas | UNVALIDATED | ja | Commodity | keine |
| fred_commodities | UNVALIDATED | ja | Commodity | keine |
| cftc_cot | UNVALIDATED | ja | Commodity | keine |

### UNIVERSE/BUCKET | N | OPTIONABLE | TRADEABLE | FORWARD N | STATUS
| Universe/Bucket | N | Optionable | Tradeable | Forward N | Status |
|---|---|---|---|---|---|
| V1 (S&P 500 + Nasdaq-100, Hard-Filter) | 344–405/Tag | – | Champion-Gates | 0 (Decision-Ledger) | ACTIVE, eingefroren |
| V2 ULTRA_MICRO | n. v. (kein Snapshot) | n. v. | n. v. | 0 | SHADOW |
| V2 MICRO | n. v. | n. v. | n. v. | 0 | SHADOW |
| V2 SMALL | n. v. | n. v. | n. v. | 0 | SHADOW |
| V2 MID | n. v. | n. v. | n. v. | 0 | SHADOW |
| V2 LARGE / MEGA | n. v. | n. v. | n. v. | 0 | Referenz (kein Segment-Vertrag) |

### COMMODITY FEATURE | EQUITY POPULATION | OOS STATUS | FORWARD STATUS | INFLUENCE
| Commodity feature | Equity population | OOS status | Forward status | Influence |
|---|---|---|---|---|
| `cmdx_oil__*` (WTI-Rendite/z, Lager, Nachfrage, Positionierung, Divergenz) | Öl-Produzenten +, Services +, Airlines/Trucking −, Raffinerien 0 | nicht bewertet (erste Ablation im nächsten full-Lauf) | keine | NONE |
| `cmdx_natural_gas__*` | Gas-Produzenten +, Chemie/Dünger/Versorger − | nicht bewertet | keine | NONE |
| `cmdx_copper__*` | Minen +, Elektroausrüster − | nicht bewertet | keine | NONE |
| `cmdx_gold__*`, `cmdx_silver__*` (COT) | Minen, Royalty + | nicht bewertet | keine | NONE |
| `cmdx_corn/wheat/soybeans__*` | Landmaschinen, Pflanzenschutz +; Lebensmittel −; Verarbeiter 0 | nicht bewertet | keine | NONE |

Status: **Commodity Intelligence: RESEARCH ONLY – NO VALIDATED INCREMENTAL ALPHA.**

---

## Abschlussfragen

1. **Kann das Repo ohne manuelle Eingriffe regelmäßig laufen?** Ja. Alle Kernjobs haben Crons, und die Laufhistorie belegt tägliche grüne Läufe. Einschränkung: Crons starten 4–6 h verspätet, und es gibt keinen Watchdog für ausgebliebene Läufe (sichtbar erst als STALLED).
2. **Ist der gesamte US-Optionsmarkt technisch korrekt abgedeckt?** Der Code ja (SEC-Listings → Chains → Buckets, getestet). Live **noch nicht belegt**, weil der erste V2-Lauf aussteht.
3. **Sind optionable und tradeable sauber getrennt?** Ja: getrennte Gates, Risikoflags, Phase-6-Test.
4. **Bleibt V1 vollständig reproduzierbar?** Ja: eingefrorener Definitions-Hash, unveränderte V1-Verträge, Prüfung in jeder SystemState-Ableitung.
5. **Bleibt V2 vollständig von V1-Evidence getrennt?** Ja. Getrennte Ledger und Verträge; der Adapter filtert nach Population (Tests).
6. **Werden Micro- und Ultra-Micro-Caps sauber behandelt?** Ja:
   - Research ohne Cap-Untergrenze;
   - Produktion nur über Netto-Segment-Verträge mit menschlicher Freigabe;
   - Penny- und Manipulationsflags.
7. **Sind EIA, FRED/ALFRED und CFTC stabil integriert?** Code und Tests ja; FRED-WTI live HEALTHY. EIA, die neuen FRED-Reihen und CFTC sind live erst mit dem Lauf aus dem Nachtrag belegt.
8. **Werden EIA- und COT-Release-Zeiten korrekt berücksichtigt?** Ja, mit konservativen Regeln inklusive Feiertagen (Tests).
9. **Gibt es Commodity Look-Ahead?** Kein bekannter. Phase 5 beweist, dass die PIT-Engine Lookahead-Scheinalpha eliminiert.
10. **Gibt es PIT-Probleme beim Commodity→Equity-Mapping?** Ja, systembedingt: Das Mapping ist NON_PIT. Es ist gekennzeichnet, in Verträgen versioniert, und Promotion zählt nur Forward-Daten.
11. **Werden Commodity-Features wirklich von Research konsumiert?** Ja:
   - Fabrik (9 Familien), Lab-Signalsprache, ML-Kandidaten, Alt-Ablation, Commodity-Ablation, Memory.
   - Erste echte Werte erst nach Ingestion und ml_research.
12. **Gibt es Commodity-Code ohne Consumer?** Nein. Jetzt auch Entscheidungs-Env und V2-Ledger. Die EIA-Spots sind bewusst nur Gegenprobe.
13. **Kann Commodity Research die Produktion umgehen?** Nein. Wirkung nur über einen Vertrag → Forward → Controller (max. SCORE_LIMITED, Freigabe je Stufe). Verhaltenstest: Entscheidungen identisch.
14. **Werden echte Outcomes automatisch zurückgeführt?** Ja, zweimal werktäglich (feedback.yml).
15. **Lernt mindestens ein System tatsächlich daraus?** Ja:
   - PromotionController (Forward-Evidenz, Demotion);
   - abstention_proposals;
   - Feature-Bins (SHADOW);
   - robuster PPO (SHADOW).
   - Produktionswirksames Lernen ist bisher nicht eingetreten (keine Forward-Daten).
16. **Werden Gate-Rejects weiter beobachtet?** Ja: Shadow-Trades und Archiv, Counterfactual-Trades, Final-MC-Survivor-Ledger (startet 05.10.).
17. **Kann das System erkennen, wenn ein Gate Alpha vernichtet?** Ja, per Gate-Effektivität je Gate (Final-MC-Ledger) und Alpha-Discovery-Gate-Efficacy. Bisher zu wenig Daten.
18. **Kann V2 selbst lernen, welche Cap- und Liquidity-Buckets funktionieren?** Ja: Segment-Verträge je Bucket auf Nettobasis, Breakdowns im Ledger.
19. **Können Commodity-Beziehungen nach Market-Cap-Bucket untersucht werden?** Ja, neu: Der V2-Ledger trägt Commodity-Exposure und -Features je Kandidat. Die Ablation zerlegt nach Bucket, Sektor, Exposure, Regime und Liquidität (Market Cap nur in V2).
20. **Können neue Research-Ergebnisse kontrolliert produktiv werden?** Ja:
   - Abstinenz automatisch bis ABSTENTION_ONLY;
   - Rerank, Score, Weight und Segmente nur per Freigabe je Stufe (neu für Rerank/Score/Weight).
21. **Können schlechte Erkenntnisse wieder demotiert werden?** Ja, die volle Leiter ist im Test belegt.
22. **Kann ein fehlerhaftes ML-, RL- oder Commodity-Modell Schaden verursachen?** Nein. Kein Modell hat Einfluss. Caps, Safe Mode und Rerank bei fehlendem Signal lassen die Champion-Reihenfolge stehen; das RL-Veto ist aus (jetzt auch per Default).
23. **Sind Wahrscheinlichkeiten ehrlich kalibriert?** Sie werden ehrlich als **UNCALIBRATED** ausgewiesen (OOS n = 8). Die rohe MC-Zahl wird nicht als Wahrscheinlichkeit gezeigt.
24. **Sind Production Decisions ausreichend reproduzierbar?** Ja, mit Versionen, Hashes, State-Version und Env-Hash. Nicht exakt reproduzierbar (markiert): Live-Chains, LLM-Antworten, Live-Quotes.
25. **Funktionieren Promotion UND Demotion End-to-End?** Ja, synthetisch über echte Pfade. Mit echten Daten noch nicht eingetreten.
26. **Funktioniert Crash Recovery?** Ja, jetzt:
   - atomare State- und History-Writes;
   - Ledger mit abgeschnittener Zeile lesbar;
   - korrupte history.json fail-closed;
   - Transition-Log fail-closed.
27. **Sind Jobs idempotent?** Ja: Controller-Neustart, Final-MC-, V2- und Commodity-Ingestion (`is_due`/Dedupe) sowie Archive append-only.
28. **Gibt es Survivorship Bias?** Rest im ML-Panel (ausgewiesen); V2 über Snapshot-Diffs.
29. **Gibt es Delisting Bias?** In V2 nein (Worst-Case-Outcome). Im ML-Panel teilweise.
30. **Gibt es Revision Leakage?** Kein bekanntes: ALFRED-Vintages, EIA-Revisionen ab Abruf, Monats-Backfill markiert.
31. **Gibt es Release-Time Leakage?** Kein bekanntes (Tests EIA/COT).
32. **Gibt es Holdout Contamination?** Ja, bekannt und markiert: Das ML-Locked-Fenster ist CONTAMINATED; Bestätigung nur prospektiv.
33. **Gibt es Komponenten, die nur scheinbar intelligent sind?** Ja: QuasiML/Pearson-Gewichte (lernen ohne Abnehmer), PPO (degeneriert), Final-MC-Hit-Rate als „Wahrscheinlichkeit“ (fehlkalibriert). Alle sind als SHADOW/UNCALIBRATED markiert.
34. **Ist Commodity Intelligence aktuell RESEARCH, SHADOW oder validiert?** RESEARCH.
35. **Ist Universe V2 aktuell SHADOW oder teilweise promotet?** SHADOW. Kein Segment ist über NONE.
36. **Welche neuen Daten müssen jetzt hauptsächlich nur noch gesammelt werden?**
   - Decision-Ledger-Zeilen echter Champion-Trades;
   - Final-MC-Survivor-Outcomes;
   - V2-Netto-Outcomes je Bucket;
   - EIA- und COT-Historie;
   - prequentielle Kalibrierungs-Outcomes (≥ 30).
37. **Was verhindert aktuell am stärksten echten Lernfortschritt?**
   1. Zu wenige echte Forward-Beobachtungen: wenige Champion-Trades, keine Decision-Ledger-Zeilen.
   2. Drift MODERATE pausiert jede Promotion (Owner-Entscheidung).
   3. Das Exposure-Mapping deckt wenige Titel ab.
38. **Welche Änderungen sollte man jetzt ausdrücklich NICHT mehr vornehmen?**
   - Keine neuen Intelligence-Module, Datenquellen oder Modelle.
   - Keine Änderung an Promotion-Kriterien, Forward-Starts, Verträgen, Gate-Schwellen, Risk-Limits oder der Champion-Definition.
   - Kein Lockern von `max_automatic_influence`.
   - Kein Aktivieren des RL-Vetos.
   - Keine Umdeutung von Altdaten zur „Beschleunigung“.

---

## Nachtrag: Live-Verifikation
(wird nach Abschluss der Läufe ergänzt)
