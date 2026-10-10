# Expectation Alpha (EA) – Architektur (V1, SHADOW)

Ziel: Messen, ob ein datengetriebener Erwartungs-Kontext den bestehenden News/Event-Alpha **inkrementell** verbessert:
- besser filtern;
- besser timen;
- besser ranken;
- schlechte Trades vermeiden;
- später eventuell eigenständiges Macro-Alpha liefern.

Jede Komponente ist eine **Hypothese**, kein Wissen. V1 verändert keinen Champion-Pfad.

```
NEWS/EVENT ENGINE (bestehend)             EXPECTATION ALPHA ENGINE (neu, SHADOW)
Deep Analysis: direction/impact/          PIT-Daten (Archiv + Marktschlüsse)
surprise/catalyst/ttm                     -> current state -> rate of change -> model future state
        |                                 -> market-implied expectation -> expectation gap
        |                                 -> regime (World Model) -> cross-asset confirmation
        v                                            |
   NEWS_EDGE ------------------+---------------------+
                               v
           DECISION INTELLIGENCE (Research-Label, deterministisch, kein LLM)
           NEWS_EDGE · EXPECTATION_CONTEXT · CONFIRMATION · TIMING
           -> TRADE | WAIT | ABSTAIN | ERROR   (+ wait_trigger, kill_conditions, expressions)
                               |
           append-only EA-Ledger -> Outcomes 20/60/120/250 Handelstage je Expression
           -> Evaluation (Gruppen A–D, Abstention-, Context-, Expression-Wert)
           -> EA001–EA007 (Verträge, Population EA_NEWS_CANDIDATE) -> PromotionController
```

## 1. Wiederverwendete Komponenten (keine Parallelarchitektur)
| Bedarf | Bestehend | Nutzung |
|---|---|---|
| PIT-Makro mit Vintages | `modules/external/archive.ExternalArchive`, ALFRED-Connectoren, `world_model.pit_snapshots` (`available_at < Stichtag`) | unverändert; neue Reihen nur als zusätzliche FRED-Quelle im bestehenden Connector-Muster |
| Indikatoren, z-Score ohne Look-ahead, Regime-Dimensionen | `world_model.market_frame`, `macro_frame`, `expanding_z`, `world_states`, `build_world` | Regime-State und Unsicherheit **aus dem World Model**; kein zweites World Model |
| Commodity-Fundamental vs. Preis | `commodity_intelligence.build` / `decision_snapshot` (PIT, Frische, Qualität) | Öl-Gap aus Lager-vs-5J und WTI-z |
| Kandidaten + Deep-Analysis-Ergebnis | `pipeline.py` Stufe 4, `candidate_ledger` | EA liest das Ergebnis read-only nach Stufe 4 |
| Population-Ledger, Outcomes, Evidenz, Block-Bootstrap über Tage und Cluster | `final_mc_ledger` (Vorlage), `promotion_controller.group_metrics/_delta/_outlier_trims` | gleiches Interface (`read_rows`, `read_outcomes`, `evidence`), eigene Population |
| Verträge, Registrierung, Hash-Kette, H0/H_alt | `hypothesis_contract` (`config/promotion_hypotheses.yaml`, `contract_registry.jsonl`) | EA001–EA007 als normale Verträge; neue `eligible_stage` EA_NEWS_CANDIDATE |
| Gatekeeper | `promotion_controller` + `cap_population` (Nicht-Champion-Population: höchstens FORWARD_VALIDATED, Einfluss NONE) + Adapter, der nur Champion-Verträge anwendet | unverändert; EA kann strukturell keinen Champion-Trade verändern |
| Safe Mode / Datenstatus | `source_health.effective_safe_mode`, SystemState | als Kontext im Ledger |
| Mail, Montagsbericht | `email_reporter`, `reports/weekly.py` | je ein klar markierter SHADOW-Abschnitt |
| Atomare, append-only IO | `modules/atomic_io` | alle EA-Schreibvorgänge |
| Outcome-Lauf | `feedback.py` (wie Final-MC/V2) | EA-Outcomes im selben Lauf |

## 2. Echte Lücken
1. **Marktimplizite Erwartungen fehlen im Archiv.** Es gibt keine Breakevens, keine 2J-Rendite und keine Fed Funds.
   - Neu ist die FRED-Quelle `fred_market_expectations` (DGS2, DGS10, DFF, T5YIE, T10YIE, T5YIFR).
   - Ingest erfolgt im bestehenden Connector-Muster mit ALFRED-Vintages und `available_at = realtime_start`.
   - Bis die Historie archiviert ist, sind Inflations- und Zins-Gap `INSUFFICIENT_DATA`.
2. **Kein Rate-of-Change-Satz je Indikator.** Gemeint sind velocity, acceleration, change_of_change, Perzentil und Regimewechsel-Wahrscheinlichkeit.
3. **Keine Gap-Definition** (Modell − Markt) mit Rohwert, z und Perzentil.
4. **Keine thesenspezifische Cross-Asset-Bestätigung** (erwartete Richtungen, Missing ≠ negativ).
5. **Kein Research-Status TRADE/WAIT/ABSTAIN/ERROR** mit WAIT-Trigger und unveränderlichen Kill Conditions.
6. **Keine vorregistrierten Expressions** und keine Thesis-vs-Expression-Attribution auf Underlyings/ETFs.
7. **Keine Population**, die Deep-Analysis-Kandidaten mit Kontext bis 250 Tage nachverfolgt (append-only).

## 3. Neue Module (`modules/expectation_alpha/`)
| Datei | Inhalt | Verantwortung |
|---|---|---|
| `schemas.py` | Status-Enums, `FeatureValue` mit Provenienz (value, unit, source, observed_at, published_at, available_at, retrieved_at, vintage, transformation, freshness, confidence), Schema-Version | Opus |
| `config.py` | lädt `config/expectation_alpha.yaml` (Modus, vorregistrierte Schwellen) und `config_hash` | Opus |
| `data.py` | PIT-Zugriff: Archiv-Reihen bis `available_at < decision_time`, Marktschlüsse nur abgeschlossener Tage | Opus |
| `future_state.py` | level, delta_1m/3m, velocity, acceleration, change_of_change, Perzentil, z (nur Vergangenheit), Unsicherheit | Opus |
| `expectation_gap.py` | Domänen-Gaps (Inflation, Zinsen/Policy, Wachstum, Öl): roh + z + Perzentil + Fenster; Earnings `UNAVAILABLE` | Opus |
| `regime_change.py` | Regime aus `world_model.build_world`, Zustandswechsel, empirische Wechselwahrscheinlichkeit (nur Vergangenheit) | Opus |
| `cross_asset_confirmation.py` | Thesen-Spezifikation erwarteter Richtungen; n_expected / n_available / n_confirming / n_conflicting / ratio / weighted / missing / confidence | Opus |
| `thesis.py` | Kandidat → These, macro_alignment, sector_alignment (versioniertes NON_PIT-Sektor-Mapping), context_status, Research-Gruppe A–D | Opus |
| `timing.py` | deterministische TRADE/WAIT/ABSTAIN/ERROR-Regel, WAIT-Trigger, Kill Conditions | Opus |
| `expression.py` | vorregistrierte Expressions, deterministische Auswahlregel | Opus |
| `ledger.py` | append-only Kontext-/Kandidaten-/Downstream-/Outcome-Dateien, Outcome-Auflösung inkl. WAIT-Replay, Promotion-Evidenz | Opus |
| `evaluation.py` | Gruppen A–D je Horizont, Abstention-, Context- und Expression-Wert, Kalibrierung, Vorschläge (nie automatisch angewendet) | Opus (Statistik), Sonnet (Report) |
| `__init__.py` | öffentliche Hooks: `enrich_candidates` (pipeline), `record_downstream`, `resolve_outcomes` (feedback), `evaluate` | Opus |

Kleine, gezielte Erweiterungen bestehender Dateien:
- `hypothesis_contract` (+Stage);
- `promotion_controller` (+Evidenz-Zweig);
- `pipeline.py` (ein Hook nach Stufe 4, ein Downstream-Hook);
- `feedback.py` (Outcome-Hook);
- Report und Mail (je ein SHADOW-Block);
- neue FRED-Quelle.

## 4. Datenfluss je Scanner-Lauf
1. **Nach Stufe 4 (Deep Analysis)** wird nur der Analysestand eingefroren (tiefe Kopie, ohne Rechenzeit).
   - **Am Laufende** läuft `expectation_alpha.enrich_candidates(...)` einmal je Lauf (`pipeline.save_stats_snapshot`, jeder Exit-Pfad), also nach allen Champion-Entscheidungen.
   - Es nutzt die Finalisierungsreserve und verbraucht nie das Laufzeitbudget späterer Champion-Stufen.
   - Der Einstieg für Outcomes ist der erste US-Schlusskurs nach der Entscheidung (16:00 New York).
   - **Kontext:** Gaps, Regime, Confirmation-Rohsignale; eine Zeile in `outputs/expectation_alpha/context/YYYY-MM.jsonl`.
   - **Je Kandidat eine Zeile** in `outputs/expectation_alpha/candidates/YYYY-MM-DD.jsonl` mit:
     - Pflichtfeldern, Status, `wait_trigger`, `kill_conditions`, `expressions` und `selected_expression`;
     - `data_snapshot`-Hash, `code_version` und `config_hash`.
   - Rückgabe ist nur eine Zusammenfassung für `stats`. Kandidatenliste und Entscheidungen werden nicht verändert.
2. **Direkt danach:** `record_downstream` hält beschreibend fest, was der Champion tat (Reject-Grund bzw. finaler TRADE/NO_TRADE).
3. **`feedback.py`:** `resolve_outcomes` je (Beobachtung, Expression, Horizont) genau einmal, append-only.
   - Horizonte 20/60/120/250 Handelstage.
   - Einstieg zum Schlusskurs des Entscheidungstags. Dieser liegt nach der Entscheidung, die vor Börsenschluss fällt.
   - Erfasst werden Rendite richtungsbereinigt, kostenbereinigt, MFE, MAE und Pfad-Drawdown je registrierter Expression.
   - Varianten (nur UNDERLYING):
     - `kill_managed`: Ausstieg beim ersten eingefrorenen Kill-Ereignis.
     - bei WAIT zusätzlich `triggered`: Der Trigger wird je Schlusskurs nur mit bis dahin bekannten Kursen geprüft. Der Einstieg erfolgt zum Schluss des **Folgetags**; Signal und Ausführung liegen nie auf demselben Schlusskurs. Der Ausstiegstag ist derselbe wie bei `immediate`. Ohne Trigger gilt `NO_ENTRY` mit Rendite 0 (Cash, gekennzeichnet).
4. **`promotion_controller`:** EA-Verträge werten nur EA-Ledger-Zeilen ab `forward_start` mit gleichem `spec_hash` aus. `cap_population` hält den Einfluss bei NONE.
5. **Montagsbericht und Mail:** zeigen den SHADOW-Abschnitt („Research – keine Handelsempfehlung“).

## 5. PIT-Risiken und Gegenmaßnahmen
| Risiko | Maßnahme |
|---|---|
| Publikationsverzug, Revisionen | nur Vintages mit `available_at < decision_date 00:00 UTC` (`pit_snapshots`); revidierte Werte nie rückwirkend |
| fehlende Vintage | `vintage_quality=FIRST_RELEASE_UNKNOWN` (ALFRED-Kappung) → Confidence reduziert |
| Marktschluss | nur abgeschlossene Tagesbalken (`date < decision_date`, außer nach 21:00 UTC) |
| Dividenden-bereinigte Schlusskurse | nur Renditen/Verhältnis-Änderungen, nie Niveaus; als Transformation dokumentiert |
| Feature-Fenster / Normierung | z und Perzentil expanding mit `shift(1)` (nur Vergangenheit), Mindesthistorie |
| Trigger-Timing | WAIT-Replay nutzt je Tag nur Schlusskurse bis zu diesem Tag; Einstieg erst am Folgetag |
| 3M-Raten monatlicher Reihen | exakt 3 Kalendermonate zurück (nicht 91 Tage, die je nach Monatslänge 4 Perioden überspannen) |
| Datenfehler | Ausfall von Kurs- oder Archivquelle -> alle Kandidaten ERROR (nie ein ABSTAIN/WAIT aus einem Teilbild) |
| Outcome-Kontamination | Outcomes in separater Datei, nie im Entscheidungs-Datensatz; Entscheidung ist vor dem ersten Outcome-Tag eingefroren |
| Konsensdaten | Earnings-Konsens PIT nicht vorhanden → `UNAVAILABLE`, nie simuliert |
| Sektor-Mapping | heutiges Mapping (NON_PIT), versioniert und gehasht |

Pflicht-Leakage-Tests stehen in `tests/test_expectation_alpha_pit.py`.

## 6. Produktionsgrenzen
- Default `mode: shadow` in `config/expectation_alpha.yaml`; `off` deaktiviert alles.
- Kein EA-Wert fließt in Proposals, Scores, Gates, Sizing oder Kandidatenlisten zurück. Ein statischer Test und ein Verhaltenstest prüfen das: identische Champion-Entscheidungen mit EA `off` und `shadow`.
- **Einfluss nur über die bestehende Leiter:** Vertrag → Forward-Evidenz → PromotionController.
  - EA-Population: höchstens FORWARD_VALIDATED mit Empfehlung.
  - Champion-Wirkung nur über einen neuen Champion-Vertrag per menschlichem PR.
- Die gewünschte Leiter SHADOW → REPORT_ONLY → ABSTENTION_ONLY → RERANK → LIMITED → PRODUCTION entspricht den vorhandenen Stufen:
  - SHADOW und REPORT_ONLY = Einfluss NONE (Report zeigt Status);
  - ABSTENTION_ONLY / RERANK_ONLY / SCORE_LIMITED / WEIGHT_10 bleiben wie bisher.
  - Neue Stufen werden nicht eingeführt.
- Kein LLM in Gap, Regime, Confirmation, Status, Outcome oder Promotion.

## 7. Forschungshypothesen (vorab registriert, `config/promotion_hypotheses.yaml`)
| ID | H1 | Vergleich innerhalb EA_NEWS_CANDIDATE |
|---|---|---|
| EA001_EXPECTATION_GAP | PIT-Gaps in Thesenrichtung haben inkrementelle Prognosekraft | gap-aligned vs. Rest |
| EA002_CONFIRMATION | Confirmation verbessert Outcomes bei ähnlich großen Gaps | große Gaps: ratio ≥ 0,7 vs. < 0,7 |
| EA003_ACCELERATION | Beschleunigung in Thesenrichtung liefert Zusatz zur Level-Baseline | gap-aligned: accel-aligned vs. nicht |
| EA004_WAIT | Triggerbasierter Einstieg verbessert Rendite/MAE gegenüber Sofort-Einstieg | gepaart je WAIT-Beobachtung |
| EA005_CONTEXT_FILTER | Starke News mit positivem Kontext schlagen starke News mit negativem Kontext | starke News: Kontext +1 vs. −1 |
| EA006_ABSTENTION | ABSTAIN-Fälle sind schlechter (Abstinenz verbessert Portfolioqualität) | ABSTAIN vs. Rest (direction −1) |
| EA007_EXPRESSION | Regelbasierte Expression schlägt Default-Expression (Underlying) | gepaart je Beobachtung |

Jeder Vertrag legt fest:
- H0/H_alt, primäre und sekundäre Metriken, Population;
- primären und sekundäre Horizonte, min_n, unabhängige Signaltage, Kalenderspanne, Regime-Breite;
- `forward_start`, Familie (Bonferroni `research_only@EA_NEWS_CANDIDATE` × Looks), Promotion- und Failure-Kriterien.

Keine Schwelle wird nach Ergebnissen gewählt.

## 8. Delegationsplan (SPEC → SUBAGENT → OPUS REVIEW → TEST → FIX → INTEGRATE)
| Aufgabe | Modell | Grenze |
|---|---|---|
| FRED-Quelle `fred_market_expectations`: Connector, Config, Tests | Sonnet | nur `real_economy.py` (neue Klasse), `real_economy.yaml` (neuer Eintrag), eigener Test |
| SHADOW-Abschnitte Montagsbericht und Mail, Report-Tests | Sonnet | nur Rendering nach festem Datenvertrag; keine Logik |
| Test-Fixtures (synthetische PIT-Archive, Preisrahmen) | Haiku | nur `tests/ea_fixtures.py` |
| Gap-, Regime-, Confirmation-, Status-, Ledger-, Outcome-, Evidenz-, Vertrags- und Promotion-Logik | **Opus** | nie delegiert |

Subagents ändern weder Architektur, Kern-Schemas, PIT-Semantik, Risk Limits, Promotion-Regeln, Champion-Logik, Research-Verträge, Produktionseinfluss noch Learning-Guardrails.

## 9. Phasen
1. **P0:** Audit und dieses Dokument.
2. **P1:** PIT-Datenzugriff und FRED-Quelle.
3. **P2:** Future State und Rate of Change.
4. **P3:** Gap.
5. **P4:** Confirmation.
6. **P5:** Kandidaten-Enrichment.
7. **P6:** Status, WAIT und Kill Conditions.
8. **P7:** Expressions und Attribution.
9. **P8:** EA001–EA007.
10. **P9:** Ledger und Outcomes.
11. **P10:** Report und Mail.
12. **P11:** Tests, E2E und Regression.
13. **P12:** Opus-Gesamtaudit und Fixes.

Nach jeder Phase: Tests → Review → Fix.

Umsetzungsstand, Testergebnisse, Datenlücken und offene Punkte: `docs/EXPECTATION_ALPHA_V1_REPORT.md`.
