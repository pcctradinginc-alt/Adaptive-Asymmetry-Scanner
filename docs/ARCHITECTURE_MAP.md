# Architektur-Karte: fünf Verantwortungsbereiche, Aufrufpfade, Redundanzen, Migrationsplan (Stand 2026-10-10)

Grundlage ist eine AST-basierte Erreichbarkeitsanalyse aller 290 `.py`-Dateien. Sie erfasst auch lazy Imports in tatsächlich aufgerufenen Funktionen. Dazu kommt ein Inventar aller Ledger und Outcome-Speicher mit den Datenmengen auf der Platte.

Die fünf Engines sind **logische Zuständigkeiten, kein neues Framework**. Bewährte Module bleiben bestehen. Konsolidiert wird über gemeinsame Schnittstellen.

## 1. Produktive Aufrufpfade (Workflows)
| Workflow | Takt | Ausführung |
|---|---|---|
| `source_health.yml` → `scanner.yml` | täglich 12:41 → ab 13:30 UTC | Source-Health-Gate, dann `pipeline.py` (Champion), danach `feedback.py` und `universe_v2_scan` |
| `feedback.yml` | 15:30 und 19:30 UTC | `feedback.py` (alle Resolver), `challenger_registrar`, **`promotion_controller`**, `abstention_proposals` |
| `ml_research.yml` | samstags + am 3. des Monats | SEC/TED/Commodity-Build, `ml_research`, Research-Director/-Factory/-Lab, `meta_learning`, `hc_scanner`, `trade_memory` |
| `weekly_report.yml` / `monthly_report.yml` | montags / monatlich | Berichte, `alpha_discovery`, `factor_monitor` |
| `external_data.yml`, `alt_data.yml`, `world_model.yml`, `universe_v2.yml`, `research.yml` | täglich / wöchentlich | Archiv-Ingest, World Model, Universum V2, Ereignisstudien |

Erreichbarkeit der 144 Module unter `modules/`:
- 102 sind live aus `pipeline.py`.
- 15 sind statisch aus `pipeline.py` importiert, werden aber nur von anderen Workflows tatsächlich aufgerufen.
- 25 sind nur aus anderen Workflows erreichbar.
- 2 werden nur von Tests genutzt: `external/policy.py` und `external/governance.py`. `external/context.py` codiert den Policy-Modus hart; beide bleiben als dokumentierter Research-Rahmen.
- **Kein Modul ist unerreichbar.**

## 2. Die fünf Engines (Zuordnung bestehender Module)
| Engine | Kernmodule (bestehend) | Schnittstelle |
|---|---|---|
| **Data** | `data_ingestion`, `universe`/`universe_v2`, `external/*` (Archiv mit `available_at`, `retrieved_at`, `vintage_time`, `payload_hash`), `sec_*`, `commodity_intelligence`, `world_model.pit_snapshots`, `source_health`, `request_cache`, `market_snapshot`, **neu `option_quotes`** | `Observation` (`external/pit.py`), Quote-Record `option-quote-v1` |
| **Signal** | `prescreener`, `deep_analysis`, `mismatch_scorer`, `alpha_sources`, `surprise_engine`, `expectation_alpha` (SHADOW), `trade_scorer`, `quasi_ml` | Kandidat mit `deep_analysis`, `features`, `trade_score` |
| **Trade** | `risk_gates`, `options_designer`, `mirofish_simulation`, `spread_execution`, `position_sizing`, Korrelations-Check | Proposal mit `option`, `roi_analysis`, `position_sizing` |
| **Evidence** | Ledger: `candidate_ledger`, `shadow_ledger`, `final_mc_ledger`, `universe_v2_ledger`, `expectation_alpha/ledger`, Decision-Ledger. Dazu `outcomes` (Reliability und Preistyp), `factor_monitor`, `hypothesis_contract`, **neu `evidence_ledger`** (einheitliche Sicht) und **`evidence_core`** (gemeinsame Statistik) | Evidence-Record `evidence-v1` |
| **Learning** | `promotion_controller`, `production_intelligence_adapter` (einzige Wirkungsstelle), `challenger*`, `ml_research`, `meta_learning`, `learning_ledger` (Consume-once), **neu `meta_label`** (Modell-Registry, Training-Manifest, Rollback) | Vertrag (`spec_hash`) → Forward-Evidenz → Stufe → Adapter |

## 3. Befunde (Redundanzen, Brüche, Lecks)
| # | Befund | Schwere | Maßnahme |
|---|---|---|---|
| 1 | Verfallene Spreads: dividendenbereinigter Schluss gegen unbereinigte Strikes, gestempelt als „Quote“ (zählte RELIABLE) | HIGH | **behoben**: `expiry_intrinsic`, unbereinigt, exakter Verfallstag, Preistyp |
| 2 | `evaluate_shadow_trades` ohne Fehlerisolation: Fehler bricht den Feedback-Lauf vor `save_history` ab | HIGH | **behoben** |
| 3 | Entscheidungs-Ledger ohne Dedupe; Neustart erzeugt doppelte Entscheidungen | MEDIUM | **behoben** (je Tag und Ticker) |
| 4 | Schatten-Näherungen flossen in `feature_stats_external` | MEDIUM | **behoben** (nur RELIABLE) |
| 5 | **Kein Ledger-Outcome erreicht ein trainiertes Modell.** ML lernt nur aus Kurs-Panels; Champion-Lerner haben 3 RELIABLE-Trades | HIGH (Lernkreislauf offen) | **neu `meta_label`** auf der einheitlichen Evidence-Sicht, mit Manifest-Nachweis |
| 6 | Keine logistische Baseline, kein Meta-Labeling, nirgends Embargo; Champion-Splits ohne Purge | MEDIUM | `evidence_core.purged_walk_forward` (Purge + Embargo) + `meta_label` (L2-Logit-Baseline, HGB-Challenger) |
| 7 | Kein dauerhaftes Options-Quote-Archiv; Exit-Quotes nie persistiert; Preistyp nur implizit | HIGH (DoD 5) | **neu `option_quotes`** + Hooks (Auswahl, Entry, Marks, Exit) + Trade-Outcome-Record mit Kosten |
| 8 | Kosten (Kommission) fehlen im realisierten Champion-Outcome | MEDIUM | `outcome_net` im Trade-Outcome-Record (Brutto bleibt für Vergleichbarkeit) |
| 9 | Rund 15 Bootstrap-Varianten; 4 parallele Trios aus `read_outcomes`, `resolve_outcomes` und `evidence`; doppelte Walk-Forward-Funktion | LOW–MEDIUM | `evidence_core` (eine Gruppen-Bootstrap-Funktion) für Research-Pfade; Walk-Forward zusammengeführt. Die Promotion-Statistik bleibt bewusst unverändert, weil sie ein eingefrorenes Produktions-Gate ist. |
| 10 | 12 Roh-JSONL-Writer ohne atomaren Einmal-Write | LOW–MEDIUM | auf `atomic_io.append_jsonl` umgestellt |
| 11 | Kein globaler Kill-Switch für Produktionseinfluss | MEDIUM | **behoben**: `config/production_kill_switch.yaml` (CODEOWNERS); aktiv oder unlesbar → Adapter = Champion, Auswertung und Logging laufen weiter |
| 12 | Sizing- und Strategie-Stufe fehlt im Promotion-Prozess | MEDIUM | **behoben (Sizing)**: Klasse `sizing`, Stufe NONE → SIZE_DOWN_ONLY (nur verkleinern, Faktor ≥ 0,25), nur per menschlicher Freigabe. Neue Strategien bleiben bewusst nur per PR möglich. |
| 13 | Survivorship: keine freie PIT-Indexhistorie für Nasdaq-100, keine Delisting-Preise | Datenlücke | dokumentiert; V2 sammelt prospektiv Snapshots |
| 14 | `payload_hash` in den `external/*`-Archivzeilen leer, `quality_status` fehlt | LOW | **offen**: Das Options-Quote-Archiv hat beides von Anfang an (`option-quote-v1`). Das Nachrüsten der `external/*`-Archive ist ein eigener Schritt, weil es das Format bestehender Archive ändert. |

## 4. Migrationsplan (Reihenfolge = Risiko aufsteigend, je Schritt Tests)
1. Korrektheit der Evidenz: Befunde 1–4, erledigt.
2. Daten: Options-Quote-Archiv und Hooks; Trade-Outcome-Record mit Kosten und Preistyp (Befunde 7, 8).
3. Evidence: `evidence_core` (Bootstrap, Purge/Embargo) und `evidence_ledger` (einheitliche Sicht über alle Ledger: `prediction_id`, `population`, `decision_time`, `label_available_at`, `outcome`, `outcome_net`, `outcome_method`, `price_type`, `quality`).
4. Learning: `meta_label`.
   - Baseline-Logit gegen HGB-Challenger, Purged/Embargo-Walk-Forward.
   - Versionierte Artefakte mit Manifest (welche Outcomes trainiert wurden), aktiver Zeiger, Rollback.
   - Shadow-Prognose-Ledger (unveränderlich).
5. Promotion: Kill-Switch; Sizing-Stufe (nur Verkleinerung); Meta-Label-Wirkung nur über einen registrierten Vertrag (Abstinenz). Forward-Evidenz über den bestehenden Controller.
6. Konsolidierung: Befunde 9, 10 und 14.

Status 2026-10-10: Schritte 1–5 sind umgesetzt und technisch getestet. In Schritt 6 sind die Befunde 9 und 10
umgesetzt; Befund 14 ist offen. Details zum Lernkreislauf und zur Datenlage stehen in `docs/LEARNING_ENGINE.md`.

Leitplanken unverändert:
- keine echten Orders;
- Champion-Pfad unverändert, solange kein Vertrag eine Stufe erreicht;
- automatische Obergrenze `max_automatic_influence: ABSTENTION_ONLY`;
- Sizing nach oben, neue Strategien und Champion-Wechsel nur per menschlichem PR.
