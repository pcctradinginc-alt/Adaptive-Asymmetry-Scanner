# Gap-Analyse: Spezifikation „autonomes Quant-Research- und Decision-System“ vs. Bestand (2026-10-03)

Grundregel: Neue Komponenten laufen SHADOW. Wirkung entsteht nur über PromotionController und ProductionIntelligenceAdapter, nach Forward-Validierung.

| Anforderung | Bestand (Modul) | Status |
|---|---|---|
| Lernkreislauf DATA → WORLD MODEL → … → LEARNING LOOP | `external/*`, `world_model`, `meta_cognition`, `blind_spots`, `hypothesis_factory`, `research_director`, `research_lab`, `promotion_controller`, `production_intelligence_adapter`, `feedback`, `meta_learning` | vorhanden |
| Hypothesis Factory: Quellen der Ideen (Fehler, Blind Spots, Drift, Cross-Source, unorthodox) | `hypothesis_factory` (cross_domain, source_condition, cross_source_divergence, drift) | vorhanden |
| Vollständige Hypothesen-Spezifikation (Population, Exposure, Lag, Horizont, Kontrollgruppe, Metrik, H0/H_alt, Failure Condition) | `hypothesis_factory`, `hypothesis_contract` | vorhanden |
| DATA_GAP statt Halluzination, kostenlose Quellen vorgeschlagen | Factory `readiness`, `active_learning` | vorhanden |
| Forschung: Walk-Forward, BH, Placebo/Lag, Regime, Ablation, Replikation | `research_lab`, `ml_research`, `alt_data.evaluate`, `price_event_study` | vorhanden |
| Research Memory mit Ähnlichkeitsprüfung | `research_memory` | vorhanden; **neu:** Surprise-Studie als Quelle |
| Active Learning / Priorisierung EIG × Relevanz × Neuheit × Datenqualität ÷ Kosten ÷ Overfit | `research_director.priority` | vorhanden |
| Meta-Learning über Features, Quellen, Richtungen, Regime; Research-Value-Attribution | `meta_learning`, `meta_cognition` (research_efficiency), `source_scoreboard`, Factory-Richtungen (Posterior) | vorhanden |
| **Expectation / Surprise Engine** | – | **neu:** `surprise_engine` (präregistrierte Studie, Lauf in `research.yml`) |
| **Abstention Intelligence (P(model wrong), Data Quality, Regime Mismatch, Unknown, Disagreement, Counterfactual Fragility, Alpha Decay)** | Einzelverträge PROM-ABST-001…005 | **neu:** `abstention_intelligence` als Risikovektor je Champion-Trade → Decision-Ledger und Trade-Features → `abstention_proposals` (Walk-Forward) |
| Promotion-Leiter bis 10 %/25 %, Demotion stufenweise | `promotion_controller` (Einfluss-Leiter, Demotion eine Stufe tiefer, Rollback) | vorhanden |
| **RL nur als Prospective Challenger; „RL PROMOTION CANDIDATE“ melden** | `rl_robust_shadow`, Challenger `ppo_robust_shadow` | **neu:** `rl_promotion` (Kollaps, Stabilität, Forward-Mehrwert; Veto bleibt aus). Erster Befund: NOT_READY (Policy kollabiert) |
| Claude-Analyse unabhängig; Memory-Reviewer nur für prospektiv validierte Erkenntnisse | `deep_analysis`, Stufe 4b Memory-Reviewer | vorhanden |
| **Monday Intelligence Report mit 7 Abschnitten** | `reports/weekly.py` (19 Detailabschnitte) | **neu strukturiert:** 7 Hauptabschnitte und Anhang A1–A19; `NO HIGH-CONFIDENCE TRADE THIS WEEK` |
| Safe Mode, Alerts, keine Orderausführung | `system_state`, `drift`, `source_health`, `hc_scanner`, Mailer | vorhanden |

## Validierungsstand der neuen Komponenten
- **Surprise Engine:** Entscheidung erst nach dem ersten Lauf in GitHub Actions (KEEP/MODIFY/REJECT nach Präregistrierung).
- **Abstention-Risikovektor:** Es gibt keine historischen Werte, deshalb nur Forward-Messung. Ein Vertrag entsteht erst, wenn `abstention_proposals` den Vektor auf echten Outcomes per Walk-Forward bestätigt.
- **RL:** NOT_READY. Walk-Forward und In-Sample sind kollabiert (100 % BOOST bzw. 100 % SKIP).

## Kernfrage als durchgängige Kette (`modules/inquiry.py`, seit 2026-10-03)

Für jede gemessene Unklarheit beantwortet das System die fünf Teilfragen der Kernfrage. Quellen der Unklarheiten sind Blind-Spot-Cluster, Modell-Drift, Merkmale außerhalb des Trainingsbereichs und die häufigsten Verlust-Ursachen.

1. **Nicht verstanden:** die Unklarheit selbst.
2. **Erklärung:** verknüpfte Hypothesen aus Director, Fabrik, Champion-Fehlerregeln und Verträgen.
3. **Daten:** benötigt, verfügbar, DATA_GAP mit freien Quellen.
4. **Neue Daten:** historisch oder prospektiv/Forward.
5. **Verhalten:** Produktionseinfluss über den PromotionController.

Kettenstatus: OPEN_QUESTION → NEEDS_DATA → UNDER_TEST / HISTORICALLY_REJECTED → FORWARD_TEST → BEHAVIOUR_CHANGED. Die Ketten erscheinen im Montagsreport, Abschnitt 2.

**Schleife geschlossen:**
- Eine OPEN_QUESTION zu einem Merkmal außerhalb des Trainingsbereichs erzeugt in der Fabrik eine falsifizierbare Hypothese (`idea_source: open_question`). Die Schwelle ist das p99 des Trainingsbereichs aus dem Drift-Befund.
- Erster Fall: `tnx` (10-jährige Rendite 5,28 > p99 4,78). Daraus wird `rank(mom_12_1) * step(tnx - 4.7817)`.

## Lernt die Forschung, besser zu forschen? (`research_memory.learning_curve`)
- **Hierarchischer Prior:** Eine neue Familie übernimmt die Erfolgsrate ihres Ideentyps (cross_domain, drift, source_condition, open_question, …). Bisher startete sie immer beim neutralen Prior. Ideentypen, die Kapazität verschwenden, verlieren damit automatisch Priorität.
- **Messung:** Erfolgsquote je Quartal und Kalibrierung der Priorisierung (Spearman: vergebene Priorität vs. Erfolg). Ideentyp und Priorität werden dafür jetzt je Fabrik-Ergebnis gespeichert.
- **Befund 2026-10-03:** 86 getestete Hypothesen, 0 Erfolge. Die Priorisierung ist noch nicht kalibrierbar (keine Prioritäten gespeichert).
