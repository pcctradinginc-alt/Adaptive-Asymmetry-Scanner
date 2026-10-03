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
