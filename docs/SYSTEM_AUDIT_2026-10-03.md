# System-Audit 2026-10-03 – Status, Fehler, Lernmechanismen

Leitfrage je Komponente: Wird das System auf neuen Daten tatsächlich besser oder besser darin,
schlechte Entscheidungen zu vermeiden? Grundlage: Code-Pfade (Input → Verarbeitung → Output →
Downstream → Validierung), Workflows, `outputs/history.json` (119 geschlossene Trades), ML-Lauf
37064729996 (PIT-Universum), Repro-Lauf 37067675084 (604 Kernwerte, 0 Abweichungen).

## 1. Komponentenstatus

**ACTIVE** = beeinflusst Trade-Entscheidungen · **SHADOW** = läuft, misst, beeinflusst nichts ·
**UNUSED** = Output ohne Abnehmer · **UNVALIDATED** = ohne belastbaren OOS-Nachweis.

| Komponente | Status | Begründung (Downstream / Validierung) |
|---|---|---|
| Risk Gates, Hard-Filter, Prescreen (LLM), Deep Analysis (LLM), Mismatch, Quick/Final MC, Intraday-Delta, Options Design + ROI-Gate, trade_score-Gate, Korrelations-Check, Sizing | ACTIVE, UNVALIDATED | erzeugen `active_trades`; Champion ohne belastbaren Vorteil (s. 3.) |
| Final-MC-Hit-Rate als „Wahrscheinlichkeit“ | ACTIVE, fehlkalibriert | vorhergesagt 0,71 → realisiert 0,22 (Band 0,65–0,75); nur ≥ 0,75 besser (0,41) |
| QuasiML (feature_stats-Bins, Pearson-Gewichte) | SHADOW, UNVALIDATED | `final_score` steuert nur, welche 2 Kandidaten Premium-Daten bekommen, und den Report – nicht Gates/Score/Auswahl |
| PPO RL-Agent (`rl_agent`) | UNUSED | `rl.veto_enabled: false`; Policy war degeneriert (Immer-SKIP). Tägliches Nachtraining jetzt nur bei aktivem Veto |
| Robuster PPO (`rl_robust_shadow`) | SHADOW | Walk-Forward-Diagnose |
| Premium Signals (FLASH/Eulerpool) | SHADOW | nur Log-Warnung (IV-Crush) und `final_score`-Anpassung → ohne Entscheidungswirkung |
| External Context, Shadow-Relation (LLM) | SHADOW | Ledger/Report |
| Candidate Ledger, Shadow Trades, Exit-/Trailing-Sim | SHADOW (Messinfrastruktur) | Counterfactual-Basis für Gate-Challenger |
| Challenger Registry (`challenger.py`, Registrar) | SHADOW | nur `promote_recommended` (menschlicher PR) |
| Alpha Discovery | SHADOW | Ledger-basierte Suche, BH |
| ML-Research (6 Modelle) | SHADOW, kein Champion | WF netto +0,29…+0,39 %/20 T, alle t < 1,7; Locked kontaminiert; `rejected_so_far` |
| Meta-Learning | SHADOW | kein Gate bestanden (alle Bootstrap-CIs schließen 0 ein) |
| Research Lab / Director / Hypothesis DB | SHADOW | 0 ACCEPTED, 20 REJECTED |
| World Model, Causal Research, Knowledge Graph | SHADOW | Causal: REJECT (keine Beziehung überlebt BH OOS) |
| Counterfactual, Blind Spots, Decision Intel, Next Intelligence | SHADOW | Verdikte MODIFY; Abstinenz-Bestätigung CONTAMINATED, Forward ACCUMULATING |
| Meta-Cognition + Safe Mode | SHADOW → Eingang der Brücke | Safe Mode aktiv (tnx-Drift); als Abstinenz-Vertrag PROM-ABST-001 prospektiv im Test |
| HC-Scanner | SHADOW | Research-Alerts, im Safe Mode deaktiviert |
| Trade-/Prediction-Memory, Failure Analyzer, Factor Monitor | SHADOW | Attribution; speisen jetzt `abstention_proposals` |
| Alternative Data (SEC/GLEIF) | SHADOW, **REJECTED** | Ingestion vollständig (Form 345 2014Q1–2026Q2, Filings 609/616 CIKs). OOS-Ablation (s. 6.): kein inkrementeller Nutzen; ALT-SEC-001…004 im Research-Lab REJECTED (zwei signifikant negativ) |
| PromotionController + ProductionIntelligenceAdapter | ACTIVE (Protokoll), Einfluss NONE | einzige Wirkungsstelle; 5 Verträge PROSPECTIVE_CHALLENGER, Forward ab 2026-10-05 |
| `abstention_proposals` (neu) | SHADOW | Vertragsentwürfe nur nach bestandenem Walk-Forward; Registrierung per PR |

Kein Modul ist vollständig ungenutzt (Import-Graph + Workflows geprüft).

## 2. Priorisierte Fehlerliste

| Prio | Befund | Status |
|---|---|---|
| P0 | Erste echte SEC-Features ließen den kompletten ML-Research-Lauf abbrechen (pandas `MergeError`, Datums-Einheiten `M8[s]`/`M8[us]` in `alt_data.feature_store.attach`, Lauf 37103770836) | **behoben**: Einheiten normalisiert, optionale Quelle fail-safe (NaN, verfügbar 0) |
| P0 | Blockierte Champion-Trades wurden als Schatten-Trade zum festen Stichtag ohne TP/SL bewertet, echte Trades mit Exit-Regeln → „blockiert vs. durchgelassen“ systematisch verzerrt (Abstinenz-Evaluation der Brücke ungültig) | **behoben**: `counterfactual_trades` mit identischem Lebenszyklus (`pipeline.build_trade_record`, `feedback.advance_counterfactual_trades`), ohne Lern-Updates |
| P0 | Promotion-Evidenz/Evaluation akzeptierte Näherungs-Outcomes (Delta-Approx bis +500 %, Aktien-Fallback, unbekannt) | **behoben**: nur Quote-basierte Outcomes (`modules/outcomes.py`); Ausschluss gezählt |
| P1 | Lern-Loop (Feature-Bins, Pearson-Gewichte, PPO, robuster PPO) lernte aus genäherten/rekonstruierten Outcomes (40 von 119) | **behoben**: `outcome_reliable` je Close; Lernen nur aus verlässlichen Outcomes |
| P1 | Keine kontrollierte Quelle neuer, falsifizierbarer Produktions-Hypothesen aus echten Fehlern | **behoben**: `abstention_proposals` (Kalibrierung 60 % / Walk-Forward 40 %, Bonferroni, Similarity) |
| P1 | Unbenutztes PPO wurde täglich nachtrainiert und committet | **behoben** (nur bei aktivem Veto) |
| P1 | Entity Resolution brach bei mehreren Tickern je CIK ab (GOOG/GOOGL) | **behoben** (#78) |
| P1 | SEC Form 3/4/5: fester URL-Aufbau → 404 für alle Quartale → Insider-Features leer (Lauf 37102688451) | **behoben**: Links von der offiziellen Übersichtsseite |
| P2 | Alt-Data-Entity-Map: 147 von 767 PIT-Tickern ohne SEC-Zuordnung (delistete Firmen fehlen in `company_tickers.json`) → für diese NaN statt Werte (Survivorship in Alt-Features) | offen, dokumentiert |
| P1 | Champion ohne nachgewiesenen Vorteil; Final-MC-Wahrscheinlichkeit fehlkalibriert | **offen – bewusst nicht direkt geändert**: nur über Challenger/Hypothesen (Gate-Challenger `final_mc_dte_shadow` läuft; Abstinenz-Verträge prospektiv) |
| P2 | Restlicher Survivorship-Bias im ML-Panel: 135 von 255 entfernten Titeln ohne Yahoo-Kurse | offen, im Bericht ausgewiesen |
| P2 | Sektor-Zuordnung im ML-Panel nicht PIT | offen, dokumentiert (Audit P2-3) |
| P2 | Alt-Data-Bewertung: Regime-/Sektor-Aufschlüsselung immer „unknown“ (Positionen trugen bereits sector/vix → `vix_x/vix_y`) | **behoben** + Test |
| P2 | Ergebnisse der HGB-Modelle schwanken zwischen zwei Yahoo-Downloads (10 h Abstand) um bis zu 0,003/20 T (hgb_asym20: 0,00304 → 0,00002); ENet identisch | offen – Befund: Baum-Modell-„Edge“ liegt im Datenrauschen; Reproduzierbarkeit nur über Panel-Snapshot (`repro.yml`) |
| P2 | Alt-Data-Protokoll alt-v1: Dev-Jahre schließen 2025H2 ein (= kontaminierter Locked-Bereich) | offen; bindend ist ohnehin Forward |

Frühere Audits (PR #74 ff.): PIT-Universum, Replay/Repro, Locked-Kontamination gekennzeichnet,
Champion/Challenger-Protokolle gepinnt – hier nicht wiederholt.

## 3. Champion – nüchterne OOS-Lage

Verlässliche Outcomes n = 79 (40 rekonstruierte ausgeschlossen): Win Rate 35,4 %, Mittel +4,2 %,
Median −31,3 %, Profit Factor 1,12; Ertrag fast vollständig aus April 2026 (PF 1,64), Mai–Sep
negativ. Kein Nachweis eines robusten Vorteils. Walk-Forward der Abstinenz-Kandidaten
(Kalibrierung 11.04.–04.05., Test 06.05.–21.09.): 6 Regeln getestet, **keine** bestätigt –
In-Sample-Muster (z. B. niedrige MC-Hit-Rate Δ −0,75, hoher z-Score Δ −0,79) verschwinden OOS.

## 4. Echte Lernmechanismen (Prediction → Entscheidung → Outcome → Attribution → Update)

1. **Produktion:** Entscheidung (`pipeline`) → Decision-Ledger mit eingefrorener Regel-Auswertung →
   Outcome (`feedback`, nur Quotes) → Attribution je Hypothese (`promotion_controller.evidence`) →
   Update ausschließlich über PromotionController (Default max. Abstinenz) mit Demotion/Rollback.
2. **Fehlerlernen:** verlässliche Champion-Outcomes → `abstention_proposals` (Walk-Forward) →
   Vertragsentwurf → menschliche Registrierung → prospektiver Test.
3. **Gate-Challenger:** Candidate Ledger → `challenger.py` (Alpha-Spending) → Empfehlung → PR.
4. **Research:** ML/Meta/World Model mit präregistrierten Protokollen, Repro bestätigt; ohne
   Produktionswirkung, solange kein Vertrag mit Champion-Population Forward besteht.

## 5. Leakage-/Overfitting-Risiken (Stand)

* Kleine Stichprobe (79 verlässliche Trades, 6 Monate, ein dominanter Monat) → jede In-Sample-
  Regel ist verdächtig; deshalb Walk-Forward + Bonferroni + prospektive Pflicht.
* Viele Research-Hypothesen (29 gezählt) → Familien-Bonferroni und Look-Spending im Controller.
* Locked-Holdout des ML-Panels kontaminiert → bindend nur Forward.
* Final-MC-Wahrscheinlichkeit fehlkalibriert → Brier/ECE im Controller verhindern „bessere“
  Kalibrierung als Promotion-Argument nur, wenn Policy-Brier nicht schlechter ist.

## 6. Nachweisbarer OOS-Mehrwert (identische OOS-Zeilen, Ablation mit/ohne Komponente)

| Komponente | Vergleich | Ergebnis |
|---|---|---|
| SEC Deep Events (Lauf 37105528299) | ENet-Baseline vs. +5 SEC-Features, 2019–2025, 77 Monate | Δ Expectancy −0,00027, Δ Sharpe −0,016, Δ IC −0,0017, Δ Brier −0,00002, Δ ECE −0,0011, Δ Prec@K −0,0006, Trades gleich; CI Monatsrendite [−0,0010; +0,0005] → **REJECT** |
| SEC Deep Events | HGB-Baseline vs. +SEC | Δ Expectancy +0,00029, Δ Sharpe +0,031, aber Δ Max DD −0,046 (−21,5 % vs. −17,0 %), Δ IC −0,0032, Δ LogLoss +0,0005; CI [−0,0015; +0,0023] → **REJECT** |
| SEC-Hypothesen ALT-SEC-001…004 | Research-Lab Walk-Forward, BH | alle REJECTED; Insider-Cluster t = −2,96, negative 8-K-Items t = −3,9 (Gegenrichtung) |
| ML-Modelle vs. Referenz | Walk-Forward 2019–2025 | kein Modell signifikant (max. t 1,95); Champion: keiner |
| Meta-Learning vs. statisches Ensemble | identische OOS-Zeilen, Ablationen | NEED_MORE_DATA: Δ Sharpe +0,155, aber Bootstrap-CI [−0,0026; +0,0048], Hälften uneinheitlich → kein Gate |
| Gesamtvalidierung A–G | Ablation je Komponente | KEEP_CHAMPION |
| Abstinenz-Regeln auf Champion-Trades | Walk-Forward 60/40 | 6 Regeln, keine bestätigt |

Ergebnis: Keine Komponente zeigt einen robusten inkrementellen Nutzen. Die Ablationen haben
damit den eigentlichen Zweck erfüllt – sie verhindern, dass Rauschen (In-Sample-Muster,
Download-Schwankungen) als Verbesserung in die Produktion gelangt. Das System ist jetzt so gebaut, dass es einen echten Mehrwert auf neuen Daten
erkennen und begrenzt nutzen **kann** – und ohne ihn keinen Einfluss erhält.

## 7. Wichtigste verbleibende Schwäche

Die Datenbasis des Champions: wenige, zeitlich geklumpte, teils rekonstruierte Outcomes und eine
fehlkalibrierte Erfolgswahrscheinlichkeit. Bis genügend Quote-basierte Forward-Outcomes vorliegen
(≥ 60 Trades, ≥ 90 Tage je Hypothese), kann keine Komponente statistisch belastbar besser werden.
