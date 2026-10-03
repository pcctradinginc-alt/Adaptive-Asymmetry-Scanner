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
| QuasiML (feature_stats-Bins, Pearson-Gewichte) | SHADOW, UNVALIDATED | `final_score` nur im Tagesreport (als SHADOW beschriftet) – nicht Gates/Score/Auswahl |
| PPO RL-Agent (`rl_agent`) | UNUSED | `rl.veto_enabled: false`; Policy war degeneriert (Immer-SKIP). Tägliches Nachtraining jetzt nur bei aktivem Veto |
| Robuster PPO (`rl_robust_shadow`) | SHADOW | Walk-Forward-Diagnose |
| Premium Signals (FLASH/Eulerpool) | entfernt (§9) | waren nur Log-Warnung und `final_score`-Anpassung → ohne Entscheidungswirkung |
| External Context, Shadow-Relation (LLM) | SHADOW | Ledger/Report |
| Candidate Ledger, Shadow Trades, Exit-/Trailing-Sim | SHADOW (Messinfrastruktur) | Counterfactual-Basis für Gate-Challenger |
| Challenger Registry (`challenger.py`, Registrar) | SHADOW | nur `promote_recommended` (menschlicher PR) |
| Alpha Discovery | SHADOW | Ledger-basierte Suche, BH |
| ML-Research (6 Modelle) | SHADOW, kein Champion | WF netto +0,29…+0,39 %/20 T, alle t < 1,7; Locked kontaminiert; `rejected_so_far` |
| Meta-Learning | SHADOW | kein Gate bestanden (alle Bootstrap-CIs schließen 0 ein) |
| Research Lab / Director / Hypothesis DB | SHADOW | 0 ACCEPTED, 20 REJECTED |
| World Model, Causal Research, Knowledge Graph | SHADOW | Causal: REJECT (keine Beziehung überlebt BH OOS) |
| Counterfactual, Blind Spots, Decision Intel, Next Intelligence | SHADOW | Verdikte MODIFY; Abstinenz-Bestätigung CONTAMINATED, Forward ACCUMULATING |
| Meta-Cognition + Safe Mode | SHADOW → Eingang der Brücke | kanonisch in `system_state` (Stand: MODERATE, Safe Mode aus); als Abstinenz-Vertrag PROM-ABST-001 prospektiv im Test |
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

## 8. Härtungsdurchgang (Konsistenz, Lern-Loops, Gate-Audit)

Kernregel: keine neue Komplexität ohne nachgewiesenen Nutzen. Es wurde kein neues Intelligence-Modul
gebaut; ein eigenes „LearningEvent"-Format wurde bewusst **nicht** eingeführt. Die vorhandenen,
gehashten Ketten (Vertragsregister mit `spec_hash`, Transition-Log mit `entry_hash`, Decision-Ledger,
`research_memory`) decken Quelle, Claim, Evidenz, n, Regime, Horizont, Zeitstempel und Daten-/Code-
Version bereits ab. Ein weiteres Format wäre eine zweite Wahrheit.

### 8.1 Komponentenmatrix

IMPL = implementiert · CALLED = in einem Workflow/Lauf aufgerufen · FLOW = Output hat einen Abnehmer ·
TEST = Unit-/Integrationstests · OOS = belastbarer OOS-/Forward-Nachweis · PROD = beeinflusst Trades.

| Komponente | IMPL | CALLED | FLOW | TEST | OOS | PROD | Status |
|---|---|---|---|---|---|---|---|
| Champion-Regelwerk (Gates, LLM, MC, Options, ROI, Sizing) | ja | täglich | Trades | ja | **nein** (PF 1,12, n=79) | ja | ACTIVE, UNVALIDATED |
| Final-MC-Hit-Rate | ja | täglich | Gate + Mail | ja | **fehlkalibriert** (0,71→0,22) | ja (Gate) | ACTIVE; Mail zeigt jetzt kalibrierte Quote |
| ROI-Gate (Sammel: roi_initial/theta/vega/edge/mc_pnl) | ja | täglich | Gate | ja | Schatten n=20: Ø −39 %, WR 20 % | ja | ACTIVE, filtert Verlierer (s. 8.3) |
| SystemState / Drift / Source Health | ja | täglich | Adapter, HC, Report | ja | – (Steuerung) | indirekt (Deckel/Block) | ACTIVE, kanonisch |
| PromotionController + Adapter | ja | täglich/wöchentl. | einzige Wirkungsstelle | ja | 5 Verträge, Forward n=0 | Einfluss NONE | ACTIVE (Protokoll) |
| Abstention-Risikovektor | ja | täglich | Trade-Record → Vorschläge | ja | nein | nein | SHADOW |
| QuasiML / model_weights / feature_stats (feedback.py) | ja | täglich | nur Tagesreport (als SHADOW beschriftet) | ja | nein | **nein** | SHADOW, lernt täglich ohne Abnehmer |
| Premium-Signale (FLASH Alpha/Eulerpool) | – | – | – | – | – | – | **entfernt** (§9): veränderten nur `final_score`, kein Abnehmer |
| PPO-Veto | ja | nein (veto_enabled=false) | – | ja | degeneriert | nein | UNUSED |
| Robuster PPO + RL-Promotion-Bewertung | ja | wöchentlich | Report | ja | WF 100 % BOOST → degeneriert | nein | SHADOW, NOT_READY |
| ML-Research / Meta-Learning / World Model / Causal | ja | Workflows | Report/Research | ja | kein Gate bestanden | nein | SHADOW |
| Hypothesis Factory / Research Memory / Inquiry | ja | wöchentlich | Plan, Ketten | ja | 86 getestet, 0 bestätigt | nein | SHADOW |
| Surprise Engine (S1–S4, S5 Holdout) | ja | research.yml | Memory/Report | ja | Lauf 1 ohne KEEP | nein | SHADOW |
| HC-Scanner | ja | täglich | Research-Alerts | ja | nein | nein | SHADOW |
| Alt-Data SEC/GLEIF/TED/Wetter/… | ja | täglich | External Context | ja | SEC: REJECT | nein | SHADOW |
| Kosten-Telemetrie / Routing / Cache / Prefilter | ja | täglich | Monatsreport, Routing-Entscheid | ja | A/B läuft | Kosten, nicht Trades | ACTIVE (Betrieb) |
| `meta_learning.meta_fallback_check` | – | – | – | – | – | – | **entfernt** (ohne Aufrufer; §9) |

### 8.2 Behobene Befunde dieses Durchgangs

| Prio | Befund | Fix |
|---|---|---|
| P1 | Tages-Stats hatten zwei `safe_mode`-Flags (Daten vs. kanonisch); die Log-Warnung las das Daten-Flag | Daten-Flag → `data_health.data_safe_mode`; Warnung nur aus `system_state.active` |
| P1 | Tages-Mail färbte die rohe MC-Quote ab 65 % grün und nannte fest codierte, veraltete Kalibrierzahlen (86 %→41 %) | neutral dargestellt + gemessene Band-Kalibrierung (dieselbe Funktion wie Montagsreport); fehlend → „n/a" |
| P1 | `shadow_trades` auf 300 gekappt → bewertete Gate-Evidenz ging verloren | verdrängte Einträge append-only in `outputs/shadow_trades_archive.jsonl` |
| P1 | „roi_gate" war ein Sammel-Label für 5 Teil-Gates → Wert einzelner Gates nicht messbar | `fail_gates` je Tier im Reject-Log und Schatten-Trade |
| P2 | Monatsreport mischte alle Schatten-Gründe zu einer Win-Rate und urteilte „Gates arbeiten korrekt"/„filtern Gewinner" | je Grund getrennt, n<10 → keine Aussage, Hinweis „ohne TP/SL" |
| P2 | Kostenabschnitt unterschied nicht zwischen gemessen und geschätzt | Labels MEASURED / ESTIMATED je Zeile |
| P2 | `SYSTEM_STATE.md` meldete Safe Mode aktiv (SEVERE), Code-Zustand MODERATE/aus | Doku auf Datei verwiesen, Stand korrigiert |
| P3 | `meta_learning.safe_mode_check` – Namenskollision mit kanonischem Safe Mode | → `meta_fallback_check` |
| P1 | Kein End-to-End-Test von `pipeline.main()` | `tests/test_pipeline_orchestration.py`: Data-Health kaputt, SystemState kaputt, VIX fehlt, leeres Universum, Ingestion-Absturz |

### 8.3 Gate-Audit Final-MC → Options → ROI-Gate (Schatten-Outcomes, `outputs/history.json`)

| Population | n | Ø Outcome | Win Rate | PF |
|---|---|---|---|---|
| Final-MC-Survivor ohne Strategie (Aktienrendite) | 37 | −2,7 % | 37,8 % | 0,50 |
| ROI-Gate-Rejects gesamt (Optionsrendite bestes Tier) | 20 | −39,0 % | 20,0 % | 0,21 |
| … davon Initial-ROI unter Hurdle (`roi_gap` < 0) | 4 | +37,1 % | 3/4 | – |
| … davon Initial-ROI bestanden, an Edge/MC-P&L gescheitert | 16 | −58,0 % | 1/16 | – |
| Echte (Paper-)Trades gesamt | 119 | +10,2 % | 34,5 % | 1,24 |
| Echte Trades seit 03.07. (gleicher Zeitraum) | 8 | −27,0 % | 12,5 % | 0,28 |

Befund: Kein Gate mit nachgewiesen **negativem** Inkrementalwert. Edge/MC-P&L-Teil-Gates filtern
deutlich Verlierer. Die reine ROI-Hurdle zeigt bei n=4 positive Rejects – das ist keine Evidenz
(n winzig, Schatten ohne TP/SL, 88 Rejects noch unbewertet), aber genau die Frage, die `fail_gates`
ab jetzt beantwortbar macht. Konsequenz: **keine Schwelle gelockert**; ab n ≥ 30 je Teil-Gate
als Gate-Challenger (`challenger.py`) prüfen.

### 8.4 Offene Risiken und fehlende Verbindungen

* **Signalfrequenz vs. Promotion-Untergrenzen:** seit Juli ≈ 4 Champion-Trades/Monat; Verträge
  verlangen n ≥ 50 und ≥ 20 unabhängige Tage → realistisch ≈ 12 Monate bis zur ersten
  Forward-Entscheidung. Die Policy darf nur strenger werden; eine Erweiterung der Beobachtungs-
  population (z. B. Final-MC-Survivor als zusätzliche Abstinenz-Population) wäre eine neue
  Vertragsversion und gehört per PR in menschliche Hand.
* Schatten-Outcomes ohne TP/SL (anders als echte Trades) – nur innerhalb eines Grundes vergleichbar.
* QuasiML/Pearson-Gewichte lernen täglich, ohne Produktionswirkung – Kosten ohne Nutzen, aber harmlos.
* Kalibrierungsbänder sind in-sample über alle Paper-Trades geschätzt (deskriptiv, nicht OOS).
* Entity-Map-Survivorship (147 Ticker), Sektor nicht PIT (s. 2.) – unverändert offen.

### 8.5 Pflichtfragen

1. **Widersprüchliche Systemzustände?** Im Code nein: ein kanonischer `system_state`, alle Leser über
   `safe_mode_view`, Test gegen Legacy-Leser. Behoben: Doppel-Flag in Tages-Stats und veraltete Doku.
2. **Lernsysteme am PromotionController vorbei?** Ein Loop lernt ohne Controller: `feedback.py`
   (feature_stats-Bins, Pearson-Gewichte). Seine Ausgabe (`final_score`) wählt nur die 2 Kandidaten für
   Premium-Daten; diese Felder liest kein Gate → kein Bypass. Kein Modul schreibt `config.yaml`/Gates;
   Challenger/Alpha-Discovery schreiben nur Vorschlagsdateien.
3. **Features unsicherer Herkunft?** Jede Panel-/Alt-Feature hat eine dokumentierte Quelle
   (`FEATURE_DEPENDENCIES.md`, Test). Unsicher: 40 rekonstruierte Outcomes (vom Lernen ausgeschlossen),
   Schatten-Outcomes ohne `outcome_method` (Altbestand), Sektor im ML-Panel nicht PIT.
4. **Leakage-Risiken?** Vorwärts-Shifts nur in Labels mit `label_end`-Purge; PIT-Tests für Surprise
   Engine, External Context, XBRL. Restrisiko: kontaminierter Locked-Holdout (bindend nur Forward),
   in-sample Kalibrierungsbänder, Survivorship im ML-Panel.
5. **Ungenutzte oder scheinbar aktive Intelligenz?** PPO-Veto (aus), QuasiML-Gewichte (lernen, ohne
   Wirkung), Premium-Signale, `meta_fallback_check` (ohne Aufrufer). Alles andere ist ehrlich SHADOW.
6. **Gates, die Alpha filtern?** Nicht nachweisbar. Edge/MC-P&L filtern Verlierer (n=16, Ø −58 %);
   reine ROI-Hurdle n=4 positiv → unentschieden, jetzt messbar über `fail_gates`.
7. **Wahrscheinlichkeiten kalibriert?** Nein. MC-Hit-Rate 0,71 → 0,22 realisiert. Alle Berichte
   (Montag, jetzt auch Tages-Mail) zeigen die gemessene Band-Win-Rate statt der Rohzahl.
8. **Berichte konsistent?** Safe Mode/Drift aus derselben Quelle; Kalibrierung aus derselben Funktion;
   Schattenstatistik je Grund; Kosten mit MEASURED/ESTIMATED.
9. **Kann ein fehlerhaftes Modell Produktion beeinflussen?** Nur über den Adapter, mit Einfluss NONE,
   Safe-Mode-/Drift-Deckel, Hard Caps und Demotion. ML/RL/Meta haben keinen Pfad. Restrisiko ist das
   Champion-Regelwerk selbst (LLM + unkalibrierte MC) – unvalidiert, aber nicht lernend.
10. **Ist jede Produktionsentscheidung reproduzierbar?** Teilweise: Candidate Ledger, Decision-Ledger
    mit eingefrorener Regelauswertung, `state_version`, `spec_hash`, Prompt-Version und Kosten-Ledger
    sind gespeichert. Nicht reproduzierbar sind LLM-Antworten (stochastisch) und Live-Optionsketten.
11. **Forward getrennt von Backtests?** Ja: Promotion-Evidenz nur ab `forward_start` und nach
    `registered_at`; Backtest/WF in Research-Artefakten; Paper-Trades in `closed_trades`;
    Schatten-/Counterfactual-Trades getrennt. Es gibt keine echten (Broker-)Trades.
12. **Was verhindert am stärksten nachweisbare Verbesserung?** Die Stichprobe: ≈ 4 Champion-Trades pro
    Monat und ein Champion ohne Vorteil (PF 1,12; seit Juli negativ). Dadurch braucht jede
    Forward-Entscheidung etwa ein Jahr. Hebel ohne Lockerung: Schatten-Populationen (Final-MC-Survivor,
    Gate-Rejects mit `fail_gates`) vollständig und vergleichbar bewerten – als Evidenzquelle für
    Gate-Challenger, nicht als Promotion-Abkürzung.

## 9. Nächster Schritt: schnellere Forward-Evidenz ohne Vermischung (2026-10-03)

Baseline eingefroren: `outputs/state/baselines/post-pr91_2026-10-03.json` (Commit ae46e9a, Hashes
von Config, Verträgen, Policies, Registry- und Transition-Kette). Produktionslogik unverändert.

**Neue Population statt Lockerung.** Die fünf Abstention-Regeln laufen zusätzlich als
`PROM-ABST-00x@v2` auf `FINAL_MC_SURVIVOR` (Details `docs/PROMOTION_CONTROLLER.md`, Populationen).
v1 bleibt byte-identisch (spec_hash gegen Baseline getestet). Jeder Survivor wird mit eingefrorener
Vertragsauswertung, SystemState, Regime, VIX, Modelluneinigkeit, erwartetem Drawdown, späterem
Downstream (`fail_gates`, Champion-Entscheidung) und 20/45/60-T-Outcomes inkl. MFE/MAE geführt.
Kein Survivor wird dadurch zum Trade.

**Erwartete Zeit bis zur ersten zulässigen Entscheidung** (Rate Jul–Sep: Ø 51 Survivors, 35
Ereignis-Cluster, 15 Signaltage je Monat; Sep allein 95/60/22):

| | v1 Champion | v2 Final-MC |
|---|---|---|
| Beobachtungen/Monat | ≈ 4 | ≈ 50–95 |
| Mindest-N / Cluster / Tage / Spanne | 60 / – / 20 / 90 T | 150 / 80 / 30 / 90 T |
| bindend | N (≈ 15 Monate) | Kalenderspanne 90 T |
| + Outcome-Horizont 45 T → erste Entscheidung | ≈ Anfang 2028 | ≈ Mitte Februar 2027 (nächster monatlicher Look) |

Treffer-Mindestwerte (30 Treffer, 15 Treffer-Cluster, 10 Treffer-Tage) bei geschätzten Trefferquoten
auf den bisherigen Survivors: PROM-ABST-004 (Drawdown) ≈ 27 % und 002 (Uneinigkeit)/003 (Blind-Spot-
Sektor) ≈ 15 % → in 2–4 Monaten erreichbar; 001 (Safe Mode aktiv) und 005 (VIX > 30, bisher max. 19,9)
feuern im aktuellen Regime praktisch nie → NEED_MORE_DATA bis zu einem Regimewechsel.

**ROI-Teil-Gates:** keine Schwelle geändert; Bewertung je Teil-Gate erst ab n ≥ 30 (Montagsbericht §8C).

**Holdouts:** ML-Locked ab 2025-07 CONTAMINATED/USED; Surprise-Jahre vor 2019 USED. S5 nur noch auf
`config/s5_forward_holdout.yaml` (Meldungen 2026-10-05…2027-10-04, eine Auswertung ab 2027-11-15,
Kriterien und Datei-Hash vorab gepinnt; bis dahin nur Fallzahlen).

**Datenrisiken:**
* Sektor im ML-Panel jetzt point-in-time (`sector_history.jsonl`, erst ab Beobachtung; davor
  `unknown`). Folge: historische Sektor-Attribution überwiegend `unknown` – ehrlich statt verzerrt.
* Kalibrierung: Bänder für neue Kandidaten nur aus der Vergangenheit; Güte prequential
  (Fit nur auf vor dem Entry geschlossenen Trades): n = 8 auswertbar, Brier 0,464 (roh) → 0,147
  (kalibriert), ECE 0,60 → 0,23 – Richtung klar, Stichprobe klein. Läuft jetzt wöchentlich.
* Survivorship: 135 von 255 entfernten Titeln ohne Yahoo-Kurse – nicht eliminierbar mit offiziellen
  freien Quellen; ab dem nächsten ML-Lauf quantifiziert (`universe.survivorship`: fehlender
  Mitgliederanteil je Jahr, Worst-Case-Verschiebung des Querschnittsmittels).

**Bereinigt:** Premium-Signale (Aufruf + Modul) und `meta_fallback_check` entfernt; QuasiML als
SHADOW beschriftet (der `safe_mode`-Block in der gepinnten `meta_protocol.yaml` ist damit ungenutzt).
