# Forensischer Acceptance-Audit – Adaptive-Asymmetry-Scanner

- **Stand:** 2026-10-02.
- **Geprüfter Commit:** `e5de692` (main).
- **Prüfmodus:** nur prüfen; nichts wurde repariert. Einzige Ergänzung ist
  `tests/test_forensic_audit.py`. Defekte sind dort als `xfail(strict=True)`
  mit Audit-ID markiert.
- **Wichtige Vorbemerkung:** Ein großer Teil des geprüften Codes und der Doku
  stammt aus früheren Sitzungen desselben Assistenten. Frühere Aussagen,
  auch eigene, gelten hier ausdrücklich **nicht** als Beleg. Mehrere davon
  werden unten widerlegt.

**Beweisquellen:**
- Code mit Datei und Funktion;
- ausgeführte Tests;
- echte CI-Läufe mit Lauf-ID;
- versionierte Artefakte in `outputs/` mit Commit.

**Einschränkung:** Yahoo Finance ist in der Audit-Sandbox per
Organisationsrichtlinie gesperrt (`connect_rejected`). Backtests auf echten
Kursdaten konnten daher nur in GitHub Actions laufen, nicht lokal.

---

## 0. KRITISCHE BEFUNDE (zuerst)

### CRITICAL FAILURES
| ID | Befund | Beleg |
|---|---|---|
| **F01** | **Survivorship-Bias im gesamten Research-Stack.** Das Panel 2014–2026 wird aus der *heutigen* S&P-500-/Nasdaq-100-Liste gebaut; delistete Titel werden aktiv entfernt. Er wirkt auf alle Walk-Forward-, Meta-, World-Model- und Validierungszahlen. Bekannt und dokumentiert, aber nicht behoben. | `modules/universe.py:get_universe` (Wikipedia + `_DELISTED`-Filter), aufgerufen von `ml_research.build_research_panel`; `tests/test_forensic_audit.py::test_A5_research_universe_is_survivorship_free` (XFAIL) |
| **F02** | **Locked Holdout ist KONTAMINIERT.** Er wird bei jedem Full-Lauf erneut für alle Modelle ausgewertet: `locked_evaluations_total = 28`, je Modell 4–5×. Die Ergebnisse wurden berichtet und flossen in Verdikte ein (z.B. F und Blind-Spot „Locked 1,9 % → 0,7 %“). Die Live-Wahrscheinlichkeitsabbildung `prob_map` wird **auf dem Locked-Fold gefittet**. | `ml_research.run` (`book[mid]["locked_evaluations"] += 1`), `outputs/research/ml_registry_log.json`, `meta_learning.prob_maps` (`use = "locked"`), `outputs/research/hc_thresholds.json` (`prob_map.fold = "locked"`, n = 31 179) |
| **F03** | **Die „Bestätigung auf ungesehenen Jahren 2019–2020“ ist nicht ungesehen.** Ablauf am 2026-09-29 (Commit-Zeiten): (1) Regime-Attribution `vix_lt_20 / vix_ge_20` über alle WF-Jahre ab 2019 in `ml_research.json`, 16:15 (#66), auch 16:59 (#67); (2) Regel „VIX ≥ 20 oder SPY < SMA200“ in `next_protocol.yaml`, 18:06 (#69). Die Bestätigungsjahre waren also Teil des Befunds, aus dem die Regel abgeleitet wurde. t = 3,27 bzw. 3,4 ist **kein** sauberer OOS-Beleg. | `git show 2f9de8d:outputs/research/ml_research.json` (`attribution.regimes`), `config/research_protocol.yaml:first_test_year: 2019`, `next_intelligence.confirm_abstention` |
| **F04** | **CI führt keinen einzigen Test aus.** Kein Workflow hat einen `push`- oder `pull_request`-Trigger, kein Workflow ruft `pytest` auf. PRs #69–#73 wurden ohne einen einzigen Check-Run gemerged (`get_check_runs`: `total_count 0`). | `.github/workflows/*.yml` (alle nur `schedule` / `workflow_dispatch`) |

### HIGH-RISK ISSUES
| ID | Befund | Beleg |
|---|---|---|
| **F05** | **Kalibrierungs-Leakage (klein):** `lagged_calibration` fittet die Isotonie auf dem Vorjahres-Fold ohne Purge. Etwa 4 Wochen Vorjahreslabels enden *nach* Testbeginn. | `meta_learning.lagged_calibration`; `test_A5_lagged_calibration_purges_overlapping_labels` (XFAIL, 240 von 3 180 Zeilen überlappen) |
| **F06** | **Governance:** Die „geschützten“, hash-gepinnten Protokolle (`next_protocol.yaml`, `intelligence_protocol.yaml`) wurden vom selben Agenten erstellt, der auch die Pins schreibt und die PRs ohne menschliches Review mergt. CODEOWNERS wird nicht erzwungen (Merges ohne Review gelangen). Der Hash-Pin schützt so nicht gegen den Agenten. | `git log config/next_protocol.yaml`, PR-Merges #69–#73 ohne Review |
| **F07** | **Safe Mode fail-open:** Fehlt `safe_mode.json` oder ist es beschädigt, gilt der Safe Mode als *nicht aktiv*. Andere Gates bleiben bestehen. | `hc_scanner.run` → `_load(..., {})`; 2 XFAIL-Tests |
| **F10** | **Alert-Verlust:** Schlägt der Mailversand fehl (`not_configured` oder Fehler), wird der Alert trotzdem als gemeldet in `alerts_state.json` geschrieben. Es gibt keinen Retry. | `hc_scanner.run` + `dedup`; `test_A16_failed_delivery_is_retried` (XFAIL) |
| **F11** | **Weekly Report widerspricht sich und ist veraltet.** Abschnitt 1: „Safe Mode: aus“ (aus `meta_state.safe_mode`, seit der Entkopplung fest `false`); Abschnitte 6/10: „SAFE MODE aktiv“. Der Report vom 2026-10-02 15:14 zeigt Meta-Zahlen vom **29.09.** (Checkout auf der SHA zum Dispatch-Zeitpunkt, vor dem ML-Commit 15:12). Abschnitt 3 zeigt die Kalibrierung des *verworfenen* `meta_regime_weights` statt des aktiven `static_equal`. | `reports/weekly.py:576` vs. `:484`; `outputs/reports/weekly_2026-10-02.md`; `actions/checkout` ohne `ref` |
| **F12** | **HC-System nie in Produktion aktiv, Wahrscheinlichkeiten nicht kalibriert.** `hc_thresholds.enabled = false` in allen 3 Läufen. Es wurde nie ein HC-Alert erzeugt, `alerts_log.jsonl` existiert nicht. Die Intelligenz-Prüfungen (Counterfactual, KG, Blind Spot, Portfolio) liefen **nie** auf echten Kandidaten. | `outputs/research/hc_candidates.json` (Historie 3d7f14d, 26d0456, 7fa38cc) |
| **F13** | **Zwei getrennte Systeme.** Der tägliche Produktions-Scanner `pipeline.py` (LLM-Prescreen → Deep Analysis → MC → Optionen → Mail) importiert **kein** Modul des ML-/Intelligenz-Stacks. World Model, Meta-Learning, Causal, KG, Counterfactual, Blind Spots, DI, Meta-Cognition und Safe Mode haben **keinen** Einfluss auf die täglichen Trade-Vorschläge. | Importgraph (Abschnitt 3); `scanner.yml` ruft nur `pipeline.py` und `feedback.py` auf |

### PARTIAL IMPLEMENTATIONS
- **Knowledge Graph:** 612 Kanten, alle `reference_data` oder `curated_config`, **0 gemessene Kanten**. `contradictions` kann damit nie auftreten; die HC-Prüfung ist faktisch wirkungslos. Für SNDK und MU ist der Teilgraph leer.
- **Active Learning, Research Value Attribution, Failure Analyzer:** erzeugen nur Text (Weekly/Machine-State). Keine operative Konsequenz.
- **Meta-Cognition:** `overall_calibration: GOOD` beruht nur auf aggregierter ECE ≤ 0,05 und der Intervall-Abdeckung. Den überkonfidenten Bucket 55–60 % ignoriert die Regel (F09, XFAIL). Der bestehende Test `test_machine_state_only_metric_statements` schreibt diese Aussage sogar fest.
- **Prediction Memory:** Versionen werden mit `model_versions: {m: null}` gespeichert; bisher wurde 0 Outcomes erfasst (Start 2026-09-29).
- **Alpha Decay:** wird gemessen (CUSUM, Modell-Trend). Ein Merkmal wird deshalb aber nicht heruntergewichtet; nur die Modell-Mehrheitsregel blockiert HC.
- **Safe Mode:** Auslöser für „stale data“ (STALE-Quellen) und „beschädigtes Modell“ fehlen (F08, XFAIL).

### DEAD CODE
| Modul | Befund |
|---|---|
| `modules/news_fetcher.py` (234 Z.) | von keinem Produktivcode, keinem Workflow und keinem Test importiert |
| `modules/reddit_signals.py` (261 Z.) | dito |
| KG-Widerspruchsprüfung in `hc_scanner.intelligence_checks` | Ohne gemessene Kanten kann der Code nie greifen. |
| Intelligenz-Prüfungen in `hc_scanner.intelligence_checks` | erreichbar nur bei aktiver HC-Regel; in Produktion nie ausgeführt |

### MISSING TESTS
- **Bisher ungetestet, mit dem Audit ergänzt:** E2E-Test des Alert-Pfads über `hc_scanner.run`.
- **Ohne jeden Test** (`prod`-Importe vorhanden):
  - `intraday_delta`, `macro_context`, `premium_signals`, `rl_robust_shadow`, `universe`, `news_fetcher`, `reddit_signals`;
  - im Produktions-Scanner damit `macro_context`, `premium_signals` und `intraday_delta`.
- **Kein Test gegen echte Daten** für den ML-Stack. Alle ML-Tests nutzen synthetische Panels; echte Daten laufen nur in CI-Läufen ohne Assertions.
- **Kein Reproduzierbarkeitstest** (gleicher Commit, gleiche Daten).

### LEAKAGE RISKS
F01 (Survivorship), F02 (Holdout), F03 (Regel aus gesehenen Daten), F05 (Kalibrierung). Weitere:
- **Statischer Sektor:** heutiger Sektor (yfinance) rückwirkend für alle Jahre (`sector_map.json`). Betrifft Blind-Spot-Cluster, Sektor-Attribution und DI-Sektorkappe.
- **`auto_adjust=True`:** heutige Dividenden- und Split-Faktoren rückwirkend. Renditen bleiben weitgehend korrekt, Niveaus nicht. Gering.
- **Knowledge Graph:** wird aus heutigen Configs gebaut, ohne `valid_from` je Kante. Nicht PIT für historische Nutzung.

### UNVALIDATED CLAIMS (in Doku oder Berichten, widerlegt oder unbelegt)
| Aussage | Wo | Befund |
|---|---|---|
| „Abstinenz auf ungesehenen Jahren bestätigt (t ≈ 3,3)“ | `NEXT_INTELLIGENCE_VALIDATION.md`, `hypotheses.yaml` HYP-ABST-001, `machine_state` | **CONTAMINATED** (F03) |
| „Variante B = Basis-Faktormodell“ | `NEXT_INTELLIGENCE_VALIDATION.md` | **falsch**: B = `meta_regime_weights` („Champion + Meta-Learning“) laut `next_protocol.yaml` |
| „Meta-Learning: REJECT“ | `META_LEARNING_VALIDATION.md`, Gap-Analyse | Lauf vom 2026-10-02: **NEED_MORE_DATA** (Δ Sharpe +0,208, CI [−0,44 %, +0,74 %]). Das Urteil ist instabil. |
| „Champion = statisches Ensemble“ | mehrere Docs | Die Registry hat **keinen** Champion (`champion: null`, Weekly: „Current Champion: keiner“). A ist eine Referenz, kein registrierter Champion. |
| „overall_calibration GOOD“ | `machine_state` | Bucket 55–60 %: vorhergesagt 0,572, realisiert 0,464 (F09) |
| „World Model signifikant besser“ | `WORLD_MODEL.md` | Besser nur gegen die Regime-Engine; **schlechter als naive Klimatologie** (Skill −1,11) und Drawdown-AUC **0,415 < 0,5** |
| „Locked: G +3,4 % vs. A +1,9 %“ | `NEXT_INTELLIGENCE_VALIDATION.md` | Locked ist kontaminiert (F02, F03) |
| Weekly „Safe Mode: aus“ | `weekly_2026-10-02.md` | **falsch** (F11) |

---

## 1. Tatsächliche Architekturkarte

Es gibt **zwei** unabhängige Systeme. Im Code gibt es keine Verbindung zwischen
ihnen; sie teilen sich nur `outputs/` (Weekly liest beide).

### A) Produktions-Scanner (täglich, `scanner.yml` 13:30 UTC, Mo–Fr)
```
DATA      yfinance/Finnhub/SEC/FDA/Reddit?  ── pipeline.py:run()
          universe.get_universe → data_ingestion → market_snapshot
FEATURES  alpha_sources.enrich_with_alpha_sources, data_validator.validate_candidate_data,
          finbert_sentiment, sentiment_tracker, macro_context, premium_signals
MODELS    prescreener (LLM Haiku) → deep_analysis (LLM Sonnet) → mismatch_scorer
          → mirofish_simulation (MC) → trade_scorer / quasi_ml / rl_agent (PPO, Veto aus)
SIGNAL    options_designer → risk_gates → position_sizing
ALERT     email_reporter.send_email (tägliche Mail); reporter; candidate_ledger
FEEDBACK  feedback.py (Outcomes, history.json) → engine_monitor, trade_memory, factor_monitor
```

### B) Research- und Intelligenz-Stack (`ml_research.yml`: Sa weekly, 3. des Monats full; Weekly-Mail Mo)
```
DATA      ml_research.fetch_data (yfinance, heutiges Universum!) + external archive (ALFRED-Vintages)
FEATURES  ml_research.build_panel → ticker_frame/market_features/macro_features (PIT, Querschnittsränge)
MODELS    ml_research.walk_forward / locked_eval / predict_latest (6 Modelle, Registry)
META      meta_learning.base_oos → meta-Varianten, lagged_calibration, prob_maps, hc_rule
VALIDATION next_intelligence (A–G, Ablation, Gate), world_model, causal_research, research_lab
SIGNAL    ml_research.build_cards → hc_scanner.evaluate_candidates → intelligence_checks
ALERT     hc_scanner.run(--send) → mailer.send_mail;  reports/weekly.py (Mo)
```

### Reale Verbindungen (Codepfad)
| Verbindung | Codepfad | real ausgeführt? |
|---|---|---|
| Daten → Features | `ml_research.build_research_panel` → `build_panel` | ja, CI 37021907362 (`n_rows`, `period 2014-06-06..2026-10-01`) |
| Features → Modelle | `ml_research.run` → `walk_forward`, `predict_latest` | ja, `ml_predictions/2026-10.jsonl` (6 Modelle, `code_sha 7fbb9d334681`) |
| Modelle → Meta | `meta_learning.main` (`META_RES_CACHE`) | ja, `meta_learning.json` (`panel_hash`, `meta_version meta-v1`) |
| Meta → Validierung | `next_intelligence.main` liest `meta_learning.LAST_RUN` | ja, `next_validation.json` 2026-10-02 |
| Validierung → Signal | `hc_scanner.regime_compatible` liest `next_validation.abstention_confirmation` | Code ja; in Produktion nie erreicht (globales Gate) |
| Signal → Alert | `hc_scanner.run` → `mailer.send_mail` | **nie** (`enabled false` in allen Läufen) |
| World Model → Signal | nur `meta_cognition.safe_mode` (Unsicherheit ≥ 0,6) und Challenger C | Safe-Mode-Pfad ja; kein Signalpfad |
| Research → Lab | `research_director` → `director_hypotheses.json` → `research_lab` | ja, CI 37021907362 (5 RD-Hypothesen, alle REJECTED) |
| ML-Stack → tägliche Trades | – | **existiert nicht** (F13) |

---

## 2. Component Acceptance Matrix

Spalten: IMPL = Code existiert · CONN = realer Aufrufer im Prod- oder CI-Pfad ·
TEST = Tests bestehen · OOS = echter OOS-Nutzen gemessen · PROD = beeinflusst
Produktionsentscheidungen (tägliche Mail oder HC-Alert).

| Komponente | IMPL | CONN | TEST | OOS | PROD | STATUS | Evidenz |
|---|---|---|---|---|---|---|---|
| Historical Feature Store | ja | ja | ja (synthetisch) | – | nur Research | **PARTIAL** | `ml_research.build_panel`; PIT-Tests `test_A5_features_invariant_to_future_prices` PASS; Survivorship F01 |
| Label Engine | ja | ja | ja | – | Research | **PASS** | `ticker_frame` (Einstieg Open t+1, `label_end`); `test_A5_labels_start_at_next_open` PASS |
| Walk-forward Validation | ja | ja | ja | ja | Research | **PARTIAL** | `walk_forward` jahresweise; Survivorship verzerrt alle Zahlen |
| Purging / Embargo | Purge ja, Embargo nein | ja | ja | – | Research | **PARTIAL** | `ml_research.purged` korrekt (`test_A5_purge…` PASS); kein Purge in `lagged_calibration` (F05) |
| Baseline Models | ja | ja | ja | ja | Research | **PASS** | `momentum_12_1` (Benchmark) WF Sharpe 0,40, IC 0,0018 |
| Specialized Models | ja | ja | ja | ja | Research | **NOT VALIDATED** | 5 Challenger, alle `rejected_so_far`; Registry-Champion = keiner |
| Regime Engine | ja | ja | ja | ja | Research | **PARTIAL** | `external/regime.py`, `market_features`; World Model ist nur gegen sie besser |
| Prediction Memory | ja | ja | ja | – | Research | **PARTIAL** | `prediction_memory/2026-10.jsonl` 50 Zeilen, `model_versions` null, 0 Outcomes |
| Trade Memory | ja | ja (CI) | ja | – | nur Weekly | **PARTIAL** | `trade_memory.jsonl` 179 Fälle; 2 von 5 jüngsten Trades ohne Eintrag (Weekly §7) |
| Failure Analyzer | ja | ja | ja | – | nein | **UNUSED** (operativ) | `failure_analysis.json` → nur Weekly-Text |
| Feature Factory | ja | ja | ja | ja | nein | **PASS** (als Prozess) | `research_lab` Discovery: 117 getestet, 0 akzeptiert (BH über alle) |
| Historical Analogy Engine | ja | ja | ja | **nein** | HC-Gate (nie aktiv) | **NOT VALIDATED** | `ml_research.analogs` (PIT: `purged(…, 60)`); OOS-Nutzen nie gemessen |
| Uncertainty Estimation | ja | ja | ja | ja | Karten | **PARTIAL** | Intervalle roh 0,65 → korrigiert 0,79 (Ziel 0,80); P(>10 %) Skill −0,15 (`p_up_informative false`) |
| Probability Calibration | ja | ja | ja | ja | HC (nie) | **FAIL** | Buckets: 55–60 % → 46,4 % realisiert; Brier 0,2495 ≈ Zufall; F05 |
| Research Engine | ja | ja | ja | ja | nein | **PASS** (Prozess) | `research_lab`, hypothesis_db, BH über alle Tests |
| Experiment Registry | ja | ja | ja | – | – | **PASS** | `model_registry.yaml` + `ml_registry_log.json` (`spec_hash`, `first_seen`, Zähler) |
| Champion–Challenger | ja | ja | ja | ja | nein | **PARTIAL** | Mechanik läuft; kein Champion registriert; Docs nennen A „Champion“ |
| Locked Holdout | ja | ja | ja | – | – | **CONTAMINATED** | F02, F03 |
| Performance Attribution | ja | ja | ja | – | Report | **PASS** | `ml_research.json` → `attribution.{importance,sectors,regimes}` |
| Meta-Learning | ja | ja | ja | ja | nein | **NOT VALIDATED** | Abschnitt 7; Urteil kippt zwischen Läufen |
| Dynamic Model Weighting | ja | ja | ja | ja | nein | **NOT VALIDATED** | `trailing_ic_weighted` Sharpe 0,093 < static 0,264 |
| Model Disagreement | ja | ja | ja | – | Karten, Safe Mode | **PARTIAL** | `disagreement.current_level NORMAL`; OOS-Wert (Ablation `no_disagreement`) uneinheitlich |
| World Model | ja | ja | ja | ja | nur Safe Mode | **NOT VALIDATED** | Skill gegen Klimatologie negativ; Challenger C Δ −0,02 %/M n.s. |
| Causal Research Layer | ja | ja | ja | ja | nein | **NOT VALIDATED** | 0/32 nach BH; keine aktive Relation; D Δ −0,04 %/M n.s. |
| Knowledge Graph | ja | ja | ja | ja | wirkungslos | **UNUSED** | 0 gemessene Kanten; SNDK-Teilgraph leer |
| Active Learning | ja | ja | ja | – | nein | **PARTIAL** | `active_learning.json` → Weekly §16; keine Datenquelle je onboarded |
| Autonomous Research Planner | ja | ja | ja | ja | nein | **PASS** (Prozess) | CI 37021907362: 5 RD-Hypothesen erzeugt, getestet, verworfen |
| Research Memory | ja | ja | ja | – | – | **PASS** | `hypothesis_db.json` mit Status-Vokabular, Gründen, Code-Version |
| Hypothesis Similarity Detection | ja | ja | ja | – | – | **PASS** | RD-38d323d8 (Jaccard 0,64) und RD-d0093d4e (0,65) automatisch gesperrt |
| Counterfactual Engine | ja | ja | ja | ja | HC-Gate (nie aktiv) | **PARTIAL** | F-Filter schadet (Sharpe 0,102 vs 0,262); Gate in HC real (Test PASS), aber nie ausgeführt |
| Adversarial Research / Self-Play | ja | ja | ja | – | – | **PARTIAL** | `research_lab.adversarial_review` = deterministische Prüfkette, keine getrennten Agenten |
| Unknown-Unknown Detector | ja | ja | ja | ja | HC-Gate (nie aktiv) | **PARTIAL** | 15 Cluster, binomial-z + BH, datenbasiert; als Filter schädlich; Rückfluss in Director PASS |
| Decision Intelligence | ja | ja | ja | ja | HC-Gate (nie aktiv) | **NOT VALIDATED** | Urteil KEEP → MODIFY zwischen Läufen |
| Portfolio Layer | teilweise | – | ja | – | nein | **PARTIAL** | nur `decision_intel`; Produktion nutzt `position_sizing` (Optionen) |
| Meta-Cognition | ja | ja | ja | – | Safe Mode | **PARTIAL** | F09; Aussagen tragen Metriken, Label-Regel ungenügend |
| Alpha Decay Detection | ja | ja | ja | – | HC-Gate (Mehrheit) | **PARTIAL** | CUSUM-Brüche gemeldet, keine Gewichtsanpassung |
| Safe Mode | ja | ja | ja | – | blockiert HC | **PARTIAL** | Auslöser real (`safe_mode.json` aktiv: tnx); fail-open (F07); kein STALE-Auslöser (F08); wirkt nicht auf tägliche Mail (F13) |
| High-Confidence Candidate Scanner | ja | ja | ja | **nein** | nie aktiv | **NOT VALIDATED** | F12; Buckets nicht monoton |
| Alert Deduplication | ja | ja | ja | – | – | **PARTIAL** | Duplikat/Regimewechsel PASS (E2E); Verlust bei Fehlversand (F10) |
| Weekly Intelligence Email | ja | ja | ja | – | ja | **PARTIAL** | echte Daten, Quellen getrennt ausgewiesen; F11 |
| High-Confidence Email Alerts | ja | ja | ja (Dry-Run) | – | nie | **UNUSED** | nie gesendet |

---

## 3. Dead-Code-Audit (Importgraph, ohne Tests)

Erhoben per statischer Importanalyse über alle `.py`-Dateien und Workflows.

- **Kein Aufrufer:** `news_fetcher`, `reddit_signals`.
- **Nur Workflow, kein Modul-Aufrufer (Einstiegspunkte, korrekt):** `hc_scanner`, `meta_cognition`, `next_intelligence`, `research_director`, `trade_memory`, `price_event_study`, `alpha_discovery`.
- **Berechnet, aber ignoriert:**
  - `world_model.json`: kein Signalpfad, nur Safe Mode und Report;
  - `causal_research.json`: nur Challenger D/E und KG, ohne Effekt;
  - `knowledge_graph.json`: Widerspruchsprüfung ohne Messkanten;
  - `active_learning.json` und `research_value`: nur Text;
  - `failure_analysis.json`: nur Text;
  - `machine_state.*`: Text; der Safe Mode kommt aus `safe_mode.json`.
- **Counterfactual, Blind Spot, DI in HC:** Codepfad real und getestet, aber hinter einem nie geöffneten globalen Gate.
- **Gesamtes Intelligenz-System:** keine Verbindung zum täglichen Produktions-Scanner (F13).

---

## 4. End-to-End-Trace (realer Ticker, realer Stichtag)

**SNDK, Stichtag 2026-10-01**, Lauf CI 37021907362, Code `7fbb9d334681`. SNDK
ist Platz 1 im statischen Ensemble.

| Schritt | Input | Output | Zeit / Version | Funktion |
|---|---|---|---|---|
| Rohdaten | yfinance-Tageskurse bis Close 2026-10-01 (SPY < heute gefiltert) | – | Abruf 2026-10-02 ~14:45 UTC | `ml_research.fetch_data` |
| PIT-Features | Kurse ≤ t | 13 Querschnittsränge + 10 Datumsmerkmale | `features_hash 15b4e20867490267` | `ticker_frame`, `build_panel` |
| World State | ALFRED-Vintages ≤ t, Märkte | z.B. `interest_rates high`, `breadth low`, `uncertainty 0.327` | `world_state.jsonl` mit `version` | `world_model` (**ohne Einfluss auf SNDK**) |
| Basis-Prognosen | Features | Ränge 0,994–1,000 (6 Modelle) | `train_cutoff 2026-09-28`, `spec_hash` je Modell | `predict_latest` |
| Meta-Gewichte | – | nicht aktiv: `static_equal` | `meta-v1`, `meta_active false` | `meta_state.json` |
| Unsicherheit | Quantilmodelle (Labels < t) | E[R60] +2,2 %, 80-%-Intervall [−30,2 %, +83,0 %], Disagreement LOW (sd 0,002) | Karte 2026-10-01 | `build_cards` |
| Kalibrierung | Rang → `prob_map` (**Locked-Fold**, F02) | P = 0,590 | `hc_thresholds.prob_map` | `hc_scanner._interp` |
| Analogien | 25 nächste Nachbarn mit fertigem 60d-Label | 56 % positiv, Median +7,0 %, MAE −18,4 %, MFE +27,7 % | 2017–2026 | `ml_research.analogs` |
| Causal Evidence | – | **keine** (0 Relationen) | – | – |
| KG Evidence | Graph `7e9edfcf500a10f3` | **leer** (`exposures [], leading [], contradictions []`) | – | `knowledge_graph.ticker_evidence` |
| Counterfactual | – | **nicht ausgeführt** (Gate davor geschlossen); Karte: trägt `beta_126`, spricht dagegen `vol_60` | – | `build_cards.counterfactual_drivers` |
| Risiko / Final | – | P(>+10 %) 0,141, `p_up_informative false` | – | – |
| HC-Entscheidung | Regel, Safe Mode, Drift | **kein HC**: Regel deaktiviert, Feature-Drift, Safe Mode (tnx 5,24 außerhalb [1,10; 4,78]) | `hc_candidates.json` 2026-10-02 | `hc_scanner.global_gate` |
| Alert | – | **kein Alert** | – | – |

**PIT-Nachweis:**
- `train_cutoff` 2026-09-28 liegt vor dem Stichtag.
- Labels mit `label_end` < Stichtag (`purged`).
- ALFRED `as_of = Stichtag − 1 s`.
- Analogien nur aus `purged(panel, latest+1, 60)`.

**PIT-Verletzungen im Trace:**
- `prob_map` ist auf dem Locked-Fold gefittet. Das ist zwar Vergangenheit, aber
  der Holdout.
- Das Universum ist heutig (F01).
- Der Sektor ist heutig.

---

## 5. Look-Ahead- und Leakage-Angriff

| Risiko | Ergebnis | Beleg |
|---|---|---|
| Zukunftspreise in Features | **sauber** | `test_A5_features_invariant_to_future_prices` PASS (Preise nach t ×3 → Features zu t identisch) |
| Revidierte Makrodaten | **sauber** | ALFRED-Vintages, `macro_features(as_of = d − 1 s)` |
| Publikationszeitpunkte | sauber für Makro; Earnings-Termine nur live (Finnhub) | `regime.regime_state` |
| Survivorship / Universum | **LEAKAGE** | F01 |
| Forward-Fill künftiger Info | sauber (`reindex(cal)` ohne ffill für Titel; VIX/TNX ffill nur rückwärtsgerichtet) | `ticker_frame`, `market_features` |
| Skalierung vor Split | **sauber**: Querschnittsränge je Stichtag | `build_panel` |
| Feature-Selektion auf Gesamtdaten | **teilweise**: Feature-Listen fest präregistriert, aber nach Ansicht früherer Ergebnisse definiert | `model_registry.yaml` |
| Hyperparameter auf Testdaten | **sauber**: inneres Validierungsjahr innerhalb des Trainings | `select_params` |
| Stacking-Leakage | **sauber**: Meta trainiert nur auf Basis-OOS mit `label_end` < Testbeginn | `meta_learning.json.leakage_checks.ok = true`, `meta_provenance` |
| Kalibrierungs-Leakage | **klein**, F05 | XFAIL-Test |
| Analogien mit Zukunftsergebnissen | **sauber** | `analogs(train = purged(…, 60))` |
| Retrospektive Regime-Definition | **LEAKAGE**: Abstinenz-Regel aus gesehenen Jahren | F03 |
| Labels in Features | **sauber** | Feature-Whitelist ohne `fwd_*` |
| Holdout-Kontamination | **KONTAMINIERT** | F02 |

Automatisierte Tests: `tests/test_forensic_audit.py` (5 Leakage-Tests, davon 2 XFAIL).

---

## 6. Locked-Holdout-Audit

| Frage | Befund |
|---|---|
| Definition | `config/research_protocol.yaml: periods.locked_from = 2025-07-01` |
| Zeitraum | 2025-07-01 bis heute (wachsend) |
| Zugriff | `ml_research.locked_eval`, `meta_learning.base_oos` (Fold „locked“), `prob_maps`, `next_intelligence`, `research_lab.locked_check` |
| Training darauf verhindert? | ja für Modellfits (`purged(panel, locked_from)`); **nein** für `prob_maps` (Isotonie auf Locked) |
| Feature Engineering sieht ihn? | Panel enthält ihn; Querschnittsränge sind je Stichtag, also kein Lernen |
| Hyperparameter darauf optimiert? | nicht direkt; aber Verdikte und Protokolle wurden nach Locked-Ergebnissen gebildet |
| Wiederholt gesehen? | **ja**: 28 Auswertungen; Ergebnisse in jedem Full-Lauf berichtet und in Verdikten zitiert |
| **Urteil** | **KONTAMINIERT**. Ein neuer, wirklich unberührter Holdout ist nur noch vorwärts möglich. |

---

## 7. Meta-Learning-Audit

**Stacking-Leakage:** Das Meta-Modell nutzt nur historische OOS-Prognosen.
Belegt durch `meta_provenance` (`meta_train_max_label_end` < `test_start` je
Jahr) und `leakage_checks.ok = true`.

**Vergleich auf identischen OOS-Zeilen** (Meta-Testjahre 2021–2025H1, 11 587
Positionen, 230 Kohorten; Lauf 2026-10-02):

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | Hit | PF | Expectancy | Brier | LogLoss | ECE | Prec@K | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best Single (ex ante) | −2,65 % | −0,137 | −0,147 | −32,6 % | −0,081 | 45,8 % | 0,967 | −0,13 % | 0,24993 | 0,69301 | 0,0232 | 0,248 | 11 587 |
| Static Ensemble | 2,77 % | 0,264 | 0,294 | −17,9 % | 0,155 | 47,6 % | 1,080 | 0,32 % | 0,24952 | 0,69219 | 0,0062 | 0,267 | 11 587 |
| Weighted (trailing IC) | 0,38 % | 0,093 | 0,089 | −21,4 % | 0,018 | 47,2 % | 1,013 | 0,05 % | 0,24951 | 0,69217 | 0,0024 | 0,260 | 11 587 |
| Meta (Regime-Gewichte) | 4,77 % | 0,472 | 0,539 | −17,2 % | 0,277 | 48,4 % | 1,126 | 0,48 % | 0,24972 | 0,69260 | 0,0132 | 0,270 | 11 587 |
| Meta (Stacking) | −0,15 % | 0,057 | 0,064 | −23,4 % | −0,006 | 47,6 % | 1,029 | 0,12 % | 0,24936 | 0,69187 | 0,0020 | 0,268 | 11 587 |

**Urteil: NOT VALIDATED.**
- Meta (Regime-Gewichte) hat die höchste Dev-Sharpe. Das Bootstrap-CI für Δ
  Monatsrendite liegt aber bei [−0,44 %, +0,74 %].
- Ohne die besten 5 % der Monate ist der Effekt negativ.
- Locked ist schlechter: 0,766 gegen 1,504, und Locked ist ohnehin
  kontaminiert.
- Am 29.09. lautete das Urteil REJECT, am 02.10. NEED_MORE_DATA.
- Alle Brier-Werte liegen bei ≈ 0,2495, also auf Zufallsniveau
  (Basisrate ≈ 0,47).

---

## 8. World-Model-Audit

- **Mathematischer Zustand:** 17 Dimensionen, expandierende z-Scores, Score,
  State und Unsicherheit je Dimension.
- **PIT und Versionierung:** append-only `world_state.jsonl` (666 Zeilen) mit
  `version`. Mehr als eine Textzusammenfassung: **ja**.
- **Downstream:** nur der Safe-Mode-Auslöser und Challenger C. Die
  Abstinenz-Regel nutzt VIX und SPY-Trend direkt aus dem Panel, **nicht** das
  World Model.
- **Validierung:** gegen die Regime-Engine besser bei dd60 und Momentum-IC.
  Gegen naive Klimatologie ist der Skill aber negativ (dd60: −1,11), und die
  AUC für Drawdown binär liegt bei 0,415.
- **Ablation, mit gegen ohne (C gegen A):** Δ −0,02 %/Monat, CI
  [−0,63 %, +0,65 %], ECE 0,0165 gegen 0,0062, also schlechter.

**Urteil: NOT VALIDATED.**

## 9. Causal-Layer-Audit

- **Trennung von Kausalität und Prediction:** korrekt. `causal_evidence` ist
  nie automatisch gesetzt; die Stufen sind explizit.
- **Aktive Relationen:** 0. Keine von 32 Kombinationen besteht BH (q = 0,1).
- **Methode:** OOS-Granger (AR gegen AR + Treiber), Lead/Lag, Stabilität,
  Replikation über Horizonte.
- **Kontrollen:** nur eigene Lags, keine konditionalen Kontrollen.
- **Ablation (D gegen A):** Δ −0,04 %/Monat n.s.; Precision@K 0,255 gegen
  0,267.

**Urteil: NOT VALIDATED, ohne Downstream-Wirkung.**

## 10. Knowledge-Graph-Audit

- **Umfang:** 603 Knoten, 612 Kanten. Jede Kante hat `source`, `timestamp`,
  `confidence`, `evidence_type` und `version`. Keine LLM-Kanten.
- **Evidenztypen:** `reference_data` 546, `curated_config` 66,
  `measured_oos` 0.
- **Timestamp:** Bauzeit, nicht Gültigkeitsbeginn. Nicht PIT.
- **Realer Kandidat:**
  - SNDK und MU: Teilgraph **leer**.
  - MRNA: 2 Pfade (road_freight, weather), Konfidenz LOW, `uncertain`.
- **Auswertung bei Kandidaten:** nur in `intelligence_checks`, dort nie
  ausgeführt, und ohne Messkanten wirkungslos.

**Urteil: UNUSED.**

## 11. Active Learning / Research Director

**Echtes Beispiel**, Lauf CI 37021907362:
1. **Informationslücke:** Blind-Spot-Cluster „extreme 5T-Bewegung × Nähe
   52W-Hoch“ (Lift 1,66, BH-signifikant).
2. **Priorisierung:** EIG × Wert × Neuheit × (1 − Overfit) / Kosten.
3. **Hypothese:** RD-5f792ced `-(step(abs(ret_5d) - 0.4) * step(dist_52w_high - 0.3))`.
4. **Experiment:** Walk-Forward im Lab.
5. **Ergebnis:** netto −0,19 %, t = −1,85, nur 29 % der Jahre positiv → **REJECTED**.

Wiederholungstests: RD-38d323d8 und RD-d0093d4e wurden per Jaccard-Sperre
**nicht** getestet. **PASS.**

Active Learning im engeren Sinn ist **PARTIAL**:
- `active_learning.json` listet Datenlücken.
- Keine Quelle durchlief bisher die Aufnahmeprüfung.
- Es gibt keine Informationsgewinn-Messung nach Aufnahme.

## 12. Counterfactual-Audit

- **Reale HC-Signale:** keine vorhanden (F12). Counterfactuals wurden in
  Produktion nie für einen HC-Kandidaten ausgeführt.
- **Code-Konsequenz:** real. Fragil führt zur Ablehnung, das zeigt der E2E-Test
  `test_A16_fragile_counterfactual_blocks` (PASS).
- **Wirkung auf die Confidence:** keine. Es gibt nur ein binäres Veto.
- **OOS als Filter:** schädlich (F: Sharpe 0,102 gegen 0,262; 94 % der
  Top-Dezil-Positionen „fragil“).

**Urteil: PARTIAL.**

## 13. Unknown-Unknown-Audit

- **Clustering:** existiert und ist datenbasiert. Binomial-z gegen alle
  Top-Dezil-Positionen, BH q = 0,1, Lift ≥ 1,5, n ≥ 30.
- **Reproduzierbarkeit:** 15 Cluster je Fold in beiden Läufen. Die Top-Cluster
  (Energy × 12M-Verlierer, Lotterie, extreme 5T-Bewegung) wiederholen sich.
  Reihenfolge und Teile ändern sich mit einer weiteren Datenwoche.
- **Rückfluss in den Director:** nachgewiesen (5 RD-Hypothesen).
- **LLM-Narrative:** keine.

**Urteil: PARTIAL** (als Filter schädlich, als Diagnose funktionsfähig).

## 14. Meta-Cognition-Audit

| Claim | Metrik | Schwelle | Daten | Ergebnis |
|---|---|---|---|---|
| overall_calibration GOOD | ECE (aggregiert), interval_calibrated | ECE ≤ 0,05 | 0,0062; Intervalle korrigiert | **irreführend**: Top-Bucket überkonfident (F09) |
| world_model_uncertainty MODERATE | mittlere Dimensions-Unsicherheit | ≥ 0,3 / ≥ 0,6 | 0,327 | belegt |
| momentum_12_1 verliert Kraft | Trend-t des IC (13 gegen 52 W) | t ≤ −2 | −2,65 | belegt |
| rev_1m Strukturbruch 2021-06 | CUSUM | > 1,36 | 1,5 | belegt |
| Blind Spot Energy × Verlierer | Lift, BH | ≥ 1,5, q 0,1 | 2,13 | belegt |
| Research-Tracks ohne Ertrag | akzeptiert/getestet | Effizienz-Regel | discovery 0/117, director 0/5 | belegt |
| „tnx veraltete Annahme“ | Wert außerhalb Trainingsband | min/max Training | 5,24 gegen [1,10; 4,78] | belegt |

Es gibt keine unbelegten Freitexte. Eine Label-Regel (Kalibrierung) ist aber
ungenügend.

## 15. High-Confidence-Signal-Audit (Konfidenz-Buckets)

**Quelle:** `meta_learning.json.calibration_buckets.static_equal`, Lauf
2026-10-02, lagged-OOS-Kalibrierung, Meta-Testjahre. Die Wahrscheinlichkeit
kommt aus einer Vorjahres-Isotonie; die Schwellen sind nicht nachträglich
optimiert, Restrisiko F05.

| Bucket | N | P vorhergesagt | Trefferquote | Kal.-Fehler | Ø Rendite | Median | Ø MFE | Ø MAE | Expectancy |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15 659 | 0,505 | 0,461 | −0,045 | +0,18 % | −0,81 % | n/a | −7,4 % (DD) | +0,18 % |
| 55–60 % | 356 | 0,572 | 0,464 | −0,108 | +0,11 % | −0,58 % | n/a | −6,9 % | +0,11 % |
| 60–65 % | 0 | – | – | – | – | – | – | – | – |
| 65–70 % | 61 | 0,682 | 0,574 | −0,108 | +12,7 % | +2,2 % | n/a | −11,4 % | +12,7 % (low n) |
| 70–90 %, ≥ 90 % | 0 | – | – | – | – | – | – | – | – |

Die Felder MFE und MAE werden je Bucket nicht gespeichert. Gespeichert ist nur
der Durchschnitts-Drawdown; das ist eine Lücke.

**Steigt die Erfolgsrate mit der Confidence?**
- Von 50–55 % auf 55–60 %: **nein** (46,1 % → 46,4 %, vorhergesagt +6,7 pp).
- Nur 61 Fälle über 65 %.
- 99 % aller Wahrscheinlichkeiten liegen unter 60 %; die Prognose
  differenziert praktisch nicht.

**Urteil:** Das High-Confidence-System ist **nicht validiert**. Die
Wahrscheinlichkeiten sind im relevanten Bereich überkonfident.

## 16. Alert-Audit

Dry-Run und Test-Transport, `tests/test_forensic_audit.py`, E2E über
`hc_scanner.run`:

| Fall | Ergebnis |
|---|---|
| guter Kandidat → Alert | PASS |
| ruhiges Regime / schlechter Kandidat → kein Alert | PASS |
| Datenqualität LOW → kein Alert | PASS |
| Kalibrierung (ECE 0,09 oder Intervalle) → kein Alert | PASS |
| Safe Mode → kein Alert | PASS |
| veraltete Prognosen oder fehlende Karten → kein Alert | PASS |
| Counterfactual fragil → kein Alert | PASS |
| Duplikat → kein erneuter Alert | PASS |
| materielle Änderung (Regimewechsel) → erneuter Alert | PASS |
| Versandfehler → späterer Retry | **FAIL** (F10, XFAIL) |
| beschädigtes oder fehlendes `safe_mode.json` → kein Alert | **FAIL** (F07, XFAIL) |

## 17. Weekly-Email-Audit (echter Dry-Run, CI 37021911593)

**Quellen getrennt:**
- §2 Paper/Forward: `history.json`, ausdrücklich „kein Backtest“.
- §3/§5 Backtest/Walk-Forward: ausgewiesen.
- ML-Forward: „keine Daten“.

Die Trennung ist **korrekt**.

**Echte Paper-Performance des Produktions-Scanners**, nur zuverlässige
Outcomes:

| Fenster | n | Trefferquote | Expectancy | PF |
|---|---|---|---|---|
| gesamt | 79 | 35,4 % | +4,2 % | 1,12 |
| letzte 3 Monate | 12 | 25 % | −16,1 % | 0,42 |
| letzte 4 Wochen | 5 | 0 % | −53,7 % | – |

**Fehler:** F11 (Safe-Mode-Widerspruch, veraltete Meta-Zahlen, falsches Modell
in §3). „Current Champion: keiner“ widerspricht den Docs.

## 18. Safe-Mode-Angriff

| Simulation | Safe Mode aktiv? | HC blockiert? |
|---|---|---|
| Feature-Drift | ja | ja (globales Gate) |
| Modell-Drift (≥ 50 % deteriorating) | ja | ja |
| Kalibrierungsfehler | ja | ja |
| Pipeline-Fehler (≥ 25 % FAIL) | ja | – |
| Disagreement HIGH | ja | – |
| World-Model-Instabilität | ja | – |
| Stale-Daten (STALE) | **nein** (F08) | über „Prognosen veraltet“ ja |
| Beschädigte Regel- oder Modelldatei | – | ja (fail-closed) |
| Beschädigtes oder fehlendes `safe_mode.json` | **nein** | **nein** (F07) |
| Wirkung auf die tägliche Produktions-Mail | **keine** (F13) | – |

Real ausgelöst: `safe_mode.json` vom 2026-10-02 zeigt `active: true` wegen
tnx-Drift. HC war blockiert (`hc_candidates.json`).

## 19. Reproduzierbarkeit

SIEHE_REPRO

## 20. Ablationsmatrix (Lauf 2026-10-02, Referenz A, identische OOS-Zeilen)

Δ = Variante − A.

| Komponente | Δ Sharpe | Δ Expectancy | Δ MaxDD | Δ Brier | Δ ECE | Δ Prec@K | Δ HC-Präzision | Urteil |
|---|---|---|---|---|---|---|---|---|
| Meta-Learning (B) | +0,206 | +0,16 pp | +0,7 pp | +0,0002 | +0,0070 | +0,002 | +21,8 pp (n = 22!) | kein Nutzen belegt (CI ∋ 0) |
| World Model (C) | +0,018 | +0,03 pp | +0,1 pp | +0,0008 | +0,0103 | −0,001 | −2,0 pp | kein Nutzen |
| Causal (D) | +0,010 | −0,01 pp | +1,1 pp | +0,0008 | +0,0073 | −0,012 | −6,9 pp | kein Nutzen |
| Knowledge Graph (E) | +0,010 | −0,01 pp | +1,1 pp | +0,0008 | +0,0083 | −0,012 | −7,2 pp | kein Nutzen |
| Counterfactual-Filter (F) | −0,160 | −0,26 pp | +0,1 pp | −0,0002 | −0,0018 | −0,028 | −0,3 pp | schädlich |
| Unknown-Unknown-Filter | −0,036 | −0,24 pp | +6,4 pp | −0,0004 | +0,0081 | −0,029 | +0,3 pp | schädlich (nur Exposure) |
| Abstinenz (aus Regime) | +0,444 | +0,83 pp | +5,6 pp | −0,0013 | +0,0008 | +0,014 | +16,0 pp | Effekt groß, aber **kontaminiert** (F03) |
| Historical Analogies | – | – | – | – | – | – | – | **nicht gemessen** |
| Failure Memory (Meta-Ablation) | Stacking ohne Failure Memory: Locked 2,37 gegen 2,96 | – | – | – | – | – | – | uneinheitlich |
| Dynamic Weighting (trailing IC) | −0,171 | −0,27 pp | −3,5 pp | −0,0000 | −0,0038 | −0,008 | – | schädlich |

## 21. Testqualität

TESTQUAL

## 22. CI / Automation

- **Gibt es Checks bei jedem Commit?** **Nein.** Kein Workflow läuft auf `push`
  oder `pull_request`. Kein Workflow ruft pytest auf (`grep pytest .github`:
  leer).
- **Branch-Schutz und CODEOWNERS:** nicht erzwungen; PRs wurden ohne Review
  und ohne Checks gemerged.
- **Was automatisch läuft:** die Pipelines selbst (scanner, feedback,
  ml_research, weekly, external_data, world_model, research, monthly). Sie sind
  der einzige „Integrationstest“, ohne Assertions.
- **Queueing:** Workflows checken die SHA zum Auslösezeitpunkt aus. In der
  Warteschlange wartende Läufe arbeiten mit veralteten Artefakten (F11).

---

## 23. Finale Scorecard

| COMPONENT | IMPLEMENTED | CONNECTED | TESTED | OOS VALIDATED | PRODUCTION ACTIVE | STATUS | EVIDENCE |
|---|---|---|---|---|---|---|---|
| Historical Feature Store | ja | ja | ja | – | Research | PARTIAL | F01; PIT-Test PASS |
| Label Engine | ja | ja | ja | – | Research | PASS | `test_A5_labels_start_at_next_open` |
| Walk-forward Validation | ja | ja | ja | ja | Research | PARTIAL | F01 |
| Purging / Embargo | teilw. | ja | ja | – | Research | PARTIAL | F05, kein Embargo |
| Baseline Models | ja | ja | ja | ja | Research | PASS | ml_research.json |
| Specialized Models | ja | ja | ja | nein | nein | NOT VALIDATED | alle rejected_so_far |
| Regime Engine | ja | ja | ja | teilw. | Research | PARTIAL | – |
| Prediction Memory | ja | ja | ja | – | Research | PARTIAL | Versionen null, 0 Outcomes |
| Trade Memory | ja | ja | ja | – | nein | PARTIAL | Lücken |
| Failure Analyzer | ja | ja | ja | – | nein | UNUSED | nur Text |
| Feature Factory | ja | ja | ja | ja | nein | PASS | 0/117 akzeptiert (korrekt streng) |
| Historical Analogy Engine | ja | ja | ja | nein | nein | NOT VALIDATED | – |
| Uncertainty Estimation | ja | ja | ja | teilw. | nein | PARTIAL | P(>10 %) Skill −0,15 |
| Probability Calibration | ja | ja | ja | ja | nein | FAIL | Buckets |
| Research Engine | ja | ja | ja | ja | nein | PASS | hypothesis_db |
| Experiment Registry | ja | ja | ja | – | – | PASS | registry_log |
| Champion–Challenger | ja | ja | ja | ja | nein | PARTIAL | kein Champion |
| Locked Holdout | ja | ja | ja | – | – | CONTAMINATED | F02, F03 |
| Performance Attribution | ja | ja | ja | – | Report | PASS | attribution |
| Meta-Learning | ja | ja | ja | nein | nein | NOT VALIDATED | §7 |
| Dynamic Model Weighting | ja | ja | ja | nein | nein | NOT VALIDATED | §20 |
| Model Disagreement | ja | ja | ja | uneinheitl. | Safe Mode | PARTIAL | – |
| World Model | ja | ja | ja | nein | nur Safe Mode | NOT VALIDATED | §8 |
| Causal Research Layer | ja | ja | ja | nein | nein | NOT VALIDATED | §9 |
| Knowledge Graph | ja | teilw. | ja | nein | wirkungslos | UNUSED | §10 |
| Active Learning | ja | ja | ja | – | nein | PARTIAL | §11 |
| Autonomous Research Planner | ja | ja | ja | ja | nein | PASS | CI 37021907362 |
| Research Memory | ja | ja | ja | – | – | PASS | hypothesis_db |
| Hypothesis Similarity Detection | ja | ja | ja | – | – | PASS | 2 Sperren |
| Counterfactual Engine | ja | ja | ja | ja (negativ) | nie | PARTIAL | §12 |
| Adversarial Research / Self-Play | ja | ja | ja | – | – | PARTIAL | deterministische Prüfkette |
| Unknown-Unknown Detector | ja | ja | ja | ja (negativ) | nie | PARTIAL | §13 |
| Decision Intelligence | ja | ja | ja | instabil | nie | NOT VALIDATED | KEEP→MODIFY |
| Portfolio Layer | teilw. | teilw. | ja | nein | nein | PARTIAL | – |
| Meta-Cognition | ja | ja | ja | – | Safe Mode | PARTIAL | F09 |
| Alpha Decay Detection | ja | ja | ja | – | HC-Gate | PARTIAL | keine Gewichtung |
| Safe Mode | ja | ja | ja | – | nur HC | PARTIAL | F07, F08, F13 |
| High-Confidence Candidate Scanner | ja | ja | ja | nein | nie | NOT VALIDATED | F12 |
| Alert Deduplication | ja | ja | ja | – | nie | PARTIAL | F10 |
| Weekly Intelligence Email | ja | ja | ja | – | ja | PARTIAL | F11 |
| High-Confidence Email Alerts | ja | ja | Dry-Run | – | nie | UNUSED | nie gesendet |
| (Produktion) tägliche Scanner-Mail | ja | ja | ja | **Forward: PF 1,12 gesamt, 0,42 letzte 3 M** | ja | NOT VALIDATED | Weekly §2 |

---

## Abschluss: Antworten

ANTWORTEN
