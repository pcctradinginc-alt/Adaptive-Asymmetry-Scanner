# Audit-Remediation: Umsetzungsstand

Grundlage: `docs/AUDIT_REMEDIATION_PLAN.md` und
`docs/FORENSIC_ACCEPTANCE_AUDIT.md`. Für jeden Punkt sind Änderung,
Nachweis-Test und ggf. offener menschlicher Schritt angegeben. Die
xfail-markierten Defekttests in `tests/test_forensic_audit.py` sind entfernt,
weil die Defekte behoben sind; sie laufen jetzt als normale Regressionstests.

## P0: macht Ergebnisse ungültig

| ID | Status | Umsetzung | Nachweis |
|---|---|---|---|
| P0-1 CI-Tests | **umgesetzt**; Branch-Schutz offen (Mensch) | `.github/workflows/tests.yml`: pytest bei jedem Push und PR, Audit-/Leakage-/Protokolltests zuerst | Workflow-Lauf auf diesem PR |
| P0-2 Survivorship | **umgesetzt** (Rest-Bias dokumentiert) | `universe.parse_sp500_changes`, `membership_intervals`, `is_member`, `research_universe`, `get_universe(as_of=…)`. Panel nur während der Indexmitgliedschaft, inkl. später entfernter Titel. Kein stiller Rückfall auf die heutige Liste. Ebenso in `price_event_study`. Abdeckung entfernter Titel mit Kursen wird gemessen (`ml_research.json → universe`). | `test_A5_sp500_changes_parser_and_pit_membership`, `…_research_universe_includes_removed_names`, `…_panel_rows_only_during_membership`, `…_never_falls_back_to_todays_list` |
| P0-3 Holdout | **umgesetzt** | `research_protocol.yaml`: `locked_status: CONTAMINATED`, `forward_holdout_from`. Locked je `spec_hash` nur einmal auswertbar (Ergebnis gecacht). Locked kann nur noch ablehnen, nie bestätigen. Bindend ist der Forward-Shadow (≥ 26 Kohorten, bestand schon). `prob_maps` nie mehr auf Locked. | `test_contaminated_holdout_never_reset`, Protokoll-Pin |
| P0-4 Abstinenz | **umgesetzt** | `next_protocol.yaml`: `confirmation_status: CONTAMINATED`, `forward_from`, je ≥ 26 aktive und inaktive Kohorten. `next_intelligence.forward_abstention` (unveränderliches Prognose-Ledger, nur fertige Labels). G enthält die Regel nur bei Vorwärts-Bestätigung. HYP-ABST-001 auf INCONCLUSIVE. HC, Weekly und Machine-State zeigen „historisch (zählt nicht)“ und „vorwärts“ getrennt. | `test_P04_contaminated_confirmation_never_counts`, `test_P04_forward_abstention_ledger_counts_only_matured_cohorts` |
| P0-5 Governance | **vorbereitet**, Aktivierung durch den Owner | CODEOWNERS um `.github/`, Audit-Tests, `next_protocol`, `data_catalog` erweitert. Dieser PR wird **nicht** vom Agenten gemergt. | – |

## P1: wichtige Funktion fehlt oder ist falsch

| ID | Status | Umsetzung | Nachweis |
|---|---|---|---|
| P1-1 Weekly | **umgesetzt** | Safe Mode aus `safe_mode.json` („UNBEKANNT“, nie „aus“, wenn die Datei fehlt). §3 zeigt Buckets des aktiven Ensembles. Datenstand je Artefakt mit VERALTET-Warnung. Hinweis „tägliche Mail wird nicht vom Research-Stack gesteuert“. Workflows `ml_research`/`weekly_report` checken `ref: main` aus. | `tests/test_weekly_report.py` |
| P1-2 Safe Mode fail-closed | **umgesetzt** | Fehlendes oder unlesbares `safe_mode.json` blockiert HC. | `test_A18_missing_…`, `test_A18_corrupt_safe_mode_…` |
| P1-3 Alert-Verlust | **umgesetzt** | Dedup-Zustand nur nach Status `sent`; sonst Rückrollen und Retry im nächsten Lauf. | `test_A16_failed_delivery_is_retried` |
| P1-4 Wahrscheinlichkeiten | **umgesetzt** | `meta_learning.probability_validation`: ≥ 2 Buckets mit n ≥ 100, \|Fehler\| ≤ 0,05, monotone Trefferquote. Sonst blockiert das HC-Gate. Die tägliche Mail markiert die MC-Trefferquote als nicht kalibriert. | `test_P14_…` (echte Buckets vom 02.10. → nicht validiert), `test_trade_mail_marks_mc_hit_rate_as_uncalibrated` |
| P1-5 Kalibrierungslabel | **umgesetzt** | `overall_calibration` ist WEAK bei schlecht kalibriertem Bucket mit n ≥ 100. Der festschreibende Test ist korrigiert. | `test_A14_…`, `test_machine_state_only_metric_statements` |
| P1-6 Rolle des Research-Stacks | **Entscheidung des Menschen offen** | Bis zur Entscheidung steht im Weekly und in der Doku ausdrücklich: Der Stack steuert die tägliche Mail **nicht**. Empfehlung in `PAPER_PERFORMANCE_ANALYSIS.md`. | – |
| P1-7 Doku | **umgesetzt** | B = Meta-Learning; „Champion“ → Referenz; Abstinenz und Locked korrigiert; Meta-Urteil als instabil gekennzeichnet (`NEXT_INTELLIGENCE_VALIDATION.md`, `META_LEARNING_VALIDATION.md`, Gap-Analyse, Machine-State-Text). | – |
| P1-8 Produktionstests | **umgesetzt** | Neue Verhaltenstests mit Negativfällen. Coverage siehe unten. | `test_production_gates.py`, `test_mirofish_behaviour.py`, `test_email_reporter_behaviour.py`, `test_options_designer_behaviour.py` |
| P1-9 Paper-Ursachen | **umgesetzt** | `scripts/paper_performance_analysis.py`, `docs/PAPER_PERFORMANCE_ANALYSIS.md`. MC-Trefferquote stark überkonfident; LLM-Scores nicht monoton; der Erfolg stammt aus einem Monat. | `test_paper_performance_analysis.py` |

## P2: Robustheit

| ID | Status | Umsetzung / Nachweis |
|---|---|---|
| P2-1 Kalibrierungs-Purge | **umgesetzt** | `lagged_calibration` nutzt nur Vorjahreslabels mit `label_end` < Testbeginn. `test_A5_lagged_calibration_purges_overlapping_labels` |
| P2-2 Safe-Mode-Auslöser | **umgesetzt** | STALE-Quellen (≥ 25 % oder eine kritische) und beschädigte Artefakte (`hc_thresholds`, `ml_cards`, Meta/ML/Next-JSON). `test_A18_stale_sources_…`, `test_A18_corrupt_artifact_…` |
| P2-3 Sektor/KG nicht PIT | **gekennzeichnet** | KG-Kanten tragen `point_in_time` und `valid_from` (nur gemessene Kanten sind PIT). `ml_research.json → universe.sector_assignment` benennt den Sektor-Bias. Historische GICS-Zuordnung bräuchte eine lizenzierte Quelle (REVIEW_REQUIRED). |
| P2-4 Modellversionen | **umgesetzt** | Prediction Memory speichert `spec_hash` je Modell statt `null`. |
| P2-5 Reproduzierbarkeit | **umgesetzt** | Panel-Snapshot als CI-Artefakt (90 Tage), Replay via `ML_PANEL_SNAPSHOT`, `.github/workflows/repro.yml` (zwei Läufe auf demselben Snapshot), `scripts/repro_check.py`, `panel_hash` und `code_sha` in `ml_research.json`. `test_P25_meta_evaluate_is_deterministic_on_same_panel` |
| P2-6 Bucket-MFE/MAE | **umgesetzt** | `calibration_buckets` mit `avg_mfe` und `avg_mae`. |
| P2-7 Embargo | **geprüft: nicht nötig** | Expandierender Walk-Forward nur vorwärts: Training liegt immer vor dem Test, Purge über `label_end`. Ein Embargo ist nur bei Training *nach* dem Test (k-fold) nötig. Belegt durch `test_A5_purge_excludes_labels_overlapping_test` und `meta_provenance` (Training endet vor Testbeginn). |

## P3: Verbesserung

| ID | Status | Umsetzung |
|---|---|---|
| P3-1 Dead Code | **umgesetzt** | `news_fetcher.py` und `reddit_signals.py` entfernt, `praw` aus `requirements.txt`. |
| P3-2 KG-Check | **umgesetzt** | Der Widerspruchs-Check greift nur bei gemessenen Kanten; sonst `kg_check: inaktiv (0 gemessene Kanten)`. |
| P3-3 Analogien | **gekennzeichnet** | Karten `analogs.oos_validated = false`, Alerts tragen `historical_analogues_note`. Eine OOS-Messung steht aus (neue Hypothese für den Director). |
| P3-4 Self-Play | **umgesetzt** | Jede Rolle hat `objection` und `finding` aus Messwerten, dazu die Liste `objections`. Die Entscheidung bleibt bei der Prüfkette. `test_self_play_roles_have_own_measured_verdicts` |
| P3-5 Coverage | **umgesetzt** | `pipeline.py` und `feedback.py` werden mitgemessen (siehe unten). |

## Coverage (lokaler Gesamtlauf)

Gesamtlauf am 2026-10-02: **1194 bestanden, 1 übersprungen** (boto3 fehlt),
0 fehlgeschlagen.

| Bereich | vorher | nachher |
|---|---|---|
| Gesamt (`modules/`, `reports/`) | 73 % | 74 % (inkl. `pipeline.py`, `feedback.py`) |
| options_designer | 30 % | **85 %** |
| risk_gates | 34 % | **88 %** |
| prescreener | 40 % | **94 %** |
| mirofish_simulation | 40 % | **90 %** |
| email_reporter | 20 % | **92 %** |
| universe | 18 % | 45 % (PIT-Teil getestet, Wikipedia-Abruf nicht) |
| pipeline.py | nicht gemessen | 17 % (Orchestrierung; Stufen in Modulen getestet) |
| feedback.py | nicht gemessen | 47 % |

`pipeline.py` bleibt die größte Testlücke. Eine Stufenkette mit Fakes ist der
nächste sinnvolle Schritt (P3, nicht Teil dieses Plans).

## Schritte, die nur der Owner ausführen kann
1. **Branch-Protection für `main`** (Settings → Branches):
   - „Require status checks“: `Tests (Pflicht-Check) / pytest`;
   - „Require review from Code Owners“;
   - „Do not allow bypassing“.
2. **P1-6 entscheiden:** Soll die Abstinenz-Regel, sobald sie vorwärts
   bestätigt ist, auch die tägliche Mail gaten?
3. **Neuberechnung anstoßen:** ein `ml_research`-Full-Lauf auf dem
   PIT-Universum. Alle bisherigen Research-Zahlen waren survivorship-verzerrt.
   Danach `repro.yml` mit der Run-ID dieses Laufs ausführen.
