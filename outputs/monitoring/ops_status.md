# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-15 · erzeugt 2026-10-15T17:54:47+00:00 · Schema ops-status-v1  
Warnungen: 1 neu, 0 geändert, 2 laufend, 1 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-10-15T11:32:56Z | 3.9 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | PENDING | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-10-15 (von heute: ja)
- **Universum / analysiert**: 514 / 82
- **Champion-Kandidaten (Final-MC)**: 14
- **Kontrakte entworfen**: 7
- **Trades**: 2
- **Haiku/Sonnet-Widerspruch**: 4 (Anteil 0.0488)
- **Session**: regular
- **Stop-Grund**: 

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | 100.0 % | 2/2 | Altbestand vor Archiv: 0; ohne Altbestand 100.0 % (2/2) |
| Entry 30d | 28.6 % | 2/7 | Altbestand vor Archiv: 5; ohne Altbestand 100.0 % (2/2) |
| Exit geschlossen 7d | 100.0 % | 2/2 | EXIT_OBSERVED_NOT_ARCHIVED: 1, EXIT_OBSERVED_ARCHIVED: 1 |
| Exit geschlossen 30d | 66.7 % | 4/6 | EXIT_NOT_REALIZABLE: 2, EXIT_OBSERVED_NOT_ARCHIVED: 3, EXIT_OBSERVED_ARCHIVED: 1 |
| Vollständig bewertbar 7d | 100.0 % | 2/2 | keine |
| Vollständig bewertbar 30d | 66.7 % | 4/6 | EXIT_PRICE_UNKNOWN: 2 |
| Exit aktiv (Mark heute) | 100.0 % | 5/5 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_5d | 273 | 272 | 1 | 50 | 0 | 10 |
| real_strat_ret_5d | 273 | 272 | 1 | 50 | 0 | 10 |

## Lebenszyklus

| Champion-Zustand | Anzahl |
|---|---|
| SIGNAL | 0 |
| CONTRACT_SELECTED | 0 |
| ENTRY_OBSERVED | 0 |
| EXIT_PENDING | 5 |
| OUTCOME_FINAL | 110 |
| LEARNING_ELIGIBLE | 3 |
| LEARNED | 1 |

- **Probleme (Champion)**: ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE: 117, LEGACY_NO_STORED_EVALUATION: 113, OUTCOME_UNUSABLE: 107, OUTCOME_DATA_BAD: 3, LEGACY_LEARNED_UNTRACKED: 3
- **Exactly-once ok**: ja
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 2/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 626, CONTRACT_SELECTED: 353, ENTRY_OBSERVED: 353, OUTCOME_FINAL: 272, LATE_MARK: 0, EXIT_OVERDUE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 273, EXIT_PENDING: 80

## Forward-Performance

- **Champion realisiert (nur beobachtet): n**: 4
- **mean_net**: -0.5419
- **sum_net**: -2.1677
- **Trefferquote**: 0
- **schlechtester**: -0.7443
- **max. Drawdown (kum.)**: -2.1677
- **nicht bewertbar**: UNUSABLE: 107, DATA_BAD: 3

Verdikt NEED_MORE_DATA: keine Alpha-Behauptung (n=4, unabhängige Tage=4; Mindestwerte n≥30, Tage≥20).

| Challenger (Shadow) | vermiedene Verluste | entgangene Gewinne | Δ vs. Champion (Σ) | greift/bewertbar | Population | Verdikt |
|---|---|---|---|---|---|---|
| EA001_EXPECTATION_GAP@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA002_CONFIRMATION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA003_ACCELERATION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA004_WAIT@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA005_CONTEXT_FILTER@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA006_ABSTENTION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA007_EXPRESSION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| PROM-ABST-001@v1 | N/A | N/A | N/A | 0/1 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-001@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-002@v1 | 0.5079 | 0 | 0.5079 | 1/1 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-002@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-003@v1 | N/A | N/A | N/A | 0/1 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-003@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-004@v1 | N/A | N/A | N/A | 0/1 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-004@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-005@v1 | N/A | N/A | N/A | 0/1 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-005@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-META-001@v1 | N/A | N/A | N/A | 0/0 | CHAMPION_TRADE | NEED_MORE_DATA |
| UNIV-V2-SEG-MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-MID@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-SMALL@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-ULTRA_MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |

22 von 22 Challenger-Verdikten NEED_MORE_DATA – dort keine Alpha-Behauptung.

| Scorecard | n | mean_net | Trefferquote | Verdikt |
|---|---|---|---|---|
| CANDIDATE_5d\|ALL | 478 | -0.0007 | 0.4791 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 3 | -0.0076 | 0.3333 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 133 | -0.0005 | 0.5338 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 272 | -0.126 | 0.2537 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 3 | -0.1617 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 80 | -0.1065 | 0.3625 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 222 | -0.1758 | 0.2252 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 3 | -0.2853 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 64 | -0.2404 | 0.1875 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 3 | -0.6168 | 0 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 0 / 0 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (DRIFT MODERATE: Promotion pausiert)
- **Champion-Version**: scanner_rules@d20d5a4ff52a
- **Zustände**: PROSPECTIVE_CHALLENGER: 22
- **Aktiver Einfluss**: keine
- **Ungerechtfertigter Einfluss**: keine

| Hypothese | Zustand | Einfluss | Entscheidung | Blockierungsgründe |
|---|---|---|---|---|
| EA001_EXPECTATION_GAP@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| EA002_CONFIRMATION@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| EA003_ACCELERATION@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| EA004_WAIT@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/60; unabhängige Tage 0/30; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/40; Treffer-Clust |
| EA005_CONTEXT_FILTER@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| EA006_ABSTENTION@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| EA007_EXPRESSION@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/60; unabhängige Tage 0/30; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/40; Treffer-Clust |
| PROM-ABST-001@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 1/60; unabhängige Tage 1/20; Kalenderspanne 0/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-001@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-002@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 1/60; unabhängige Tage 1/20; Kalenderspanne 0/90 T; Regel-Treffer 1/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet; Sektor Technology dominiert (100%) – n |
| PROM-ABST-002@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-003@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 1/60; unabhängige Tage 1/20; Kalenderspanne 0/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-003@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-004@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 1/60; unabhängige Tage 1/20; Kalenderspanne 0/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-004@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-005@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 1/60; unabhängige Tage 1/20; Kalenderspanne 0/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-005@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-META-001@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/60; unabhängige Tage 0/20; Kalenderspanne 0/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 0 Marktregime beobachtet |
| UNIV-V2-SEG-MICRO@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| UNIV-V2-SEG-MID@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| UNIV-V2-SEG-SMALL@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| UNIV-V2-SEG-ULTRA_MICRO@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |

## Integrität

- **Probleme**: keine
- **Promotion-Hash ok**: ja
- **Kette/Registry**: keine / keine
- **Duplikate Decision-Ledger / Trade-Outcomes**: 0 / 0
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 549 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 27013, legacy: 209604

## Daten

- **Quellenstatus**: HEALTHY: 41, BROKEN: 3, STALE: 1, DEGRADED: 2, UNVALIDATED: 8
- **Stand Quellenstatus**: 2026-10-15T17:20:08+00:00
- **Externe Daten**: keine
- **Quote-Archiv heute / Median 14 T**: 88 / 139.5
- **Quote-Archiv-Verzeichnis vorhanden**: ja

## Timing-Policy

- **Policy neuer Entscheidungen**: timing-v2
- **Bestand je Policy**: active: timing-v1: 3, timing-v2: 2, exit_pending: keine, closed: timing-v1: 114, unusable: keine, counterfactual_exit_pending: keine, counterfactual_unusable: keine
- **timing-v2 Entry offener Trades**: FILLED: 2
- **timing-v2 Exit ausstehend (ältester, Tage)**: 0 (N/A)
- **Gründe ausstehend**: keine
- **timing-v2 ausgeführt / UNUSABLE**: 0 / 0
- **Gründe UNUSABLE**: keine
- **Abweichung Ausführung vs. Entscheid**: n: 0, mean: N/A, definition: Outcome zur zulässigen Ausführung minus Return zum Entscheid

## Lern-Prognose

| Ziel | Stand | benötigt | Zulauf/Tag | voraussichtlich ab |
|---|---|---|---|---|
| Meta-Label-Training | 0 · Tage 0 | 200 · Tage 20 | 9.5 | 2026-11-25 |
| RL-Freischaltung (closed trades) | 10 | 30 | 0.2 | 2027-01-23 |
| Champion-Evidenz | 4 · Tage 4 | 30 · Tage 20 | 0.133 | 2027-04-28 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 158 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-10-14T15:54:47Z (26 h)
- **Läufe / fehlgeschlagen**: 61 / 2
- **Scanner-Start heute (New York)**: 2026-10-15T13:20-04:00

| Workflow | erstellt | Ergebnis | fehlgeschlagene Schritte | letzte Fehlerzeilen |
|---|---|---|---|---|
| Tests (Pflicht-Check) | 2026-10-15T07:15:38Z | failure | Gesamte Testsuite | E       AssertionError: assert {'real_opt_re...rks': 1, ...}} == {'real_opt_re...rks': 1, ...}} / /home/runner/work/Adaptive-Asymmetry-Scanner/Adaptive-Asymmetry-Scanner/tests/test_ops_monitor.py:506: AssertionError / FAILED tests/test_ops_monitor.py::test_candidate_option_marks_due_pending_and_late - AssertionError: assert {'real_opt_re...rks': 1, ...}} == {'real_opt_re...rks': 1, ...}} / ##[error]Process completed with exit code 1. |
| Tests (Pflicht-Check) | 2026-10-15T07:15:06Z | failure | Gesamte Testsuite | E       AssertionError: assert {'real_opt_re...rks': 1, ...}} == {'real_opt_re...rks': 1, ...}} / /home/runner/work/Adaptive-Asymmetry-Scanner/Adaptive-Asymmetry-Scanner/tests/test_ops_monitor.py:506: AssertionError / FAILED tests/test_ops_monitor.py::test_candidate_option_marks_due_pending_and_late - AssertionError: assert {'real_opt_re...rks': 1, ...}} == {'real_opt_re...rks': 1, ...}} / ##[error]Process completed with exit code 1. |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Lauf | WORKFLOW_FAILED:Tests (Pflicht-Check) | „Tests (Pflicht-Check)“: 2 fehlgeschlagene(r) Lauf/Läufe seit 2026-10-14T15:54:47Z; Schritte: Gesamte Testsuite; letzte Fehlerzeile: ##[error]Process completed with exit code 1.; https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/38697489319 | 2026-10-15 |
| RESOLVED | CRITICAL | Integrität | LEDGER_INTEGRITY | Nicht mehr aktiv – Bedingung behoben. | 2026-10-14 |
| ONGOING | WARNING | Lebenszyklus | OUTCOMES_OVERDUE:candidates | 1 Kandidat(en) mit überfälligem 5-Tage-Real-Option-Mark (> 2 Tage nach Fälligkeit). | 2026-10-14 |
| ONGOING | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_5d 1 von 273 offen, ältester 10 Tage; real_strat_ret_5d 1 von 273 offen, ältester 10 Tage (Schwelle > 3 Tage). | 2026-10-14 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.quote_archive_median_14d | 139.5 | 164 (-24.5) | N/A | N/A |
| data.quote_archive_today | 88 | 164 (-76) | N/A | N/A |
| data.source_health.BROKEN | 3 | 1 (+2) | N/A | N/A |
| data.source_health.DEGRADED | 2 | 7 (-5) | N/A | N/A |
| data.source_health.HEALTHY | 41 | 28 (+13) | N/A | N/A |
| data.source_health.STALE | 1 | N/A | N/A | N/A |
| data.source_health.UNVALIDATED | 8 | 8 (0) | N/A | N/A |
| data.source_health.n_bad | 6 | 8 (-2) | N/A | N/A |
| forward.champion.mean_net | -0.5419 | -0.6168 (+0.0748) | N/A | N/A |
| forward.champion.n | 4 | 3 (+1) | N/A | N/A |
| forward.champion.n_independent_days | 4 | 3 (+1) | N/A | N/A |
| forward.champion.sum_net | -2.1677 | -1.8503 (-0.3174) | N/A | N/A |
| integrity.issues | 0 | 1 (-1) | N/A | N/A |
| learning.champion_evidence.have | 4 | N/A | N/A | N/A |
| learning.meta_label.have | 0 | N/A | N/A | N/A |
| learning.rl_arming.have | 10 | N/A | N/A | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 1 | 1 (0) | N/A | N/A |
| lifecycle.candidates.EXIT_PENDING | 80 | 40 (+40) | N/A | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 272 | 272 (0) | N/A | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.state.EXIT_PENDING | 5 | 5 (0) | N/A | N/A |
| lifecycle.champion.state.LEARNED | 1 | 0 (+1) | N/A | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | N/A | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | N/A | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | N/A | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | 10 | 9 (+1) | N/A | N/A |
| marks.real_opt_ret_5d.pending | 1 | 1 (0) | N/A | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | 10 | 9 (+1) | N/A | N/A |
| marks.real_strat_ret_5d.pending | 1 | 1 (0) | N/A | N/A |
| promotion.active_influence | 0 | 0 (0) | N/A | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | N/A | N/A |
| quote.complete.30d.coverage | 0.6667 | 0.6 (+0.0667) | N/A | N/A |
| quote.complete.30d.denominator | 6 | 5 (+1) | N/A | N/A |
| quote.complete.30d.evaluable | 4 | 3 (+1) | N/A | N/A |
| quote.complete.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.complete.7d.denominator | 2 | 1 (+1) | N/A | N/A |
| quote.complete.7d.evaluable | 2 | 1 (+1) | N/A | N/A |
| quote.entry.30d.coverage | 0.2857 | 0.1667 (+0.119) | N/A | N/A |
| quote.entry.30d.denominator | 7 | 6 (+1) | N/A | N/A |
| quote.entry.30d.legacy | 5 | 5 (0) | N/A | N/A |
| quote.entry.30d.observed | 2 | 1 (+1) | N/A | N/A |
| quote.entry.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.entry.7d.denominator | 2 | 1 (+1) | N/A | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | N/A | N/A |
| quote.entry.7d.observed | 2 | 1 (+1) | N/A | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | N/A | N/A |
| quote.exit_active.denominator | 5 | 5 (0) | N/A | N/A |
| quote.exit_active.mark_today | 5 | 5 (0) | N/A | N/A |
| quote.exit_closed.30d.coverage | 0.6667 | 0.6 (+0.0667) | N/A | N/A |
| quote.exit_closed.30d.denominator | 6 | 5 (+1) | N/A | N/A |
| quote.exit_closed.30d.observed | 4 | 3 (+1) | N/A | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.exit_closed.7d.denominator | 2 | 1 (+1) | N/A | N/A |
| quote.exit_closed.7d.observed | 2 | 1 (+1) | N/A | N/A |
| run.scanner.analyzed | 82 | 66 (+16) | N/A | N/A |
| run.scanner.champion_candidates | 14 | 11 (+3) | N/A | N/A |
| run.scanner.contracts_designed | 7 | 5 (+2) | N/A | N/A |
| run.scanner.direction_conflict_share | 0.0488 | N/A | N/A | N/A |
| run.scanner.direction_conflicts | 4 | N/A | N/A | N/A |
| run.scanner.trades | 2 | 1 (+1) | N/A | N/A |
| run.scanner.universe | 514 | 514 (0) | N/A | N/A |
| training.n_labels | 0 | 0 (0) | N/A | N/A |
| workflows.failed_runs | 2 | N/A | N/A | N/A |
