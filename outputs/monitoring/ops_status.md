# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-09 · erzeugt 2026-11-09T17:01:32+00:00 · Schema ops-status-v1  
Warnungen: 1 neu, 0 geändert, 0 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-09T16:25:14Z | 10.1 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-09 (von heute: ja)
- **Universum / analysiert**: 518 / 3
- **Champion-Kandidaten (Final-MC)**: 0
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 1 (Anteil 0.3333)
- **Session**: regular
- **Stop-Grund**: Alle durch Intraday-Delta-Filter verworfen (>7% Bewegung).

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Entry 30d | 100.0 % | 3/3 | Altbestand vor Archiv: 0; ohne Altbestand 100.0 % (3/3) |
| Exit geschlossen 7d | 100.0 % | 1/1 | EXIT_OBSERVED_ARCHIVED: 1 |
| Exit geschlossen 30d | 100.0 % | 5/5 | EXIT_OBSERVED_ARCHIVED: 5 |
| Vollständig bewertbar 7d | 100.0 % | 1/1 | keine |
| Vollständig bewertbar 30d | 100.0 % | 5/5 | keine |
| Exit aktiv (Mark heute) | 100.0 % | 2/2 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 473 | 471 | 1 | 50 | 1 | 0 |
| real_opt_ret_5d | 817 | 816 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 473 | 471 | 1 | 50 | 1 | 0 |
| real_strat_ret_5d | 817 | 815 | 1 | 50 | 1 | 1 |

## Lebenszyklus

| Champion-Zustand | Anzahl |
|---|---|
| SIGNAL | 0 |
| CONTRACT_SELECTED | 0 |
| ENTRY_OBSERVED | 0 |
| EXIT_PENDING | 2 |
| OUTCOME_FINAL | 110 |
| LEARNING_ELIGIBLE | 3 |
| LEARNED | 5 |

- **Probleme (Champion)**: ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE: 117, LEGACY_NO_STORED_EVALUATION: 113, OUTCOME_UNUSABLE: 107, OUTCOME_DATA_BAD: 3, LEGACY_LEARNED_UNTRACKED: 3
- **Exactly-once ok**: ja
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 50/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 1039, CONTRACT_SELECTED: 673, ENTRY_OBSERVED: 673, OUTCOME_FINAL: 664, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 8

## Forward-Performance

- **Champion realisiert (nur beobachtet): n**: 8
- **mean_net**: -0.2966
- **sum_net**: -2.3728
- **Trefferquote**: 0.125
- **schlechtester**: -0.7443
- **max. Drawdown (kum.)**: -2.3728
- **nicht bewertbar**: UNUSABLE: 107, DATA_BAD: 3

Verdikt NEED_MORE_DATA: keine Alpha-Behauptung (n=8, unabhängige Tage=7; Mindestwerte n≥30, Tage≥20).

| Challenger (Shadow) | vermiedene Verluste | entgangene Gewinne | Δ vs. Champion (Σ) | greift/bewertbar | Population | Verdikt |
|---|---|---|---|---|---|---|
| EA001_EXPECTATION_GAP@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA002_CONFIRMATION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA003_ACCELERATION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA004_WAIT@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA005_CONTEXT_FILTER@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA006_ABSTENTION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA007_EXPRESSION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| PROM-ABST-001@v1 | N/A | N/A | N/A | 0/3 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-001@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-002@v1 | 0.976 | 0 | 0.976 | 2/3 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-002@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-003@v1 | N/A | N/A | N/A | 0/3 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-003@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-004@v1 | 0.4681 | 0 | 0.4681 | 1/3 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-004@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-005@v1 | N/A | N/A | N/A | 0/3 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-005@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-META-001@v1 | N/A | N/A | N/A | 0/0 | CHAMPION_TRADE | NEED_MORE_DATA |
| UNIV-V2-SEG-MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-MID@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-SMALL@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-ULTRA_MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |

22 von 22 Challenger-Verdikten NEED_MORE_DATA – dort keine Alpha-Behauptung.

| Scorecard | n | mean_net | Trefferquote | Verdikt |
|---|---|---|---|---|
| CANDIDATE_20d\|ALL | 758 | -0.0036 | 0.5409 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 7 | -0.036 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 174 | -0.0227 | 0.523 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1029 | 0 | 0.5277 | INCONCLUSIVE |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 235 | -0.0009 | 0.5149 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 392 | -0.0411 | 0.426 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 5 | -0.138 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 97 | -0.1449 | 0.3505 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 661 | -0.0938 | 0.3177 | NEGATIVE |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 154 | -0.0926 | 0.3377 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 392 | -0.0847 | 0.4005 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 5 | -0.2819 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 97 | -0.2576 | 0.2784 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 611 | -0.1363 | 0.2831 | NEGATIVE |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 138 | -0.1848 | 0.2391 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 8 | -0.2966 | 0.125 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 165 / 11 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (DATA: 2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports']; MODEL: PERFORMANCE BREAKDOWN: Forward-Expectancy t=-5.19)
- **Champion-Version**: scanner_rules@fa6cc74e0def
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
| PROM-ABST-001@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-001@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-002@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 2/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet |
| PROM-ABST-002@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-003@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-003@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-004@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 1/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet; Sektor Financial Services dominiert (1 |
| PROM-ABST-004@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-005@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 4522 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 12942, legacy: 5926

## Daten

- **Quellenstatus**: HEALTHY: 49, DEGRADED: 3, UNVALIDATED: 9, STALE: 2
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 2, DEFERRED: 5, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-09T16:42:28+00:00
- **Externe Daten**: PASS: 42, DEFERRED: 14, FAIL: 1
- **Quote-Archiv heute / Median 14 T**: 516 / 222.5
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-09 (Quelle Produktion: nasdaq_official)

## Timing-Policy

- **Policy neuer Entscheidungen**: timing-v2
- **Bestand je Policy**: active: timing-v2: 2, exit_pending: keine, closed: timing-v1: 117, timing-v2: 1, unusable: keine, counterfactual_exit_pending: keine, counterfactual_unusable: keine
- **timing-v2 Entry offener Trades**: FILLED: 2
- **timing-v2 Exit ausstehend (ältester, Tage)**: 0 (N/A)
- **Gründe ausstehend**: keine
- **timing-v2 ausgeführt / UNUSABLE**: 1 / 0
- **Gründe UNUSABLE**: keine
- **Abweichung Ausführung vs. Entscheid**: n: 1, mean: 0.697, definition: Outcome zur zulässigen Ausführung minus Return zum Entscheid

## Lern-Prognose

| Ziel | Stand | benötigt | Zulauf/Tag | voraussichtlich ab |
|---|---|---|---|---|
| Meta-Label-Training | 174 · Tage 12 | 200 · Tage 20 | 0.93 | 2026-12-08 |
| RL-Freischaltung (closed trades) | 14 | 30 | 0.167 | 2027-02-13 |
| Champion-Evidenz | 8 · Tage 7 | 30 · Tage 20 | 0.167 | 2027-03-21 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 62 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-11-06T15:01:32Z (74 h)
- **Läufe / fehlgeschlagen**: 81 / 1
- **Scanner-Start heute (New York)**: 2026-11-09T11:43-05:00

| Workflow | erstellt | Ergebnis | fehlgeschlagene Schritte | letzte Fehlerzeilen |
|---|---|---|---|---|
| PortWatch HDX Shadow (OCHA mirror, no fallback) | 2026-11-07T10:29:20Z | failure | Fail check on stale, empty or unverified HDX data | ##[error]HDX shadow failed validation (no primary-source recovery and no Safe Mode override) / ##[error]Process completed with exit code 1. |

Grün, aber nicht vollständig ausgeführt (jüngster Lauf je Kern-Workflow; technischer Status, kein Nachweis von Trades, Quotes oder Alpha):

| Workflow | effektiver Status | GitHub-Ergebnis | ausgefallene Stufen | übersprungene Jobs |
|---|---|---|---|---|
| Adaptive Asymmetry Scanner | UNVERIFIED | unknown | N/A | N/A |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Lauf | WORKFLOW_FAILED:PortWatch HDX Shadow (OCHA mirror, no fallback) | „PortWatch HDX Shadow (OCHA mirror, no fallback)“: 1 fehlgeschlagene(r) Lauf/Läufe seit 2026-11-06T15:01:32Z; Schritte: Fail check on stale, empty or unverified HDX data; letzte Fehlerzeile: ##[error]Process completed with exit code 1.; https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/42364150832 | 2026-11-09 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | N/A |
| data.quote_archive_median_14d | 222.5 | 222.5 (0) | 207.5 (+15) | N/A |
| data.quote_archive_today | 516 | 117 (+399) | 206 (+310) | N/A |
| data.source_health.BROKEN | N/A | 2 (N/A) | 2 (N/A) | N/A |
| data.source_health.DEGRADED | 3 | 2 (+1) | 2 (+1) | N/A |
| data.source_health.HEALTHY | 49 | 49 (0) | 42 (+7) | N/A |
| data.source_health.STALE | 2 | 1 (+1) | 1 (+1) | N/A |
| data.source_health.UNVALIDATED | 9 | 9 (0) | 8 (+1) | N/A |
| data.source_health.n_bad | 5 | 5 (0) | 5 (0) | N/A |
| forward.champion.mean_net | -0.2966 | -0.2966 (0) | -0.3322 (+0.0356) | N/A |
| forward.champion.n | 8 | 8 (0) | 7 (+1) | N/A |
| forward.champion.n_independent_days | 7 | 7 (0) | 6 (+1) | N/A |
| forward.champion.sum_net | -2.3728 | -2.3728 (0) | -2.3256 (-0.0472) | N/A |
| integrity.issues | 0 | 0 (0) | 0 (0) | N/A |
| learning.champion_evidence.have | 8 | 8 (0) | 7 (+1) | N/A |
| learning.meta_label.have | 174 | 165 (+9) | 133 (+41) | N/A |
| learning.rl_arming.have | 14 | 14 (0) | 13 (+1) | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.candidates.EXIT_PENDING | 8 | 8 (0) | 6 (+2) | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 664 | 661 (+3) | 656 (+8) | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.EXIT_PENDING | 2 | 2 (0) | 3 (-1) | N/A |
| lifecycle.champion.state.LEARNED | 5 | 5 (0) | 4 (+1) | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | 1 | N/A | N/A | N/A |
| marks.real_strat_ret_5d.pending | 1 | 0 (+1) | 0 (+1) | N/A |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | N/A |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.complete.30d.denominator | 5 | 6 (-1) | 5 (0) | N/A |
| quote.complete.30d.evaluable | 5 | 6 (-1) | 5 (0) | N/A |
| quote.complete.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.complete.7d.denominator | 1 | 1 (0) | 0 (+1) | N/A |
| quote.complete.7d.evaluable | 1 | 1 (0) | 0 (+1) | N/A |
| quote.entry.30d.coverage | 1 | 1 (0) | 0.6 (+0.4) | N/A |
| quote.entry.30d.denominator | 3 | 3 (0) | 5 (-2) | N/A |
| quote.entry.30d.legacy | 0 | 0 (0) | 2 (-2) | N/A |
| quote.entry.30d.observed | 3 | 3 (0) | 3 (0) | N/A |
| quote.entry.7d.denominator | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.observed | 0 | 0 (0) | 0 (0) | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_active.denominator | 2 | 2 (0) | 3 (-1) | N/A |
| quote.exit_active.mark_today | 2 | 2 (0) | 3 (-1) | N/A |
| quote.exit_closed.30d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_closed.30d.denominator | 5 | 6 (-1) | 5 (0) | N/A |
| quote.exit_closed.30d.observed | 5 | 6 (-1) | 5 (0) | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.exit_closed.7d.denominator | 1 | 1 (0) | 0 (+1) | N/A |
| quote.exit_closed.7d.observed | 1 | 1 (0) | 0 (+1) | N/A |
| run.scanner.analyzed | 3 | 2 (+1) | 1 (+2) | N/A |
| run.scanner.champion_candidates | 0 | 1 (-1) | 0 (0) | N/A |
| run.scanner.contracts_designed | N/A | 1 (N/A) | N/A | N/A |
| run.scanner.direction_conflict_share | 0.3333 | 0 (+0.3333) | 1 (-0.6667) | N/A |
| run.scanner.direction_conflicts | 1 | 0 (+1) | 1 (0) | N/A |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | N/A |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | N/A |
| training.n_labels | 165 | 165 (0) | 133 (+32) | N/A |
| workflows.failed_runs | 1 | 0 (+1) | 0 (+1) | N/A |

2 Kennzahl(en) ohne Wert (N/A, nie 0): marks.real_opt_ret_5d.oldest_pending_days, quote.entry.7d.coverage
