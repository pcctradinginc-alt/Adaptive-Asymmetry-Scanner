# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-10 · erzeugt 2026-11-10T17:38:56+00:00 · Schema ops-status-v1  
Warnungen: 2 neu, 0 geändert, 0 laufend, 1 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-10T14:55:37Z | 8.6 |
| feedback | OK | success | 2026-11-10T02:18:45Z | -13.2 |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-10 (von heute: ja)
- **Universum / analysiert**: 518 / 2
- **Champion-Kandidaten (Final-MC)**: 1
- **Kontrakte entworfen**: 1
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 0 (Anteil 0)
- **Session**: regular
- **Stop-Grund**: Alle Options-Kontrakte scheitern am ROI-Gate.

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
| real_opt_ret_20d | 513 | 510 | 2 | 50 | 1 | 1 |
| real_opt_ret_5d | 855 | 854 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 513 | 510 | 2 | 50 | 1 | 1 |
| real_strat_ret_5d | 855 | 854 | 0 | 50 | 1 | N/A |

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
- **Kandidaten (Real-Option)**: SIGNAL: 1041, CONTRACT_SELECTED: 675, ENTRY_OBSERVED: 675, OUTCOME_FINAL: 667, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 7

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
| CANDIDATE_20d\|ALL | 805 | -0.0039 | 0.5416 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 7 | -0.036 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 181 | -0.0256 | 0.5138 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1030 | 0 | 0.5282 | INCONCLUSIVE |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 235 | -0.0009 | 0.5149 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 471 | -0.0442 | 0.4246 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 5 | -0.138 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 109 | -0.1604 | 0.3303 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 664 | -0.0941 | 0.3178 | NEGATIVE |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 154 | -0.0926 | 0.3377 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 471 | -0.0854 | 0.3992 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 5 | -0.2819 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 109 | -0.2609 | 0.2661 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 614 | -0.1364 | 0.2834 | NEGATIVE |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 138 | -0.1848 | 0.2391 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 8 | -0.2966 | 0.125 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 181 / 13 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (MODEL: PERFORMANCE BREAKDOWN: Forward-Expectancy t=-5.33)
- **Champion-Version**: scanner_rules@9aaf7a892de0
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 5004 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 13038, legacy: 5628

## Daten

- **Quellenstatus**: HEALTHY: 46, UNVALIDATED: 13, STALE: 2, DEGRADED: 2
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 2, DEFERRED: 9, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-10T17:17:24+00:00
- **Externe Daten**: PASS: 39, DEFERRED: 17, WARN: 1
- **Quote-Archiv heute / Median 14 T**: 448 / 222.5
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-10 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 181 · Tage 13 | 200 · Tage 20 | 0.14 | 2026-12-01 |
| Champion-Evidenz | 8 · Tage 7 | 30 · Tage 20 | 0.167 | 2027-03-22 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 56 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-11-09T15:38:56Z (26 h)
- **Läufe / fehlgeschlagen**: 36 / 0
- **Scanner-Start heute (New York)**: 2026-11-10T12:17-05:00

Keine fehlgeschlagenen Läufe.

Grün, aber nicht vollständig ausgeführt (jüngster Lauf je Kern-Workflow; technischer Status, kein Nachweis von Trades, Quotes oder Alpha):

| Workflow | effektiver Status | GitHub-Ergebnis | ausgefallene Stufen | übersprungene Jobs |
|---|---|---|---|---|
| Adaptive Asymmetry Scanner | UNVERIFIED | unknown | N/A | N/A |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Governance | BRANCH_UNPROTECTED | Branch „main“ ist nicht geschützt: CODEOWNERS-Reviews (Bewertungsprotokolle, Gates, Datenkatalog, Workflows) werden nicht erzwungen. Nur der Owner kann das ändern (Settings → Rules). | 2026-11-10 |
| NEW | WARNING | Governance | UNREVIEWED_CRITICAL_MERGE | 14 PR(s) mit CODEOWNERS-Pfaden ohne freigebendes Review gemergt (letzte 14 Tage): #187 (.github/CODEOWNERS, .github/workflows/ml_research.yml, .github/workflows/weekly_report.yml); #186 (.github/workflows/feature_discovery_tests.yml, .github/workflows/ml_research.yml, modules/research_memory.py); #185 (.github/workflows/feature_discovery_tests.yml, .github/workflows/ml_research.yml, modules/research_memory.py); #184 (.github/workflows/feature_discovery_tests.yml, .github/workflows/ml_research.yml); #183 (.github/workflows/feature_discovery_tests.yml, .github/workflows/ml_research.yml, modules/research_memory.py) | 2026-11-10 |
| RESOLVED | WARNING | Lauf | WORKFLOW_FAILED:PortWatch HDX Shadow (OCHA mirror, no fallback) | Nicht mehr aktiv – Bedingung behoben. | 2026-11-09 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | N/A |
| data.quote_archive_median_14d | 222.5 | 222.5 (0) | 193 (+29.5) | N/A |
| data.quote_archive_today | 448 | 516 (-68) | 172 (+276) | N/A |
| data.source_health.BROKEN | N/A | N/A | 2 (N/A) | N/A |
| data.source_health.DEGRADED | 2 | 3 (-1) | 2 (0) | N/A |
| data.source_health.HEALTHY | 46 | 49 (-3) | 42 (+4) | N/A |
| data.source_health.STALE | 2 | 2 (0) | 1 (+1) | N/A |
| data.source_health.UNVALIDATED | 13 | 9 (+4) | 8 (+5) | N/A |
| data.source_health.n_bad | 4 | 5 (-1) | 5 (-1) | N/A |
| forward.champion.mean_net | -0.2966 | -0.2966 (0) | -0.3322 (+0.0356) | N/A |
| forward.champion.n | 8 | 8 (0) | 7 (+1) | N/A |
| forward.champion.n_independent_days | 7 | 7 (0) | 6 (+1) | N/A |
| forward.champion.sum_net | -2.3728 | -2.3728 (0) | -2.3256 (-0.0472) | N/A |
| integrity.issues | 0 | 0 (0) | 0 (0) | N/A |
| learning.champion_evidence.have | 8 | 8 (0) | 7 (+1) | N/A |
| learning.meta_label.have | 181 | 174 (+7) | 133 (+48) | N/A |
| learning.rl_arming.have | N/A | 14 (N/A) | 13 (N/A) | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.candidates.EXIT_PENDING | 7 | 8 (-1) | 4 (+3) | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 667 | 664 (+3) | 659 (+8) | N/A |
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
| marks.real_strat_ret_5d.oldest_pending_days | N/A | 1 (N/A) | N/A | N/A |
| marks.real_strat_ret_5d.pending | 0 | 1 (-1) | 0 (0) | N/A |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | N/A |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.complete.30d.denominator | 5 | 5 (0) | 5 (0) | N/A |
| quote.complete.30d.evaluable | 5 | 5 (0) | 5 (0) | N/A |
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
| quote.exit_closed.30d.denominator | 5 | 5 (0) | 5 (0) | N/A |
| quote.exit_closed.30d.observed | 5 | 5 (0) | 5 (0) | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.exit_closed.7d.denominator | 1 | 1 (0) | 0 (+1) | N/A |
| quote.exit_closed.7d.observed | 1 | 1 (0) | 0 (+1) | N/A |
| run.scanner.analyzed | 2 | 3 (-1) | 1 (+1) | N/A |
| run.scanner.champion_candidates | 1 | 0 (+1) | 0 (+1) | N/A |
| run.scanner.contracts_designed | 1 | N/A | N/A | N/A |
| run.scanner.direction_conflict_share | 0 | 0.3333 (-0.3333) | 0 (0) | N/A |
| run.scanner.direction_conflicts | 0 | 1 (-1) | 0 (0) | N/A |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | N/A |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | N/A |
| training.n_labels | 181 | 165 (+16) | 133 (+48) | N/A |
| workflows.failed_runs | 0 | 1 (-1) | 0 (0) | N/A |

2 Kennzahl(en) ohne Wert (N/A, nie 0): marks.real_opt_ret_5d.oldest_pending_days, quote.entry.7d.coverage
