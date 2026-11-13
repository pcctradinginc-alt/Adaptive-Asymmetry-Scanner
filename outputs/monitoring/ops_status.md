# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-13 · erzeugt 2026-11-13T16:51:03+00:00 · Schema ops-status-v1  
Warnungen: 1 neu, 0 geändert, 2 laufend, 3 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-13T14:37:38Z | 8.3 |
| feedback | OK | success | 2026-11-13T00:35:16Z | -14.9 |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-13 (von heute: ja)
- **Universum / analysiert**: 518 / 2
- **Champion-Kandidaten (Final-MC)**: 0
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 0 (Anteil 0)
- **Session**: regular
- **Stop-Grund**: Kein Signal hat Mismatch-Filter bestanden.

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
| real_opt_ret_20d | 593 | 590 | 2 | 50 | 1 | 4 |
| real_opt_ret_45d | 50 | 50 | 0 | 31 | 0 | N/A |
| real_opt_ret_5d | 894 | 893 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 593 | 590 | 2 | 50 | 1 | 4 |
| real_strat_ret_45d | 50 | 50 | 0 | 31 | 0 | N/A |
| real_strat_ret_5d | 894 | 893 | 0 | 50 | 1 | N/A |

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
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 58/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 1049, CONTRACT_SELECTED: 683, ENTRY_OBSERVED: 683, OUTCOME_FINAL: 669, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 13

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
| CANDIDATE_20d\|ALL | 958 | -0.003 | 0.5511 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 7 | -0.036 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 219 | -0.0293 | 0.5068 | NEED_MORE_DATA |
| CANDIDATE_45d\|ALL | 18 | 0.0127 | 0.5556 | NEED_MORE_DATA |
| CANDIDATE_45d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_45d\|LATE_STAGE | 8 | 0.0554 | 0.75 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1035 | 0 | 0.5275 | INCONCLUSIVE |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 236 | -0.0009 | 0.5127 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 589 | -0.0406 | 0.4329 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 5 | -0.138 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 138 | -0.1749 | 0.3188 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|ALL | 18 | 0.0685 | 0.4444 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_OPTION_45d\|LATE_STAGE | 8 | 0.2711 | 0.625 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 669 | -0.0938 | 0.3184 | NEGATIVE |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 155 | -0.0937 | 0.3355 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 589 | -0.0822 | 0.4024 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 5 | -0.2819 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 138 | -0.2614 | 0.2681 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|ALL | 18 | 0.0423 | 0.4444 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_STRATEGY_45d\|LATE_STAGE | 8 | 0.2711 | 0.625 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 619 | -0.1374 | 0.2811 | NEGATIVE |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 139 | -0.1863 | 0.2374 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 8 | -0.2966 | 0.125 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 219 / 16 (min_labels: 200, min_dates: 20)
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 5622 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 0
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 12945, legacy: 5435

## Daten

- **Quellenstatus**: HEALTHY: 49, UNVALIDATED: 13, DEGRADED: 1
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 2, DEFERRED: 9, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-13T16:30:46+00:00
- **Externe Daten**: PASS: 40, DEFERRED: 17
- **Quote-Archiv heute / Median 14 T**: 168 / 259
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-13 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 219 · Tage 16 | 200 · Tage 20 | 0.29 | 2026-12-01 |
| Champion-Evidenz | 8 · Tage 7 | 30 · Tage 20 | 0.167 | 2027-03-25 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 20 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-11-12T14:51:03Z (26 h)
- **Läufe / fehlgeschlagen**: 35 / 0
- **Scanner-Start heute (New York)**: 2026-11-13T11:31-05:00

Keine fehlgeschlagenen Läufe.

Grün, aber nicht vollständig ausgeführt (jüngster Lauf je Kern-Workflow; technischer Status, kein Nachweis von Trades, Quotes oder Alpha):

| Workflow | effektiver Status | GitHub-Ergebnis | ausgefallene Stufen | übersprungene Jobs |
|---|---|---|---|---|
| Adaptive Asymmetry Scanner | UNVERIFIED | unknown | N/A | N/A |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_20d 2 von 593 offen, ältester 4 Tage; real_strat_ret_20d 2 von 593 offen, ältester 4 Tage (Schwelle > 3 Tage). | 2026-11-13 |
| RESOLVED | WARNING | Lauf | SCANNER_FAILED:scanner | Nicht mehr aktiv – Bedingung behoben. | 2026-11-12 |
| RESOLVED | WARNING | Lauf | WORKFLOW_FAILED:PortWatch HDX Shadow (OCHA mirror, no fallback) | Nicht mehr aktiv – Bedingung behoben. | 2026-11-12 |
| RESOLVED | WARNING | Lauf | WORKFLOW_FAILED:Tests (Pflicht-Check) | Nicht mehr aktiv – Bedingung behoben. | 2026-11-12 |
| ONGOING | WARNING | Governance | BRANCH_UNPROTECTED | Branch „main“ ist nicht geschützt: CODEOWNERS-Reviews (Bewertungsprotokolle, Gates, Datenkatalog, Workflows) werden nicht erzwungen. Nur der Owner kann das ändern (Settings → Rules). | 2026-11-10 |
| ONGOING | WARNING | Governance | UNREVIEWED_CRITICAL_MERGE | 12 PR(s) mit CODEOWNERS-Pfaden ohne freigebendes Review gemergt (letzte 14 Tage): #196 (config/champion_protocol.yaml, modules/champion_protocol.py); #194 (.github/workflows/llm_census_offline.yml, .github/workflows/options_learning_evidence.yml, .github/workflows/scanner.yml); #189 (config/champion_protocol.yaml, modules/champion_protocol.py); #187 (.github/CODEOWNERS, .github/workflows/ml_research.yml, .github/workflows/weekly_report.yml); #186 (.github/workflows/feature_discovery_tests.yml, .github/workflows/ml_research.yml, modules/research_memory.py) | 2026-11-10 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | N/A |
| data.quote_archive_median_14d | 259 | 255.5 (+3.5) | 222.5 (+36.5) | 164 (+95) |
| data.quote_archive_today | 168 | 285 (-117) | 117 (+51) | 164 (+4) |
| data.source_health.BROKEN | N/A | N/A | 2 (N/A) | 1 (N/A) |
| data.source_health.DEGRADED | 1 | 3 (-2) | 2 (-1) | 7 (-6) |
| data.source_health.HEALTHY | 49 | 47 (+2) | 49 (0) | 28 (+21) |
| data.source_health.STALE | N/A | N/A | 1 (N/A) | N/A |
| data.source_health.UNVALIDATED | 13 | 13 (0) | 9 (+4) | 8 (+5) |
| data.source_health.n_bad | 1 | 3 (-2) | 5 (-4) | 8 (-7) |
| forward.champion.mean_net | -0.2966 | -0.2966 (0) | -0.2966 (0) | -0.6168 (+0.3202) |
| forward.champion.n | 8 | 8 (0) | 8 (0) | 3 (+5) |
| forward.champion.n_independent_days | 7 | 7 (0) | 7 (0) | 3 (+4) |
| forward.champion.sum_net | -2.3728 | -2.3728 (0) | -2.3728 (0) | -1.8503 (-0.5225) |
| integrity.issues | 0 | 0 (0) | 0 (0) | 1 (-1) |
| learning.champion_evidence.have | 8 | 8 (0) | 8 (0) | N/A |
| learning.meta_label.have | 219 | 207 (+12) | 165 (+54) | N/A |
| learning.rl_arming.have | N/A | N/A | 14 (N/A) | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 1 (-1) |
| lifecycle.candidates.EXIT_PENDING | 13 | 11 (+2) | 8 (+5) | 40 (-27) |
| lifecycle.candidates.OUTCOME_FINAL | 669 | 669 (0) | 661 (+8) | 272 (+397) |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.EXIT_PENDING | 2 | 2 (0) | 2 (0) | 5 (-3) |
| lifecycle.champion.state.LEARNED | 5 | 5 (0) | 5 (0) | 0 (+5) |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | 3 (0) |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | 110 (0) |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | 0 (0) |
| marks.real_opt_ret_5d.oldest_pending_days | N/A | N/A | N/A | 9 (N/A) |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 0 (0) | 1 (-1) |
| marks.real_strat_ret_5d.oldest_pending_days | N/A | N/A | N/A | 9 (N/A) |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 0 (0) | 1 (-1) |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | 0.6 (+0.4) |
| quote.complete.30d.denominator | 5 | 5 (0) | 6 (-1) | 5 (0) |
| quote.complete.30d.evaluable | 5 | 5 (0) | 6 (-1) | 3 (+2) |
| quote.complete.7d.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.complete.7d.denominator | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.complete.7d.evaluable | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.entry.30d.coverage | 1 | 1 (0) | 1 (0) | 0.1667 (+0.8333) |
| quote.entry.30d.denominator | 3 | 3 (0) | 3 (0) | 6 (-3) |
| quote.entry.30d.legacy | 0 | 0 (0) | 0 (0) | 5 (-5) |
| quote.entry.30d.observed | 3 | 3 (0) | 3 (0) | 1 (+2) |
| quote.entry.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.entry.7d.denominator | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.entry.7d.observed | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_active.denominator | 2 | 2 (0) | 2 (0) | 5 (-3) |
| quote.exit_active.mark_today | 2 | 2 (0) | 2 (0) | 5 (-3) |
| quote.exit_closed.30d.coverage | 1 | 1 (0) | 1 (0) | 0.6 (+0.4) |
| quote.exit_closed.30d.denominator | 5 | 5 (0) | 6 (-1) | 5 (0) |
| quote.exit_closed.30d.observed | 5 | 5 (0) | 6 (-1) | 3 (+2) |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_closed.7d.denominator | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_closed.7d.observed | 1 | 1 (0) | 1 (0) | 1 (0) |
| run.scanner.analyzed | 2 | 3 (-1) | 2 (0) | 66 (-64) |
| run.scanner.champion_candidates | 0 | 1 (-1) | 1 (-1) | 11 (-11) |
| run.scanner.contracts_designed | N/A | 1 (N/A) | 1 (N/A) | 5 (N/A) |
| run.scanner.direction_conflict_share | 0 | 0 (0) | 0 (0) | N/A |
| run.scanner.direction_conflicts | 0 | 0 (0) | 0 (0) | N/A |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | 1 (-1) |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | 514 (+4) |
| training.n_labels | 219 | 207 (+12) | 165 (+54) | 0 (+219) |
| workflows.failed_runs | 0 | 2 (-2) | 0 (0) | N/A |
