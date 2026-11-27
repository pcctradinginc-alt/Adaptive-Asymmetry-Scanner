# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-27 · erzeugt 2026-11-27T17:05:38+00:00 · Schema ops-status-v1  
Warnungen: 2 neu, 1 geändert, 2 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-27T16:42:17Z | 10.4 |
| feedback | OK | success | 2026-11-27T02:18:59Z | -13.2 |
| scanner | MISSED | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-27 (von heute: ja)
- **Universum / analysiert**: 518 / 1
- **Champion-Kandidaten (Final-MC)**: 0
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 0 (Anteil 0)
- **Session**: regular
- **Stop-Grund**: Alle Signale bearish (Bearish-Trades deaktiviert).

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Entry 30d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Exit geschlossen 7d | N/A | 0/0 | keine |
| Exit geschlossen 30d | 100.0 % | 1/1 | EXIT_OBSERVED_ARCHIVED: 1 |
| Vollständig bewertbar 7d | N/A | 0/0 | keine |
| Vollständig bewertbar 30d | 100.0 % | 1/1 | keine |
| Exit aktiv (Mark heute) | 100.0 % | 2/2 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 894 | 890 | 1 | 50 | 3 | 1 |
| real_opt_ret_45d | 273 | 271 | 2 | 50 | 0 | 13 |
| real_opt_ret_5d | 1290 | 1289 | 0 | 50 | 1 | N/A |
| real_opt_ret_60d | 18 | 0 | 18 | 0 | 0 | 0 |
| real_strat_ret_20d | 894 | 890 | 1 | 50 | 3 | 1 |
| real_strat_ret_45d | 273 | 270 | 3 | 50 | 0 | 13 |
| real_strat_ret_5d | 1290 | 1289 | 0 | 50 | 1 | N/A |
| real_strat_ret_60d | 18 | 0 | 18 | 0 | 0 | 0 |

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
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 134/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 1057, CONTRACT_SELECTED: 691, ENTRY_OBSERVED: 691, OUTCOME_FINAL: 688, LATE_MARK: 0, PRICE_BASIS_CHANGED: 35, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 2

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
| CANDIDATE_20d\|ALL | 1035 | -0.0038 | 0.5459 | INCONCLUSIVE |
| CANDIDATE_20d\|CHAMPION | 7 | -0.036 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 236 | -0.0307 | 0.5 | NEED_MORE_DATA |
| CANDIDATE_45d\|ALL | 475 | -0.0462 | 0.4126 | NEED_MORE_DATA |
| CANDIDATE_45d\|CHAMPION | 3 | -0.2926 | 0 | NEED_MORE_DATA |
| CANDIDATE_45d\|LATE_STAGE | 133 | -0.0985 | 0.3684 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1054 | -0.0003 | 0.5228 | INCONCLUSIVE |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 241 | -0.0014 | 0.5062 | INCONCLUSIVE |
| CANDIDATE_OPTION_20d\|ALL | 665 | -0.0515 | 0.4271 | INCONCLUSIVE |
| CANDIDATE_OPTION_20d\|CHAMPION | 5 | -0.138 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 154 | -0.1817 | 0.3182 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|ALL | 271 | -0.1159 | 0.3395 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|CHAMPION | 3 | -0.7778 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|LATE_STAGE | 80 | -0.3293 | 0.2375 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 688 | -0.0962 | 0.314 | NEGATIVE |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 160 | -0.0983 | 0.325 | NEGATIVE |
| CANDIDATE_STRATEGY_20d\|ALL | 665 | -0.0983 | 0.3865 | NEGATIVE |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 5 | -0.2819 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 154 | -0.2624 | 0.2662 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|ALL | 270 | -0.1488 | 0.3333 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|CHAMPION | 3 | -0.8161 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|LATE_STAGE | 79 | -0.3952 | 0.2278 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 638 | -0.1428 | 0.2743 | NEGATIVE |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 144 | -0.1915 | 0.2292 | NEGATIVE |
| CHAMPION_TRADE\|ALL | 8 | -0.2966 | 0.125 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 236 / 19 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (MODEL: PERFORMANCE BREAKDOWN: Forward-Expectancy t=-4.81; DRIFT: FEATURE DRIFT SEVERE: ['vix', 'vix_chg_21', 'tnx'])
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
| PROM-ABST-001@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 50/150; unabhängige Tage 4/30; Kalenderspanne 3/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; Ereignis-Cluster 37/80; Treffer-Cluster 0/15 |
| PROM-ABST-002@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 2/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet |
| PROM-ABST-002@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 45/150; unabhängige Tage 4/30; Kalenderspanne 3/90 T; Regel-Treffer 14/30; Treffer-Tage 4/10; Ereignis-Cluster 34/80; Treffer-Cluster 12/15 |
| PROM-ABST-003@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-003@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 50/150; unabhängige Tage 4/30; Kalenderspanne 3/90 T; Regel-Treffer 5/30; Treffer-Tage 4/10; Ereignis-Cluster 37/80; Treffer-Cluster 3/15 |
| PROM-ABST-004@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 1/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet; Sektor Financial Services dominiert (1 |
| PROM-ABST-004@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 45/150; unabhängige Tage 4/30; Kalenderspanne 3/90 T; Regel-Treffer 16/30; Treffer-Tage 4/10; Sektor Technology dominiert (75%) – nur sektor-gescopte neue Hyp |
| PROM-ABST-005@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-005@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 50/150; unabhängige Tage 4/30; Kalenderspanne 3/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; Ereignis-Cluster 37/80; Treffer-Cluster 0/15 |
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 6901 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 0
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 10730, legacy: 4800

## Daten

- **Quellenstatus**: HEALTHY: 49, UNVALIDATED: 13, STALE: 1, DEGRADED: 1
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 2, DEFERRED: 9, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-27T16:52:11+00:00
- **Externe Daten**: PASS: 39, DEFERRED: 18
- **Quote-Archiv heute / Median 14 T**: 237 / 241
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-27 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 236 · Tage 19 | 200 · Tage 20 | 0.14 | 2026-12-01 |
| Champion-Evidenz | 8 · Tage 7 | 30 · Tage 20 | 0.033 | nicht absehbar |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 5 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-11-25T15:05:38Z (50 h)
- **Läufe / fehlgeschlagen**: 69 / 1
- **Scanner-Start heute (New York)**: 2026-11-27T11:52-05:00

| Workflow | erstellt | Ergebnis | fehlgeschlagene Schritte | letzte Fehlerzeilen |
|---|---|---|---|---|
| PortWatch HDX Shadow (OCHA mirror, no fallback) | 2026-11-26T02:25:53Z | failure | Fail check on stale, empty or unverified HDX data | ##[error]HDX shadow failed validation (no primary-source recovery and no Safe Mode override) / ##[error]Process completed with exit code 1. |

Grün, aber nicht vollständig ausgeführt (jüngster Lauf je Kern-Workflow; technischer Status, kein Nachweis von Trades, Quotes oder Alpha):

| Workflow | effektiver Status | GitHub-Ergebnis | ausgefallene Stufen | übersprungene Jobs |
|---|---|---|---|---|
| Adaptive Asymmetry Scanner (nach Source Health) | UNVERIFIED | unknown | N/A | N/A |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Lauf | SCANNER_FAILED:scanner | Workflow „scanner“ heute nicht ausgeführt (verpasst): Status MISSED, Ergebnis N/A, letzter Lauf N/A | 2026-11-27 |
| NEW | WARNING | Lauf | WORKFLOW_FAILED:PortWatch HDX Shadow (OCHA mirror, no fallback) | „PortWatch HDX Shadow (OCHA mirror, no fallback)“: 1 fehlgeschlagene(r) Lauf/Läufe seit 2026-11-25T15:05:38Z; Schritte: Fail check on stale, empty or unverified HDX data; letzte Fehlerzeile: ##[error]Process completed with exit code 1.; https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/45742367465 | 2026-11-27 |
| CHANGED | WARNING | Governance | UNREVIEWED_CRITICAL_MERGE | 6 PR(s) mit CODEOWNERS-Pfaden ohne freigebendes Review gemergt (letzte 14 Tage): #225 (.github/workflows/scanner.yml, .github/workflows/source_health.yml, config/source_health.yaml); #224 (.github/workflows/llm_census_offline.yml, modules/llm_census.py); #220 (.github/workflows/options_capture.yml); #217 (config/production_kill_switch.yaml, modules/production_intelligence_adapter.py); #211 (.github/workflows/portwatch_recovery.yml, .github/workflows/portwatch_reliability.yml) | 2026-11-10 |
| ONGOING | WARNING | Governance | BRANCH_UNPROTECTED | Branch „main“ ist nicht geschützt: CODEOWNERS-Reviews (Bewertungsprotokolle, Gates, Datenkatalog, Workflows) werden nicht erzwungen. Nur der Owner kann das ändern (Settings → Rules). | 2026-11-10 |
| ONGOING | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_45d 2 von 273 offen, ältester 13 Tage; real_strat_ret_45d 3 von 273 offen, ältester 13 Tage (Schwelle > 3 Tage). | 2026-11-13 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | 0 (0) |
| data.quote_archive_median_14d | 241 | 253.5 (-12.5) | 292 (-51) | 220 (+21) |
| data.quote_archive_today | 237 | 268 (-31) | 241 (-4) | 238 (-1) |
| data.source_health.DEGRADED | 1 | 2 (-1) | 1 (0) | N/A |
| data.source_health.HEALTHY | 49 | 48 (+1) | 48 (+1) | 46 (+3) |
| data.source_health.STALE | 1 | N/A | 1 (0) | 1 (0) |
| data.source_health.UNVALIDATED | 13 | 13 (0) | 13 (0) | 8 (+5) |
| data.source_health.n_bad | 2 | 2 (0) | 2 (0) | 1 (+1) |
| forward.champion.mean_net | -0.2966 | -0.2966 (0) | -0.2966 (0) | -0.3322 (+0.0356) |
| forward.champion.n | 8 | 8 (0) | 8 (0) | 7 (+1) |
| forward.champion.n_independent_days | 7 | 7 (0) | 7 (0) | 6 (+1) |
| forward.champion.sum_net | -2.3728 | -2.3728 (0) | -2.3728 (0) | -2.3256 (-0.0472) |
| integrity.issues | 0 | 0 (0) | 0 (0) | 0 (0) |
| learning.champion_evidence.have | 8 | 8 (0) | 8 (0) | 7 (+1) |
| learning.meta_label.have | 236 | 235 (+1) | 235 (+1) | 83 (+153) |
| learning.rl_arming.have | N/A | N/A | N/A | 13 (N/A) |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.candidates.EXIT_PENDING | 2 | 1 (+1) | 6 (-4) | 64 (-62) |
| lifecycle.candidates.OUTCOME_FINAL | 688 | 688 (0) | 682 (+6) | 592 (+96) |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.EXIT_PENDING | 2 | 2 (0) | 2 (0) | 3 (-1) |
| lifecycle.champion.state.LEARNED | 5 | 5 (0) | 5 (0) | 4 (+1) |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | 3 (0) |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | 110 (0) |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | 0 (0) |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 0 (0) | 0 (0) |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 0 (0) | 0 (0) |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.complete.30d.denominator | 1 | 1 (0) | 2 (-1) | 7 (-6) |
| quote.complete.30d.evaluable | 1 | 1 (0) | 2 (-1) | 7 (-6) |
| quote.complete.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.complete.7d.denominator | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.complete.7d.evaluable | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.entry.30d.coverage | N/A | N/A | N/A | 0.6 (N/A) |
| quote.entry.30d.denominator | 0 | 0 (0) | 0 (0) | 5 (-5) |
| quote.entry.30d.legacy | 0 | 0 (0) | 0 (0) | 2 (-2) |
| quote.entry.30d.observed | 0 | 0 (0) | 0 (0) | 3 (-3) |
| quote.entry.7d.denominator | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.entry.7d.observed | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_active.denominator | 2 | 2 (0) | 2 (0) | 3 (-1) |
| quote.exit_active.mark_today | 2 | 2 (0) | 2 (0) | 3 (-1) |
| quote.exit_closed.30d.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_closed.30d.denominator | 1 | 1 (0) | 2 (-1) | 7 (-6) |
| quote.exit_closed.30d.observed | 1 | 1 (0) | 2 (-1) | 7 (-6) |
| quote.exit_closed.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.exit_closed.7d.denominator | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.exit_closed.7d.observed | 0 | 0 (0) | 0 (0) | 1 (-1) |
| run.scanner.analyzed | 1 | 0 (+1) | 2 (-1) | 1 (0) |
| run.scanner.champion_candidates | 0 | 0 (0) | 1 (-1) | 0 (0) |
| run.scanner.contracts_designed | N/A | N/A | 1 (N/A) | N/A |
| run.scanner.direction_conflict_share | 0 | N/A | 0.5 (-0.5) | 0 (0) |
| run.scanner.direction_conflicts | 0 | N/A | 1 (-1) | 0 (0) |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | 0 (0) |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | 518 (0) |
| training.n_labels | 236 | 235 (+1) | 235 (+1) | 66 (+170) |
| workflows.failed_runs | 1 | 0 (+1) | 1 (0) | 2 (-1) |

3 Kennzahl(en) ohne Wert (N/A, nie 0): marks.real_opt_ret_5d.oldest_pending_days, marks.real_strat_ret_5d.oldest_pending_days, quote.entry.7d.coverage
