# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-19 · erzeugt 2026-11-19T16:22:13+00:00 · Schema ops-status-v1  
Warnungen: 0 neu, 0 geändert, 4 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-19T15:47:30Z | 9.5 |
| feedback | OK | success | 2026-11-19T01:18:34Z | -14.2 |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-19 (von heute: ja)
- **Universum / analysiert**: 518 / 2
- **Champion-Kandidaten (Final-MC)**: 0
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 1 (Anteil 0.5)
- **Session**: regular
- **Stop-Grund**: Kein Signal hat Mismatch-Filter bestanden.

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Entry 30d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Exit geschlossen 7d | N/A | 0/0 | keine |
| Exit geschlossen 30d | 100.0 % | 2/2 | EXIT_OBSERVED_ARCHIVED: 2 |
| Vollständig bewertbar 7d | N/A | 0/0 | keine |
| Vollständig bewertbar 30d | 100.0 % | 2/2 | keine |
| Exit aktiv (Mark heute) | 100.0 % | 2/2 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 700 | 697 | 2 | 50 | 1 | 10 |
| real_opt_ret_45d | 113 | 112 | 1 | 50 | 0 | 5 |
| real_opt_ret_5d | 1094 | 1093 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 700 | 697 | 2 | 50 | 1 | 10 |
| real_strat_ret_45d | 113 | 112 | 1 | 50 | 0 | 5 |
| real_strat_ret_5d | 1094 | 1093 | 0 | 50 | 1 | N/A |

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
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 77/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 1053, CONTRACT_SELECTED: 687, ENTRY_OBSERVED: 687, OUTCOME_FINAL: 682, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 4

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
| CANDIDATE_20d\|ALL | 1025 | -0.0035 | 0.5483 | INCONCLUSIVE |
| CANDIDATE_20d\|CHAMPION | 7 | -0.036 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 235 | -0.0305 | 0.5021 | NEED_MORE_DATA |
| CANDIDATE_45d\|ALL | 143 | -0.0306 | 0.4266 | NEED_MORE_DATA |
| CANDIDATE_45d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_45d\|LATE_STAGE | 47 | -0.0443 | 0.4681 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1048 | -0.0004 | 0.5239 | INCONCLUSIVE |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 239 | -0.0012 | 0.5105 | INCONCLUSIVE |
| CANDIDATE_OPTION_20d\|ALL | 657 | -0.0494 | 0.4277 | INCONCLUSIVE |
| CANDIDATE_OPTION_20d\|CHAMPION | 5 | -0.138 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 154 | -0.1817 | 0.3182 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|ALL | 112 | -0.1376 | 0.3125 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_OPTION_45d\|LATE_STAGE | 38 | -0.1805 | 0.3421 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 682 | -0.0958 | 0.3152 | NEGATIVE |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 158 | -0.096 | 0.3291 | NEGATIVE |
| CANDIDATE_STRATEGY_20d\|ALL | 657 | -0.0949 | 0.3896 | NEGATIVE |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 5 | -0.2819 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 154 | -0.2624 | 0.2662 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|ALL | 112 | -0.158 | 0.3125 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_STRATEGY_45d\|LATE_STAGE | 38 | -0.188 | 0.3421 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 632 | -0.141 | 0.2769 | NEGATIVE |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 142 | -0.1896 | 0.2324 | NEGATIVE |
| CHAMPION_TRADE\|ALL | 8 | -0.2966 | 0.125 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 235 / 18 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (MODEL: EXCESSIVE MODEL DISAGREEMENT; MODEL: PERFORMANCE BREAKDOWN: Forward-Expectancy t=-7.70)
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 6432 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 0
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 12873, legacy: 5423

## Daten

- **Quellenstatus**: HEALTHY: 46, DEGRADED: 3, UNVALIDATED: 13, STALE: 1
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 2, DEFERRED: 9, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-19T15:59:49+00:00
- **Externe Daten**: WARN: 2, PASS: 36, FAIL: 1, DEFERRED: 18
- **Quote-Archiv heute / Median 14 T**: 174 / 292
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-19 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 235 · Tage 18 | 200 · Tage 20 | 0.36 | 2026-12-01 |
| Champion-Evidenz | 8 · Tage 7 | 30 · Tage 20 | 0.067 | 2027-10-15 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 5 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-11-18T14:22:13Z (26 h)
- **Läufe / fehlgeschlagen**: 35 / 1
- **Scanner-Start heute (New York)**: 2026-11-19T11:00-05:00

| Workflow | erstellt | Ergebnis | fehlgeschlagene Schritte | letzte Fehlerzeilen |
|---|---|---|---|---|
| PortWatch Recovery (read-only source, fail-closed) | 2026-11-19T09:31:56Z | failure | Verify actual recovery of both important PortWatch sources | ##[error]GitHub success is not source recovery: imf_portwatch_ports / ##[error]Process completed with exit code 1. |

Grün, aber nicht vollständig ausgeführt (jüngster Lauf je Kern-Workflow; technischer Status, kein Nachweis von Trades, Quotes oder Alpha):

| Workflow | effektiver Status | GitHub-Ergebnis | ausgefallene Stufen | übersprungene Jobs |
|---|---|---|---|---|
| Adaptive Asymmetry Scanner | UNVERIFIED | unknown | N/A | N/A |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| ONGOING | WARNING | Governance | BRANCH_UNPROTECTED | Branch „main“ ist nicht geschützt: CODEOWNERS-Reviews (Bewertungsprotokolle, Gates, Datenkatalog, Workflows) werden nicht erzwungen. Nur der Owner kann das ändern (Settings → Rules). | 2026-11-10 |
| ONGOING | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_20d 2 von 700 offen, ältester 10 Tage; real_opt_ret_45d 1 von 113 offen, ältester 5 Tage; real_strat_ret_20d 2 von 700 offen, ältester 10 Tage; real_strat_ret_45d 1 von 113 offen, ältester 5 Tage (Schwelle > 3 Tage). | 2026-11-13 |
| ONGOING | WARNING | Governance | UNREVIEWED_CRITICAL_MERGE | 10 PR(s) mit CODEOWNERS-Pfaden ohne freigebendes Review gemergt (letzte 14 Tage): #205 (.github/workflows/forward_cohort_audit.yml); #201 (.github/workflows/portwatch_live_contract.yml, .github/workflows/portwatch_recovery.yml); #200 (.github/workflows/portwatch_recovery.yml, config/external_sources/maritime.yaml, modules/source_health.py); #199 (.github/workflows/ml_research.yml, .github/workflows/portwatch_reliability.yml, config/source_health.yaml); #198 (.github/workflows/ml_research.yml, .github/workflows/option_feature_diagnostics.yml) | 2026-11-10 |
| ONGOING | WARNING | Lauf | WORKFLOW_FAILED:PortWatch Recovery (read-only source, fail-closed) | „PortWatch Recovery (read-only source, fail-closed)“: 1 fehlgeschlagene(r) Lauf/Läufe seit 2026-11-18T14:22:13Z; Schritte: Verify actual recovery of both important PortWatch sources; letzte Fehlerzeile: ##[error]Process completed with exit code 1.; https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/44497270214 | 2026-11-18 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | 0 (0) |
| data.quote_archive_median_14d | 292 | 292 (0) | 255.5 (+36.5) | 191 (+101) |
| data.quote_archive_today | 174 | 255 (-81) | 285 (-111) | 206 (-32) |
| data.source_health.DEGRADED | 3 | 2 (+1) | 3 (0) | N/A |
| data.source_health.HEALTHY | 46 | 47 (-1) | 47 (-1) | 47 (-1) |
| data.source_health.STALE | 1 | 1 (0) | N/A | N/A |
| data.source_health.UNVALIDATED | 13 | 13 (0) | 13 (0) | 8 (+5) |
| data.source_health.n_bad | 4 | 3 (+1) | 3 (+1) | 0 (+4) |
| forward.champion.mean_net | -0.2966 | -0.2966 (0) | -0.2966 (0) | -0.3095 (+0.0129) |
| forward.champion.n | 8 | 8 (0) | 8 (0) | 6 (+2) |
| forward.champion.n_independent_days | 7 | 7 (0) | 7 (0) | 5 (+2) |
| forward.champion.sum_net | -2.3728 | -2.3728 (0) | -2.3728 (0) | -1.8569 (-0.5159) |
| integrity.issues | 0 | 0 (0) | 0 (0) | 0 (0) |
| learning.champion_evidence.have | 8 | 8 (0) | 8 (0) | 6 (+2) |
| learning.meta_label.have | 235 | 235 (0) | 207 (+28) | 19 (+216) |
| learning.rl_arming.have | N/A | N/A | N/A | 12 (N/A) |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.candidates.EXIT_PENDING | 4 | 2 (+2) | 11 (-7) | 120 (-116) |
| lifecycle.candidates.OUTCOME_FINAL | 682 | 682 (0) | 669 (+13) | 352 (+330) |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.champion.state.EXIT_PENDING | 2 | 2 (0) | 2 (0) | 4 (-2) |
| lifecycle.champion.state.LEARNED | 5 | 5 (0) | 5 (0) | 3 (+2) |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | 3 (0) |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | 110 (0) |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | 0 (0) |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 0 (0) | 0 (0) |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 0 (0) | 0 (0) |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | 0.75 (+0.25) |
| quote.complete.30d.denominator | 2 | 2 (0) | 5 (-3) | 8 (-6) |
| quote.complete.30d.evaluable | 2 | 2 (0) | 5 (-3) | 6 (-4) |
| quote.complete.7d.coverage | N/A | N/A | 1 (N/A) | 1 (N/A) |
| quote.complete.7d.denominator | 0 | 0 (0) | 1 (-1) | 3 (-3) |
| quote.complete.7d.evaluable | 0 | 0 (0) | 1 (-1) | 3 (-3) |
| quote.entry.30d.coverage | N/A | N/A | 1 (N/A) | 0.375 (N/A) |
| quote.entry.30d.denominator | 0 | 0 (0) | 3 (-3) | 8 (-8) |
| quote.entry.30d.legacy | 0 | 0 (0) | 0 (0) | 5 (-5) |
| quote.entry.30d.observed | 0 | 0 (0) | 3 (-3) | 3 (-3) |
| quote.entry.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.entry.7d.denominator | 0 | 0 (0) | 0 (0) | 3 (-3) |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.entry.7d.observed | 0 | 0 (0) | 0 (0) | 3 (-3) |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_active.denominator | 2 | 2 (0) | 2 (0) | 4 (-2) |
| quote.exit_active.mark_today | 2 | 2 (0) | 2 (0) | 4 (-2) |
| quote.exit_closed.30d.coverage | 1 | 1 (0) | 1 (0) | 0.75 (+0.25) |
| quote.exit_closed.30d.denominator | 2 | 2 (0) | 5 (-3) | 8 (-6) |
| quote.exit_closed.30d.observed | 2 | 2 (0) | 5 (-3) | 6 (-4) |
| quote.exit_closed.7d.coverage | N/A | N/A | 1 (N/A) | 1 (N/A) |
| quote.exit_closed.7d.denominator | 0 | 0 (0) | 1 (-1) | 3 (-3) |
| quote.exit_closed.7d.observed | 0 | 0 (0) | 1 (-1) | 3 (-3) |
| run.scanner.analyzed | 2 | 0 (+2) | 3 (-1) | 47 (-45) |
| run.scanner.champion_candidates | 0 | 0 (0) | 1 (-1) | 7 (-7) |
| run.scanner.contracts_designed | N/A | N/A | 1 (N/A) | N/A |
| run.scanner.direction_conflict_share | 0.5 | N/A | 0 (+0.5) | 0.0638 (+0.4362) |
| run.scanner.direction_conflicts | 1 | N/A | 0 (+1) | 3 (-2) |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | 0 (0) |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | 518 (0) |
| training.n_labels | 235 | 235 (0) | 207 (+28) | 8 (+227) |
| workflows.failed_runs | 1 | 1 (0) | 2 (-1) | 0 (+1) |

2 Kennzahl(en) ohne Wert (N/A, nie 0): marks.real_opt_ret_5d.oldest_pending_days, marks.real_strat_ret_5d.oldest_pending_days
