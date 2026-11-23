# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-23 · erzeugt 2026-11-23T18:51:40+00:00 · Schema ops-status-v1  
Warnungen: 0 neu, 1 geändert, 4 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-23T18:14:10Z | 12 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | MISSED | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-23 (von heute: ja)
- **Universum / analysiert**: 518 / 1
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
| Entry 30d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Exit geschlossen 7d | N/A | 0/0 | keine |
| Exit geschlossen 30d | 100.0 % | 1/1 | EXIT_OBSERVED_ARCHIVED: 1 |
| Vollständig bewertbar 7d | N/A | 0/0 | keine |
| Vollständig bewertbar 30d | 100.0 % | 1/1 | keine |
| Exit aktiv (Mark heute) | 100.0 % | 2/2 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 778 | 775 | 2 | 50 | 1 | 14 |
| real_opt_ret_45d | 273 | 271 | 2 | 50 | 0 | 9 |
| real_opt_ret_5d | 1212 | 1211 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 778 | 775 | 2 | 50 | 1 | 14 |
| real_strat_ret_45d | 273 | 270 | 3 | 50 | 0 | 9 |
| real_strat_ret_5d | 1212 | 1210 | 1 | 50 | 1 | 2 |

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
- **Kandidaten (Real-Option)**: SIGNAL: 1056, CONTRACT_SELECTED: 690, ENTRY_OBSERVED: 690, OUTCOME_FINAL: 684, LATE_MARK: 0, PRICE_BASIS_CHANGED: 35, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 5

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
| CANDIDATE_20d\|ALL | 1028 | -0.0035 | 0.5477 | INCONCLUSIVE |
| CANDIDATE_20d\|CHAMPION | 7 | -0.036 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 235 | -0.0305 | 0.5021 | NEED_MORE_DATA |
| CANDIDATE_45d\|ALL | 380 | -0.0513 | 0.4026 | NEED_MORE_DATA |
| CANDIDATE_45d\|CHAMPION | 3 | -0.2926 | 0 | NEED_MORE_DATA |
| CANDIDATE_45d\|LATE_STAGE | 112 | -0.1021 | 0.3571 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1050 | -0.0004 | 0.5238 | INCONCLUSIVE |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 240 | -0.0014 | 0.5083 | INCONCLUSIVE |
| CANDIDATE_OPTION_20d\|ALL | 659 | -0.0508 | 0.4264 | INCONCLUSIVE |
| CANDIDATE_OPTION_20d\|CHAMPION | 5 | -0.138 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 154 | -0.1817 | 0.3182 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|ALL | 152 | -0.1836 | 0.2895 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|CHAMPION | 2 | -0.8481 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_45d\|LATE_STAGE | 50 | -0.2688 | 0.28 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 682 | -0.0958 | 0.3152 | NEGATIVE |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 158 | -0.096 | 0.3291 | NEGATIVE |
| CANDIDATE_STRATEGY_20d\|ALL | 659 | -0.0961 | 0.3885 | NEGATIVE |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 5 | -0.2819 | 0.4 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 154 | -0.2624 | 0.2662 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|ALL | 152 | -0.2153 | 0.2895 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|CHAMPION | 2 | -0.8559 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_45d\|LATE_STAGE | 50 | -0.3132 | 0.28 | NEED_MORE_DATA |
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
| PROM-ABST-001@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 19/150; unabhängige Tage 1/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 1 Marktregime beobachtet; Ereignis-Cluster 19/80; Treffer-Clu |
| PROM-ABST-002@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 2/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet |
| PROM-ABST-002@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 18/150; unabhängige Tage 1/30; Kalenderspanne 0/90 T; Regel-Treffer 10/30; Treffer-Tage 1/10; nur 1 Marktregime beobachtet; Sektor Technology dominiert (80%)  |
| PROM-ABST-003@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-003@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 19/150; unabhängige Tage 1/30; Kalenderspanne 0/90 T; Regel-Treffer 1/30; Treffer-Tage 1/10; nur 1 Marktregime beobachtet; Sektor Consumer Cyclical dominiert  |
| PROM-ABST-004@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 1/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet; Sektor Financial Services dominiert (1 |
| PROM-ABST-004@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 18/150; unabhängige Tage 1/30; Kalenderspanne 0/90 T; Regel-Treffer 10/30; Treffer-Tage 1/10; nur 1 Marktregime beobachtet; Sektor Technology dominiert (80%)  |
| PROM-ABST-005@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 3/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-005@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 19/150; unabhängige Tage 1/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 1 Marktregime beobachtet; Ereignis-Cluster 19/80; Treffer-Clu |
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 6834 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 0
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 10385, legacy: 4932

## Daten

- **Quellenstatus**: HEALTHY: 48, UNVALIDATED: 13, DEGRADED: 2
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 2, DEFERRED: 9, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-23T18:25:53+00:00
- **Externe Daten**: FAIL: 1, PASS: 38, DEFERRED: 18
- **Quote-Archiv heute / Median 14 T**: 843 / 277
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-23 (Quelle Produktion: nasdaq_official)

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
| Champion-Evidenz | 8 · Tage 7 | 30 · Tage 20 | 0.033 | nicht absehbar |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 6 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-11-20T16:51:40Z (74 h)
- **Läufe / fehlgeschlagen**: 64 / 3
- **Scanner-Start heute (New York)**: 2026-11-23T13:26-05:00

| Workflow | erstellt | Ergebnis | fehlgeschlagene Schritte | letzte Fehlerzeilen |
|---|---|---|---|---|
| PortWatch Recovery (read-only source, fail-closed) | 2026-11-22T08:53:25Z | failure | Verify actual recovery of both important PortWatch sources | [36;1m        except (ValueError, TypeError, AttributeError):[0m / ##[error]PortWatch recovery not proven: imf_portwatch_ports: provider observations not current; imf_portwatch_ports: archival observation did not advance; Houston: no current archived portcalls for po / ##[error]Process completed with exit code 1. |
| PortWatch Recovery (read-only source, fail-closed) | 2026-11-22T08:29:48Z | failure | Verify actual recovery of both important PortWatch sources | [36;1m        except (ValueError, TypeError, AttributeError):[0m / ##[error]PortWatch recovery not proven: imf_portwatch_ports: provider observations not current; imf_portwatch_ports: archival observation did not advance; Houston: no current archived portcalls for po / ##[error]Process completed with exit code 1. |
| PortWatch Recovery (read-only source, fail-closed) | 2026-11-21T08:50:08Z | failure | Verify actual recovery of both important PortWatch sources | ##[error]GitHub success is not source recovery: imf_portwatch_ports / ##[error]Process completed with exit code 1. |

Grün, aber nicht vollständig ausgeführt (jüngster Lauf je Kern-Workflow; technischer Status, kein Nachweis von Trades, Quotes oder Alpha):

| Workflow | effektiver Status | GitHub-Ergebnis | ausgefallene Stufen | übersprungene Jobs |
|---|---|---|---|---|
| Adaptive Asymmetry Scanner | UNVERIFIED | unknown | N/A | N/A |

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| CHANGED | WARNING | Governance | UNREVIEWED_CRITICAL_MERGE | 9 PR(s) mit CODEOWNERS-Pfaden ohne freigebendes Review gemergt (letzte 14 Tage): #211 (.github/workflows/portwatch_recovery.yml, .github/workflows/portwatch_reliability.yml); #209 (.github/workflows/portwatch_recovery.yml); #205 (.github/workflows/forward_cohort_audit.yml); #201 (.github/workflows/portwatch_live_contract.yml, .github/workflows/portwatch_recovery.yml); #200 (.github/workflows/portwatch_recovery.yml, config/external_sources/maritime.yaml, modules/source_health.py) | 2026-11-10 |
| ONGOING | WARNING | Governance | BRANCH_UNPROTECTED | Branch „main“ ist nicht geschützt: CODEOWNERS-Reviews (Bewertungsprotokolle, Gates, Datenkatalog, Workflows) werden nicht erzwungen. Nur der Owner kann das ändern (Settings → Rules). | 2026-11-10 |
| ONGOING | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_20d 2 von 778 offen, ältester 14 Tage; real_opt_ret_45d 2 von 273 offen, ältester 9 Tage; real_strat_ret_20d 2 von 778 offen, ältester 14 Tage; real_strat_ret_45d 3 von 273 offen, ältester 9 Tage (Schwelle > 3 Tage). | 2026-11-13 |
| ONGOING | WARNING | Lauf | SCANNER_FAILED:scanner | Workflow „scanner“ heute nicht ausgeführt (verpasst): Status MISSED, Ergebnis N/A, letzter Lauf N/A | 2026-11-20 |
| ONGOING | WARNING | Lauf | WORKFLOW_FAILED:PortWatch Recovery (read-only source, fail-closed) | „PortWatch Recovery (read-only source, fail-closed)“: 3 fehlgeschlagene(r) Lauf/Läufe seit 2026-11-20T16:51:40Z; Schritte: Verify actual recovery of both important PortWatch sources; letzte Fehlerzeile: ##[error]Process completed with exit code 1.; https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/44886722979 | 2026-11-18 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | 0 (0) |
| data.quote_archive_median_14d | 277 | 292 (-15) | 292 (-15) | 184 (+93) |
| data.quote_archive_today | 843 | 241 (+602) | 714 (+129) | 70 (+773) |
| data.source_health.BROKEN | N/A | N/A | 3 (N/A) | N/A |
| data.source_health.DEGRADED | 2 | 1 (+1) | 1 (+1) | N/A |
| data.source_health.HEALTHY | 48 | 48 (0) | 46 (+2) | 47 (+1) |
| data.source_health.STALE | N/A | 1 (N/A) | N/A | N/A |
| data.source_health.UNVALIDATED | 13 | 13 (0) | 13 (0) | 8 (+5) |
| data.source_health.n_bad | 2 | 2 (0) | 4 (-2) | 0 (+2) |
| forward.champion.mean_net | -0.2966 | -0.2966 (0) | -0.2966 (0) | -0.3322 (+0.0356) |
| forward.champion.n | 8 | 8 (0) | 8 (0) | 7 (+1) |
| forward.champion.n_independent_days | 7 | 7 (0) | 7 (0) | 6 (+1) |
| forward.champion.sum_net | -2.3728 | -2.3728 (0) | -2.3728 (0) | -2.3256 (-0.0472) |
| integrity.issues | 0 | 0 (0) | 0 (0) | 0 (0) |
| learning.champion_evidence.have | 8 | 8 (0) | 8 (0) | 7 (+1) |
| learning.meta_label.have | 235 | 235 (0) | 223 (+12) | 47 (+188) |
| learning.rl_arming.have | N/A | N/A | N/A | 13 (N/A) |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | 0 (0) |
| lifecycle.candidates.EXIT_PENDING | 5 | 6 (-1) | 6 (-1) | 200 (-195) |
| lifecycle.candidates.OUTCOME_FINAL | 684 | 682 (+2) | 677 (+7) | 392 (+292) |
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
| marks.real_strat_ret_5d.oldest_pending_days | 2 | N/A | N/A | N/A |
| marks.real_strat_ret_5d.pending | 1 | 0 (+1) | 0 (+1) | 0 (+1) |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | 0.875 (+0.125) |
| quote.complete.30d.denominator | 1 | 2 (-1) | 2 (-1) | 8 (-7) |
| quote.complete.30d.evaluable | 1 | 2 (-1) | 2 (-1) | 7 (-6) |
| quote.complete.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.complete.7d.denominator | 0 | 0 (0) | 0 (0) | 2 (-2) |
| quote.complete.7d.evaluable | 0 | 0 (0) | 0 (0) | 2 (-2) |
| quote.entry.30d.coverage | N/A | N/A | N/A | 0.5 (N/A) |
| quote.entry.30d.denominator | 0 | 0 (0) | 0 (0) | 6 (-6) |
| quote.entry.30d.legacy | 0 | 0 (0) | 0 (0) | 3 (-3) |
| quote.entry.30d.observed | 0 | 0 (0) | 0 (0) | 3 (-3) |
| quote.entry.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.entry.7d.denominator | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | 0 (0) |
| quote.entry.7d.observed | 0 | 0 (0) | 0 (0) | 1 (-1) |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | 1 (0) |
| quote.exit_active.denominator | 2 | 2 (0) | 2 (0) | 3 (-1) |
| quote.exit_active.mark_today | 2 | 2 (0) | 2 (0) | 3 (-1) |
| quote.exit_closed.30d.coverage | 1 | 1 (0) | 1 (0) | 0.875 (+0.125) |
| quote.exit_closed.30d.denominator | 1 | 2 (-1) | 2 (-1) | 8 (-7) |
| quote.exit_closed.30d.observed | 1 | 2 (-1) | 2 (-1) | 7 (-6) |
| quote.exit_closed.7d.coverage | N/A | N/A | N/A | 1 (N/A) |
| quote.exit_closed.7d.denominator | 0 | 0 (0) | 0 (0) | 2 (-2) |
| quote.exit_closed.7d.observed | 0 | 0 (0) | 0 (0) | 2 (-2) |
| run.scanner.analyzed | 1 | 2 (-1) | 1 (0) | 41 (-40) |
| run.scanner.champion_candidates | 0 | 1 (-1) | 1 (-1) | 12 (-12) |
| run.scanner.contracts_designed | N/A | 1 (N/A) | 1 (N/A) | 4 (N/A) |
| run.scanner.direction_conflict_share | 0 | 0.5 (-0.5) | 0 (0) | 0.0732 (-0.0732) |
| run.scanner.direction_conflicts | 0 | 1 (-1) | 0 (0) | 3 (-3) |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | 0 (0) |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | 518 (0) |
| training.n_labels | 235 | 235 (0) | 219 (+16) | 25 (+210) |
| workflows.failed_runs | 3 | 1 (+2) | 0 (+3) | 0 (+3) |

1 Kennzahl(en) ohne Wert (N/A, nie 0): marks.real_opt_ret_5d.oldest_pending_days
