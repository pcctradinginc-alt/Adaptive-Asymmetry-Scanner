# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-14 · erzeugt 2026-10-14T18:04:21+00:00 · Schema ops-status-v1  
Warnungen: 3 neu, 0 geändert, 0 laufend, 0 behoben; davon aktiv CRITICAL: 1  

## Lauf

| Workflow | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-10-14T11:56:20Z | 5.7 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | PENDING | N/A | N/A | N/A |

- **Tagesreport**: 2026-10-14 (von heute: ja)
- **Universum / analysiert**: 514 / 66
- **Champion-Kandidaten (Final-MC)**: 11
- **Kontrakte entworfen**: 5
- **Trades**: 1
- **Session**: regular
- **Stop-Grund**: 

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | 100.0 % | 1/1 | Altbestand vor Archiv: 0; ohne Altbestand 100.0 % (1/1) |
| Entry 30d | 16.7 % | 1/6 | Altbestand vor Archiv: 5; ohne Altbestand 100.0 % (1/1) |
| Exit geschlossen 7d | 100.0 % | 1/1 | EXIT_OBSERVED_NOT_ARCHIVED: 1 |
| Exit geschlossen 30d | 60.0 % | 3/5 | EXIT_NOT_REALIZABLE: 2, EXIT_OBSERVED_NOT_ARCHIVED: 3 |
| Vollständig bewertbar 7d | 100.0 % | 1/1 | keine |
| Vollständig bewertbar 30d | 60.0 % | 3/5 | EXIT_PRICE_UNKNOWN: 2 |
| Exit aktiv (Mark heute) | 100.0 % | 5/5 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ältester offener (Tage) |
|---|---|---|---|---|---|
| real_opt_ret_5d | 273 | 272 | 1 | 50 | 9 |
| real_strat_ret_5d | 273 | 272 | 1 | 50 | 9 |

## Lebenszyklus

| Champion-Zustand | Anzahl |
|---|---|
| SIGNAL | 0 |
| CONTRACT_SELECTED | 0 |
| ENTRY_OBSERVED | 0 |
| EXIT_PENDING | 5 |
| OUTCOME_FINAL | 110 |
| LEARNING_ELIGIBLE | 3 |
| LEARNED | 0 |

- **Probleme (Champion)**: ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE: 117, LEGACY_NO_STORED_EVALUATION: 113, OUTCOME_UNUSABLE: 107, OUTCOME_DATA_BAD: 3, LEGACY_LEARNED_UNTRACKED: 3
- **Exactly-once ok**: ja
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 0/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 544, CONTRACT_SELECTED: 313, ENTRY_OBSERVED: 313, OUTCOME_FINAL: 272, LATE_MARK: 0, EXIT_OVERDUE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 231, EXIT_PENDING: 40

## Forward-Performance

- **Champion realisiert (nur beobachtet): n**: 3
- **mean_net**: -0.6168
- **sum_net**: -1.8503
- **Trefferquote**: 0
- **schlechtester**: -0.7443
- **max. Drawdown (kum.)**: -1.8503
- **nicht bewertbar**: UNUSABLE: 107, DATA_BAD: 3

Verdikt NEED_MORE_DATA: keine Alpha-Behauptung (n=3, unabhängige Tage=3; Mindestwerte n≥30, Tage≥20).

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
| CANDIDATE_5d\|ALL | 382 | 0.0012 | 0.4895 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 3 | -0.0076 | 0.3333 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 112 | 0.0042 | 0.5625 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 232 | -0.1262 | 0.2328 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 3 | -0.1617 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 73 | -0.1011 | 0.3425 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 182 | -0.1746 | 0.2033 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 3 | -0.2853 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 57 | -0.2427 | 0.1579 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 3 | -0.6168 | 0 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 0 / 0 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Safe-Mode**: nein / ja
- **Champion-Version**: scanner_rules@b60b12cedaea
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

- **Probleme**: QUOTE_ARCHIVE
- **Promotion-Hash ok**: ja
- **Kette/Registry**: keine / keine
- **Duplikate Decision-Ledger / Trade-Outcomes**: 0 / 0
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 434 / 11 / 0
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: legacy: 210663, ok: 815

## Daten

- **Quellenstatus**: HEALTHY: 28, DEGRADED: 7, BROKEN: 1, UNVALIDATED: 8
- **Stand Quellenstatus**: 2026-10-14T17:34:54+00:00
- **Externe Daten**: keine
- **Quote-Archiv heute / Median 14 T**: 164 / 164
- **Quote-Archiv-Verzeichnis vorhanden**: ja

## Timing-Policy

- **Policy neuer Entscheidungen**: timing-v2
- **Bestand je Policy**: active: timing-v1: 4, timing-v2: 1, exit_pending: keine, closed: timing-v1: 113, unusable: keine, counterfactual_exit_pending: keine, counterfactual_unusable: keine
- **timing-v2 Entry offener Trades**: FILLED: 1
- **timing-v2 Exit ausstehend (ältester, Tage)**: 0 (N/A)
- **Gründe ausstehend**: keine
- **timing-v2 ausgeführt / UNUSABLE**: 0 / 0
- **Gründe UNUSABLE**: keine
- **Abweichung Ausführung vs. Entscheid**: n: 0, mean: N/A, definition: Outcome zur zulässigen Ausführung minus Return zum Entscheid

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | CRITICAL | Integrität | LEDGER_INTEGRITY | Ledger-/Hash-Integrität verletzt: QUOTE_ARCHIVE. | 2026-10-14 |
| NEW | WARNING | Lebenszyklus | OUTCOMES_OVERDUE:candidates | 1 Kandidat(en) mit überfälligem 5-Tage-Real-Option-Mark (> 2 Tage nach Fälligkeit). | 2026-10-14 |
| NEW | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_5d 1 von 273 offen, ältester 9 Tage; real_strat_ret_5d 1 von 273 offen, ältester 9 Tage (Schwelle > 3 Tage). | 2026-10-14 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.quote_archive_median_14d | 164 | N/A | N/A | N/A |
| data.quote_archive_today | 164 | N/A | N/A | N/A |
| data.source_health.BROKEN | 1 | N/A | N/A | N/A |
| data.source_health.DEGRADED | 7 | N/A | N/A | N/A |
| data.source_health.HEALTHY | 28 | N/A | N/A | N/A |
| data.source_health.UNVALIDATED | 8 | N/A | N/A | N/A |
| data.source_health.n_bad | 8 | N/A | N/A | N/A |
| forward.champion.mean_net | -0.6168 | N/A | N/A | N/A |
| forward.champion.n | 3 | N/A | N/A | N/A |
| forward.champion.n_independent_days | 3 | N/A | N/A | N/A |
| forward.champion.sum_net | -1.8503 | N/A | N/A | N/A |
| integrity.issues | 1 | N/A | N/A | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 1 | N/A | N/A | N/A |
| lifecycle.candidates.EXIT_PENDING | 40 | N/A | N/A | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 272 | N/A | N/A | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | N/A | N/A | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | N/A | N/A | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.EXIT_PENDING | 5 | N/A | N/A | N/A |
| lifecycle.champion.state.LEARNED | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | N/A | N/A | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | N/A | N/A | N/A |
| lifecycle.champion.state.SIGNAL | 0 | N/A | N/A | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | 9 | N/A | N/A | N/A |
| marks.real_opt_ret_5d.pending | 1 | N/A | N/A | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | 9 | N/A | N/A | N/A |
| marks.real_strat_ret_5d.pending | 1 | N/A | N/A | N/A |
| promotion.active_influence | 0 | N/A | N/A | N/A |
| promotion.unjustified_influence | 0 | N/A | N/A | N/A |
| quote.complete.30d.coverage | 0.6 | N/A | N/A | N/A |
| quote.complete.30d.denominator | 5 | N/A | N/A | N/A |
| quote.complete.30d.evaluable | 3 | N/A | N/A | N/A |
| quote.complete.7d.coverage | 1 | N/A | N/A | N/A |
| quote.complete.7d.denominator | 1 | N/A | N/A | N/A |
| quote.complete.7d.evaluable | 1 | N/A | N/A | N/A |
| quote.entry.30d.coverage | 0.1667 | N/A | N/A | N/A |
| quote.entry.30d.denominator | 6 | N/A | N/A | N/A |
| quote.entry.30d.legacy | 5 | N/A | N/A | N/A |
| quote.entry.30d.observed | 1 | N/A | N/A | N/A |
| quote.entry.7d.coverage | 1 | N/A | N/A | N/A |
| quote.entry.7d.denominator | 1 | N/A | N/A | N/A |
| quote.entry.7d.legacy | 0 | N/A | N/A | N/A |
| quote.entry.7d.observed | 1 | N/A | N/A | N/A |
| quote.exit_active.coverage | 1 | N/A | N/A | N/A |
| quote.exit_active.denominator | 5 | N/A | N/A | N/A |
| quote.exit_active.mark_today | 5 | N/A | N/A | N/A |
| quote.exit_closed.30d.coverage | 0.6 | N/A | N/A | N/A |
| quote.exit_closed.30d.denominator | 5 | N/A | N/A | N/A |
| quote.exit_closed.30d.observed | 3 | N/A | N/A | N/A |
| quote.exit_closed.7d.coverage | 1 | N/A | N/A | N/A |
| quote.exit_closed.7d.denominator | 1 | N/A | N/A | N/A |
| quote.exit_closed.7d.observed | 1 | N/A | N/A | N/A |
| run.scanner.analyzed | 66 | N/A | N/A | N/A |
| run.scanner.champion_candidates | 11 | N/A | N/A | N/A |
| run.scanner.contracts_designed | 5 | N/A | N/A | N/A |
| run.scanner.trades | 1 | N/A | N/A | N/A |
| run.scanner.universe | 514 | N/A | N/A | N/A |
| training.n_labels | 0 | N/A | N/A | N/A |
