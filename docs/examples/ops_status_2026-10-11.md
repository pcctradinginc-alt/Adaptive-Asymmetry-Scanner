> **Beispielbericht aus echten Repo-Daten** (Stand 2026-10-11, Sonntag; read-only erzeugt mit `python -m modules.ops_report --out <tmp>`). Keine Testdaten. Der Workflow-Status des Tages fehlt, weil an einem Sonntag kein Scanner-Lauf geplant ist.

# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-11 · erzeugt 2026-10-11T07:35:53+00:00 · Schema ops-status-v1  
Warnungen: 2 neu, 0 geändert, 0 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

Workflow-Status für heute: N/A

- **Tagesreport**: 2026-10-09 (von heute: nein)
- **Universum / analysiert**: 372 / 96
- **Champion-Kandidaten (Final-MC)**: 21
- **Kontrakte entworfen**: 15
- **Trades**: 0
- **Session**: regular
- **Stop-Grund**: Alle Options-Kontrakte scheitern am ROI-Gate.

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | 0.0 % | 0/2 | Altbestand vor Archiv: 2; ohne Altbestand N/A (0/0) |
| Entry 30d | 0.0 % | 0/5 | Altbestand vor Archiv: 5; ohne Altbestand N/A (0/0) |
| Exit geschlossen 7d | 100.0 % | 1/1 | EXIT_OBSERVED_NOT_ARCHIVED: 1 |
| Exit geschlossen 30d | 60.0 % | 3/5 | EXIT_NOT_REALIZABLE: 2, EXIT_OBSERVED_NOT_ARCHIVED: 3 |
| Vollständig bewertbar 7d | 100.0 % | 1/1 | keine |
| Vollständig bewertbar 30d | 60.0 % | 3/5 | EXIT_PRICE_UNKNOWN: 2 |
| Exit aktiv (Mark heute) | 0.0 % | 0/4 | Handelstag: nein; ältester Mark: N/A Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ältester offener (Tage) |
|---|---|---|---|---|---|
| real_opt_ret_5d | 153 | 112 | 41 | 0 | 6 |
| real_strat_ret_5d | 153 | 26 | 127 | 0 | 8 |

## Lebenszyklus

| Champion-Zustand | Anzahl |
|---|---|
| SIGNAL | 0 |
| CONTRACT_SELECTED | 0 |
| ENTRY_OBSERVED | 0 |
| EXIT_PENDING | 4 |
| OUTCOME_FINAL | 117 |
| LEARNING_ELIGIBLE | 3 |
| LEARNED | 0 |

- **Probleme (Champion)**: ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE: 124, LEGACY_NO_STORED_EVALUATION: 120, OUTCOME_UNUSABLE: 114, OUTCOME_DATA_BAD: 3, LEGACY_LEARNED_UNTRACKED: 3
- **Exactly-once ok**: ja
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 0/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 478, CONTRACT_SELECTED: 273, ENTRY_OBSERVED: 273, OUTCOME_FINAL: 112, LATE_MARK: 0, EXIT_OVERDUE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 205, EXIT_PENDING: 160

## Forward-Performance

- **Champion realisiert (nur beobachtet): n**: 3
- **mean_net**: -0.6126
- **sum_net**: -1.8378
- **Trefferquote**: 0
- **schlechtester**: -0.7381
- **max. Drawdown (kum.)**: -1.8378
- **nicht bewertbar**: UNUSABLE: 114, DATA_BAD: 3

Verdikt NEED_MORE_DATA: keine Alpha-Behauptung (n=3, unabhängige Tage=3; Mindestwerte n≥30, Tage≥20).

| Challenger (Shadow) | vermiedene Verluste | entgangene Gewinne | Δ vs. Champion (Σ) | greift/bewertbar | Population | Verdikt |
|---|---|---|---|---|---|---|
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
| UNIV-V2-SEG-MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-MID@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-SMALL@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-ULTRA_MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |

14 von 14 Challenger-Verdikten NEED_MORE_DATA – dort keine Alpha-Behauptung.

| Scorecard | n | mean_net | Trefferquote | Verdikt |
|---|---|---|---|---|
| CANDIDATE_5d\|ALL | 144 | 0.0072 | 0.4375 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_5d\|LATE_STAGE | 47 | 0.0077 | 0.5319 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 112 | -0.1057 | 0.2679 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 38 | -0.0644 | 0.3947 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 26 | -0.1933 | 0.2308 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 11 | -0.2057 | 0.2727 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 0 | N/A | N/A | NO_DATA |

## Training

- **Status**: NEVER_RUN
- **Labels / Tage (benötigt)**: N/A / N/A (N/A)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Safe-Mode**: nein / ja
- **Champion-Version**: scanner_rules@b60b12cedaea
- **Zustände**: PROSPECTIVE_CHALLENGER: 14
- **Aktiver Einfluss**: keine
- **Ungerechtfertigter Einfluss**: keine

| Hypothese | Zustand | Einfluss | Entscheidung | Blockierungsgründe |
|---|---|---|---|---|
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
| UNIV-V2-SEG-MICRO@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| UNIV-V2-SEG-MID@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| UNIV-V2-SEG-SMALL@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |
| UNIV-V2-SEG-ULTRA_MICRO@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/100; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/100; Treffer-Tage 0/30; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Clus |

## Integrität

- **Probleme**: keine
- **Promotion-Hash ok**: ja
- **Kette/Registry**: keine / keine
- **Duplikate Decision-Ledger / Trade-Outcomes**: 0 / 0
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 0 / 0 / 0
- **Quarantäne geschlossener Trades**: 0
- **Externes Archiv**: legacy: 8899

## Daten

- **Quellenstatus**: HEALTHY: 33, DEGRADED: 2, UNVALIDATED: 9
- **Stand Quellenstatus**: 2026-10-10T17:27:40+00:00
- **Externe Daten**: keine
- **Quote-Archiv heute / Median 14 T**: N/A / N/A
- **Quote-Archiv-Verzeichnis vorhanden**: nein

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Lebenszyklus | OUTCOMES_OVERDUE:candidates | 1 Kandidat(en) mit überfälligem 5-Tage-Real-Option-Mark (> 2 Tage nach Fälligkeit). | 2026-10-11 |
| NEW | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_5d 41 von 153 offen, ältester 6 Tage; real_strat_ret_5d 127 von 153 offen, ältester 8 Tage (Schwelle > 3 Tage). | 2026-10-11 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.source_health.DEGRADED | 2 | N/A | N/A | N/A |
| data.source_health.HEALTHY | 33 | N/A | N/A | N/A |
| data.source_health.UNVALIDATED | 9 | N/A | N/A | N/A |
| data.source_health.n_bad | 2 | N/A | N/A | N/A |
| forward.champion.mean_net | -0.6126 | N/A | N/A | N/A |
| forward.champion.n | 3 | N/A | N/A | N/A |
| forward.champion.n_independent_days | 3 | N/A | N/A | N/A |
| forward.champion.sum_net | -1.8378 | N/A | N/A | N/A |
| integrity.issues | 0 | N/A | N/A | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 1 | N/A | N/A | N/A |
| lifecycle.candidates.EXIT_PENDING | 160 | N/A | N/A | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 112 | N/A | N/A | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | N/A | N/A | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | N/A | N/A | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.EXIT_PENDING | 4 | N/A | N/A | N/A |
| lifecycle.champion.state.LEARNED | 0 | N/A | N/A | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | N/A | N/A | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 117 | N/A | N/A | N/A |
| lifecycle.champion.state.SIGNAL | 0 | N/A | N/A | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | 6 | N/A | N/A | N/A |
| marks.real_opt_ret_5d.pending | 41 | N/A | N/A | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | 8 | N/A | N/A | N/A |
| marks.real_strat_ret_5d.pending | 127 | N/A | N/A | N/A |
| promotion.active_influence | 0 | N/A | N/A | N/A |
| promotion.unjustified_influence | 0 | N/A | N/A | N/A |
| quote.complete.30d.coverage | 0.6 | N/A | N/A | N/A |
| quote.complete.30d.denominator | 5 | N/A | N/A | N/A |
| quote.complete.30d.evaluable | 3 | N/A | N/A | N/A |
| quote.complete.7d.coverage | 1 | N/A | N/A | N/A |
| quote.complete.7d.denominator | 1 | N/A | N/A | N/A |
| quote.complete.7d.evaluable | 1 | N/A | N/A | N/A |
| quote.entry.30d.coverage | 0 | N/A | N/A | N/A |
| quote.entry.30d.denominator | 5 | N/A | N/A | N/A |
| quote.entry.30d.legacy | 5 | N/A | N/A | N/A |
| quote.entry.30d.observed | 0 | N/A | N/A | N/A |
| quote.entry.7d.coverage | 0 | N/A | N/A | N/A |
| quote.entry.7d.denominator | 2 | N/A | N/A | N/A |
| quote.entry.7d.legacy | 2 | N/A | N/A | N/A |
| quote.entry.7d.observed | 0 | N/A | N/A | N/A |
| quote.exit_active.coverage | 0 | N/A | N/A | N/A |
| quote.exit_active.denominator | 4 | N/A | N/A | N/A |
| quote.exit_active.mark_today | 0 | N/A | N/A | N/A |
| quote.exit_closed.30d.coverage | 0.6 | N/A | N/A | N/A |
| quote.exit_closed.30d.denominator | 5 | N/A | N/A | N/A |
| quote.exit_closed.30d.observed | 3 | N/A | N/A | N/A |
| quote.exit_closed.7d.coverage | 1 | N/A | N/A | N/A |
| quote.exit_closed.7d.denominator | 1 | N/A | N/A | N/A |
| quote.exit_closed.7d.observed | 1 | N/A | N/A | N/A |
| run.scanner.analyzed | 96 | N/A | N/A | N/A |
| run.scanner.champion_candidates | 21 | N/A | N/A | N/A |
| run.scanner.contracts_designed | 15 | N/A | N/A | N/A |
| run.scanner.trades | 0 | N/A | N/A | N/A |
| run.scanner.universe | 372 | N/A | N/A | N/A |

3 Kennzahl(en) ohne Wert (N/A, nie 0): data.quote_archive_median_14d, data.quote_archive_today, training.n_labels
