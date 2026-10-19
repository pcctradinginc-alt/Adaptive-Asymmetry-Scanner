# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-19 · erzeugt 2026-10-19T19:22:52+00:00 · Schema ops-status-v1  
Warnungen: 0 neu, 0 geändert, 2 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | RUNNING | N/A | 2026-10-19T13:21:18Z | 7.1 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | PENDING | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-10-19 (von heute: ja)
- **Universum / analysiert**: 518 / 56
- **Champion-Kandidaten (Final-MC)**: 9
- **Kontrakte entworfen**: 2
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 1 (Anteil 0.0179)
- **Session**: regular
- **Stop-Grund**: Alle Options-Kontrakte scheitern am ROI-Gate.

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | 100.0 % | 3/3 | Altbestand vor Archiv: 0; ohne Altbestand 100.0 % (3/3) |
| Entry 30d | 37.5 % | 3/8 | Altbestand vor Archiv: 5; ohne Altbestand 100.0 % (3/3) |
| Exit geschlossen 7d | 100.0 % | 3/3 | EXIT_OBSERVED_ARCHIVED: 3 |
| Exit geschlossen 30d | 75.0 % | 6/8 | EXIT_NOT_REALIZABLE: 2, EXIT_OBSERVED_NOT_ARCHIVED: 3, EXIT_OBSERVED_ARCHIVED: 3 |
| Vollständig bewertbar 7d | 100.0 % | 3/3 | keine |
| Vollständig bewertbar 30d | 75.0 % | 6/8 | EXIT_PRICE_UNKNOWN: 2 |
| Exit aktiv (Mark heute) | 100.0 % | 4/4 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 50 | 50 | 0 | 31 | 0 | N/A |
| real_opt_ret_5d | 313 | 312 | 1 | 50 | 0 | 14 |
| real_strat_ret_20d | 50 | 50 | 0 | 31 | 0 | N/A |
| real_strat_ret_5d | 313 | 312 | 1 | 50 | 0 | 14 |

## Lebenszyklus

| Champion-Zustand | Anzahl |
|---|---|
| SIGNAL | 0 |
| CONTRACT_SELECTED | 0 |
| ENTRY_OBSERVED | 0 |
| EXIT_PENDING | 4 |
| OUTCOME_FINAL | 110 |
| LEARNING_ELIGIBLE | 3 |
| LEARNED | 3 |

- **Probleme (Champion)**: ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE: 117, LEGACY_NO_STORED_EVALUATION: 113, OUTCOME_UNUSABLE: 107, OUTCOME_DATA_BAD: 3, LEGACY_LEARNED_UNTRACKED: 3
- **Exactly-once ok**: ja
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 6/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 759, CONTRACT_SELECTED: 433, ENTRY_OBSERVED: 433, OUTCOME_FINAL: 312, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, EXIT_OVERDUE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 326, EXIT_PENDING: 120

## Forward-Performance

- **Champion realisiert (nur beobachtet): n**: 6
- **mean_net**: -0.3095
- **sum_net**: -1.8569
- **Trefferquote**: 0.1667
- **schlechtester**: -0.7443
- **max. Drawdown (kum.)**: -2.3653
- **nicht bewertbar**: UNUSABLE: 107, DATA_BAD: 3

Verdikt NEED_MORE_DATA: keine Alpha-Behauptung (n=6, unabhängige Tage=5; Mindestwerte n≥30, Tage≥20).

| Challenger (Shadow) | vermiedene Verluste | entgangene Gewinne | Δ vs. Champion (Σ) | greift/bewertbar | Population | Verdikt |
|---|---|---|---|---|---|---|
| EA001_EXPECTATION_GAP@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA002_CONFIRMATION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA003_ACCELERATION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA004_WAIT@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA005_CONTEXT_FILTER@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA006_ABSTENTION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| EA007_EXPRESSION@v1 | N/A | N/A | N/A | 0/0 | EA_NEWS_CANDIDATE | NEED_MORE_DATA |
| PROM-ABST-001@v1 | N/A | N/A | N/A | 0/2 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-001@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-002@v1 | 0.5079 | 0 | 0.5079 | 1/2 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-002@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-003@v1 | N/A | N/A | N/A | 0/2 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-003@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-004@v1 | N/A | N/A | N/A | 0/2 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-004@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-ABST-005@v1 | N/A | N/A | N/A | 0/2 | CHAMPION_TRADE | NEED_MORE_DATA |
| PROM-ABST-005@v2 | N/A | N/A | N/A | 0/0 | FINAL_MC_SURVIVOR | NEED_MORE_DATA |
| PROM-META-001@v1 | N/A | N/A | N/A | 0/0 | CHAMPION_TRADE | NEED_MORE_DATA |
| UNIV-V2-SEG-MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-MID@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-SMALL@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |
| UNIV-V2-SEG-ULTRA_MICRO@v1 | N/A | N/A | N/A | 0/0 | V2_CANDIDATE | NEED_MORE_DATA |

22 von 22 Challenger-Verdikten NEED_MORE_DATA – dort keine Alpha-Behauptung.

| Scorecard | n | mean_net | Trefferquote | Verdikt |
|---|---|---|---|---|
| CANDIDATE_20d\|ALL | 18 | -0.0321 | 0.3889 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_20d\|LATE_STAGE | 8 | -0.0478 | 0.375 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 477 | -0.0024 | 0.478 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 3 | -0.0076 | 0.3333 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 133 | -0.0005 | 0.5338 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 272 | -0.126 | 0.2537 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 3 | -0.1617 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 80 | -0.1065 | 0.3625 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 222 | -0.1758 | 0.2252 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 3 | -0.2853 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 64 | -0.2404 | 0.1875 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 6 | -0.3095 | 0.1667 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 0 / 0 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (DRIFT MODERATE: Promotion pausiert)
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
| PROM-ABST-001@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 2/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-001@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-002@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 2/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 1/15; Treffer-Tage 1/8; nur 1 Marktregime beobachtet; Sektor Technology dominiert (100%) – n |
| PROM-ABST-002@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-003@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 2/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-003@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-004@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 2/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
| PROM-ABST-004@v2 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 0/150; unabhängige Tage 0/30; Kalenderspanne 0/90 T; Regel-Treffer 0/30; Treffer-Tage 0/10; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/80; Treffer-Clust |
| PROM-ABST-005@v1 | PROSPECTIVE_CHALLENGER | NONE | KEEP_SHADOW | n 2/60; unabhängige Tage 2/20; Kalenderspanne 8/90 T; Regel-Treffer 0/15; Treffer-Tage 0/8; nur 1 Marktregime beobachtet |
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 941 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 29590, legacy: 103843

## Daten

- **Quellenstatus**: HEALTHY: 47, UNVALIDATED: 8
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 1, DEFERRED: 5, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-10-19T18:50:25+00:00
- **Externe Daten**: PASS: 36, DEFERRED: 13
- **Quote-Archiv heute / Median 14 T**: 261 / 153.5
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-10-19 (Quelle Produktion: nasdaq_official)

## Timing-Policy

- **Policy neuer Entscheidungen**: timing-v2
- **Bestand je Policy**: active: timing-v1: 2, timing-v2: 2, exit_pending: keine, closed: timing-v1: 115, timing-v2: 1, unusable: keine, counterfactual_exit_pending: keine, counterfactual_unusable: keine
- **timing-v2 Entry offener Trades**: FILLED: 2
- **timing-v2 Exit ausstehend (ältester, Tage)**: 0 (N/A)
- **Gründe ausstehend**: keine
- **timing-v2 ausgeführt / UNUSABLE**: 1 / 0
- **Gründe UNUSABLE**: keine
- **Abweichung Ausführung vs. Entscheid**: n: 1, mean: 0.697, definition: Outcome zur zulässigen Ausführung minus Return zum Entscheid

## Lern-Prognose

| Ziel | Stand | benötigt | Zulauf/Tag | voraussichtlich ab |
|---|---|---|---|---|
| Meta-Label-Training | 8 · Tage 1 | 200 · Tage 20 | 9.07 | 2026-11-24 |
| RL-Freischaltung (closed trades) | 12 | 30 | 0.267 | 2026-12-26 |
| Champion-Evidenz | 6 · Tage 5 | 30 · Tage 20 | 0.2 | 2027-02-16 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 166 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-10-16T17:22:52Z (74 h)
- **Läufe / fehlgeschlagen**: 47 / 0
- **Scanner-Start heute (New York)**: 2026-10-19T14:51-04:00

Keine fehlgeschlagenen Läufe.

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| ONGOING | WARNING | Lebenszyklus | OUTCOMES_OVERDUE:candidates | 1 Kandidat(en) mit überfälligem 5-Tage-Real-Option-Mark (> 2 Tage nach Fälligkeit). | 2026-10-14 |
| ONGOING | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_5d 1 von 313 offen, ältester 14 Tage; real_strat_ret_5d 1 von 313 offen, ältester 14 Tage (Schwelle > 3 Tage). | 2026-10-14 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | N/A | N/A | N/A |
| data.quote_archive_median_14d | 153.5 | 116 (+37.5) | N/A | N/A |
| data.quote_archive_today | 261 | 77 (+184) | N/A | N/A |
| data.source_health.BROKEN | N/A | 3 (N/A) | N/A | N/A |
| data.source_health.DEGRADED | N/A | 1 (N/A) | N/A | N/A |
| data.source_health.HEALTHY | 47 | 42 (+5) | N/A | N/A |
| data.source_health.STALE | N/A | 1 (N/A) | N/A | N/A |
| data.source_health.UNVALIDATED | 8 | 8 (0) | N/A | N/A |
| data.source_health.n_bad | 0 | 5 (-5) | N/A | N/A |
| forward.champion.mean_net | -0.3095 | -0.3095 (0) | N/A | N/A |
| forward.champion.n | 6 | 6 (0) | N/A | N/A |
| forward.champion.n_independent_days | 5 | 5 (0) | N/A | N/A |
| forward.champion.sum_net | -1.8569 | -1.8569 (0) | N/A | N/A |
| integrity.issues | 0 | 0 (0) | N/A | N/A |
| learning.champion_evidence.have | 6 | 6 (0) | N/A | N/A |
| learning.meta_label.have | 8 | 0 (+8) | N/A | N/A |
| learning.rl_arming.have | 12 | 12 (0) | N/A | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 1 | 1 (0) | N/A | N/A |
| lifecycle.candidates.EXIT_PENDING | 120 | 120 (0) | N/A | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 312 | 272 (+40) | N/A | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | N/A | N/A |
| lifecycle.champion.state.EXIT_PENDING | 4 | 4 (0) | N/A | N/A |
| lifecycle.champion.state.LEARNED | 3 | 3 (0) | N/A | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | N/A | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | N/A | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | N/A | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | 14 | 11 (+3) | N/A | N/A |
| marks.real_opt_ret_5d.pending | 1 | 1 (0) | N/A | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | 14 | 11 (+3) | N/A | N/A |
| marks.real_strat_ret_5d.pending | 1 | 1 (0) | N/A | N/A |
| promotion.active_influence | 0 | 0 (0) | N/A | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | N/A | N/A |
| quote.complete.30d.coverage | 0.75 | 0.75 (0) | N/A | N/A |
| quote.complete.30d.denominator | 8 | 8 (0) | N/A | N/A |
| quote.complete.30d.evaluable | 6 | 6 (0) | N/A | N/A |
| quote.complete.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.complete.7d.denominator | 3 | 3 (0) | N/A | N/A |
| quote.complete.7d.evaluable | 3 | 3 (0) | N/A | N/A |
| quote.entry.30d.coverage | 0.375 | 0.375 (0) | N/A | N/A |
| quote.entry.30d.denominator | 8 | 8 (0) | N/A | N/A |
| quote.entry.30d.legacy | 5 | 5 (0) | N/A | N/A |
| quote.entry.30d.observed | 3 | 3 (0) | N/A | N/A |
| quote.entry.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.entry.7d.denominator | 3 | 3 (0) | N/A | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | N/A | N/A |
| quote.entry.7d.observed | 3 | 3 (0) | N/A | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | N/A | N/A |
| quote.exit_active.denominator | 4 | 4 (0) | N/A | N/A |
| quote.exit_active.mark_today | 4 | 4 (0) | N/A | N/A |
| quote.exit_closed.30d.coverage | 0.75 | 0.75 (0) | N/A | N/A |
| quote.exit_closed.30d.denominator | 8 | 8 (0) | N/A | N/A |
| quote.exit_closed.30d.observed | 6 | 6 (0) | N/A | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | N/A | N/A |
| quote.exit_closed.7d.denominator | 3 | 3 (0) | N/A | N/A |
| quote.exit_closed.7d.observed | 3 | 3 (0) | N/A | N/A |
| run.scanner.analyzed | 56 | 77 (-21) | N/A | N/A |
| run.scanner.champion_candidates | 9 | 7 (+2) | N/A | N/A |
| run.scanner.contracts_designed | 2 | 3 (-1) | N/A | N/A |
| run.scanner.direction_conflict_share | 0.0179 | 0.0649 (-0.047) | N/A | N/A |
| run.scanner.direction_conflicts | 1 | 5 (-4) | N/A | N/A |
| run.scanner.trades | 0 | 1 (-1) | N/A | N/A |
| run.scanner.universe | 518 | 514 (+4) | N/A | N/A |
| training.n_labels | 0 | 0 (0) | N/A | N/A |
| workflows.failed_runs | 0 | 0 (0) | N/A | N/A |
