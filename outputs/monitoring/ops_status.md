# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-22 · erzeugt 2026-10-22T18:09:34+00:00 · Schema ops-status-v1  
Warnungen: 0 neu, 0 geändert, 0 laufend, 1 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-10-22T11:56:29Z | 5.7 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | OK | success | 2026-10-22T15:53:03Z | 3.2 |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-10-22 (von heute: ja)
- **Universum / analysiert**: 518 / 62
- **Champion-Kandidaten (Final-MC)**: 9
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 3 (Anteil 0.0484)
- **Session**: regular
- **Stop-Grund**: Alle Options-Kontrakte scheitern am ROI-Gate.

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | 100.0 % | 2/2 | Altbestand vor Archiv: 0; ohne Altbestand 100.0 % (2/2) |
| Entry 30d | 42.9 % | 3/7 | Altbestand vor Archiv: 4; ohne Altbestand 100.0 % (3/3) |
| Exit geschlossen 7d | 100.0 % | 4/4 | EXIT_OBSERVED_ARCHIVED: 4 |
| Exit geschlossen 30d | 87.5 % | 7/8 | EXIT_NOT_REALIZABLE: 1, EXIT_OBSERVED_NOT_ARCHIVED: 3, EXIT_OBSERVED_ARCHIVED: 4 |
| Vollständig bewertbar 7d | 100.0 % | 4/4 | keine |
| Vollständig bewertbar 30d | 87.5 % | 7/8 | EXIT_PRICE_UNKNOWN: 1 |
| Exit aktiv (Mark heute) | 100.0 % | 3/3 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 113 | 112 | 1 | 50 | 0 | 2 |
| real_opt_ret_5d | 393 | 392 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 113 | 112 | 1 | 50 | 0 | 2 |
| real_strat_ret_5d | 393 | 392 | 0 | 50 | 1 | N/A |

## Lebenszyklus

| Champion-Zustand | Anzahl |
|---|---|
| SIGNAL | 0 |
| CONTRACT_SELECTED | 0 |
| ENTRY_OBSERVED | 0 |
| EXIT_PENDING | 3 |
| OUTCOME_FINAL | 110 |
| LEARNING_ELIGIBLE | 3 |
| LEARNED | 4 |

- **Probleme (Champion)**: ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE: 117, LEGACY_NO_STORED_EVALUATION: 113, OUTCOME_UNUSABLE: 107, OUTCOME_DATA_BAD: 3, LEGACY_LEARNED_UNTRACKED: 3
- **Exactly-once ok**: ja
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 8/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 918, CONTRACT_SELECTED: 553, ENTRY_OBSERVED: 553, OUTCOME_FINAL: 392, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 365, EXIT_PENDING: 160

## Forward-Performance

- **Champion realisiert (nur beobachtet): n**: 7
- **mean_net**: -0.3322
- **sum_net**: -2.3256
- **Trefferquote**: 0.1429
- **schlechtester**: -0.7443
- **max. Drawdown (kum.)**: -2.3653
- **nicht bewertbar**: UNUSABLE: 107, DATA_BAD: 3

Verdikt NEED_MORE_DATA: keine Alpha-Behauptung (n=7, unabhängige Tage=6; Mindestwerte n≥30, Tage≥20).

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
| CANDIDATE_20d\|ALL | 72 | -0.0304 | 0.4028 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_20d\|LATE_STAGE | 25 | -0.0592 | 0.36 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 702 | 0.0005 | 0.5228 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 165 | 0.002 | 0.5455 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 72 | -0.2148 | 0.2917 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 25 | -0.2946 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 392 | -0.1016 | 0.2857 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 97 | -0.0891 | 0.3608 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 72 | -0.226 | 0.2778 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 25 | -0.305 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 342 | -0.1379 | 0.2602 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 81 | -0.2098 | 0.2099 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 6 | -0.3095 | 0.1667 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 25 / 3 (min_labels: 200, min_dates: 20)
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 1544 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 30629, legacy: 73447

## Daten

- **Quellenstatus**: HEALTHY: 47, UNVALIDATED: 8
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 1, DEFERRED: 5, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-10-22T17:35:40+00:00
- **Externe Daten**: PASS: 36, DEFERRED: 13
- **Quote-Archiv heute / Median 14 T**: 184 / 184
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-10-22 (Quelle Produktion: nasdaq_official)

## Timing-Policy

- **Policy neuer Entscheidungen**: timing-v2
- **Bestand je Policy**: active: timing-v1: 1, timing-v2: 2, exit_pending: keine, closed: timing-v1: 116, timing-v2: 1, unusable: keine, counterfactual_exit_pending: keine, counterfactual_unusable: keine
- **timing-v2 Entry offener Trades**: FILLED: 2
- **timing-v2 Exit ausstehend (ältester, Tage)**: 0 (N/A)
- **Gründe ausstehend**: keine
- **timing-v2 ausgeführt / UNUSABLE**: 1 / 0
- **Gründe UNUSABLE**: keine
- **Abweichung Ausführung vs. Entscheid**: n: 1, mean: 0.697, definition: Outcome zur zulässigen Ausführung minus Return zum Entscheid

## Lern-Prognose

| Ziel | Stand | benötigt | Zulauf/Tag | voraussichtlich ab |
|---|---|---|---|---|
| Meta-Label-Training | 25 · Tage 3 | 200 · Tage 20 | 6.79 | 2026-11-21 |
| RL-Freischaltung (closed trades) | 13 | 30 | 0.267 | 2026-12-25 |
| Champion-Evidenz | 7 · Tage 6 | 30 · Tage 20 | 0.233 | 2027-01-29 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 182 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-10-21T16:09:34Z (26 h)
- **Läufe / fehlgeschlagen**: 34 / 0
- **Scanner-Start heute (New York)**: 2026-10-22T13:36-04:00

Keine fehlgeschlagenen Läufe.

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| RESOLVED | WARNING | Daten | DATA_DEGRADATION:source_health | Nicht mehr aktiv – Bedingung behoben. | 2026-10-21 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | N/A | N/A |
| data.quote_archive_median_14d | 184 | 172 (+12) | 139.5 (+44.5) | N/A |
| data.quote_archive_today | 184 | 153 (+31) | 88 (+96) | N/A |
| data.source_health.BROKEN | N/A | 1 (N/A) | 3 (N/A) | N/A |
| data.source_health.DEGRADED | N/A | 1 (N/A) | 2 (N/A) | N/A |
| data.source_health.HEALTHY | 47 | 45 (+2) | 41 (+6) | N/A |
| data.source_health.STALE | N/A | N/A | 1 (N/A) | N/A |
| data.source_health.UNVALIDATED | 8 | 8 (0) | 8 (0) | N/A |
| data.source_health.n_bad | 0 | 2 (-2) | 6 (-6) | N/A |
| forward.champion.mean_net | -0.3322 | -0.3095 (-0.0227) | -0.5419 (+0.2097) | N/A |
| forward.champion.n | 7 | 6 (+1) | 4 (+3) | N/A |
| forward.champion.n_independent_days | 6 | 5 (+1) | 4 (+2) | N/A |
| forward.champion.sum_net | -2.3256 | -1.8569 (-0.4687) | -2.1677 (-0.1578) | N/A |
| integrity.issues | 0 | 0 (0) | 0 (0) | N/A |
| learning.champion_evidence.have | 7 | 6 (+1) | 4 (+3) | N/A |
| learning.meta_label.have | 25 | 25 (0) | 0 (+25) | N/A |
| learning.rl_arming.have | 13 | 12 (+1) | 10 (+3) | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 1 (-1) | N/A |
| lifecycle.candidates.EXIT_PENDING | 160 | 120 (+40) | 80 (+80) | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 392 | 392 (0) | 272 (+120) | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.EXIT_PENDING | 3 | 4 (-1) | 5 (-2) | N/A |
| lifecycle.champion.state.LEARNED | 4 | 3 (+1) | 1 (+3) | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | N/A | N/A | 10 (N/A) | N/A |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 1 (-1) | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | N/A | N/A | 10 (N/A) | N/A |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 1 (-1) | N/A |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | N/A |
| quote.complete.30d.coverage | 0.875 | 0.75 (+0.125) | 0.6667 (+0.2083) | N/A |
| quote.complete.30d.denominator | 8 | 8 (0) | 6 (+2) | N/A |
| quote.complete.30d.evaluable | 7 | 6 (+1) | 4 (+3) | N/A |
| quote.complete.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.complete.7d.denominator | 4 | 3 (+1) | 2 (+2) | N/A |
| quote.complete.7d.evaluable | 4 | 3 (+1) | 2 (+2) | N/A |
| quote.entry.30d.coverage | 0.4286 | 0.375 (+0.0536) | 0.2857 (+0.1429) | N/A |
| quote.entry.30d.denominator | 7 | 8 (-1) | 7 (0) | N/A |
| quote.entry.30d.legacy | 4 | 5 (-1) | 5 (-1) | N/A |
| quote.entry.30d.observed | 3 | 3 (0) | 2 (+1) | N/A |
| quote.entry.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.entry.7d.denominator | 2 | 3 (-1) | 2 (0) | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.observed | 2 | 3 (-1) | 2 (0) | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_active.denominator | 3 | 4 (-1) | 5 (-2) | N/A |
| quote.exit_active.mark_today | 3 | 4 (-1) | 5 (-2) | N/A |
| quote.exit_closed.30d.coverage | 0.875 | 0.75 (+0.125) | 0.6667 (+0.2083) | N/A |
| quote.exit_closed.30d.denominator | 8 | 8 (0) | 6 (+2) | N/A |
| quote.exit_closed.30d.observed | 7 | 6 (+1) | 4 (+3) | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_closed.7d.denominator | 4 | 3 (+1) | 2 (+2) | N/A |
| quote.exit_closed.7d.observed | 4 | 3 (+1) | 2 (+2) | N/A |
| run.scanner.analyzed | 62 | 50 (+12) | 82 (-20) | N/A |
| run.scanner.champion_candidates | 9 | 17 (-8) | 14 (-5) | N/A |
| run.scanner.contracts_designed | N/A | 3 (N/A) | 7 (N/A) | N/A |
| run.scanner.direction_conflict_share | 0.0484 | 0.04 (+0.0084) | 0.0488 (-0.0004) | N/A |
| run.scanner.direction_conflicts | 3 | 2 (+1) | 4 (-1) | N/A |
| run.scanner.trades | 0 | 0 (0) | 2 (-2) | N/A |
| run.scanner.universe | 518 | 518 (0) | 514 (+4) | N/A |
| training.n_labels | 25 | 19 (+6) | 0 (+25) | N/A |
| workflows.failed_runs | 0 | 0 (0) | 2 (-2) | N/A |
