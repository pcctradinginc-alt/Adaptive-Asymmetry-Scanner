# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-11-02 · erzeugt 2026-11-02T16:23:00+00:00 · Schema ops-status-v1  
Warnungen: 0 neu, 0 geändert, 1 laufend, 1 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-11-02T15:58:48Z | 9.7 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | RUNNING | N/A | 2026-11-02T16:06:52Z | 2.6 |
| source_health | OK | success | 2026-11-02T16:05:13Z | 3.4 |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-11-02 (von heute: ja)
- **Universum / analysiert**: 518 / 1
- **Champion-Kandidaten (Final-MC)**: 0
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 1 (Anteil 1)
- **Session**: regular
- **Stop-Grund**: Alle Signale bearish (Bearish-Trades deaktiviert).

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Entry 30d | 60.0 % | 3/5 | Altbestand vor Archiv: 2; ohne Altbestand 100.0 % (3/3) |
| Exit geschlossen 7d | N/A | 0/0 | keine |
| Exit geschlossen 30d | 100.0 % | 5/5 | EXIT_OBSERVED_NOT_ARCHIVED: 1, EXIT_OBSERVED_ARCHIVED: 4 |
| Vollständig bewertbar 7d | N/A | 0/0 | keine |
| Vollständig bewertbar 30d | 100.0 % | 5/5 | keine |
| Exit aktiv (Mark heute) | 100.0 % | 3/3 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 273 | 272 | 1 | 50 | 0 | 13 |
| real_opt_ret_5d | 657 | 656 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 273 | 272 | 1 | 50 | 0 | 13 |
| real_strat_ret_5d | 657 | 656 | 0 | 50 | 1 | N/A |

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
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 31/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 1029, CONTRACT_SELECTED: 663, ENTRY_OBSERVED: 663, OUTCOME_FINAL: 656, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 6

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
| CANDIDATE_20d\|ALL | 477 | -0.011 | 0.5199 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 3 | -0.1111 | 0 | NEED_MORE_DATA |
| CANDIDATE_20d\|LATE_STAGE | 133 | -0.0264 | 0.5113 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 1021 | 0 | 0.5269 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 235 | -0.0009 | 0.5149 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 272 | -0.0996 | 0.3787 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 3 | -0.4607 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 80 | -0.1799 | 0.325 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 592 | -0.0979 | 0.3091 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 138 | -0.0936 | 0.3406 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 272 | -0.1476 | 0.3456 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 3 | -0.5565 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 80 | -0.2827 | 0.2375 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 542 | -0.1349 | 0.286 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 122 | -0.1887 | 0.2377 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 7 | -0.3322 | 0.1429 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 133 / 8 (min_labels: 200, min_dates: 20)
- **Versionen / aktiv / Hold**: 0 / N/A / N/A
- **Aktives Artefakt ok**: N/A
- **Manifest-Verstöße**: keine

## Promotion

- **Max. automatischer Einfluss**: ABSTENTION_ONLY
- **Kill-Switch / Promotion pausiert**: nein / ja (MODEL: EXCESSIVE MODEL DISAGREEMENT)
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 3100 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 33521, legacy: 7907

## Daten

- **Quellenstatus**: HEALTHY: 42, BROKEN: 2, DEGRADED: 2, UNVALIDATED: 8, STALE: 1
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 1, DEFERRED: 5, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-11-02T16:06:43+00:00
- **Externe Daten**: SCHEMA_CHANGED: 1, PASS: 34, DEFERRED: 13, FAIL: 1
- **Quote-Archiv heute / Median 14 T**: 206 / 207.5
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-11-02 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 133 · Tage 8 | 200 · Tage 20 | 4.36 | 2026-11-28 |
| RL-Freischaltung (closed trades) | 13 | 30 | 0.167 | 2027-02-12 |
| Champion-Evidenz | 7 · Tage 6 | 30 · Tage 20 | 0.167 | 2027-03-20 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 102 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-10-30T14:23:00Z (74 h)
- **Läufe / fehlgeschlagen**: 72 / 0
- **Scanner-Start heute (New York)**: 2026-11-02T11:07-05:00

Keine fehlgeschlagenen Läufe.

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| RESOLVED | WARNING | Daten | DATA_DEGRADATION:source_health | Nicht mehr aktiv – Bedingung behoben. | 2026-10-29 |
| ONGOING | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_20d 1 von 273 offen, ältester 13 Tage; real_strat_ret_20d 1 von 273 offen, ältester 13 Tage (Schwelle > 3 Tage). | 2026-10-26 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | N/A |
| data.quote_archive_median_14d | 207.5 | 220 (-12.5) | 200 (+7.5) | N/A |
| data.quote_archive_today | 206 | 79 (+127) | 445 (-239) | N/A |
| data.source_health.BROKEN | 2 | 1 (+1) | N/A | N/A |
| data.source_health.DEGRADED | 2 | 3 (-1) | N/A | N/A |
| data.source_health.HEALTHY | 42 | 42 (0) | 46 (-4) | N/A |
| data.source_health.STALE | 1 | 1 (0) | 1 (0) | N/A |
| data.source_health.UNVALIDATED | 8 | 8 (0) | 8 (0) | N/A |
| data.source_health.n_bad | 5 | 5 (0) | 1 (+4) | N/A |
| forward.champion.mean_net | -0.3322 | -0.3322 (0) | -0.3322 (0) | N/A |
| forward.champion.n | 7 | 7 (0) | 7 (0) | N/A |
| forward.champion.n_independent_days | 6 | 6 (0) | 6 (0) | N/A |
| forward.champion.sum_net | -2.3256 | -2.3256 (0) | -2.3256 (0) | N/A |
| integrity.issues | 0 | 0 (0) | 0 (0) | N/A |
| learning.champion_evidence.have | 7 | 7 (0) | 7 (0) | N/A |
| learning.meta_label.have | 133 | 133 (0) | 47 (+86) | N/A |
| learning.rl_arming.have | 13 | 13 (0) | 13 (0) | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.candidates.EXIT_PENDING | 6 | 69 (-63) | 115 (-109) | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 656 | 592 (+64) | 512 (+144) | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.EXIT_PENDING | 3 | 3 (0) | 3 (0) | N/A |
| lifecycle.champion.state.LEARNED | 4 | 4 (0) | 4 (0) | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 0 (0) | N/A |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | N/A |
| quote.complete.30d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.complete.30d.denominator | 5 | 6 (-1) | 7 (-2) | N/A |
| quote.complete.30d.evaluable | 5 | 6 (-1) | 7 (-2) | N/A |
| quote.complete.7d.coverage | N/A | N/A | 1 (N/A) | N/A |
| quote.complete.7d.denominator | 0 | 0 (0) | 1 (-1) | N/A |
| quote.complete.7d.evaluable | 0 | 0 (0) | 1 (-1) | N/A |
| quote.entry.30d.coverage | 0.6 | 0.6 (0) | 0.6 (0) | N/A |
| quote.entry.30d.denominator | 5 | 5 (0) | 5 (0) | N/A |
| quote.entry.30d.legacy | 2 | 2 (0) | 2 (0) | N/A |
| quote.entry.30d.observed | 3 | 3 (0) | 3 (0) | N/A |
| quote.entry.7d.denominator | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.observed | 0 | 0 (0) | 0 (0) | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_active.denominator | 3 | 3 (0) | 3 (0) | N/A |
| quote.exit_active.mark_today | 3 | 3 (0) | 3 (0) | N/A |
| quote.exit_closed.30d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_closed.30d.denominator | 5 | 6 (-1) | 7 (-2) | N/A |
| quote.exit_closed.30d.observed | 5 | 6 (-1) | 7 (-2) | N/A |
| quote.exit_closed.7d.coverage | N/A | N/A | 1 (N/A) | N/A |
| quote.exit_closed.7d.denominator | 0 | 0 (0) | 1 (-1) | N/A |
| quote.exit_closed.7d.observed | 0 | 0 (0) | 1 (-1) | N/A |
| run.scanner.analyzed | 1 | 2 (-1) | 35 (-34) | N/A |
| run.scanner.champion_candidates | 0 | 0 (0) | 4 (-4) | N/A |
| run.scanner.direction_conflict_share | 1 | 0.5 (+0.5) | 0.0286 (+0.9714) | N/A |
| run.scanner.direction_conflicts | 1 | 1 (0) | 1 (0) | N/A |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | N/A |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | N/A |
| training.n_labels | 133 | 112 (+21) | 47 (+86) | N/A |
| workflows.failed_runs | 0 | 0 (0) | 0 (0) | N/A |

4 Kennzahl(en) ohne Wert (N/A, nie 0): marks.real_opt_ret_5d.oldest_pending_days, marks.real_strat_ret_5d.oldest_pending_days, quote.entry.7d.coverage, run.scanner.contracts_designed
