# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-21 · erzeugt 2026-10-21T18:05:59+00:00 · Schema ops-status-v1  
Warnungen: 1 neu, 0 geändert, 0 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-10-21T11:52:47Z | 5.6 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | PENDING | N/A | N/A | N/A |
| source_health | RECOVERY_TRIGGERED | N/A | N/A | N/A |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-10-21 (von heute: ja)
- **Universum / analysiert**: 518 / 50
- **Champion-Kandidaten (Final-MC)**: 17
- **Kontrakte entworfen**: 3
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 2 (Anteil 0.04)
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
| real_opt_ret_20d | 73 | 72 | 1 | 50 | 0 | 1 |
| real_opt_ret_5d | 393 | 392 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 73 | 72 | 1 | 50 | 0 | 1 |
| real_strat_ret_5d | 393 | 392 | 0 | 50 | 1 | N/A |

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
- **Kandidaten (Real-Option)**: SIGNAL: 856, CONTRACT_SELECTED: 513, ENTRY_OBSERVED: 513, OUTCOME_FINAL: 392, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 343, EXIT_PENDING: 120

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
| CANDIDATE_20d\|ALL | 72 | -0.0304 | 0.4028 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_20d\|LATE_STAGE | 25 | -0.0592 | 0.36 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 625 | -0.0003 | 0.5104 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 6 | -0.0046 | 0.3333 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 158 | 0.0001 | 0.5253 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 72 | -0.2148 | 0.2917 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 25 | -0.2946 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 352 | -0.105 | 0.2756 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 4 | -0.1323 | 0 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 93 | -0.0987 | 0.3333 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 72 | -0.226 | 0.2778 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 25 | -0.305 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 302 | -0.1436 | 0.2517 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 4 | -0.3055 | 0 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 77 | -0.2242 | 0.1818 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 6 | -0.3095 | 0.1667 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 19 / 2 (min_labels: 200, min_dates: 20)
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 1336 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 30379, legacy: 79102

## Daten

- **Quellenstatus**: HEALTHY: 45, BROKEN: 1, UNVALIDATED: 8, DEGRADED: 1
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 1, DEFERRED: 5, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-10-21T17:37:41+00:00
- **Externe Daten**: PASS: 35, FAIL: 1, DEFERRED: 13
- **Quote-Archiv heute / Median 14 T**: 153 / 172
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-10-21 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 25 · Tage 3 | 200 · Tage 20 | 8.21 | 2026-11-22 |
| RL-Freischaltung (closed trades) | 12 | 30 | 0.267 | 2026-12-28 |
| Champion-Evidenz | 6 · Tage 5 | 30 · Tage 20 | 0.2 | 2027-02-18 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 173 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-10-20T16:05:59Z (26 h)
- **Läufe / fehlgeschlagen**: 34 / 0
- **Scanner-Start heute (New York)**: 2026-10-21T13:38-04:00

Keine fehlgeschlagenen Läufe.

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Daten | DATA_DEGRADATION:source_health | Quellen mit Status FAILED/BROKEN/STALE/DEGRADED: 2 (Vorlauf 0, +2). | 2026-10-21 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | N/A | N/A |
| data.quote_archive_median_14d | 172 | 191 (-19) | 164 (+8) | N/A |
| data.quote_archive_today | 153 | 206 (-53) | 164 (-11) | N/A |
| data.source_health.BROKEN | 1 | N/A | 1 (0) | N/A |
| data.source_health.DEGRADED | 1 | N/A | 7 (-6) | N/A |
| data.source_health.HEALTHY | 45 | 47 (-2) | 28 (+17) | N/A |
| data.source_health.UNVALIDATED | 8 | 8 (0) | 8 (0) | N/A |
| data.source_health.n_bad | 2 | 0 (+2) | 8 (-6) | N/A |
| forward.champion.mean_net | -0.3095 | -0.3095 (0) | -0.6168 (+0.3073) | N/A |
| forward.champion.n | 6 | 6 (0) | 3 (+3) | N/A |
| forward.champion.n_independent_days | 5 | 5 (0) | 3 (+2) | N/A |
| forward.champion.sum_net | -1.8569 | -1.8569 (0) | -1.8503 (-0.0065) | N/A |
| integrity.issues | 0 | 0 (0) | 1 (-1) | N/A |
| learning.champion_evidence.have | 6 | 6 (0) | N/A | N/A |
| learning.meta_label.have | 25 | 19 (+6) | N/A | N/A |
| learning.rl_arming.have | 12 | 12 (0) | N/A | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 1 (-1) | N/A |
| lifecycle.candidates.EXIT_PENDING | 120 | 120 (0) | 40 (+80) | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 392 | 352 (+40) | 272 (+120) | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.EXIT_PENDING | 4 | 4 (0) | 5 (-1) | N/A |
| lifecycle.champion.state.LEARNED | 3 | 3 (0) | 0 (+3) | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | N/A | N/A | 9 (N/A) | N/A |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 1 (-1) | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | N/A | N/A | 9 (N/A) | N/A |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 1 (-1) | N/A |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | N/A |
| quote.complete.30d.coverage | 0.75 | 0.75 (0) | 0.6 (+0.15) | N/A |
| quote.complete.30d.denominator | 8 | 8 (0) | 5 (+3) | N/A |
| quote.complete.30d.evaluable | 6 | 6 (0) | 3 (+3) | N/A |
| quote.complete.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.complete.7d.denominator | 3 | 3 (0) | 1 (+2) | N/A |
| quote.complete.7d.evaluable | 3 | 3 (0) | 1 (+2) | N/A |
| quote.entry.30d.coverage | 0.375 | 0.375 (0) | 0.1667 (+0.2083) | N/A |
| quote.entry.30d.denominator | 8 | 8 (0) | 6 (+2) | N/A |
| quote.entry.30d.legacy | 5 | 5 (0) | 5 (0) | N/A |
| quote.entry.30d.observed | 3 | 3 (0) | 1 (+2) | N/A |
| quote.entry.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.entry.7d.denominator | 3 | 3 (0) | 1 (+2) | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.observed | 3 | 3 (0) | 1 (+2) | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_active.denominator | 4 | 4 (0) | 5 (-1) | N/A |
| quote.exit_active.mark_today | 4 | 4 (0) | 5 (-1) | N/A |
| quote.exit_closed.30d.coverage | 0.75 | 0.75 (0) | 0.6 (+0.15) | N/A |
| quote.exit_closed.30d.denominator | 8 | 8 (0) | 5 (+3) | N/A |
| quote.exit_closed.30d.observed | 6 | 6 (0) | 3 (+3) | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_closed.7d.denominator | 3 | 3 (0) | 1 (+2) | N/A |
| quote.exit_closed.7d.observed | 3 | 3 (0) | 1 (+2) | N/A |
| run.scanner.analyzed | 50 | 47 (+3) | 66 (-16) | N/A |
| run.scanner.champion_candidates | 17 | 7 (+10) | 11 (+6) | N/A |
| run.scanner.contracts_designed | 3 | N/A | 5 (-2) | N/A |
| run.scanner.direction_conflict_share | 0.04 | 0.0638 (-0.0238) | N/A | N/A |
| run.scanner.direction_conflicts | 2 | 3 (-1) | N/A | N/A |
| run.scanner.trades | 0 | 0 (0) | 1 (-1) | N/A |
| run.scanner.universe | 518 | 518 (0) | 514 (+4) | N/A |
| training.n_labels | 19 | 8 (+11) | 0 (+19) | N/A |
| workflows.failed_runs | 0 | 0 (0) | N/A | N/A |
