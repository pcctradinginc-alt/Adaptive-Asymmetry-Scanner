# Scanner-Monitor

**READ-ONLY Monitoring – keine Handelsempfehlung**

Stand 2026-10-26 · erzeugt 2026-10-26T19:32:38+00:00 · Schema ops-status-v1  
Warnungen: 1 neu, 0 geändert, 0 laufend, 0 behoben; davon aktiv CRITICAL: 0  

## Lauf

| Workflow (Stand Watchdog) | Status | Ergebnis | Letzter Lauf | Verzug (h) |
|---|---|---|---|---|
| external_data | OK | success | 2026-10-26T13:07:26Z | 6.8 |
| feedback | PENDING | N/A | N/A | N/A |
| scanner | RUNNING | N/A | 2026-10-26T18:52:00Z | 5.4 |
| source_health | OK | success | 2026-10-26T18:50:30Z | 3.6 |

- **Dieser Lauf (Schritte)**: feedback=success, rth_marks=success, scanner=success
- **Tagesreport**: 2026-10-26 (von heute: ja)
- **Universum / analysiert**: 518 / 35
- **Champion-Kandidaten (Final-MC)**: 4
- **Kontrakte entworfen**: N/A
- **Trades**: 0
- **Haiku/Sonnet-Widerspruch**: 1 (Anteil 0.0286)
- **Session**: regular
- **Stop-Grund**: Alle Options-Kontrakte scheitern am ROI-Gate.

## Quote-Abdeckung

| Kennzahl | Abdeckung | Beobachtet/Nenner | Hinweis |
|---|---|---|---|
| Entry 7d | N/A | 0/0 | Altbestand vor Archiv: 0; ohne Altbestand N/A (0/0) |
| Entry 30d | 60.0 % | 3/5 | Altbestand vor Archiv: 2; ohne Altbestand 100.0 % (3/3) |
| Exit geschlossen 7d | 100.0 % | 1/1 | EXIT_OBSERVED_ARCHIVED: 1 |
| Exit geschlossen 30d | 100.0 % | 7/7 | EXIT_OBSERVED_NOT_ARCHIVED: 3, EXIT_OBSERVED_ARCHIVED: 4 |
| Vollständig bewertbar 7d | 100.0 % | 1/1 | keine |
| Vollständig bewertbar 30d | 100.0 % | 7/7 | keine |
| Exit aktiv (Mark heute) | 100.0 % | 3/3 | Handelstag: ja; ältester Mark: 0 Tage |

| Kandidaten-Mark | fällig | gemarkt | offen | spät | ohne Preis (Endzustand) | ältester offener (Tage) |
|---|---|---|---|---|---|---|
| real_opt_ret_20d | 153 | 152 | 1 | 50 | 0 | 6 |
| real_opt_ret_5d | 513 | 512 | 0 | 50 | 1 | N/A |
| real_strat_ret_20d | 153 | 152 | 1 | 50 | 0 | 6 |
| real_strat_ret_5d | 513 | 512 | 0 | 50 | 1 | N/A |

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
- **gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt**: 17/0/0/0
- **Altbestand**: Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne Schlüssel gelernt – keine Exactly-once-Verletzung
- **Kandidaten (Real-Option)**: SIGNAL: 994, CONTRACT_SELECTED: 628, ENTRY_OBSERVED: 628, OUTCOME_FINAL: 512, LATE_MARK: 0, PRICE_BASIS_CHANGED: 2, MARK_UNAVAILABLE: 1, EXCLUDED_BUDGET_RANDOM_EXCLUSION: 366, EXIT_PENDING: 115

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
| CANDIDATE_20d\|ALL | 143 | -0.0226 | 0.4406 | NEED_MORE_DATA |
| CANDIDATE_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_20d\|LATE_STAGE | 47 | -0.0584 | 0.383 | NEED_MORE_DATA |
| CANDIDATE_5d\|ALL | 805 | 0.0015 | 0.5354 | NEED_MORE_DATA |
| CANDIDATE_5d\|CHAMPION | 7 | -0.0031 | 0.4286 | NEED_MORE_DATA |
| CANDIDATE_5d\|LATE_STAGE | 181 | 0.0022 | 0.5525 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|ALL | 112 | -0.1652 | 0.3125 | NEED_MORE_DATA |
| CANDIDATE_OPTION_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_OPTION_20d\|LATE_STAGE | 38 | -0.2788 | 0.2105 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|ALL | 392 | -0.1016 | 0.2857 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|CHAMPION | 5 | -0.1037 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_OPTION_5d\|LATE_STAGE | 97 | -0.0891 | 0.3608 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|ALL | 112 | -0.1896 | 0.2946 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_20d\|CHAMPION | 0 | N/A | N/A | NO_DATA |
| CANDIDATE_STRATEGY_20d\|LATE_STAGE | 38 | -0.3167 | 0.1842 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|ALL | 342 | -0.1379 | 0.2602 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|CHAMPION | 5 | -0.2422 | 0.2 | NEED_MORE_DATA |
| CANDIDATE_STRATEGY_5d\|LATE_STAGE | 81 | -0.2098 | 0.2099 | NEED_MORE_DATA |
| CHAMPION_TRADE\|ALL | 7 | -0.3322 | 0.1429 | NEED_MORE_DATA |

## Training

- **Status**: NEED_MORE_DATA
- **Labels / Tage (benötigt)**: 47 / 4 (min_labels: 200, min_dates: 20)
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
- **Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise**: 2106 / 0 / 0
- **Quote-Archiv 30 T: Mehrfachbeobachtungen (unkritisch)**: 11
- **Quarantäne geschlossener Trades**: 7
- **Externes Archiv**: ok: 31204, legacy: 8671

## Daten

- **Quellenstatus**: HEALTHY: 46, STALE: 1, UNVALIDATED: 8
- **davon UNVALIDATED, weil bewusst nicht aktiv**: disabled: 1, DEFERRED: 5, REVIEW_REQUIRED: 2
- **Stand Quellenstatus**: 2026-10-26T18:51:46+00:00
- **Externe Daten**: PASS: 36, DEFERRED: 13
- **Quote-Archiv heute / Median 14 T**: 445 / 200
- **Quote-Archiv-Verzeichnis vorhanden**: ja
- **Nasdaq-100 offiziell vs. Produktionsliste**: 0 Abweichung(en) am 2026-10-26 (Quelle Produktion: nasdaq_official)

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
| Meta-Label-Training | 47 · Tage 4 | 200 · Tage 20 | 6.43 | 2026-11-21 |
| RL-Freischaltung (closed trades) | 13 | 30 | 0.233 | 2027-01-07 |
| Champion-Evidenz | 7 · Tage 6 | 30 · Tage 20 | 0.233 | 2027-02-02 |

Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage. Meta-Label: 176 offene Labels reifen noch; Zielpopulation ['CHAMPION', 'LATE'], Horizont 20 T; offene Labels reifen an label_available_at, Zulauf aus den letzten 14 T.

## Workflow-Fehler (GitHub Actions)

- **Zeitraum**: seit 2026-10-23T17:32:38Z (74 h)
- **Läufe / fehlgeschlagen**: 52 / 0
- **Scanner-Start heute (New York)**: 2026-10-26T14:52-04:00

Keine fehlgeschlagenen Läufe.

## Warnungen

| Status | Schwere | Komponente | ID | Ursache | seit |
|---|---|---|---|---|---|
| NEW | WARNING | Quote-Abdeckung | OUTCOMES_OVERDUE:marks | Fällige Marks offen: real_opt_ret_20d 1 von 153 offen, ältester 6 Tage; real_strat_ret_20d 1 von 153 offen, ältester 6 Tage (Schwelle > 3 Tage). | 2026-10-26 |

## Trends

| Kennzahl | Wert | Vorlauf (Δ) | ≥7 T zurück (Δ) | ≥30 T zurück (Δ) |
|---|---|---|---|---|
| data.nasdaq100_mismatch | 0 | 0 (0) | 0 (0) | N/A |
| data.quote_archive_median_14d | 200 | 184 (+16) | 153.5 (+46.5) | N/A |
| data.quote_archive_today | 445 | 70 (+375) | 261 (+184) | N/A |
| data.source_health.HEALTHY | 46 | 47 (-1) | 47 (-1) | N/A |
| data.source_health.STALE | 1 | N/A | N/A | N/A |
| data.source_health.UNVALIDATED | 8 | 8 (0) | 8 (0) | N/A |
| data.source_health.n_bad | 1 | 0 (+1) | 0 (+1) | N/A |
| forward.champion.mean_net | -0.3322 | -0.3322 (0) | -0.3095 (-0.0227) | N/A |
| forward.champion.n | 7 | 7 (0) | 6 (+1) | N/A |
| forward.champion.n_independent_days | 6 | 6 (0) | 5 (+1) | N/A |
| forward.champion.sum_net | -2.3256 | -2.3256 (0) | -1.8569 (-0.4687) | N/A |
| integrity.issues | 0 | 0 (0) | 0 (0) | N/A |
| learning.champion_evidence.have | 7 | 7 (0) | 6 (+1) | N/A |
| learning.meta_label.have | 47 | 47 (0) | 8 (+39) | N/A |
| learning.rl_arming.have | 13 | 13 (0) | 12 (+1) | N/A |
| lifecycle.candidates.EXIT_OVERDUE | 0 | 0 (0) | 1 (-1) | N/A |
| lifecycle.candidates.EXIT_PENDING | 115 | 200 (-85) | 120 (-5) | N/A |
| lifecycle.candidates.OUTCOME_FINAL | 512 | 392 (+120) | 312 (+200) | N/A |
| lifecycle.champion.issue.ELIGIBLE_NOT_LEARNED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.EXIT_OVERDUE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.issue.LEARNED_NOT_ELIGIBLE | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.CONTRACT_SELECTED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.ENTRY_OBSERVED | 0 | 0 (0) | 0 (0) | N/A |
| lifecycle.champion.state.EXIT_PENDING | 3 | 3 (0) | 4 (-1) | N/A |
| lifecycle.champion.state.LEARNED | 4 | 4 (0) | 3 (+1) | N/A |
| lifecycle.champion.state.LEARNING_ELIGIBLE | 3 | 3 (0) | 3 (0) | N/A |
| lifecycle.champion.state.OUTCOME_FINAL | 110 | 110 (0) | 110 (0) | N/A |
| lifecycle.champion.state.SIGNAL | 0 | 0 (0) | 0 (0) | N/A |
| marks.real_opt_ret_5d.oldest_pending_days | N/A | N/A | 14 (N/A) | N/A |
| marks.real_opt_ret_5d.pending | 0 | 0 (0) | 1 (-1) | N/A |
| marks.real_strat_ret_5d.oldest_pending_days | N/A | N/A | 14 (N/A) | N/A |
| marks.real_strat_ret_5d.pending | 0 | 0 (0) | 1 (-1) | N/A |
| promotion.active_influence | 0 | 0 (0) | 0 (0) | N/A |
| promotion.unjustified_influence | 0 | 0 (0) | 0 (0) | N/A |
| quote.complete.30d.coverage | 1 | 0.875 (+0.125) | 0.75 (+0.25) | N/A |
| quote.complete.30d.denominator | 7 | 8 (-1) | 8 (-1) | N/A |
| quote.complete.30d.evaluable | 7 | 7 (0) | 6 (+1) | N/A |
| quote.complete.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.complete.7d.denominator | 1 | 2 (-1) | 3 (-2) | N/A |
| quote.complete.7d.evaluable | 1 | 2 (-1) | 3 (-2) | N/A |
| quote.entry.30d.coverage | 0.6 | 0.5 (+0.1) | 0.375 (+0.225) | N/A |
| quote.entry.30d.denominator | 5 | 6 (-1) | 8 (-3) | N/A |
| quote.entry.30d.legacy | 2 | 3 (-1) | 5 (-3) | N/A |
| quote.entry.30d.observed | 3 | 3 (0) | 3 (0) | N/A |
| quote.entry.7d.coverage | N/A | 1 (N/A) | 1 (N/A) | N/A |
| quote.entry.7d.denominator | 0 | 1 (-1) | 3 (-3) | N/A |
| quote.entry.7d.legacy | 0 | 0 (0) | 0 (0) | N/A |
| quote.entry.7d.observed | 0 | 1 (-1) | 3 (-3) | N/A |
| quote.exit_active.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_active.denominator | 3 | 3 (0) | 4 (-1) | N/A |
| quote.exit_active.mark_today | 3 | 3 (0) | 4 (-1) | N/A |
| quote.exit_closed.30d.coverage | 1 | 0.875 (+0.125) | 0.75 (+0.25) | N/A |
| quote.exit_closed.30d.denominator | 7 | 8 (-1) | 8 (-1) | N/A |
| quote.exit_closed.30d.observed | 7 | 7 (0) | 6 (+1) | N/A |
| quote.exit_closed.7d.coverage | 1 | 1 (0) | 1 (0) | N/A |
| quote.exit_closed.7d.denominator | 1 | 2 (-1) | 3 (-2) | N/A |
| quote.exit_closed.7d.observed | 1 | 2 (-1) | 3 (-2) | N/A |
| run.scanner.analyzed | 35 | 41 (-6) | 56 (-21) | N/A |
| run.scanner.champion_candidates | 4 | 12 (-8) | 9 (-5) | N/A |
| run.scanner.contracts_designed | N/A | 4 (N/A) | 2 (N/A) | N/A |
| run.scanner.direction_conflict_share | 0.0286 | 0.0732 (-0.0446) | 0.0179 (+0.0107) | N/A |
| run.scanner.direction_conflicts | 1 | 3 (-2) | 1 (0) | N/A |
| run.scanner.trades | 0 | 0 (0) | 0 (0) | N/A |
| run.scanner.universe | 518 | 518 (0) | 518 (0) | N/A |
| training.n_labels | 47 | 25 (+22) | 0 (+47) | N/A |
| workflows.failed_runs | 0 | 0 (0) | 0 (0) | N/A |
