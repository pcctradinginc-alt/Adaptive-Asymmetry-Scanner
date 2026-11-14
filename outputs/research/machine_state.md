# Machine Intelligence State – 2026-11-14T15:23:46+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00075 je Position, Sharpe 0.097, Trefferquote 0.4873
- Regime-Abhängigkeit: vix_lt_20 -0.00317, vix_ge_20 0.00691, spy_uptrend -0.00184, spy_downtrend 0.00903
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01658 vs. inaktiv 0.00311 (t=1.68)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 2 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6691, korrigiert 0.7818; P(>+10 %) Skill -0.1344
- World-Model-Unsicherheit 0.4565; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'} (Fehler -0.1868, Lift 2.55, Abdeckung LOW)
- UNKNOWN_CLUSTER_002: {'volatility': 'high_vol', 'near_high': 'near_52w_high'} (Fehler -0.16, Lift 2.08, Abdeckung LOW)
- UNKNOWN_CLUSTER_003: {'lottery': 'lottery_profile'} (Fehler -0.1802, Lift 2.01, Abdeckung LOW)
- UNKNOWN_CLUSTER_004: {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'} (Fehler -0.1692, Lift 1.77, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1631, Lift 1.71, Abdeckung LOW)
- UNKNOWN_CLUSTER_006: {'momentum': 'loser_12m', 'trend': 'downtrend'} (Fehler -0.1807, Lift 1.7, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'} (Fehler -0.1636, Lift 1.64, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'volatility': 'high_vol'} (Fehler -0.1655, Lift 1.64, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'momentum': 'loser_12m', 'beta': 'high_beta'} (Fehler -0.1672, Lift 1.59, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'momentum': 'winner_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1644, Lift 1.53, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'} (Fehler -0.1638, Lift 1.5, Abdeckung LOW)
- Überkonfident im Bucket 60–65 %: prognostiziert 0.6249, realisiert 0.4479 (n=317)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.154 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- momentum_12_1: Leave-one-out-Beitrag -0.001 (≤ 0 = entbehrlich)
- enet_xs20_v1: Leave-one-out-Beitrag -0.00022 (≤ 0 = entbehrlich)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.00112 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell enet_xs20_v1: 0.0503 -> -0.1912 (t=-2.46)
- vol_20: Strukturbruch um 2018-11-30 (CUSUM 1.43)
- vol_60: Strukturbruch um 2019-08-02 (CUSUM 1.553)
- max_ret_21: Strukturbruch um 2022-12-02 (CUSUM 1.441)

## Wo ist die Datenqualität niedrig?

- imf_portwatch_ports: FAIL
- imf_portwatch_chokepoints: FAIL
- imf_portwatch_disruptions: FAIL

## Wo ist die Uneinigkeit hoch?

- aktuell HIGH (mittlere Rang-Streuung 0.2752)

## Research-Tracks mit echtem OOS-Wert

- (keine gemessene Aussage)

## Research-Tracks ohne Ertrag

- literature: 0/5 akzeptiert
- director: 0/6 akzeptiert
- factory: 0/15 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

## Options-Lernmodell: erkannte Lücken (research_director --options)

- [1.0] Merkmal news_novelty nur in 0% der 130 geprüften Labels befüllt – vom Modell verworfen
- [1.0] Merkmal news_novelty nur in 0% der 495 geprüften Labels befüllt – vom Modell verworfen
- [0.803] 33 Merkmale auf 130 geprüfte Labels (4 je Merkmal, Faustregel >= 20)
- [0.4] für 20 T wurde noch keine Modell-, Merkmals- oder Auswahlvariante prospektiv gegen die Basis getestet
- [0.3] NEED_MORE_DATA – walk_forward: 0/20 Testtage, 0/5 Perioden, 0/30 Fälle; forward: 0/20 Testtage, 0/5 Perioden, 0/30 Fälle
- [0.3] NEED_MORE_DATA – walk_forward: 0/20 Testtage, 0/5 Perioden, 0/30 Fälle; forward: 0/20 Testtage, 0/5 Perioden, 0/30 Fälle
- [0.25] 33 Merkmale auf 495 geprüfte Labels (15 je Merkmal, Faustregel >= 20)

## Offene technische Verbesserungsaufgaben (--notify)

- [P1] IMP-4a0d2d31 Merkmal news_novelty im Options-Lernmodell leer (NOTIFIED)
- [P2] IMP-b4f3b983 Workflow source_health muss regelmäßig nachgestartet werden (NOTIFIED)
- [P2] IMP-ce5d574e Datenquelle imf_portwatch_ports seit 7 Tagen BROKEN (NOTIFIED)

## Alpha-to-Options: Prüfpläne und Lernbefunde

- Status NEED_MORE_DATA · Prüfpläne 0 · Einstufungen {} · Abdeckung NEED_MORE_DATA · Ökonomie NEED_MORE_DATA
- Fehlende Daten: {'ABSTAIN:CAPITAL_ABOVE_MAX': 11, 'ABSTAIN:LONG_OPEN_INTEREST_BELOW_MIN': 6, 'ABSTAIN:SHORT_OPEN_INTEREST_BELOW_MIN': 1, 'ABSTAIN:SHORT_SPREAD_TOO_WIDE': 3, 'NO_EXECUTED_TRADES_WITH_EX_ANTE_SIGNAL_ID': 1}

## Fehlergetriebene Forschung: Forward-Fehler, Hypothesen, Lücken

- 5 T: 8588 geprüfte Forward-Fehler an 26 Stichtagen; Ensemble-IC je unabhängigem Fenster -0.0352 (t=-0.315, 3 Fenster); Muster: DATA_GAP
- 20 T: 4012 geprüfte Forward-Fehler an 11 Stichtagen; Ensemble-IC je unabhängigem Fenster None (t=None, 0 Fenster); Muster: DATA_GAP
- Testbudget: 1/24 Slots belegt, Slot-Alpha 0.00208 (familienweise <= 0.05)
- ERR-d6695229 sentiment_score (5 T): PENDING_FORWARD – 0 unabhängige Forward-Fenster fertig; nächste Prüfung 1/1 bei 4 Fenstern

Letzter Versand der Verbesserungsvorschläge: critical **nothing_to_send**, weekly **sent**

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.5)?

Safe Mode: **AKTIV** – ['MODEL: EXCESSIVE MODEL DISAGREEMENT', 'MODEL: PERFORMANCE BREAKDOWN: Forward-Expectancy t=-7.70']
