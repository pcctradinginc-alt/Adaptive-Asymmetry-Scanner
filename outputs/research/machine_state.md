# Machine Intelligence State – 2026-11-21T16:26:42+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00116 je Position, Sharpe 0.162, Trefferquote 0.4861
- Regime-Abhängigkeit: vix_lt_20 -0.00313, vix_ge_20 0.0079, spy_uptrend -0.00127, spy_downtrend 0.00892
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01658 vs. inaktiv 0.00311 (t=1.68)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 2 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6691, korrigiert 0.7818; P(>+10 %) Skill -0.1344
- World-Model-Unsicherheit 0.4727; nicht verfügbar: ['earnings_momentum', 'inventories']

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
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5759, realisiert 0.4651 (n=215)

## Welche Annahmen sind veraltet?

- vix: Wert 35.89 außerhalb Trainingsband [12.2374, 33.3101]
- vix_chg_21: Wert 16.86 außerhalb Trainingsband [-11.8762, 13.7307]
- tnx: Wert 4.912 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- hgb_asym20_v1: Leave-one-out-Beitrag -5e-05 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell enet_xs20_v1: 0.0376 -> -0.188 (t=-2.3)
- vol_20: Strukturbruch um 2018-11-30 (CUSUM 1.413)
- vol_60: Strukturbruch um 2019-08-02 (CUSUM 1.535)
- max_ret_21: Strukturbruch um 2022-12-02 (CUSUM 1.409)

## Wo ist die Datenqualität niedrig?

- imf_portwatch_ports: WARN

## Wo ist die Uneinigkeit hoch?

- aktuell LOW (mittlere Rang-Streuung 0.185)

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

- [1.0] Merkmal news_novelty nur in 0% der 356 geprüften Labels befüllt – vom Modell verworfen
- [1.0] Merkmal news_novelty nur in 0% der 963 geprüften Labels befüllt – vom Modell verworfen
- [0.541] Merkmal dealer_oi_ratio nur in 46% der 963 geprüften Labels befüllt – vom Modell verworfen
- [0.513] Merkmal impact nur in 49% der 963 geprüften Labels befüllt – vom Modell verworfen
- [0.513] Merkmal options_skew_25d nur in 49% der 963 geprüften Labels befüllt – vom Modell verworfen
- [0.513] Merkmal surprise nur in 49% der 963 geprüften Labels befüllt – vom Modell verworfen
- [0.511] Merkmal options_skew_ratio nur in 49% der 963 geprüften Labels befüllt – vom Modell verworfen
- [0.461] 33 Merkmale auf 356 geprüfte Labels (11 je Merkmal, Faustregel >= 20)

## Offene technische Verbesserungsaufgaben (--notify)

- [P1] IMP-4a0d2d31 Merkmal news_novelty im Options-Lernmodell leer (NOTIFIED)
- [P2] IMP-8c83b133 Merkmal impact im Options-Lernmodell kaum befüllt (OPEN)
- [P2] IMP-8ea10f58 Merkmal dealer_oi_ratio im Options-Lernmodell kaum befüllt (OPEN)
- [P2] IMP-b4f3b983 Workflow source_health muss regelmäßig nachgestartet werden (NOTIFIED)

## Alpha-to-Options: Prüfpläne und Lernbefunde

- Status NEED_MORE_DATA · Prüfpläne 0 · Einstufungen {} · Abdeckung NEED_MORE_DATA · Ökonomie NEED_MORE_DATA
- Fehlende Daten: {'ABSTAIN:CAPITAL_ABOVE_MAX': 19, 'ABSTAIN:LONG_OPEN_INTEREST_BELOW_MIN': 36, 'ABSTAIN:LONG_SPREAD_TOO_WIDE': 6, 'ABSTAIN:NO_EXPIRY_IN_DTE_WINDOW': 6, 'ABSTAIN:SHORT_BID_BELOW_MIN': 1}

## Fehlergetriebene Forschung: Forward-Fehler, Hypothesen, Lücken

- 5 T: 9471 geprüfte Forward-Fehler an 31 Stichtagen; Ensemble-IC je unabhängigem Fenster -0.0352 (t=-0.315, 3 Fenster); Muster: DATA_GAP
- 20 T: 5618 geprüfte Forward-Fehler an 16 Stichtagen; Ensemble-IC je unabhängigem Fenster None (t=None, 0 Fenster); Muster: DATA_GAP
- Testbudget: 1/24 Slots belegt, Slot-Alpha 0.00208 (familienweise <= 0.05)
- ERR-d6695229 sentiment_score (5 T): PENDING_FORWARD – 0 unabhängige Forward-Fenster fertig; nächste Prüfung 1/1 bei 4 Fenstern

Letzter Versand der Verbesserungsvorschläge: critical **nothing_to_send**, weekly **sent**

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.5)?

Safe Mode: **AKTIV** – ['MODEL: PERFORMANCE BREAKDOWN: Forward-Expectancy t=-4.81', "DRIFT: FEATURE DRIFT SEVERE: ['vix', 'vix_chg_21', 'tnx']"]
