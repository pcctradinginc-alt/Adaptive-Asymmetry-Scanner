# Machine Intelligence State – 2026-10-31T14:49:54+00:00

## Selbsteinschätzung

- overall_calibration: **GOOD**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **False**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00028 je Position, Sharpe 0.04, Trefferquote 0.4834
- Regime-Abhängigkeit: vix_lt_20 -0.00392, vix_ge_20 0.0069, spy_uptrend -0.00204, spy_downtrend 0.00768
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01573 vs. inaktiv 0.00178 (t=1.66)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 0 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6582, korrigiert 0.7887; P(>+10 %) Skill -0.156
- World-Model-Unsicherheit 0.4169; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.1609, Lift 2.87, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_002: {'sector': 'Energy', 'beta': 'high_beta'} (Fehler -0.1654, Lift 2.82, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_003: {'sector': 'Energy', 'volatility': 'high_vol'} (Fehler -0.1618, Lift 2.53, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_004: {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'} (Fehler -0.1845, Lift 2.46, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'sector': 'Energy', 'liquidity': 'less_liquid'} (Fehler -0.1688, Lift 2.32, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_006: {'sector': 'Energy', 'vix': 'vix_ge_20'} (Fehler -0.1526, Lift 2.24, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_007: {'lottery': 'lottery_profile'} (Fehler -0.1782, Lift 1.96, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'sector': 'Energy'} (Fehler -0.1541, Lift 1.85, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_009: {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'} (Fehler -0.1696, Lift 1.75, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'sector': 'Consumer Cyclical', 'beta': 'high_beta'} (Fehler -0.1708, Lift 1.74, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'volatility': 'high_vol'} (Fehler -0.1652, Lift 1.7, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'momentum': 'loser_12m', 'trend': 'downtrend'} (Fehler -0.1861, Lift 1.66, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1724, Lift 1.65, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'momentum': 'loser_12m', 'beta': 'high_beta'} (Fehler -0.1726, Lift 1.64, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'} (Fehler -0.1821, Lift 1.61, Abdeckung LOW)

## Welche Annahmen sind veraltet?

- tnx: Wert 4.929 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- momentum_12_1: Leave-one-out-Beitrag -0.00059 (≤ 0 = entbehrlich)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.0001 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell enet_xs20_v1: 0.0679 -> -0.1769 (t=-2.5)
- vol_20: Strukturbruch um 2018-11-30 (CUSUM 1.415)
- vol_60: Strukturbruch um 2019-08-02 (CUSUM 1.529)
- max_ret_21: Strukturbruch um 2022-12-02 (CUSUM 1.415)

## Wo ist die Datenqualität niedrig?

- imf_portwatch_ports: FAIL
- imf_portwatch_disruptions: FAIL

## Wo ist die Uneinigkeit hoch?

- aktuell LOW (mittlere Rang-Streuung 0.2185)

## Research-Tracks mit echtem OOS-Wert

- (keine gemessene Aussage)

## Research-Tracks ohne Ertrag

- literature: 0/5 akzeptiert
- director: 0/16 akzeptiert
- factory: 0/18 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

## Options-Lernmodell: erkannte Lücken (research_director --options)

- (keine gemessene Aussage)

## Offene technische Verbesserungsaufgaben (--notify)

- (keine gemessene Aussage)

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.76)?

Safe Mode: **aus** – –
