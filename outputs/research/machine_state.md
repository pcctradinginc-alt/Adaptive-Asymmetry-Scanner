# Machine Intelligence State – 2026-10-03T13:39:07+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **False**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00154 je Position, Sharpe 0.216, Trefferquote 0.4877
- Regime-Abhängigkeit: vix_lt_20 -0.00275, vix_ge_20 0.00828, spy_uptrend -0.00109, spy_downtrend 0.0099
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01573 vs. inaktiv 0.00178 (t=1.66)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 0 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6582, korrigiert 0.7887; P(>+10 %) Skill -0.156
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

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
- Überkonfident im Bucket 50–55 %: prognostiziert 0.5186, realisiert 0.462 (n=29764)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5762, realisiert 0.4447 (n=425)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.277 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)

## Welche Merkmale verlieren Kraft?

- Modell momentum_12_1: 0.0652 -> -0.1045 (t=-2.44)
- Modell enet_xs20_v1: 0.0976 -> -0.0958 (t=-2.76)
- vol_20: Strukturbruch um 2018-11-30 (CUSUM 1.38)
- vol_60: Strukturbruch um 2019-08-02 (CUSUM 1.493)

## Wo ist die Datenqualität niedrig?

- imf_portwatch_ports: WARN

## Wo ist die Uneinigkeit hoch?

- aktuell NORMAL (mittlere Rang-Streuung 0.2592)

## Research-Tracks mit echtem OOS-Wert

- (keine gemessene Aussage)

## Research-Tracks ohne Ertrag

- literature: 0/5 akzeptiert
- director: 0/16 akzeptiert
- factory: 0/18 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.76)?

Safe Mode: **aus** – –
