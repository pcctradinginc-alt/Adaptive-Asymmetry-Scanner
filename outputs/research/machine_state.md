# Machine Intelligence State – 2026-10-03T09:31:56+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00093 je Position, Sharpe 0.127, Trefferquote 0.4862
- Regime-Abhängigkeit: vix_lt_20 -0.00373, vix_ge_20 0.00825, spy_uptrend -0.00193, spy_downtrend 0.01002
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01634 vs. inaktiv 0.00428 (t=1.41)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 0 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6594, korrigiert 0.7859; P(>+10 %) Skill -0.1673
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.1556, Lift 2.75, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_002: {'sector': 'Energy', 'beta': 'high_beta'} (Fehler -0.1606, Lift 2.73, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_003: {'sector': 'Energy', 'volatility': 'high_vol'} (Fehler -0.1596, Lift 2.48, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_004: {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'} (Fehler -0.1842, Lift 2.47, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'lottery': 'lottery_profile'} (Fehler -0.1775, Lift 1.99, Abdeckung LOW)
- UNKNOWN_CLUSTER_006: {'sector': 'Energy'} (Fehler -0.1508, Lift 1.91, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_007: {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'} (Fehler -0.1685, Lift 1.81, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1696, Lift 1.75, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'} (Fehler -0.176, Lift 1.73, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'sector': 'Consumer Cyclical', 'beta': 'high_beta'} (Fehler -0.1698, Lift 1.71, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'volatility': 'high_vol'} (Fehler -0.1652, Lift 1.7, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'momentum': 'loser_12m', 'beta': 'high_beta'} (Fehler -0.1716, Lift 1.64, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'sector': 'Technology', 'recent_move': 'extreme_5d_move'} (Fehler -0.1707, Lift 1.63, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'momentum': 'loser_12m', 'trend': 'downtrend'} (Fehler -0.1856, Lift 1.61, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'sector': 'Technology', 'momentum': 'winner_12m'} (Fehler -0.1618, Lift 1.56, Abdeckung LOW)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5731, realisiert 0.4338 (n=468)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.277 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.0002 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell momentum_12_1: 0.0652 -> -0.1045 (t=-2.44)
- Modell enet_xs20_v1: 0.0976 -> -0.0958 (t=-2.76)
- Modell hgb_asym20_v1: 0.0195 -> -0.0739 (t=-2.62)
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
- factory: 0/6 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.76)?

Safe Mode: **AKTIV** – ["FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs (['tnx'])", "MODEL DRIFT: 50% der Modelle 'deteriorating'"]
