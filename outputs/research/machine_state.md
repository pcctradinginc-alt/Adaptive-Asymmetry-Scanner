# Machine Intelligence State – 2026-10-03T12:28:35+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00094 je Position, Sharpe 0.127, Trefferquote 0.4861
- Regime-Abhängigkeit: vix_lt_20 -0.00341, vix_ge_20 0.00779, spy_uptrend -0.00175, spy_downtrend 0.00952
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01575 vs. inaktiv 0.00143 (t=1.67)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 0 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6655, korrigiert 0.7839; P(>+10 %) Skill -0.1471
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.1588, Lift 2.74, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_002: {'sector': 'Energy', 'beta': 'high_beta'} (Fehler -0.1638, Lift 2.65, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_003: {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'} (Fehler -0.1857, Lift 2.47, Abdeckung LOW)
- UNKNOWN_CLUSTER_004: {'sector': 'Energy', 'volatility': 'high_vol'} (Fehler -0.161, Lift 2.4, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_005: {'sector': 'Energy', 'liquidity': 'less_liquid'} (Fehler -0.1697, Lift 2.16, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_006: {'lottery': 'lottery_profile'} (Fehler -0.1783, Lift 2.02, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'sector': 'Energy'} (Fehler -0.1535, Lift 1.77, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_008: {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'} (Fehler -0.1692, Lift 1.77, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1671, Lift 1.75, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'sector': 'Consumer Cyclical', 'beta': 'high_beta'} (Fehler -0.1729, Lift 1.72, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'volatility': 'high_vol'} (Fehler -0.1674, Lift 1.68, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'momentum': 'loser_12m', 'trend': 'downtrend'} (Fehler -0.1831, Lift 1.66, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'} (Fehler -0.1525, Lift 1.63, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'sector': 'Technology', 'recent_move': 'extreme_5d_move'} (Fehler -0.1693, Lift 1.63, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'momentum': 'loser_12m', 'beta': 'high_beta'} (Fehler -0.1712, Lift 1.62, Abdeckung LOW)
- Überkonfident im Bucket 50–55 %: prognostiziert 0.52, realisiert 0.4644 (n=28350)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5785, realisiert 0.4275 (n=538)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.277 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- momentum_12_1: Leave-one-out-Beitrag -4e-05 (≤ 0 = entbehrlich)
- enet_xs20_v1: Leave-one-out-Beitrag -0.0003 (≤ 0 = entbehrlich)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.00073 (≤ 0 = entbehrlich)

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
- factory: 0/12 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.76)?

Safe Mode: **AKTIV** – ["FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs (['tnx'])"]
