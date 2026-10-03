# Machine Intelligence State – 2026-10-03T07:35:45+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00105 je Position, Sharpe 0.145, Trefferquote 0.4829
- Regime-Abhängigkeit: vix_lt_20 -0.00353, vix_ge_20 0.00823, spy_uptrend -0.00192, spy_downtrend 0.01049
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01612 vs. inaktiv 0.00224 (t=1.63)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 0 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6627, korrigiert 0.7854; P(>+10 %) Skill -0.1829
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.1603, Lift 2.86, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_002: {'sector': 'Energy', 'beta': 'high_beta'} (Fehler -0.1638, Lift 2.72, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_003: {'sector': 'Energy', 'volatility': 'high_vol'} (Fehler -0.1632, Lift 2.56, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_004: {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'} (Fehler -0.1884, Lift 2.56, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'sector': 'Energy', 'liquidity': 'less_liquid'} (Fehler -0.1694, Lift 2.3, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_006: {'volatility': 'high_vol', 'near_high': 'near_52w_high'} (Fehler -0.1607, Lift 2.1, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'lottery': 'lottery_profile'} (Fehler -0.1809, Lift 2.01, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'sector': 'Energy'} (Fehler -0.1563, Lift 1.88, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_009: {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'} (Fehler -0.171, Lift 1.77, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'sector': 'Consumer Cyclical', 'beta': 'high_beta'} (Fehler -0.1691, Lift 1.73, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1684, Lift 1.7, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'} (Fehler -0.1806, Lift 1.69, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'volatility': 'high_vol'} (Fehler -0.1668, Lift 1.69, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'sector': 'Technology', 'recent_move': 'extreme_5d_move'} (Fehler -0.1718, Lift 1.68, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'momentum': 'loser_12m', 'beta': 'high_beta'} (Fehler -0.1715, Lift 1.62, Abdeckung LOW)
- Überkonfident im Bucket 50–55 %: prognostiziert 0.5182, realisiert 0.4611 (n=28887)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5643, realisiert 0.4311 (n=617)

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
- director: 0/14 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Warum verliert enet_xs20_v1 an Prognosekraft (t=-2.76)?

Safe Mode: **AKTIV** – ["FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs (['tnx'])"]
