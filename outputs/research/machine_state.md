# Machine Intelligence State – 2026-10-02T21:33:12+00:00

## Selbsteinschätzung

- overall_calibration: **WEAK**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Referenz (statisches Ensemble; KEIN registrierter Champion) OOS 2021–2025H1: Expectancy 0.00058 je Position, Sharpe 0.08, Trefferquote 0.4851
- Regime-Abhängigkeit: vix_lt_20 -0.00381, vix_ge_20 0.00747, spy_uptrend -0.00214, spy_downtrend 0.00922
- Abstinenz-Regel historisch [2019, 2020] (Status CONTAMINATED, zählt nicht): aktiv 0.01659 vs. inaktiv 0.00357 (t=1.54)
- Abstinenz-Regel VORWÄRTS ab 2026-09-29: 0 aktive / 0 inaktive fertige Kohorten, Status ACCUMULATING

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6584, korrigiert 0.7843; P(>+10 %) Skill -0.1491
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.1555, Lift 2.87, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_002: {'sector': 'Energy', 'beta': 'high_beta'} (Fehler -0.1595, Lift 2.85, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_003: {'sector': 'Energy', 'volatility': 'high_vol'} (Fehler -0.1597, Lift 2.59, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_004: {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'} (Fehler -0.1863, Lift 2.5, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'sector': 'Energy', 'vix': 'vix_ge_20'} (Fehler -0.1488, Lift 2.35, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_006: {'lottery': 'lottery_profile'} (Fehler -0.1772, Lift 2.0, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'sector': 'Energy'} (Fehler -0.1522, Lift 1.96, Abdeckung MEDIUM)
- UNKNOWN_CLUSTER_008: {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'} (Fehler -0.1692, Lift 1.82, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'} (Fehler -0.1689, Lift 1.81, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'volatility': 'high_vol'} (Fehler -0.1659, Lift 1.71, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'sector': 'Consumer Cyclical', 'beta': 'high_beta'} (Fehler -0.1747, Lift 1.69, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'} (Fehler -0.1838, Lift 1.67, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'momentum': 'loser_12m', 'trend': 'downtrend'} (Fehler -0.1824, Lift 1.67, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'sector': 'Basic Materials', 'beta': 'high_beta'} (Fehler -0.1572, Lift 1.66, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'sector': 'Technology', 'recent_move': 'extreme_5d_move'} (Fehler -0.1697, Lift 1.65, Abdeckung LOW)
- Überkonfident im Bucket 50–55 %: prognostiziert 0.5197, realisiert 0.4639 (n=26974)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5689, realisiert 0.4505 (n=364)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.277 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- momentum_12_1: Leave-one-out-Beitrag -0.00048 (≤ 0 = entbehrlich)
- enet_xs20_v1: Leave-one-out-Beitrag -0.00055 (≤ 0 = entbehrlich)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.00036 (≤ 0 = entbehrlich)

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
- director: 0/9 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Warum verliert momentum_12_1 an Prognosekraft (t=-2.65)?

Safe Mode: **AKTIV** – ["FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs (['tnx'])"]
