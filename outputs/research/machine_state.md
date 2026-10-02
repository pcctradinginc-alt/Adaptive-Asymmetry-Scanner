# Machine Intelligence State – 2026-10-02T19:19:01+00:00

## Selbsteinschätzung

- overall_calibration: **GOOD**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **REJECT**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Champion (statisches Ensemble) OOS 2021–2025H1: Expectancy 0.0025 je Position, Sharpe 0.22, Trefferquote 0.4735
- Regime-Abhängigkeit: vix_lt_20 -0.00319, vix_ge_20 0.01139, spy_uptrend -0.00124, spy_downtrend 0.01444
- Abstinenz-Regel auf ungesehenen Jahren [2019, 2020]: aktiv 0.0307 vs. inaktiv 0.0027 (t=3.54) -> bestätigt=True
- Akzeptierte Hypothese HYP-ABST-001: Regime-Abstinenz – Ensemble nur bei VIX >= 20 oder SPY unter SMA200 handeln

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6483, korrigiert 0.7919; P(>+10 %) Skill -0.1389
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.227, Lift 2.06, Abdeckung LOW)
- UNKNOWN_CLUSTER_002: {'sector': 'Communication Services', 'volatility': 'high_vol'} (Fehler -0.2165, Lift 1.76, Abdeckung LOW)
- UNKNOWN_CLUSTER_003: {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'} (Fehler -0.2014, Lift 1.76, Abdeckung LOW)
- UNKNOWN_CLUSTER_004: {'lottery': 'lottery_profile'} (Fehler -0.2046, Lift 1.68, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'liquidity': 'liquid', 'momentum': 'mid'} (Fehler -0.2021, Lift 1.67, Abdeckung LOW)
- UNKNOWN_CLUSTER_006: {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'} (Fehler -0.2017, Lift 1.65, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'} (Fehler -0.1978, Lift 1.63, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'sector': 'Communication Services', 'momentum': 'loser_12m'} (Fehler -0.2091, Lift 1.62, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'sector': 'Financial Services', 'liquidity': 'liquid'} (Fehler -0.2148, Lift 1.61, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'volatility': 'high_vol', 'liquidity': 'liquid'} (Fehler -0.1986, Lift 1.59, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'volatility': 'high_vol', 'recent_move': 'extreme_5d_move'} (Fehler -0.1988, Lift 1.57, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'liquidity': 'liquid', 'vix': 'vix_lt_20'} (Fehler -0.1981, Lift 1.55, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'liquidity': 'liquid', 'trend': 'uptrend'} (Fehler -0.1976, Lift 1.53, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'sector': 'Technology', 'liquidity': 'liquid'} (Fehler -0.1949, Lift 1.52, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'sector': 'Energy'} (Fehler -0.1908, Lift 1.52, Abdeckung LOW)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5657, realisiert 0.4913 (n=230)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.237 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- momentum_12_1: Leave-one-out-Beitrag -0.00065 (≤ 0 = entbehrlich)
- hgb_xs20_v1: Leave-one-out-Beitrag -0.0011 (≤ 0 = entbehrlich)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.00051 (≤ 0 = entbehrlich)
- hgb_xs20_risk_regime_v1: Leave-one-out-Beitrag -0.00194 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell momentum_12_1: 0.0711 -> -0.1172 (t=-2.65)
- Modell hgb_asym20_v1: 0.0369 -> -0.0834 (t=-2.08)
- rev_1m: Strukturbruch um 2021-06-04 (CUSUM 1.5)
- log_dollar_vol: Strukturbruch um 2023-01-27 (CUSUM 2.009)
- max_ret_21: Strukturbruch um 2022-12-02 (CUSUM 1.487)
- beta_126: Strukturbruch um 2022-12-02 (CUSUM 1.546)

## Wo ist die Datenqualität niedrig?

- imf_portwatch_ports: WARN

## Wo ist die Uneinigkeit hoch?

- aktuell NORMAL (mittlere Rang-Streuung 0.254)

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
