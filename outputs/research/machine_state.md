# Machine Intelligence State – 2026-09-29T16:35:24+00:00

## Selbsteinschätzung

- overall_calibration: **GOOD**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **REJECT**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Champion (statisches Ensemble) OOS 2021–2025H1: Expectancy 0.00267 je Position, Sharpe 0.229, Trefferquote 0.4742
- Regime-Abhängigkeit: vix_lt_20 -0.00303, vix_ge_20 0.01158, spy_uptrend -0.00052, spy_downtrend 0.01287
- Abstinenz-Regel auf ungesehenen Jahren [2019, 2020]: aktiv 0.0302 vs. inaktiv 0.00239 (t=3.4) -> bestätigt=True

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6461, korrigiert 0.7899; P(>+10 %) Skill -0.132
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.2329, Lift 2.02, Abdeckung LOW)
- UNKNOWN_CLUSTER_002: {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'} (Fehler -0.2082, Lift 1.74, Abdeckung LOW)
- UNKNOWN_CLUSTER_003: {'sector': 'Communication Services', 'volatility': 'high_vol'} (Fehler -0.2191, Lift 1.71, Abdeckung LOW)
- UNKNOWN_CLUSTER_004: {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'} (Fehler -0.2041, Lift 1.68, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'lottery': 'lottery_profile'} (Fehler -0.2048, Lift 1.68, Abdeckung LOW)
- UNKNOWN_CLUSTER_006: {'liquidity': 'liquid', 'momentum': 'mid'} (Fehler -0.2015, Lift 1.65, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'} (Fehler -0.1993, Lift 1.62, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'volatility': 'high_vol', 'liquidity': 'liquid'} (Fehler -0.1991, Lift 1.6, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'sector': 'Communication Services', 'momentum': 'loser_12m'} (Fehler -0.2098, Lift 1.59, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'liquidity': 'liquid', 'vix': 'vix_lt_20'} (Fehler -0.1975, Lift 1.58, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'sector': 'Technology', 'beta': 'low_beta'} (Fehler -0.1765, Lift 1.58, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'liquidity': 'liquid', 'momentum': 'loser_12m'} (Fehler -0.2017, Lift 1.57, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'sector': 'Technology', 'liquidity': 'liquid'} (Fehler -0.195, Lift 1.56, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'liquidity': 'liquid', 'trend': 'uptrend'} (Fehler -0.1975, Lift 1.56, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'sector': 'Healthcare', 'beta': 'high_beta'} (Fehler -0.1984, Lift 1.55, Abdeckung LOW)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.24 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- momentum_12_1: Leave-one-out-Beitrag -0.00041 (≤ 0 = entbehrlich)
- hgb_xs20_v1: Leave-one-out-Beitrag -0.00127 (≤ 0 = entbehrlich)
- hgb_asym20_v1: Leave-one-out-Beitrag -0.0005 (≤ 0 = entbehrlich)
- hgb_xs20_risk_regime_v1: Leave-one-out-Beitrag -0.00255 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell momentum_12_1: 0.0711 -> -0.1172 (t=-2.65)
- Modell hgb_asym20_v1: 0.0361 -> -0.0645 (t=-2.04)
- rev_1m: Strukturbruch um 2021-06-04 (CUSUM 1.5)
- log_dollar_vol: Strukturbruch um 2023-01-27 (CUSUM 2.01)
- max_ret_21: Strukturbruch um 2022-12-02 (CUSUM 1.487)
- beta_126: Strukturbruch um 2022-12-02 (CUSUM 1.546)

## Wo ist die Datenqualität niedrig?

- imf_portwatch_ports: WARN
- entsoe_power: WARN

## Wo ist die Uneinigkeit hoch?

- aktuell NORMAL (mittlere Rang-Streuung 0.2512)

## Research-Tracks mit echtem OOS-Wert

- (keine gemessene Aussage)

## Research-Tracks ohne Ertrag

- literature: 0/5 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Warum verliert momentum_12_1 an Prognosekraft (t=-2.65)?

Safe Mode: **AKTIV** – ["FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs (['tnx'])"]
