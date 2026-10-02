# Machine Intelligence State – 2026-10-02T15:12:30+00:00

## Selbsteinschätzung

- overall_calibration: **GOOD**
- world_model_uncertainty: **MODERATE**
- world_model_validation: **MODIFY**
- meta_learning: **NEED_MORE_DATA**
- next_architecture: **KEEP_CHAMPION**
- safe_mode: **True**

## Was wissen wir?

- Champion (statisches Ensemble) OOS 2021–2025H1: Expectancy 0.00319 je Position, Sharpe 0.264, Trefferquote 0.4759
- Regime-Abhängigkeit: vix_lt_20 -0.00265, vix_ge_20 0.01234, spy_uptrend -0.00066, spy_downtrend 0.01552
- Abstinenz-Regel auf ungesehenen Jahren [2019, 2020]: aktiv 0.03023 vs. inaktiv 0.00336 (t=3.27) -> bestätigt=True
- Akzeptierte Hypothese HYP-ABST-001: Regime-Abstinenz – Ensemble nur bei VIX >= 20 oder SPY unter SMA200 handeln

## Worüber sind wir unsicher?

- 80-%-Intervalle: roh 0.6512, korrigiert 0.7944; P(>+10 %) Skill -0.1502
- World-Model-Unsicherheit 0.3267; nicht verfügbar: ['earnings_momentum']

## Wo liegen wir systematisch falsch?

- UNKNOWN_CLUSTER_001: {'sector': 'Energy', 'momentum': 'loser_12m'} (Fehler -0.2324, Lift 2.13, Abdeckung LOW)
- UNKNOWN_CLUSTER_002: {'sector': 'Communication Services', 'volatility': 'high_vol'} (Fehler -0.2208, Lift 1.73, Abdeckung LOW)
- UNKNOWN_CLUSTER_003: {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'} (Fehler -0.2114, Lift 1.7, Abdeckung LOW)
- UNKNOWN_CLUSTER_004: {'sector': 'Communication Services', 'momentum': 'loser_12m'} (Fehler -0.2076, Lift 1.67, Abdeckung LOW)
- UNKNOWN_CLUSTER_005: {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'} (Fehler -0.208, Lift 1.66, Abdeckung LOW)
- UNKNOWN_CLUSTER_006: {'sector': 'Financial Services', 'liquidity': 'liquid'} (Fehler -0.2181, Lift 1.66, Abdeckung LOW)
- UNKNOWN_CLUSTER_007: {'lottery': 'lottery_profile'} (Fehler -0.206, Lift 1.66, Abdeckung LOW)
- UNKNOWN_CLUSTER_008: {'liquidity': 'liquid', 'momentum': 'mid'} (Fehler -0.2034, Lift 1.62, Abdeckung LOW)
- UNKNOWN_CLUSTER_009: {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'} (Fehler -0.1998, Lift 1.62, Abdeckung LOW)
- UNKNOWN_CLUSTER_010: {'volatility': 'high_vol', 'liquidity': 'liquid'} (Fehler -0.1993, Lift 1.59, Abdeckung LOW)
- UNKNOWN_CLUSTER_011: {'sector': 'Energy'} (Fehler -0.1941, Lift 1.58, Abdeckung LOW)
- UNKNOWN_CLUSTER_012: {'liquidity': 'liquid', 'vix': 'vix_lt_20'} (Fehler -0.1986, Lift 1.57, Abdeckung LOW)
- UNKNOWN_CLUSTER_013: {'sector': 'Communication Services', 'liquidity': 'liquid'} (Fehler -0.2301, Lift 1.56, Abdeckung LOW)
- UNKNOWN_CLUSTER_014: {'volatility': 'high_vol', 'recent_move': 'extreme_5d_move'} (Fehler -0.2012, Lift 1.55, Abdeckung LOW)
- UNKNOWN_CLUSTER_015: {'liquidity': 'liquid', 'trend': 'uptrend'} (Fehler -0.1984, Lift 1.55, Abdeckung LOW)
- Überkonfident im Bucket 55–60 %: prognostiziert 0.5717, realisiert 0.4635 (n=356)

## Welche Annahmen sind veraltet?

- tnx: Wert 5.237 außerhalb Trainingsband [1.0993, 4.7817]

## Welche Modelle sind redundant?

- rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)
- hgb_xs20_v1: Leave-one-out-Beitrag -0.0002 (≤ 0 = entbehrlich)
- hgb_xs20_risk_regime_v1: Leave-one-out-Beitrag -0.00151 (≤ 0 = entbehrlich)

## Welche Merkmale verlieren Kraft?

- Modell momentum_12_1: 0.0711 -> -0.1172 (t=-2.65)
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
- director: 0/5 akzeptiert
- discovery_engine: 0/117 akzeptiert
- ml_models: 0/6 akzeptiert
- meta_learning: 0/12 akzeptiert

Höchste Research-Priorität: Unterperformt das Segment {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'} den Querschnitt systematisch?

Safe Mode: **AKTIV** – ["FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs (['tnx'])"]
