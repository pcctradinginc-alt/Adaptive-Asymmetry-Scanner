# Hypothesen-Datenbank – 2026-09-29T14:28:09+00:00

Getestet (mit p-Wert): 7 · Mehrfachtest: Benjamini-Hochberg über alle · Discovery: 117 Relationen im Fenster 2015-01-01..2019-01-01, 0 überlebt

| ID | Titel | Quelle | Status | netto (WF) | t | p | Regime | Gründe |
|---|---|---|---|---|---|---|---|---|
| HYP-0101 | Kurzfrist-Umkehr (1 Monat) | literature | not_significant_after_fdr | 0.00502 | 2.11 | 0.017429 | {'vix_lt_20': 'works', 'vix_ge_20': 'works', 'spy_uptrend': 'works', 'spy_downtrend': 'works'} | p=0.017429 nicht signifikant nach BH (q=0.1, n=7) |
| HYP-0102 | Niedrig-Volatilitäts-Anomalie | literature | rejected | -0.0067 | -2.01 | 0.977784 | {'vix_lt_20': 'fails', 'vix_ge_20': 'fails', 'spy_uptrend': 'fails', 'spy_downtrend': 'fails'} | Walk-Forward netto -0.0067 mit t=-2.01; nur 0.0 der Jahre positiv; bei Stresskosten nicht positiv; instabil über Hälften {'first': -0.0065, 'second': -0.00689}; |
| HYP-0103 | Lotterie-Effekt (MAX) | literature | rejected | -0.00653 | -2.4 | 0.991802 | {'vix_lt_20': 'fails', 'vix_ge_20': 'fails', 'spy_uptrend': 'fails', 'spy_downtrend': 'fails'} | Walk-Forward netto -0.00653 mit t=-2.4; nur 0.143 der Jahre positiv; bei Stresskosten nicht positiv; instabil über Hälften {'first': -0.00366, 'second': -0.0093 |
| HYP-0104 | Nähe zum 52-Wochen-Hoch | literature | rejected | -0.00758 | -2.83 | 0.997673 | {'vix_lt_20': 'fails', 'vix_ge_20': 'fails', 'spy_uptrend': 'fails', 'spy_downtrend': 'fails'} | Walk-Forward netto -0.00758 mit t=-2.83; nur 0.0 der Jahre positiv; bei Stresskosten nicht positiv; instabil über Hälften {'first': -0.00693, 'second': -0.00823 |
| HYP-0105 | Intakter Trend, kurzer Rücksetzer (Asymmetrie-Divergenz) | human | rejected | 0.00346 | 1.58 | 0.057053 | {'vix_lt_20': 'works', 'vix_ge_20': 'works', 'spy_uptrend': 'works', 'spy_downtrend': 'works'} | Walk-Forward netto 0.00346 mit t=1.58 |
| HYP-0106 | Momentum nur im Aufwärtstrend | literature | rejected | 0.00523 | 1.23 | 0.109349 | {'vix_lt_20': 'works', 'vix_ge_20': 'works', 'spy_uptrend': 'works', 'spy_downtrend': 'works'} | Walk-Forward netto 0.00523 mit t=1.23; nur 0.571 der Jahre positiv |
| HYP-0107 | Aufmerksamkeitsschock ohne Kursreaktion | human | rejected | -0.00321 | -3.99 | 0.999967 | {'vix_lt_20': 'fails', 'vix_ge_20': 'fails', 'spy_uptrend': 'fails', 'spy_downtrend': 'fails'} | Walk-Forward netto -0.00321 mit t=-3.99; nur 0.286 der Jahre positiv; bei Stresskosten nicht positiv; instabil über Hälften {'first': -0.00306, 'second': -0.003 |
| HYP-0201 | LKW-Maut führt europäische Industriegewinne (30–90 Tage) | human | blocked_data | None | None | None |  | Universum ist US (S&P 500); Maut-Archiv < 3 Jahre PIT; keine Gewinnrevisions-Historie (PIT) |
| HYP-0202 | Hafenaktivität steigt, Aktie fällt -> Aufholpotenzial (Divergenz Realwirtschaft vs. Preis) | human | blocked_data | None | None | None |  | PortWatch ohne belastbare Ticker-Zuordnung; Archiv-Historie zu kurz für Walk-Forward ab 2019 |
| PRIOR-ES-H1 | Drift nach Volumen-Event in Richtung des Event-Tags | prior_study | prior_result | None | None | None |  | Event-Studie 2026-09-29: -0,35 % netto (t -2,4), 1/8 Jahre positiv -> abgelehnt |
| PRIOR-ES-H2 | Kleine Bewegung am Event-Tag (Überreaktions-Gate) ist besser | prior_study | prior_result | None | None | None |  | -0,47 % netto (t -3,6), 0/8 Jahre positiv -> abgelehnt, eher umgekehrt |
| PRIOR-ES-H3 | Positive relative Stärke verstärkt die Event-Drift | prior_study | prior_result | None | None | None |  | -0,16 % netto (t -1,1) -> abgelehnt |
| PRIOR-ES-H4 | Hohes relatives Volumen verstärkt die Drift | prior_study | prior_result | None | None | None |  | -0,39 % netto (t -2,3) -> abgelehnt |
| PRIOR-LLM-VALUE | LLM-Richtung schlägt naive Preis-Baseline | prospective | prior_result | None | None | None |  | läuft prospektiv (factor_monitor.llm_value_add), Urteil frühestens Mitte November 2026 |
