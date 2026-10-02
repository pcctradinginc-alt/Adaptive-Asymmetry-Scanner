# Paper-Performance des täglichen Scanners – Ursachenanalyse

Quelle: outputs/history.json, nur zuverlässige Outcomes: n=79 von 119.

Gesamt: {'n': 79, 'win_rate': 0.354, 'mean': 0.0423, 'median': -0.3134, 'profit_factor': 1.12}

## mc_hit_rate_calibration

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 0.55-0.65 | 8 | 0.25 | -0.2884 | -0.5854 | 0.37 | 0.595 |
| 0.65-0.75 | 27 | 0.222 | -0.2061 | -0.503 | 0.42 | 0.709 |
| <0.55 | 1 | 0.0 | -0.6523 | -0.6523 | 0.0 | 0.522 |
| >=0.75 | 29 | 0.414 | 0.0903 | -0.3134 | 1.22 | 0.857 |

## by_strategy

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| BEAR_PUT_SPREAD | 3 | 0.0 | 0.0 | 0.0 | None |  |
| BULL_CALL_SPREAD | 38 | 0.342 | -0.0825 | -0.3828 | 0.79 |  |
| LONG_CALL | 36 | 0.417 | 0.2236 | -0.2218 | 1.67 |  |
| LONG_PUT | 2 | 0.0 | -0.7863 | -0.7863 | 0.0 |  |

## by_entry_month

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 2026-04 | 46 | 0.435 | 0.2178 | -0.0612 | 1.64 |  |
| 2026-05 | 25 | 0.28 | -0.1806 | -0.5263 | 0.56 |  |
| 2026-07 | 3 | 0.333 | 0.1748 | -0.1579 | 2.63 |  |
| 2026-08 | 3 | 0.0 | -0.4522 | -0.503 | 0.0 |  |
| 2026-09 | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |

## by_close_reason

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| max_holding_period | 7 | 0.286 | -0.0689 | -0.1579 | 0.52 |  |
| offen_bis_Bewertung | 45 | 0.444 | 0.2399 | 0.0 | 1.73 |  |
| stop_loss | 20 | 0.0 | -0.6436 | -0.6234 | 0.0 |  |
| take_profit | 6 | 1.0 | 0.9992 | 0.6703 | None |  |
| time_exit | 1 | 0.0 | -0.0941 | -0.0941 | 0.0 |  |

## by_catalyst

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| EARNINGS | 6 | 0.167 | -0.1773 | -0.2551 | 0.44 |  |
| INSIDER | 1 | 0.0 | -0.5063 | -0.5063 | 0.0 |  |
| OTHER | 1 | 0.0 | -0.5918 | -0.5918 | 0.0 |  |
| keiner | 71 | 0.38 | 0.0775 | -0.2218 | 1.21 |  |

## by_llm_impact

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 3 | 1 | 1.0 | 0.3016 | 0.3016 | None |  |
| 4 | 12 | 0.5 | 0.4423 | 0.3293 | 2.49 |  |
| 5 | 31 | 0.258 | -0.0933 | -0.5263 | 0.78 |  |
| 6 | 30 | 0.367 | 0.0145 | -0.2179 | 1.04 |  |
| 7 | 3 | 0.0 | -0.4812 | -0.2218 | 0.0 |  |
| 8 | 2 | 1.0 | 0.8166 | 0.8166 | None |  |

## by_llm_surprise

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 2 | 4 | 0.75 | 0.7691 | 0.5591 | 4.7 |  |
| 3 | 11 | 0.455 | 0.4508 | -0.1579 | 2.71 |  |
| 4 | 24 | 0.208 | -0.2355 | -0.5188 | 0.45 |  |
| 5 | 25 | 0.4 | 0.0028 | -0.2218 | 1.01 |  |
| 6 | 10 | 0.4 | 0.0917 | -0.4476 | 1.2 |  |
| 7 | 5 | 0.2 | -0.0054 | 0.0 | 0.97 |  |
