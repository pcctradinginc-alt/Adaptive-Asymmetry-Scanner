# Paper-Performance des täglichen Scanners – Ursachenanalyse

Quelle: outputs/history.json, nur RELIABLE-Outcomes: n=8 von 118 (Klassen {'UNKNOWN': 70, 'RECONSTRUCTED': 40, 'RELIABLE': 8}).

Gesamt: {'n': 8, 'win_rate': 0.125, 'mean': -0.2925, 'median': -0.3918, 'profit_factor': 0.18}

Kalibrierung out-of-sample (prequential): {'method': 'prequential: Band-Win-Rate nur aus vor dem Entry geschlossenen Trades (min n 10 je Band)', 'n_evaluated': 0, 'n_skipped_insufficient_history': 8, 'brier_raw': None, 'brier_calibrated': None, 'ece_raw': None, 'ece_calibrated': None, 'calibrated_better': None}

## mc_hit_rate_calibration

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 0.55-0.65 | 1 | 1.0 | 0.5102 | 0.5102 | None | 0.613 |
| 0.65-0.75 | 6 | 0.0 | -0.352 | -0.3918 | 0.0 | 0.713 |
| >=0.75 | 1 | 0.0 | -0.7381 | -0.7381 | 0.0 | 0.75 |

## by_strategy

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| BULL_CALL_SPREAD | 5 | 0.0 | -0.467 | -0.5079 | 0.0 |  |
| LONG_CALL | 3 | 0.333 | -0.0016 | -0.0469 | 0.99 |  |

## by_entry_month

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 2026-08 | 1 | 0.0 | -0.3154 | -0.3154 | 0.0 |  |
| 2026-09 | 4 | 0.25 | -0.2167 | -0.3194 | 0.37 |  |
| 2026-10 | 3 | 0.0 | -0.3859 | -0.4681 | 0.0 |  |

## by_close_reason

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| max_holding_period | 2 | 0.0 | -0.1812 | -0.1812 | 0.0 |  |
| stop_loss | 5 | 0.0 | -0.4975 | -0.5079 | 0.0 |  |
| take_profit | 1 | 1.0 | 0.5102 | 0.5102 | None |  |

## by_catalyst

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| EARNINGS | 5 | 0.0 | -0.3886 | -0.4681 | 0.0 |  |
| INSIDER | 1 | 0.0 | -0.3154 | -0.3154 | 0.0 |  |
| OTHER | 2 | 0.5 | -0.0408 | -0.0408 | 0.86 |  |

## by_llm_impact

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 4 | 7 | 0.143 | -0.2674 | -0.3154 | 0.21 |  |
| 5 | 1 | 0.0 | -0.4681 | -0.4681 | 0.0 |  |

## by_llm_surprise

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 3 | 7 | 0.0 | -0.4071 | -0.4681 | 0.0 |  |
| 4 | 1 | 1.0 | 0.5102 | 0.5102 | None |  |
