# Paper-Performance des täglichen Scanners – Ursachenanalyse

Quelle: outputs/history.json, nur RELIABLE-Outcomes: n=6 von 116 (Klassen {'UNKNOWN': 70, 'RECONSTRUCTED': 40, 'RELIABLE': 6}).

Gesamt: {'n': 6, 'win_rate': 0.167, 'mean': -0.3041, 'median': -0.4117, 'profit_factor': 0.22}

Kalibrierung out-of-sample (prequential): {'method': 'prequential: Band-Win-Rate nur aus vor dem Entry geschlossenen Trades (min n 10 je Band)', 'n_evaluated': 0, 'n_skipped_insufficient_history': 6, 'brier_raw': None, 'brier_calibrated': None, 'ece_raw': None, 'ece_calibrated': None, 'calibrated_better': None}

## mc_hit_rate_calibration

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 0.55-0.65 | 1 | 1.0 | 0.5102 | 0.5102 | None | 0.613 |
| 0.65-0.75 | 4 | 0.0 | -0.3992 | -0.4117 | 0.0 | 0.711 |
| >=0.75 | 1 | 0.0 | -0.7381 | -0.7381 | 0.0 | 0.75 |

## by_strategy

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| BULL_CALL_SPREAD | 5 | 0.0 | -0.467 | -0.5079 | 0.0 |  |
| LONG_CALL | 1 | 1.0 | 0.5102 | 0.5102 | None |  |

## by_entry_month

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 2026-08 | 1 | 0.0 | -0.3154 | -0.3154 | 0.0 |  |
| 2026-09 | 3 | 0.333 | -0.2732 | -0.5918 | 0.38 |  |
| 2026-10 | 2 | 0.0 | -0.3448 | -0.3448 | 0.0 |  |

## by_close_reason

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| max_holding_period | 1 | 0.0 | -0.3154 | -0.3154 | 0.0 |  |
| stop_loss | 4 | 0.0 | -0.5049 | -0.5498 | 0.0 |  |
| take_profit | 1 | 1.0 | 0.5102 | 0.5102 | None |  |

## by_catalyst

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| EARNINGS | 3 | 0.0 | -0.4759 | -0.5079 | 0.0 |  |
| INSIDER | 1 | 0.0 | -0.3154 | -0.3154 | 0.0 |  |
| OTHER | 2 | 0.5 | -0.0408 | -0.0408 | 0.86 |  |

## by_llm_impact

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 4 | 6 | 0.167 | -0.3041 | -0.4117 | 0.22 |  |

## by_llm_surprise

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 3 | 5 | 0.0 | -0.467 | -0.5079 | 0.0 |  |
| 4 | 1 | 1.0 | 0.5102 | 0.5102 | None |  |
