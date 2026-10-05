# Paper-Performance des täglichen Scanners – Ursachenanalyse

Quelle: outputs/history.json, nur zuverlässige Outcomes: n=2 von 119.

Gesamt: {'n': 2, 'win_rate': 0.0, 'mean': -0.6649, 'median': -0.6649, 'profit_factor': 0.0}

Kalibrierung out-of-sample (prequential): {'method': 'prequential: Band-Win-Rate nur aus vor dem Entry geschlossenen Trades (min n 10 je Band)', 'n_evaluated': 0, 'n_skipped_insufficient_history': 2, 'brier_raw': None, 'brier_calibrated': None, 'ece_raw': None, 'ece_calibrated': None, 'calibrated_better': None}

## mc_hit_rate_calibration

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 0.65-0.75 | 1 | 0.0 | -0.5918 | -0.5918 | 0.0 | 0.684 |
| >=0.75 | 1 | 0.0 | -0.7381 | -0.7381 | 0.0 | 0.75 |

## by_strategy

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| BULL_CALL_SPREAD | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |

## by_entry_month

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 2026-09 | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |

## by_close_reason

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| stop_loss | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |

## by_catalyst

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| EARNINGS | 1 | 0.0 | -0.7381 | -0.7381 | 0.0 |  |
| OTHER | 1 | 0.0 | -0.5918 | -0.5918 | 0.0 |  |

## by_llm_impact

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 4 | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |

## by_llm_surprise

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 3 | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |
