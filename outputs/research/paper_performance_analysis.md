# Paper-Performance des täglichen Scanners – Ursachenanalyse

Quelle: outputs/history.json, nur RELIABLE-Outcomes: n=3 von 120 (Klassen {'UNKNOWN': 77, 'RECONSTRUCTED': 40, 'RELIABLE': 3}).

Gesamt: {'n': 3, 'win_rate': 0.0, 'mean': -0.6126, 'median': -0.5918, 'profit_factor': 0.0}

Kalibrierung out-of-sample (prequential): {'method': 'prequential: Band-Win-Rate nur aus vor dem Entry geschlossenen Trades (min n 10 je Band)', 'n_evaluated': 0, 'n_skipped_insufficient_history': 3, 'brier_raw': None, 'brier_calibrated': None, 'ece_raw': None, 'ece_calibrated': None, 'calibrated_better': None}

## mc_hit_rate_calibration

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 0.65-0.75 | 2 | 0.0 | -0.5498 | -0.5498 | 0.0 | 0.688 |
| >=0.75 | 1 | 0.0 | -0.7381 | -0.7381 | 0.0 | 0.75 |

## by_strategy

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| BULL_CALL_SPREAD | 3 | 0.0 | -0.6126 | -0.5918 | 0.0 |  |

## by_entry_month

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 2026-09 | 2 | 0.0 | -0.6649 | -0.6649 | 0.0 |  |
| 2026-10 | 1 | 0.0 | -0.5079 | -0.5079 | 0.0 |  |

## by_close_reason

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| stop_loss | 3 | 0.0 | -0.6126 | -0.5918 | 0.0 |  |

## by_catalyst

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| EARNINGS | 2 | 0.0 | -0.623 | -0.623 | 0.0 |  |
| OTHER | 1 | 0.0 | -0.5918 | -0.5918 | 0.0 |  |

## by_llm_impact

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 4 | 3 | 0.0 | -0.6126 | -0.5918 | 0.0 |  |

## by_llm_surprise

| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |
|---|---|---|---|---|---|---|
| 3 | 3 | 0.0 | -0.6126 | -0.5918 | 0.0 |  |
