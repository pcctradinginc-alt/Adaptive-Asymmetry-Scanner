# Faktor-Report 2026-10-01

SHADOW: keine Produktionswirkung. Gewichte nur Forschung.

Ereignisse: 1038 ({'closed_trades': 119, 'shadow_trades': 67, 'ledger': 852})

## Outcome `gross_1d` (n=128, unabhängige Tage=1)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|

Walk-Forward OOS: adaptiv None vs. gleichgewichtet None

## Outcome `net` (n=186, unabhängige Tage=54)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|
| iv_rank | 59 | 26 | 0.1035 | [-0.2347, 0.4193] | 0.0353 | 0.2647 | decaying |
| surprise | 186 | 54 | -0.1012 | [-0.3202, 0.1281] | -0.0402 | 0.0247 | unstable, decaying |
| mismatch | 186 | 54 | -0.0874 | [-0.3077, 0.1418] | -0.323 | 0.2061 |  |
| impact | 186 | 54 | -0.078 | [-0.2991, 0.151] | -0.0936 | 0.229 | decaying |
| eps_drift | 186 | 54 | -0.0762 | [-0.2974, 0.1528] | -0.1028 | -0.5551 | unstable |
| z_score | 186 | 54 | 0.0546 | [-0.1739, 0.2775] | 0.1852 | -0.1142 |  |
| price_move_48h | 123 | 43 | 0.0374 | [-0.2191, 0.289] | 0.0216 | 0.2338 | unstable, decaying |
| sigma_30d | 186 | 54 | 0.0182 | [-0.209, 0.2436] | -0.0608 | 0.3215 | unstable |
| quick_mc_hit_rate | 171 | 52 | 0.0111 | [-0.2202, 0.2413] | -0.1767 | 0.304 | unstable, decaying |

Walk-Forward OOS: adaptiv {'n': 0} vs. gleichgewichtet {'n': 68, 'oos_rank_ic': -0.1918, 'positive_score': {'n': 12, 'hit_rate': 0.1667, 'mean': -0.281, 'median': -0.1841, 'sharpe_per_trade': -0.9835, 'sortino_per_trade': -0.702, 'profit_factor': 0.0191, 'max_drawdown_10pct_sizing': -0.2979}, 'all': {'n': 68, 'hit_rate': 0.2794, 'mean': -0.1942, 'median': -0.0969, 'sharpe_per_trade': -0.4764, 'sortino_per_trade': -0.4879, 'profit_factor': 0.2284, 'max_drawdown_10pct_sizing': -0.7838}}

Redundant: mismatch~price_move_48h (ρ=-0.8129); mismatch~z_score (ρ=-0.9211); price_move_48h~z_score (ρ=0.9304)

## LLM-Richtung vs. naive Preis-Baseline (gepaart)

| h | n | Tage | LLM-Mittel | Baseline-Mittel | Diff/Tag | t | Urteil |
|---|---|---|---|---|---|---|---|
| h5 | 0 | 0 | None | None | None | None | insufficient_data |
| h20 | 0 | 0 | None | None | None | None | insufficient_data |
| h45 | 0 | 0 | None | None | None | None | insufficient_data |

