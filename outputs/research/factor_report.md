# Faktor-Report 2026-09-29

SHADOW: keine Produktionswirkung. Gewichte nur Forschung.

Ereignisse: 337 ({'closed_trades': 117, 'shadow_trades': 67, 'ledger': 153})

## Outcome `net` (n=184, unabhängige Tage=52)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|
| iv_rank | 59 | 26 | 0.1418 | [-0.1976, 0.4508] | 0.0741 | 0.3359 |  |
| surprise | 184 | 52 | -0.1342 | [-0.354, 0.0997] | -0.0731 | -0.0752 | decaying |
| eps_drift | 184 | 52 | -0.1143 | [-0.3362, 0.1197] | -0.1317 | -0.5925 |  |
| mismatch | 184 | 52 | -0.1071 | [-0.3298, 0.1268] | -0.3504 | 0.2251 |  |
| impact | 184 | 52 | -0.0967 | [-0.3203, 0.1371] | -0.0911 | 0.1949 | decaying |
| z_score | 184 | 52 | 0.0778 | [-0.1558, 0.3031] | 0.2303 | -0.0273 |  |
| quick_mc_hit_rate | 169 | 50 | 0.0691 | [-0.1691, 0.2997] | -0.1275 | 0.5009 | decaying |
| price_move_48h | 121 | 41 | 0.007 | [-0.2542, 0.2672] | -0.0185 | 0.1122 | unstable, decaying |
| sigma_30d | 184 | 52 | 0.006 | [-0.2251, 0.2364] | -0.085 | 0.3597 | unstable, decaying |

Walk-Forward OOS: adaptiv {'n': 0} vs. gleichgewichtet {'n': 68, 'oos_rank_ic': -0.1918, 'positive_score': {'n': 12, 'hit_rate': 0.1667, 'mean': -0.281, 'median': -0.1841, 'sharpe_per_trade': -0.9835, 'sortino_per_trade': -0.702, 'profit_factor': 0.0191, 'max_drawdown_10pct_sizing': -0.2979}, 'all': {'n': 68, 'hit_rate': 0.2794, 'mean': -0.1942, 'median': -0.0969, 'sharpe_per_trade': -0.4764, 'sortino_per_trade': -0.4879, 'profit_factor': 0.2284, 'max_drawdown_10pct_sizing': -0.7838}}

Redundant: mismatch~price_move_48h (ρ=-0.8157); mismatch~z_score (ρ=-0.9222); price_move_48h~z_score (ρ=0.9298)

