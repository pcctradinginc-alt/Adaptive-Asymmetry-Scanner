# World Model (wm-v1) – 2026-11-08T15:58:17+00:00

Stichtag 2026-11-06 · Unsicherheit 0.4565 · Dimensionen mit Daten 15/16 · Hash 84ea5ac4c4972ff7

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.4513 | 0.903 | neutral |
| inflation | neutral | 0.2495 | 0.499 | neutral |
| liquidity | neutral | -0.1324 | 0.632 | neutral |
| interest_rates | high | 0.706 | 0.588 | high |
| credit_conditions | low | -0.6759 | 0.824 | neutral |
| risk_appetite | low | -0.7593 | 0.481 | low |
| volatility | high | 1.3571 | 0.0 | neutral |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.973 | 0.054 | low |
| industrial_activity | low | -0.7301 | 0.54 | low |
| freight_supply_chain | low | -1.2585 | 0.0 | no_data |
| inventories | low | -0.7215 | 0.557 | low |
| commodities | neutral | 0.1299 | 0.26 | neutral |
| fx_usd | high | 0.5603 | 0.879 | neutral |
| labour_market | neutral | 0.0045 | 0.009 | neutral |
| breadth | neutral | -0.3103 | 0.621 | neutral |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.00397, 0.024955] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.727 | -1.563 | 0.074 | [-0.063437, 0.111345] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.19 | -2.108 | -2.011 | [-1.326857, -0.085098] | -1.933 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.572 | -9.594 | 0.608 | [-0.008934, 0.087107] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.166 | 0.682 | [-0.006256, 0.122008] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.157111, 0.045777] | -0.205 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.562 | [0.01466, 0.199523] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.154 | -0.274 | -0.377 | 0.409 | [0.000381, 0.088173] | 0.361 | None/None |
