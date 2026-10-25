# World Model (wm-v1) – 2026-10-25T13:44:05+00:00

Stichtag 2026-10-23 · Unsicherheit 0.4169 · Dimensionen mit Daten 15/16 · Hash 68f75644d754e686

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.1121 | 0.224 | neutral |
| inflation | neutral | 0.2499 | 0.5 | neutral |
| liquidity | neutral | -0.1217 | 0.622 | neutral |
| interest_rates | high | 0.9338 | 0.132 | high |
| credit_conditions | neutral | -0.2068 | 0.707 | neutral |
| risk_appetite | low | -0.5018 | 0.996 | low |
| volatility | neutral | 0.2381 | 0.476 | neutral |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.9799 | 0.04 | low |
| industrial_activity | low | -1.488 | 0.0 | low |
| freight_supply_chain | low | -0.9789 | 0.042 | low |
| inventories | low | -0.7241 | 0.552 | low |
| commodities | neutral | 0.4666 | 0.933 | high |
| fx_usd | neutral | 0.2093 | 0.419 | neutral |
| labour_market | neutral | -0.0116 | 0.023 | neutral |
| breadth | low | -0.7064 | 0.587 | low |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.003974, 0.024959] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.726 | -1.563 | 0.074 | [-0.063394, 0.111352] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.191 | -2.108 | -2.011 | [-1.32691, -0.085137] | -1.933 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.572 | -9.594 | 0.608 | [-0.008934, 0.087106] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.167 | 0.682 | [-0.006252, 0.122008] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.157102, 0.045781] | -0.205 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.563 | [0.014681, 0.199547] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.155 | -0.274 | -0.377 | 0.409 | [0.000378, 0.088201] | 0.361 | None/None |
