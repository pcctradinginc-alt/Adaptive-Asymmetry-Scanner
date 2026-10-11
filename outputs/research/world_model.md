# World Model (wm-v1) – 2026-10-11T15:38:57+00:00

Stichtag 2026-10-09 · Unsicherheit 0.3403 · Dimensionen mit Daten 15/16 · Hash a65a3f2427e1487f

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.0965 | 0.193 | neutral |
| inflation | low | -0.786 | 0.428 | low |
| liquidity | neutral | -0.1125 | 0.612 | neutral |
| interest_rates | high | 1.3272 | 0.0 | high |
| credit_conditions | neutral | 0.4523 | 0.952 | high |
| risk_appetite | neutral | -0.2961 | 0.592 | neutral |
| volatility | low | -0.5046 | 0.991 | neutral |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.9356 | 0.129 | low |
| industrial_activity | low | -1.2574 | 0.0 | low |
| freight_supply_chain | low | -0.982 | 0.036 | low |
| inventories | low | -0.7467 | 0.507 | low |
| commodities | high | 0.874 | 0.252 | high |
| fx_usd | neutral | 0.1529 | 0.306 | neutral |
| labour_market | neutral | 0.0118 | 0.024 | neutral |
| breadth | low | -0.9586 | 0.083 | low |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.003974, 0.024959] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.726 | -1.563 | 0.074 | [-0.063394, 0.111352] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.059 | -2.191 | -2.108 | -2.011 | [-1.326874, -0.085137] | -1.934 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.572 | -9.594 | 0.608 | [-0.008934, 0.087107] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.167 | 0.682 | [-0.006253, 0.122008] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.157102, 0.045781] | -0.205 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.563 | [0.01468, 0.19955] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.155 | -0.274 | -0.377 | 0.409 | [0.000378, 0.088199] | 0.361 | None/None |
