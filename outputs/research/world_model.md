# World Model (wm-v1) – 2026-10-18T13:16:17+00:00

Stichtag 2026-10-16 · Unsicherheit 0.4917 · Dimensionen mit Daten 15/16 · Hash ab56724a73988e5b

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.4369 | 0.916 | neutral |
| inflation | neutral | 0.2502 | 0.5 | low |
| liquidity | neutral | -0.1113 | 0.611 | neutral |
| interest_rates | high | 1.1184 | 0.0 | high |
| credit_conditions | neutral | -0.1049 | 0.605 | neutral |
| risk_appetite | low | -0.6637 | 0.673 | neutral |
| volatility | neutral | 0.4066 | 0.813 | low |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | neutral | 0.1067 | 0.607 | low |
| industrial_activity | low | -2.1379 | 0.5 | low |
| freight_supply_chain | low | -0.9804 | 0.039 | low |
| inventories | low | -0.7254 | 0.549 | low |
| commodities | high | 0.5674 | 0.865 | high |
| fx_usd | neutral | 0.2811 | 0.562 | neutral |
| labour_market | neutral | 0.003 | 0.006 | neutral |
| breadth | low | -0.9356 | 0.129 | low |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.003974, 0.024959] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.726 | -1.563 | 0.074 | [-0.063394, 0.111352] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.059 | -2.191 | -2.108 | -2.011 | [-1.326956, -0.085153] | -1.934 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.573 | -9.594 | 0.608 | [-0.008936, 0.087106] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.167 | 0.682 | [-0.006254, 0.122008] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.157099, 0.045784] | -0.205 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.563 | [0.01468, 0.199551] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.155 | -0.274 | -0.377 | 0.409 | [0.000378, 0.088202] | 0.361 | None/None |
