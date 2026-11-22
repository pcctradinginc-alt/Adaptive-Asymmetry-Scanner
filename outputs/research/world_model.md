# World Model (wm-v1) – 2026-11-22T17:07:03+00:00

Stichtag 2026-11-20 · Unsicherheit 0.3442 · Dimensionen mit Daten 15/16 · Hash 9b0056551fdf5bef

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.2031 | 0.406 | neutral |
| inflation | high | 0.5263 | 0.947 | high |
| liquidity | neutral | -0.1385 | 0.638 | neutral |
| interest_rates | high | 0.7346 | 0.531 | high |
| credit_conditions | low | -0.7867 | 0.713 | neutral |
| risk_appetite | low | -1.1951 | 0.0 | neutral |
| volatility | high | 2.6529 | 0.0 | high |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.9298 | 0.14 | low |
| industrial_activity | neutral | -0.3931 | 0.786 | low |
| freight_supply_chain | low | -1.2532 | 0.0 | low |
| inventories | low | -0.9654 | 0.069 | no_data |
| commodities | low | -0.9162 | 0.168 | neutral |
| fx_usd | high | 1.1392 | 0.0 | high |
| labour_market | neutral | 0.0382 | 0.076 | neutral |
| breadth | low | -0.6553 | 0.689 | neutral |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.00397, 0.024955] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.727 | -1.563 | 0.074 | [-0.063438, 0.111344] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.19 | -2.108 | -2.011 | [-1.326869, -0.085087] | -1.933 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.572 | -9.594 | 0.608 | [-0.008934, 0.087107] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.166 | 0.682 | [-0.006256, 0.122008] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.15711, 0.045778] | -0.204 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.562 | [0.01466, 0.19953] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.155 | -0.274 | -0.377 | 0.409 | [0.000381, 0.088181] | 0.361 | None/None |
