# World Model (wm-v1) – 2026-09-29T15:24:43+00:00

Stichtag 2026-09-28 · Unsicherheit 0.3267 · Dimensionen mit Daten 15/16 · Hash b2012e8000deb86f

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | 0.0486 | 0.097 | neutral |
| inflation | low | -0.7888 | 0.422 | low |
| liquidity | neutral | -0.1124 | 0.612 | neutral |
| interest_rates | high | 1.5202 | 0.0 | high |
| credit_conditions | high | 1.3651 | 0.5 | high |
| risk_appetite | low | -0.5954 | 0.809 | neutral |
| volatility | neutral | -0.1918 | 0.384 | neutral |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.8902 | 0.22 | low |
| industrial_activity | low | -1.3759 | 0.0 | low |
| freight_supply_chain | low | -0.9017 | 0.197 | low |
| inventories | low | -0.7481 | 0.504 | low |
| commodities | high | 0.7905 | 0.419 | high |
| fx_usd | neutral | -0.3476 | 0.695 | neutral |
| labour_market | neutral | 0.0207 | 0.041 | neutral |
| breadth | low | -1.2255 | 0.0 | low |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.111 | -2.872 | 0.561 | [0.004103, 0.025059] | 0.195 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.713 | -1.59 | 0.081 | [-0.062301, 0.113031] | -0.39 | 0.515/0.415 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.148 | -2.069 | -1.971 | [-1.303329, -0.081157] | -1.896 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.566 | -9.589 | 0.609 | [-0.008875, 0.087147] | 0.093 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.77 | -7.006 | 0.684 | [-0.006097, 0.122103] | 0.087 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.567 | -0.638 | -0.15 | [-0.155079, 0.044755] | -0.202 | 0.503/0.416 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.552 | -0.339 | -3.349 | 0.475 | [0.004693, 0.148756] | -0.704 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -0.734 | -0.196 | -0.344 | 0.31 | [-0.003309, 0.053817] | 0.225 | None/None |
