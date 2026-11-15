# World Model (wm-v1) – 2026-11-15T16:20:27+00:00

Stichtag 2026-11-13 · Unsicherheit 0.4727 · Dimensionen mit Daten 14/16 · Hash ad6fd3e448de9750

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.2792 | 0.558 | neutral |
| inflation | high | 0.5269 | 0.946 | neutral |
| liquidity | neutral | -0.1334 | 0.633 | neutral |
| interest_rates | high | 1.0683 | 0.0 | high |
| credit_conditions | neutral | -0.2979 | 0.798 | low |
| risk_appetite | neutral | -0.3537 | 0.707 | low |
| volatility | high | 0.6292 | 0.742 | high |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.9696 | 0.061 | low |
| industrial_activity | low | -0.734 | 0.532 | low |
| freight_supply_chain | low | -1.2559 | 0.0 | low |
| inventories | no_data | None | None | low |
| commodities | neutral | -0.4279 | 0.856 | neutral |
| fx_usd | high | 0.8383 | 0.323 | high |
| labour_market | neutral | 0.0456 | 0.091 | neutral |
| breadth | neutral | 0.1853 | 0.371 | neutral |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.00397, 0.024955] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.727 | -1.563 | 0.074 | [-0.063437, 0.111345] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.19 | -2.108 | -2.011 | [-1.3268, -0.085084] | -1.933 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.572 | -9.594 | 0.608 | [-0.008934, 0.087107] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.166 | 0.682 | [-0.006256, 0.122009] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.157108, 0.045778] | -0.204 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.562 | [0.01466, 0.199535] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.155 | -0.274 | -0.377 | 0.409 | [0.000382, 0.088183] | 0.361 | None/None |
