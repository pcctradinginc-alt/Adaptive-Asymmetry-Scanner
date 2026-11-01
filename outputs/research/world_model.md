# World Model (wm-v1) – 2026-11-01T16:00:47+00:00

Stichtag 2026-10-30 · Unsicherheit 0.4721 · Dimensionen mit Daten 14/16 · Hash 9f0f9f987ea43848

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | -0.3107 | 0.621 | neutral |
| inflation | neutral | 0.2497 | 0.499 | neutral |
| liquidity | neutral | -0.1196 | 0.62 | neutral |
| interest_rates | high | 0.8278 | 0.344 | high |
| credit_conditions | neutral | -0.3535 | 0.854 | neutral |
| risk_appetite | low | -0.6781 | 0.644 | low |
| volatility | neutral | 0.2383 | 0.477 | neutral |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.9765 | 0.047 | low |
| industrial_activity | low | -0.9011 | 0.198 | low |
| freight_supply_chain | no_data | None | None | low |
| inventories | low | -0.7228 | 0.554 | low |
| commodities | neutral | 0.3543 | 0.709 | neutral |
| fx_usd | neutral | 0.1248 | 0.25 | neutral |
| labour_market | neutral | -0.0024 | 0.005 | neutral |
| breadth | neutral | -0.3942 | 0.788 | low |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.983 | 0.556 | [0.00397, 0.024955] | 0.172 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.727 | -1.564 | 0.074 | [-0.06344, 0.111342] | -0.375 | 0.515/0.407 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.191 | -2.108 | -2.011 | [-1.326896, -0.085103] | -1.933 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.572 | -9.594 | 0.608 | [-0.008935, 0.087106] | 0.092 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.792 | -7.166 | 0.682 | [-0.006256, 0.122008] | 0.069 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.57 | -0.642 | -0.152 | [-0.157099, 0.045785] | -0.205 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.741 | -0.199 | -4.462 | 0.563 | [0.01468, 0.199544] | -0.993 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.155 | -0.274 | -0.377 | 0.409 | [0.00038, 0.088205] | 0.361 | None/None |
