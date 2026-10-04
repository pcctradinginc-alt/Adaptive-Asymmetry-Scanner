# World Model (wm-v1) – 2026-10-04T14:48:11+00:00

Stichtag 2026-10-02 · Unsicherheit 0.3204 · Dimensionen mit Daten 15/16 · Hash aa401fd387c7ebd3

| Dimension | Zustand | Score | Unsicherheit | Vorwoche |
|---|---|---|---|---|
| growth | neutral | 0.021 | 0.042 | neutral |
| inflation | low | -0.7888 | 0.422 | low |
| liquidity | neutral | -0.1018 | 0.602 | neutral |
| interest_rates | high | 1.5068 | 0.0 | high |
| credit_conditions | high | 0.9539 | 0.546 | high |
| risk_appetite | neutral | -0.4443 | 0.889 | neutral |
| volatility | neutral | -0.3986 | 0.797 | neutral |
| earnings_momentum | unavailable | None | None | unavailable |
| consumer_demand | low | -0.9383 | 0.123 | low |
| industrial_activity | low | -1.3615 | 0.0 | low |
| freight_supply_chain | low | -0.9836 | 0.033 | low |
| inventories | low | -0.7481 | 0.504 | low |
| commodities | high | 0.8408 | 0.318 | high |
| fx_usd | neutral | -0.2435 | 0.487 | neutral |
| labour_market | neutral | 0.0216 | 0.043 | neutral |
| breadth | low | -1.1624 | 0.0 | low |

- earnings_momentum: nicht verfügbar – keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)

## Validierung gegen bestehende Regime-Engine: **MODIFY**

Signifikant besser bei: ['dd60', 'ic_mom20', 'ic_rev20'] · signifikant schlechter bei: ['rv20'] (einseitig, Bonferroni-α je Ziel 0.0063)

| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |
|---|---|---|---|---|---|---|---|---|
| dd60 (SPY max. Drawdown nächste 60 Tage) | regression | -3.813 | -1.137 | -2.98 | 0.556 | [0.003969, 0.024959] | 0.173 | None/None |
| dd60_bin (Drawdown < -8 %) | binary | -0.864 | -0.726 | -1.565 | 0.074 | [-0.063346, 0.111287] | -0.376 | 0.515/0.408 |
| rv20 (log realisierte SPY-Vola nächste 20 Tage) | regression | -0.06 | -2.189 | -2.106 | -2.01 | [-1.325799, -0.085216] | -1.932 | None/None |
| rot60 (Zyklische minus defensive Sektoren, 60 Tage) | regression | -10.669 | -3.58 | -9.615 | 0.608 | [-0.008978, 0.087077] | 0.09 | None/None |
| sb60 (SPY minus IEF (Aktien vs. Anleihen), 60 Tage) | regression | -7.771 | -1.798 | -7.176 | 0.681 | [-0.006313, 0.121978] | 0.068 | None/None |
| regime_flip20 (Wechsel des bestehenden Regime-Labels (VIX>=20 / SPY<SMA200) in 20 Tagen) | binary | -0.363 | -0.569 | -0.642 | -0.151 | [-0.156894, 0.045949] | -0.204 | 0.503/0.412 |
| ic_mom20 (Querschnitts-IC Momentum 12-1 (Wann wirkt das Signal?)) | regression | -1.728 | -0.199 | -4.413 | 0.561 | [0.014747, 0.197638] | -0.984 | None/None |
| ic_rev20 (Querschnitts-IC 1-Monats-Umkehr) | regression | -1.184 | -0.275 | -0.371 | 0.416 | [0.000239, 0.091708] | 0.372 | None/None |
