# World Model

Modul: `modules/world_model.py`. Präregistrierung:
`config/intelligence_protocol.yaml` → `world_model`. Workflow: `world_model.yml`
(sonntags). Nur Schattenbetrieb.

## Zweck
Ein versionierter, messbarer Zustand von Wirtschaft und Markt je Woche. Er wird
nicht aus isolierten Features gebildet und nie per LLM klassifiziert.

## Dimensionen und Daten (nur PIT)
| Dimension | Indikatoren | Quelle |
|---|---|---|
| Growth | INDPRO YoY, Kupfer/Gold 63T, Zyklisch − Defensiv 63T | ALFRED, Märkte |
| Inflation | CPI YoY, CPI-Beschleunigung | ALFRED |
| Liquidity | Fed-Bilanz 13W, −NFCI (ab 2025) | ALFRED |
| Interest Rates | 10J-Rendite, 63T-Änderung, Kurve 10J−3M | Märkte |
| Credit | HYG/LQD 63T, −NFCI-Credit (ab 2025) | Märkte, ALFRED |
| Risk Appetite | SPY vs. SMA200, XLY/XLP 63T, Breite 13W | Märkte, Feature-Store |
| Volatility | VIX, VIX/VIX3M, realisierte SPY-Vola | Märkte |
| Earnings Momentum | – | **nicht verfügbar**, keine PIT-Revisionen |
| Consumer | UMCSENT, Retail Sales YoY | ALFRED |
| Industrial | INDPRO 3M, XLI − SPY | ALFRED, Märkte |
| Freight/Supply Chain | US-Fracht-TSI YoY | ALFRED |
| Inventories | Lager/Umsatz YoY | ALFRED |
| Commodities | WTI 63T, Kupfer 63T | ALFRED, Märkte |
| FX | Broad Dollar 63T | ALFRED |
| Labour | Payrolls 3M, −Erstanträge 13W | ALFRED |
| Breadth | Anteil > 13W zuvor, Anteil nahe 52W-Hoch | Feature-Store |

Destatis, Eurostat und PortWatch sind historisch **nicht** PIT-fähig
(`available_at` = Abrufdatum). Sie fließen erst ein, wenn sie mindestens 3
Jahre lang vorwärts archiviert sind.

## Zustand
- **z-Score:** expandierend, nur mit Vergangenheitswerten, frühestens nach 104
  Wochen.
- **Score je Dimension:** Mittel der verfügbaren z-Scores. |Score| > 0,5 ergibt
  high/low, sonst neutral.
- **Unsicherheit je Dimension:** 1 − Abdeckung × Abstand zur Schwelle.
- **Speicherung:** append-only State-Log `outputs/research/world_state.jsonl`
  mit Version und Hash je Stichtag.

## Validierung (Lauf 2026-09-29 15:24 UTC): **MODIFY**
Walk-Forward 2019 bis 2025-06, Block-Bootstrap über Monate, Bonferroni über 8
Ziele (α = 0,0063 je Ziel). Verglichen wird das World Model allein mit der
bestehenden Regime-Engine (VIX, Trend, Zinsen, CPI, Fed-Bilanz, USD, WTI).

| Ziel | Ergebnis World Model gegen Regime-Engine |
|---|---|
| SPY-Drawdown 60T | **signifikant besser** (MSE −56 %) |
| Momentum-IC 20T („wann wirkt Momentum“) | **signifikant besser** (−48 %) |
| Rotation, Aktien vs. Anleihen, Reversal-IC | besser, aber nicht signifikant |
| realisierte Vola 20T | **signifikant schlechter** |
| Drawdown-Binär, Regimewechsel | kein Unterschied |

Beide Modelle schlagen die naive Klimatologie meist **nicht**; die
Skill-Werte sind negativ.

**Entscheidung nach Regel:** MODIFY, weil das World Model bei einem Ziel
schlechter ist. Es bleibt Bericht und Überwachung (Weekly Report,
Meta-Cognition, Safe Mode „World-Model-Unsicherheit“). Als Modell-Input wird es
nur in der Gesamtvalidierung (Challenger C) geprüft. Für die
Volatilitätsprognose bleibt die bestehende Regime-Engine die Referenz.
