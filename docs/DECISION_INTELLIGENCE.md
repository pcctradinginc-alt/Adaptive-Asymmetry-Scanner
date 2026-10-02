# Decision Intelligence

Modul: `modules/decision_intel.py`. Protokoll: `config/next_protocol.yaml`
(`decision_intelligence`). **Keine Orderausführung.**

## Je Kandidat
- **Rendite und Wahrscheinlichkeit:** erwartete Rendite, kalibrierte
  Wahrscheinlichkeit.
- **Verlustrisiko:**
  - Downside (unteres 80-%-Band);
  - Expected Shortfall (5 % der 4-Wochen-Renditen);
  - Drawdown-Schätzung.
- **Asymmetrie**
- **Portfolio-Bezug:**
  - mittlere Korrelation zu den übrigen Kandidaten;
  - Faktor-Exposures (Beta, Momentum, Vola, Liquidität);
  - Sektoranteil;
  - Liquidität;
  - Tail-Risiko-Beitrag (Grenzvarianzbeitrag am gleichgewichteten
    Kandidatenportfolio).
- **Portfolio Utility Score** = erwartete Rendite − λ · Grenzrisiko.

## Robustheit statt Mittelwert/Kovarianz
- **Kovarianz:** Ledoit-Wolf-Shrinkage über 104 Wochen, nur aus Wochen vor dem
  Stichtag.
- **Erwartete Renditen:** keine historischen Titelmittelwerte, nur die
  OOS-kalibrierte Rang-Abbildung.
- **Auswahl:** gierig, mit einer Sektor-Obergrenze von 25 %.

## Validierung
- Walk-Forward: diversifizierte Auswahl gegen das plain Top-Dezil.
- KEEP nur, wenn die Sharpe mindestens gleich ist **und** Expected Shortfall
  oder Max-Drawdown besser sind.
- Im HC-Scanner führt ein Portfolio-Nutzen ≤ 0 zur Ablehnung.

Ergebnis: `docs/NEXT_INTELLIGENCE_VALIDATION.md`.

**Ergebnis (Lauf 2026-09-29): KEEP (knapp).**

| | Top-Dezil | diversifiziert |
|---|---|---|
| Sharpe | 0,229 | 0,252 |
| Expected Shortfall (5 %, Monat) | −8,2 % | −7,7 % |
| MaxDD | −18,8 % | −18,6 % |
| Sektor-HHI | 0,179 | 0,152 |

Die Regel ist erfüllt (Sharpe nicht schlechter, ES besser). Der Effekt ist
klein und nicht signifikant. Eingesetzt wird die Auswahl nur im HC-Scanner
(Portfolio-Nutzen), nicht im Ranking.
