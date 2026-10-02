# Causal Research

Modul: `modules/causal_research.py`. Präregistrierung:
`config/intelligence_protocol.yaml` → `causal_research` (16 vorab festgelegte
Priors mit erwartetem Vorzeichen). Ausgaben: `outputs/research/causal_research.{json,md}`.

## Evidenz-Stufen
Die Stufen bauen aufeinander auf:
1. correlation
2. predictive_relationship
3. temporally_leading
4. causal_hypothesis

Eine Stufe gilt nur, wenn ihr Test **OOS** besteht. Das Feld
`causal_evidence` wird **nie automatisch** vergeben, denn Kausalität lässt sich
aus Zeitreihen allein nicht belegen. Ein Mensch setzt es nur mit externer
Begründung.

## Tests je Beziehung (Treiber → Sektor-ETF relativ zu SPY, h = 4/13 Wochen)
- Rang-Korrelation; Lead/Lag-Asymmetrie (Treiber führt gegenüber Ziel führt).
- Granger-Test, nur PIT: erweitertes gegen reines AR-Modell, OOS-MSE im
  Walk-Forward.
- Stabilität über Teilperioden, Replikation über Horizonte, ökonomische
  Plausibilität (Vorzeichen = Prior).
- Benjamini-Hochberg über alle Kombinationen (q = 0,10).

## Ergebnis (Lauf 2026-09-29): **REJECT**
Getestet wurden 32 Kombinationen. Keine Beziehung übersteht BH; der beste
Wert ist q = 0,11 (Erstanträge → XLY, jedoch mit **falschem** Vorzeichen).

Mehrere Priors drehen das Vorzeichen um: Zinsen → XLU, Öl → XLE und Fracht →
XLI. Bei einigen führt der Markt die Makrodaten an, nicht umgekehrt. Das passt
zu Märkten, die Erwartungen einpreisen.

Die Sektor-Tilts (Challenger D/E) aus **allen** Prior-Paaren bringen keinen
signifikanten Mehrwert: Monats-Δ +0,003 % (CI [−0,60 %, +0,66 %]).

**Konsequenz:**
- Es gibt keinen Produktions-Input und keinen Knowledge-Graph-Eintrag über
  `correlation` hinaus.
- Neue Priors nur per Präregistrierung. Jeder weitere Test verschärft BH.
