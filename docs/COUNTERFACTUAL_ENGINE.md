# Counterfactual Engine und Stress-Umgebung

Modul: `modules/counterfactual.py`. Protokoll: `config/next_protocol.yaml`
(`counterfactual`).

## Frage
Was müsste sich ändern, damit diese Prognose falsch wird?

## Fälle (vorab festgelegt)
| Szenario | Verschiebung |
|---|---|
| vol_spike | VIX +10, VIX-Änderung +10 |
| rate_shock | 10J +100 bp, Kurve −50 bp |
| crash | SPY-Trend −12 %, SPY 63T −15 %, VIX +15 |
| usd_shock | USD 63T +6 % |
| oil_shock | WTI 63T +25 % |
| inflation_shock | CPI YoY +2 Pp. |
| liquidity_shock | Fed-Bilanz 13W −4 % |
| momentum_reversal | Momentum-Ränge gespiegelt |
| volatility_flip | Volatilitäts-/Beta-Ränge gespiegelt |

## Methode
- **Historisch:** In jedem Walk-Forward-Fold werden die dort trainierten
  Basismodelle unter jedem Fall erneut ausgewertet. Daraus entsteht der
  Ensemble-Rang je Fall (`cfrank_*`, `meta_learning.base_oos`).
- **Aktuell:** `latest_counterfactuals` liefert Szenario-Ränge und die
  60T-Median-Rendite je Szenario, zum Beispiel Basis +4 %, Zinsschock +1 % usw.
- **Fragil** heißt: Der Titel fällt unter einem **einzigen** Fall aus dem
  Top-Dezil (Rang < 0,9). Die **dominante Annahme** ist dieser Fall.
- **High Confidence** wird nie auf eine fragile Prognose vergeben
  (`hc_scanner.intelligence_checks`).

## Stress
- Anteil des Top-Dezils, der unter jedem Szenario herausfällt.
- Performance in echten historischen Stressfenstern (Covid 2020, Zinsschock
  2022, Q4 2018, Regionalbanken 2023, Zollschock 2025), nur mit OOS-Daten.
- Synthetische Verschiebungen dienen ausschließlich der Robustheitsprüfung und
  ersetzen keine OOS-Evidenz.

## Validierung
Challenger **F** („A + fragile Top-Dezil-Positionen meiden“) wird gegen A
geprüft. Ergebnis: `docs/NEXT_INTELLIGENCE_VALIDATION.md`.

**Ergebnis (Lauf 2026-09-29): REJECT.**
- 94 % der Top-Dezil-Positionen gelten als fragil. Der Filter ist damit kaum
  selektiv.
- F gegen A: Monats-Δ −0,10 % (CI [−0,79 %, +0,59 %]). Die Locked-Rendite
  sinkt von 1,9 % auf 0,7 %.
- Kontrafaktische Ränge bleiben als Diagnose im HC-Scanner (Fragilität) und in
  der Szenario-Turnover-Analyse.
