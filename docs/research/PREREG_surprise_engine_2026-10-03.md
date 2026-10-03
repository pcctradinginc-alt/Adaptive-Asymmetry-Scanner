# Präregistrierung: Expectation / Surprise Engine (2026-10-03)

Festgelegt **vor** dem ersten Datenlauf. Code: `modules/surprise_engine.py`, Tests: `tests/test_surprise_engine.py`.

## Frage
Enthält die Differenz zwischen **fundamentaler Überraschung** (Ist-EPS vs. Konsens) und **Marktreaktion**
(abnormale Rendite im Reaktionsfenster) Information über die folgende Rendite, die über die reine
Kursreaktion hinausgeht?

## Daten
- **Universum:** PIT-S&P-500, inklusive später entfernter Titel. Events zählen nur während der Indexmitgliedschaft.
- **Kurse:** Yahoo, bereinigt. Marktbereinigt wird mit SPY im selben Fenster.
- **Konsens- und Ist-EPS:** Yahoo-Earnings-Kalender, bis zu 40 Quartale je Titel.
- **PIT-Regeln:** Der Konsens ist die Erwartung vor der Meldung. Das Reaktionsfenster ergibt sich aus der Meldezeit:
  - vor Börsenöffnung: Meldetag;
  - nach Börsenschluss: Folgetag;
  - Uhrzeit unbekannt: Meldetag und Folgetag.
- **Entry:** Open am Tag nach dem Reaktionsfenster.
- **Kosten:** 10 bp pro Seite, Stresstest mit 25 bp.

## Hypothesen (h = 20 entscheidend, h = 60 berichtet)
| ID | Regel | Richtung |
|---|---|---|
| S1_pead | Vorzeichen der fundamentalen Überraschung | long/short |
| S1L_pead_long | oberes Trainings-Terzil der Überraschung | long |
| S2_unpriced | Perzentil(Fundamental) − Perzentil(Reaktion), oberes vs. unteres Terzil | long/short |
| S2L_unpriced_long | dasselbe, nur oberes Terzil (produktionsnah: Long Calls) | long |
| S3_reaction_only | **Kontrolle:** nur die Richtung der Kursreaktion | long/short |
| S4_disagreement_long | positive Überraschung bei negativer Reaktion | long |

Perzentile, Terzile und Schwellen stammen ausschließlich aus Trainingsjahren mit Exit vor dem Testjahr. Der Test erfolgt jahresweise ab 2019.

## Entscheidungsregel (alle Kriterien müssen für KEEP gelten)
1. OOS-Mittel > 0 und t (Monatskohorten) ≥ 2.
2. ≥ 60 % der Jahre positiv.
3. Bei 25 bp pro Seite weiter positiv.
4. Signifikant nach Benjamini-Hochberg (q = 0,10) über alle 6 × 2 Tests.
5. **Placebo:** Fundamentalwerte werden innerhalb jedes Jahres permutiert (100 Durchläufe). p < 0,05 (nur Hypothesen mit Fundamentaldaten).
6. **Lag-Test:** Entry 5 Handelstage später, weiter positiv.
7. **Replikation:** beide Universumshälften (Ticker-Hash) und beide Zeithälften positiv.
8. Alle Regime (VIX < 20 / ≥ 20, Trend auf/ab) positiv.

Ergebnis: **MODIFY**, wenn Punkt 1 erfüllt ist, aber etwas anderes fehlt; sonst **REJECT**.

## Konsequenz
- Ergebnisse gehen in die Research Memory (ACCEPTED, INCONCLUSIVE oder REJECTED) und in den Montagsreport.
- Historische Evidenz hat **nie** direkte Produktionswirkung. Auch KEEP ermöglicht nur einen Vertragsentwurf. Danach zählt ausschließlich Forward-Evidenz über PromotionController und Adapter.

## Bekannte Grenzen und Datenlücken
- Für den Implied Move gibt es keine historischen IV-Daten. Er ist deshalb nicht Teil der „Markterwartung“ (DATA_GAP).
- Ebenfalls nicht verfügbar: Positionierung und Prediction Markets.
- Es gibt keine PIT-Sektorzuordnung.
- Yahoo liefert nur etwa die letzten 6–10 Jahre an Earnings-Daten.

## Nachregistrierung 2026-10-03 (nach Lauf 1, vor jeder Holdout-Auswertung)
**Ergebnis von Lauf 1 (OOS 2019+):** Alle sechs Hypothesen wurden verworfen (REJECT). Die Kontrolle S3 („Kursreaktion setzt sich fort“) war signifikant **negativ** (t = −2,59, signifikant nach Benjamini-Hochberg), H_alt hat also gewonnen.

**Regel:** Auf denselben Daten wird das Vorzeichen nicht gewechselt. Stattdessen wird eine neue Hypothese formuliert:
- **S5_reaction_reversal:** gegen die Richtung der abnormalen Earnings-Reaktion positionieren (h = 20 entscheidend).

**Prüfung ausschließlich auf dem Holdout:** die Jahre vor 2019, die nie Testperiode waren. Die Hypothese hat keine Parameter, also auch kein Training.

**KEEP**, wenn alle folgenden Kriterien gelten:
1. Mittel > 0 und t ≥ 2;
2. bei 25 bp pro Seite positiv;
3. Lag-Test positiv;
4. beide Universums- und beide Zeithälften positiv.

**Konsequenz:**
- Selbst bei KEEP entsteht nur ein Vorschlag als prospektiver Challenger. Es gibt keine Produktionswirkung, die Forward-Daten entscheiden.
- **Bedeutung:** Bestätigt sich die Umkehr, widerspricht sie der Unterreaktions-These der Produktionspipeline. Die Produktion setzt auf Kursanstiege nach Nachrichten.

## Nachtrag 2 (2026-10-03, vor jeder S5-Auswertung): S5 nur prospektiv

Der oben vorgesehene S5-Test auf den Jahren vor 2019 wird **nicht** durchgeführt (er lief nie).
Begründung: Diese Jahre waren in Lauf 1 Trainingsfenster der S1–S4-Schwellen, und S5 wurde nach
Sicht auf das S3-Ergebnis formuliert – der Zeitraum gilt als USED. Der ML-Locked-Holdout ab
2025-07-01 ist CONTAMINATED. Keiner von beiden dient der Auswahl von S5 oder einer Variante.

Neu: `config/s5_forward_holdout.yaml` (SHA-256 `e605109c98ed…`, gepinnt in
`tests/test_surprise_engine.py`). Nur Meldungen vom 2026-10-05 bis 2027-10-04, Horizont
20 Handelstage, **eine** Auswertung ab 2027-11-15 (vorher nur Fallzahlen, keine Renditen).
Kriterien: n ≥ 300, Mittel nach Basiskosten > 0, t ≥ 2, Stresskosten > 0, Lag > 0,
Replikation in beiden Universums-Hälften positiv. Ergebnis KEEP = nur Vorschlag eines
prospektiven Challengers per PR.
