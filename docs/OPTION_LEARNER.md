# Options-Lernmodell (Research): aus geprüften Options-Ergebnissen lernen und empfehlen

Stand 2026-10-30. Modul `modules/option_learner.py`, Tests `tests/test_option_learner.py`.

## Kreislauf (autonom, kein LLM, keine Markt-API, keine Order)

0. **Ausdrucksformen:** Je Kandidat werden die Produktions-Strategie (`single` oder `spread`), die Einzel-Option (`real_option`) und, sobald erfasst, die Research-Kontraktvariante (`real_option_alt`, `single_alt`) bewertet. Derselbe Kontrakt zählt nur einmal. Die Empfehlung nimmt je Kandidat die Form mit der höchsten Prognose.

1. **Daten:** Kandidaten-Ledger. Merkmale und Kontrakt sind zum Entscheidungszeitpunkt eingefroren. Kontrakt-Merkmale kommen aus der Snapshot-Quote (`snapshot_legs`), nie aus dem späteren Fill.
2. **Label:** `real_strat_ret_{h}d` minus Kommission (0,65 USD je Leg und Seite). Es zählt nur, wenn `research_evidence_discovery.provenance` das Label als VERIFIED bestätigt, Entry und Exit also aus dem Quote-Archiv nachgerechnet sind.
3. **Modell:** Median-Imputation, Standardisierung, Ridge (alpha 10). Das Ziel wird auf das 1.–99. Perzentil gekappt. Merkmale mit weniger als 50 % Abdeckung fallen weg. Die Spezifikation ist fest (`SPEC`, `spec_hash`).
4. **Walk-Forward:** Je Signaltag wird nur mit Labels trainiert, deren Mark-Tag vor dem Testtag liegt.
   - VALIDATED: mindestens 20 Testtage, mindestens 30 Fälle mit Prognose > 0, deren Netto-Mittel im 95-%-Datums-Bootstrap über 0 liegt, und Rang-IC mit t > 2.
   - Sonst NOT_VALIDATED oder NEED_MORE_DATA.
5. **Prognose-Journal:** `outputs/research/option_learner/predictions.jsonl`, append-only. Jede Tagesprognose wird geschrieben, bevor ihr Label existiert.
6. **Forward-Test:** Journal gegen die später geprüften Labels. Ergebnis ist FORWARD_POSITIVE, FORWARD_NEGATIVE, INCONCLUSIVE oder NEED_MORE_DATA.
7. **Empfehlung:**
   - Nur bei Walk-Forward VALIDATED, Forward nicht FORWARD_NEGATIVE und Prognose > 0.
   - Nur für Call-Strukturen (LONG_CALL, BULL_CALL_SPREAD), die die Produktion handeln kann.
   - Sonst eine Beobachtungsliste (Top 5, unvalidiert).
   - Mail nur bei echten Empfehlungen.

## Leistungszeugnis und Lernkurve

- **Manifest** je Auswertung: Anzahl und Hash der verwendeten geprüften Labels, Stichtag, Verteilung je Ausdrucksform. Damit ist nachvollziehbar, welche Labels in welches Modell geflossen sind.
- **Historie** `status_history_<h>d.jsonl` (append-only, ein Eintrag je neuem Stand) mit dem Vergleich zur Vorversion (`vs_previous`).
- **Vergleich mit einfachen Basisstrategien:** „nie handeln“ (0) und „alles handeln“ (Mittel aller Fälle auf ungesehenen Tagen).
- **Strategievergleich:** Mittel je Ausdrucksform auf ungesehenen Tagen (`by_expression`).
- **`report_card.{json,md}`** beantwortet die Audit-Fragen je Horizont mit Ja, Teilweise, Nein oder offen, jeweils mit Beleg. Es erscheint im Montagsbericht (Abschnitt 12).

## Automatik

| Workflow | Schritt | Wirkung |
|---|---|---|
| `scanner.yml` (täglich, nach dem Feedback) | `recommend --horizon 20 --mail` und `--horizon 5` | Journal, `recommendations/<Tag>_<h>d.{json,md}`, Mail nur bei Empfehlung |
| `ml_research.yml` (Samstag und monatlich) | `evaluate --horizon 5/20` | `status_<h>d.json` mit Walk-Forward, Forward-Test und Koeffizienten |

## Grenzen

- Es entsteht keine Order, und es gibt keinen Einfluss auf Champion, Gates, Sizing, Risk oder Promotion.
- Eine Empfehlung ist eine Research-Aussage auf Basis der eigenen Evidenz, keine Anlageberatung.
- Mit dem Datenstand vom 30.10.2026 gibt es 320 geprüfte 5-Tage-Labels an 8 Tagen und noch keine geprüften 20-Tage-Labels. Der Status lautet deshalb NEED_MORE_DATA.
- Die ersten 20-Tage-Labels sind ab dem 3.11. geprüft. Mit der Research-Stichprobe (etwa 40 Labels je Tag) ist ein Walk-Forward-Verdikt frühestens Ende Dezember möglich.
