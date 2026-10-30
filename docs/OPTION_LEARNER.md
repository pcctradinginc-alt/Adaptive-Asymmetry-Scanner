# Options-Lernmodell (Research): aus geprüften Options-Ergebnissen lernen und empfehlen

Stand 2026-10-30 (Freigabe, Versionierung, Auswahl-Validierung, Lernfortschritt). Modul `modules/option_learner.py`, Tests `tests/test_option_learner.py`.

## Kreislauf (autonom, kein LLM, keine Markt-API, keine Order)

0. **Ausdrucksformen:** Je Kandidat werden die Produktions-Strategie (`single` oder `spread`), die Einzel-Option (`real_option`) und, sobald erfasst, die Research-Kontraktvariante (`real_option_alt`, `single_alt`) bewertet. Derselbe Kontrakt zählt nur einmal. Die erste vorhandene Form (Produktions-Strategie, sonst `real_option`) ist die **Standardform**.
   - **Auswahlregel (`decide`):** Je Kandidat wird genau eine Form gewählt, nur aus den Prognosen und nie aus Ergebnissen. Gewählt wird die zulässige Form (Call-Struktur) mit der höchsten Prognose. TRADE bei Prognose > 0, sonst ABSTAIN.

1. **Daten:** Kandidaten-Ledger. Merkmale und Kontrakt sind zum Entscheidungszeitpunkt eingefroren. Kontrakt-Merkmale kommen aus der Snapshot-Quote (`snapshot_legs`), nie aus dem späteren Fill. Dazu gehören Geld-Brief-Spanne, IV, Laufzeit, Moneyness, Delta, Open Interest, IV-Rang sowie dessen Komponenten `rv_score` (IV gegen realisierte Volatilität) und `term_score` (Term-Struktur). So kann das Modell lernen, ob eine Option relativ zur Volatilität teuer oder günstig ist (Audit §8).
2. **Label:** `real_strat_ret_{h}d` minus Kommission (0,65 USD je Leg und Seite). Es zählt nur, wenn `research_evidence_discovery.provenance` das Label als VERIFIED bestätigt, Entry und Exit also aus dem Quote-Archiv nachgerechnet sind.
3. **Modell:** Median-Imputation, Standardisierung, Ridge (alpha 10). Das Ziel wird auf das 1.–99. Perzentil gekappt. Merkmale mit weniger als 50 % Abdeckung fallen weg.
   - **Experimentversion** (`experiment()`, `experiment_id`): Merkmale, Labeldefinition, Kostenmodell, Trainingsverfahren und Auswahlregel. Jede Änderung daran ergibt eine neue Version. Evidenzschwellen gehören nicht dazu, weil sie keine Prognose verändern.
   - **Modellinstanz** (`instance()`, `model_id`): Experimentversion, Trainingsstichtag, Hash der verwendeten Label-IDs (Signal, Kontrakt, Mark-Tag), letzter Mark-Tag und Fingerprint der Gewichte. Sie wird append-only in `model_instances.jsonl` registriert. Die Label-IDs sind aus dem Ledger reproduzierbar (Mark-Tag vor dem Stichtag).
4. **Walk-Forward der Auswahlregel:** Je Signaltag wird nur mit Labels trainiert, deren Mark-Tag vor dem Testtag liegt. Für jeden fälligen Kandidaten wird zuerst die Form gewählt; erst danach wird ihr geprüftes Ergebnis betrachtet.
   - **Fällig** ist ein Kandidat, wenn die Mark-Frist des Ledgers abgelaufen ist (Horizont + 10 Tage), unabhängig davon, ob ein Ergebnis vorliegt. Je Aktie und Tag zählt ein Kandidat.
   - **Nicht bewertbar:** Fehlt der gewählten Form ein geprüftes Ergebnis, wird der Fall mit Grund gezählt (`unevaluable`) und nie durch eine andere Form ersetzt. Liegt die Abdeckung unter 80 %, lautet das Verdikt INSUFFICIENT_COVERAGE. Zusätzlich wird das Mittel ausgewiesen, wenn jede Lücke als Totalverlust zählt.
   - **Leistungszeugnis:** (1) Ergebnis jeder Form (`by_expression`), (2) Ergebnis der gewählten Form (`mean_selected`, `chosen_forms`), (3) Unterschied zur Standardform bei gleicher Handelsentscheidung (`vs_default`, gepaart mit Block-Bootstrap).
   - **Repräsentativität:** getrennt nach Forschungsstichprobe und Richtungssignalen sowie nach Richtung und Zeitraum. Warnungen erscheinen, wenn mindestens 80 % aus der Forschungsstichprobe oder aus bullischen Calls stammen oder der Zeitraum unter 90 Tagen liegt.
   - VALIDATED: mindestens 20 Testtage, mindestens 5 unabhängige Perioden, mindestens 30 Fälle mit Prognose > 0, deren Netto-Mittel im 95-%-Block-Bootstrap über 0 liegt, und Rang-IC mit t > 2.
   - **Unabhängige Perioden:** Signaltage werden in nicht überlappende Haltefenster von der Länge des Horizonts (Kalendertage) gebündelt. Positionen im selben Fenster überlappen und gelten nicht als unabhängig. Der Bootstrap zieht ganze Fenster, nicht einzelne Tage. Beim 20-Tage-Modell braucht der Nachweis deshalb mindestens rund 100 Kalendertage.
   - Sonst NOT_VALIDATED oder NEED_MORE_DATA.
5. **Prognose-Journal:** `outputs/research/option_learner/predictions.jsonl`, append-only. Je Kandidat gibt es eine Prognose je Form (`kind=prediction`) und eine Entscheidung (`kind=decision`, TRADE mit genau einer Form oder ABSTAIN). Alles wird geschrieben, bevor das Label existiert, und trägt `experiment_id` und `model_id`. Alte Einträge werden nie geändert oder mit neuen Gewichten neu bewertet.
6. **Forward-Test:** Bewertet die am Signaltag festgehaltenen Entscheidungen (`kind=decision`) mit derselben Auswertung wie der Walk-Forward. Er rechnet nie neu. Ergebnis ist FORWARD_POSITIVE, FORWARD_NEGATIVE, INCONCLUSIVE, INSUFFICIENT_COVERAGE oder NEED_MORE_DATA.
   - Es zählen nur Einträge der aktuellen Experimentversion. Fortlaufend neu trainierte Modellinstanzen derselben Version bilden eine gemeinsame Lernkurve. Andere Versionen werden nie mitgezählt, aber mit Anzahl ausgewiesen (`journal_by_experiment`).
   - Ausgeschlossen und mit Grund gezählt (`excluded`): `OTHER_EXPERIMENT`, `UNKNOWN_MODEL_INSTANCE`, `INSTANCE_EXPERIMENT_MISMATCH`, `NO_TIMESTAMP`, `LATE_PREDICTION` (nicht am Signaltag in UTC geschrieben), `TRAINED_ON_FUTURE` (Instanz kennt Labels ab dem Signaltag) und `DUPLICATE` (die erste Fassung gilt).
   - Die 40 Einträge vom 30.10.2026 stammen aus der Vorversion `298be00d41bc` und zählen deshalb als `OTHER_EXPERIMENT`.
7. **Empfehlung (Freigabe `release_gate`, fail closed):**
   - Nur bei Walk-Forward VALIDATED **und** Forward FORWARD_POSITIVE. NEED_MORE_DATA, INCONCLUSIVE, FORWARD_NEGATIVE oder ein fehlender Bericht sperren.
   - Die Freigabe prüft die Mindestzahlen (Testtage, unabhängige Perioden, Fallzahl, KI über 0, Ø netto > 0, beim Walk-Forward Rang-IC t > 2) aus den Kennzahlen erneut. Ein Verdikt, das nicht zu seinen Zahlen passt, sperrt ebenfalls.
   - Je Kandidat nur bei Prognose > 0 und nur für Call-Strukturen (LONG_CALL, BULL_CALL_SPREAD), die die Produktion handeln kann.
   - Sonst nur eine als UNVALIDIERT gekennzeichnete Research-Watchlist (Top 5) mit den offenen Bedingungen.
   - Mail nur für das 20-Tage-Hauptmodell mit Empfehlungen; `mail()` prüft die Freigabe selbst erneut. Das 5-Tage-Modell ist Frühindikator ohne Handels-Mail.

## Lernfortschritt: lernende Version gegen eingefrorene Referenz

- **Referenz:** Je Experimentversion und Horizont wird die erste trainierte Modellinstanz eingefroren. Ihre Gewichte stehen einmalig in `reference/<h>d_<experiment_id>.json` und werden nie überschrieben. Eine neue Experimentversion beginnt eine neue, eigene Vergleichskohorte.
- **Täglich:** Die lernende Version (täglich nachtrainiert) und die Referenz entscheiden mit derselben Auswahlregel über dieselben Kandidaten. Beide Entscheidungen (inkl. ABSTAIN) stehen im Journal (`role=learning` bzw. `role=reference`).
- **Gleiche Bedingungen:** Gleiche Signale, Kontrakte, Quote-Prüfung, Kosten und Halteperiode. Jeder Kandidat erhält das gleiche Budget (eine Einheit Prämie, ABSTAIN = 0). Je Tag zählt das Mittel über alle Kandidaten des Tages. Fehlt das geprüfte Ergebnis einer gehandelten Wahl, fällt der Kandidat für alle Varianten heraus. Je Aktie und Tag zählt ein Kandidat.
- **Unsicherheit:** 95-%-Block-Bootstrap der mittleren Tagesdifferenz (lernend − Referenz) über unabhängige Haltefenster, nicht getrennte Durchschnittsrenditen.
- **Vorab festgelegte Verdikte:**
  - NEED_MORE_DATA: unter 20 Tagen, 5 unabhängigen Perioden oder 30 Kandidaten. Das ist nie „verbessert“.
  - LEARNING_BETTER bzw. LEARNING_WORSE: das KI liegt ganz über bzw. unter 0.
  - Sonst NO_SIGNIFICANT_DIFFERENCE (statistisch unentschieden).
- **Basisstrategien** unter denselben Bedingungen: „nie handeln“ (0) und „alles handeln“ (Standardform je Kandidat, sofern zulässig).
- Dieselbe Rechnung läuft zusätzlich retrospektiv im Walk-Forward (erste Walk-Forward-Instanz eingefroren). Sie ist nur ein Hinweis; maßgeblich ist der Forward-Vergleich.
- Verlauf (`learning_trend`) und Verdikt stehen im Leistungszeugnis und damit im Montagsbericht.

## Leistungszeugnis und Lernkurve

- **Manifest** je Auswertung: Anzahl und Hash der verwendeten geprüften Labels, Stichtag, Verteilung je Ausdrucksform. Damit ist nachvollziehbar, welche Labels in welches Modell geflossen sind.
- **Historie** `status_history_<h>d.jsonl` (append-only, ein Eintrag je neuem Stand). `vs_previous` vergleicht nur aggregierte Kennzahlen innerhalb derselben Experimentversion und ist rein beschreibend. Der Lernfortschritt wird gepaart gemessen (siehe oben).
- **Vergleich mit einfachen Basisstrategien:** „nie handeln“ und „alles handeln“ auf denselben Kandidaten und Tagen mit gleichem Budget. „Schlägt“ gilt nur, wenn das gepaarte KI über 0 liegt.
- **Strategievergleich:** Mittel je Ausdrucksform auf ungesehenen Tagen (`by_expression`).
- **Filter-Prüfung (`by_gate`, Audit §9):** realisiertes Netto der geprüften Labels je Ablehnungsgrund. Das ist beschreibend und zeigt, ob ein Filter eher schlechte oder auch gute Gelegenheiten verwirft. Eine Gruppe wird erst ab 20 Labels an 10 Tagen ausgewiesen. Die Registrierung des ROI-Shadows ist seit 2026-10-30 stillgelegt (`enabled: false`); seine Population (vom ROI-Gate verworfene Kontrakte) wird hier über die Snapshots im Ledger mitgemessen.
- **`report_card.{json,md}`** beantwortet die Audit-Fragen je Horizont mit Ja, Teilweise, Nein oder offen, jeweils mit Beleg. Es erscheint im Montagsbericht (Abschnitt 12).

## Automatik

| Workflow | Schritt | Wirkung |
|---|---|---|
| `scanner.yml` (täglich, nach dem Feedback) | `recommend --horizon 20 --mail` und `--horizon 5` | Journal, `recommendations/<Tag>_<h>d.{json,md}`, Mail nur bei Empfehlung |
| `ml_research.yml` (Samstag und monatlich) | `evaluate --horizon 5/20` | `status_<h>d.json` mit Walk-Forward, Forward-Test und Koeffizienten |

## Grenzen

- Es entsteht keine Order, und es gibt keinen Einfluss auf Champion, Gates, Sizing, Risk oder Promotion.
- Eine Empfehlung ist eine Research-Aussage auf Basis der eigenen Evidenz, keine Anlageberatung.
- **Datenstand 30.10.2026:** 359 geprüfte 5-Tage-Labels an 8 Signaltagen (14.–23.10.) und noch keine geprüften 20-Tage-Labels. Alle Verdikte lauten deshalb NEED_MORE_DATA.
- Die 40 Journal-Einträge vom 30.10. stammen aus der Vorversion und zählen nicht. Die aktuelle Experimentversion sammelt ab dem nächsten Scanner-Lauf.
- **Frühestens möglich** (bei etwa 40 Kandidaten je Tag, wenn alle Mindestzahlen erreicht werden):
  - 5-Tage-Modell: Walk-Forward- und Forward-Verdikt ab Mitte Dezember 2026, ebenso der gepaarte Vergleich mit der Referenz.
  - 20-Tage-Modell (Hauptmaßstab, einzige Handels-Mail): 5 unabhängige Perioden zu je 20 Tagen plus Fälligkeit (Horizont + 10 Tage) ergeben ein Forward-Verdikt frühestens etwa Mitte März 2027.
