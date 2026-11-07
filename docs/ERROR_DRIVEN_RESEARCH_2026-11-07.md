# Fehlergetriebene Forschung: nächste Stufe des autonomen Lernens (2026-11-07)

Die Stufe arbeitet nur in der Forschung: keine Orders, kein LLM, keine Produktionswirkung.
Grundlage ist `main` bei `e9e35846` mit echten Daten aus `outputs/` vom 07.11.

Protokoll: `config/error_research_protocol.yaml`, **Version 2**. Version 1 (erster Lauf dieses PRs) hatte vier
methodische Schwachstellen; sie sind unten beschrieben. Die einzige v1-Registrierung, **ERR-d6695229**, ist mit
unveränderter Registrierung erhalten (`outputs/research/error_hypotheses.jsonl`, Prüfsumme unverändert). Sie wird
strenger ausgewertet und nie rückwirkend bestätigt. Forward-Daten beginnen am 08.11., kein Forward-Ergebnis war bei
der Umstellung bekannt.

## Kreislauf in den bestehenden Modulen

Ein Lauf ist `python -m modules.research_director --errors`. Er läuft **wöchentlich** (samstags) und im Monatslauf in
`ml_research.yml`, vor dem Director. Ein kritischer Befund liefert Exit-Code 2: Der Schritt wird rot, und über
`ci_stage_check.sh` (CRITICAL) schlägt der ganze Lauf fehl.

| Schritt | Umsetzung |
|---|---|
| 1 Fehleranalyse | `research_director.verified_forecasts`: Jede Ensemble-Prognose im Candidate Ledger wird gegen das `ml_predictions`-Archiv geprüft. Sie muss vor dem Signal erstellt worden und höchstens 8 Tage alt sein; Abweichungen werden ausgeschlossen und gelten als KRITISCH. `forecast_errors` berechnet Rangfehler und Residuum nach Rang-OLS auf die Prognose, zusätzlich eine Fassung nach Größenkontrolle. |
| 2 Hypothesen | Fester Suchraum mit 23 Kandidaten. Eine Spezifikation wird eingefroren: Hash, H0/H1, Baseline, Falsifikation, Prüfplan, Fenstermethode, Datenstand der Entdeckung (`discovery_data_ref`). |
| 3 Datenlücken | Gemeldet werden zu kurze Historie, zu wenige unabhängige Fenster und Segmente mit größerem unerklärtem Fehler (BH). Sie werden auf Katalog-Dimensionen abgebildet. **Wöchentlich** daraus abgeleitet: `data_acquisition` (Vorschläge `propose_data_source`, Freigabe beim Menschen). |
| 4 Priorisierung | `priority()` × Gedächtnis-Prior. Budget: 2 je Lauf, 24 insgesamt, nie mehr als Slots im Testbudget. |
| 5 Test | `research_lab`: unabhängige Fenster (siehe P1). Der Walk-Forward-Screen nach Embargo kann nur falsifizieren. Danach folgen **eingefrorene Forward-Prüfungen** zu festen Zeitpunkten mit Slot-Alpha (siehe P0). |
| 6 Optionen | `option_check` (siehe P0): Auswahl zum Signalzeitpunkt, LONG_CALL und BULL_CALL_SPREAD getrennt, Coverage-Lücken. |
| 7 Gedächtnis | `research_memory.error_entries`: Hypothesen mit eingefrorener Endprüfung, durch Größe erklärte Muster, Datenlücken und Audits. `meta_cognition` zeigt Track, Budget, Audits und kritische Befunde. |

## Die vier Schwachstellen und ihre Behebung

### P0 – Mehrfachtest-Kontrolle über sequenziell eintreffende Forward-Tests

- **Vorher:** BH über alle abgeschlossenen Forward-Tests, in jedem Lauf neu berechnet. Eine spätere Hypothese konnte
  eine frühere Bestätigung nachträglich kippen.
- **Vorhanden und wiederverwendet:** das Verfahren aus `option_learner.test_budget` und `alpha_spending`. Es teilt ein
  festes Fehlerbudget `total_alpha = 0,05` in `max_tests = 24` gleich große Bonferroni-Slots. Jeder Slot verteilt sein
  Alpha nach O'Brien-Fleming auf drei vorab feste Prüfzeitpunkte (4, 8 und 12 unabhängige Fenster). Nach dem
  Union-Bound ist die familienweise Fehlerrate höchstens 0,05, und zwar über die **gesamte Testfamilie** (alle je
  möglichen Registrierungen; `max_lifetime ≤ max_tests` wird beim Laden erzwungen), auch bei abhängigen Tests.
- **Einfrieren:** `outputs/research/error_forward_journal.jsonl` ist append-only mit Hash-Kette. Es enthält das Budget
  (bei der ersten Auswertung eingefroren), die Slots (in Registrierungsreihenfolge) und jede Prüfung mit
  Prüfzeitpunkt, Fenstern und Stichtagen, n, t, p, Teil-Alpha, Verfahren, Entscheidung und Datenstand (Hash über alle
  verwendeten Zeilen, Code-SHA). Eine Entscheidung wird nur gelesen, nie neu getroffen. Eine verletzte Kette ist
  KRITISCH; dann wird nichts angehängt.
- **ERR-d6695229 (v1):** Slot 1. Die Registrierung sah α = 0,05 plus BH vor. Ausgewertet wird jetzt mit
  min(0,05; 0,05/24) = **0,00208** bei einer Prüfung nach 4 Fenstern. Das ist strenger als registriert, nie lockerer.
  Die Registrierung selbst ist unverändert.
- **Folge für die Trennschärfe:** Bei 4 Fenstern (df = 3) verlangt α = 0,00208 einseitig t ≥ 7,94. ERR-d6695229 kann an
  dieser Prüfung praktisch nur falsifiziert (Mittel ≤ 0) oder „nicht bestätigt“ werden. Das ist die ehrliche Folge
  einer korrekten Kontrolle; die Schwellen wurden nicht gelockert.

### P0 – Auswahlverzerrung bei Optionsstrategien

- **Vorher:** Das obere Drittel wurde nur unter Titeln mit bereits geprüftem Ergebnislabel gewählt. Eine fehlende
  Exit-Quote, typisch für wertlose Optionen, nahm den Titel aus der Population.
- **Jetzt:**
  - Die Population je Stichtag und Struktur besteht aus allen Kandidaten, für die zum Signalzeitpunkt ein Kontrakt
    erfasst ist (Ledger: `real_option` bzw. `real_strategy`).
  - Gewählt wird vor jedem Blick auf Labels: das obere Drittel des Signals.
  - Erst danach wird das geprüfte Netto-Label aus `option_learner` gelesen: Kauf zum Ask, Verkauf zum Bid, 0,65 USD je
    Leg und Seite, Kontraktgröße 100, gemessene Haltedauer.
  - Fehlt ein Label, ist das eine **Coverage-Lücke mit Grund** und wird nie ersetzt. Nicht fällige Labels gelten als
    ausstehend.
  - LONG_CALL und BULL_CALL_SPREAD werden getrennt nach derselben Regel ausgewertet.
  - Vergleiche: gegen „alle geeigneten Signale handeln“ und gegen „nicht handeln“, mit Block-Bootstrap über
    unabhängige Fenster.
  - Bei Coverage unter 0,8 oder weniger als 5 Fenstern lautet das Ergebnis INSUFFICIENT_DATA. Explorative Evidenz ist
    nie OPTION_EDGE.
  - Eine bestätigende Prüfung wird eingefroren, sobald alle Labels des festen Fensters fällig sind.

### P1 – Unabhängigkeit der Testfenster

- **Befund:** Die Überlappung besteht. Ein Freitags-Signal hat sein 5-Tage-Label erst am Freitag der Folgewoche, also
  im nächsten Fenster. Bei 20 Tagen reicht ein Signal vom Fensterende fast über das ganze Folgefenster. Benachbarte
  Fenster teilen Renditezeiträume. Die v1-Fenster waren daher **nicht unabhängig**, ihre t-Werte waren nicht
  belastbar.
- **Jetzt** (`research_lab.period_blocks`):
  - Nur jedes zweite Fenster zählt; das dazwischen dient als Purge/Embargo.
  - Ein Stichtag zählt nur, wenn sein Label vor dem nächsten genutzten Fenster feststeht.
  - Fenster sind erst fertig, wenn das Label fällig ist plus 3 Tage Karenz (Feiertage, Füll-Verzögerung).
  - Die Forward-Fenster, Prüfzeitpunkte und Teil-Alphas liegen vor dem ersten Forward-Datum fest.

### P1 – Einfrieren, Audit, Wochenrhythmus

- **Einfrieren:** Abgeschlossene Prüfungen stehen mit Spezifikations-Hash und Datenstand im Journal. Spätere Läufe
  lesen sie nur.
- **Audit bei Datenänderung:** Ändern sich danach Kurs-, Prognose- oder Ergebnisdaten, entsteht genau ein
  Audit-Eintrag (`audit_data_change`) mit altem und neuem Datenstand und einer getrennt gekennzeichneten Neubewertung.
  Die Entscheidung bleibt. Abweichende Prognosen gegenüber dem Archiv gelten zusätzlich als KRITISCH.
- **Wochenrhythmus:** Fehleranalyse, Auswahl, Priorisierung, Prüfungen, Optionsprüfung, Datenentscheidung,
  Gedächtnis-Sync (`rm.sync`) und `meta_cognition` laufen **wöchentlich**.
  - Der monatliche Director (`run()`) braucht das historische Panel. Er liest nur die Fehlerlücken aus
    `error_research.json`.
  - Nichts läuft doppelt. Ein wiederholter Lauf registriert und entscheidet nichts erneut (getestet).

## Echter Lauf (as of 2026-11-07, Protokoll v2)

| | Ergebnis |
|---|---|
| Prognosen | 8.976 im Ledger, **8.590 VERIFIED**, 0 Abweichungen, 386 mit weniger als 3 Modellen. Status OK. |
| 5-Tage-Fehler | 7.342 fällige Fehler an 21 Stichtagen. Nur **2 unabhängige Fenster** (28.09., 12.10.); das Fenster ab 26.10. ist mit Karenz noch nicht fertig. Ensemble-IC +0,030 (t 0,19). |
| 20-Tage-Fehler | 2.963 Fehler, 0 fertige Fenster. |
| Muster | Nicht prüfbar (DATA_GAP): Für die Entdeckung braucht es 5 unabhängige Fenster. |
| Segmentlücken | Keine. Die drei Segmentbefunde aus v1 beruhten auf überlappenden Fenstern. |
| ERR-d6695229 | PENDING_FORWARD. Slot 1, α 0,00208, eine Prüfung bei 4 Fenstern, `max_days` 150. |
| Journal | 2 Einträge: Budget und Slot. |

**Reproduzierbar:** Zwei Läufe mit demselben Datenstand ergeben identische Ergebnisse und kein neues Journal-Ereignis.
Die Registrierung ist byte-gleich, und jede künftige Prüfung lässt sich über ihren Datenstand-Hash nachrechnen.

### Was sich gegenüber v1 statistisch ändert

| Kennzahl (5 T) | v1 (überlappend, ohne Karenz) | v2 (alternierend, gepurgt, Karenz) |
|---|---|---|
| Ensemble-Rang-IC | −0,078, t −1,09, 5 Fenster | +0,030, t 0,19, 2 Fenster |
| `sentiment_score` Residual-IC (Entdeckung) | +0,036, t 3,04, 3 Fenster | +0,044, t 2,80, 2 Fenster (Entdeckung nicht möglich) |
| Segmente Technology / Comm. Services / hohe Short-Quote | t 12,0 / 3,7 / 6,7, BH-signifikant | nicht prüfbar bzw. nicht signifikant |
| `news_count*sentiment_score` „durch Größe erklärt“ | ja | nicht mehr prüfbar, daher kein Gedächtnis-Eintrag |
| Option explorativ (oberes Sentiment-Drittel) | −9,8 pp vs. „nicht handeln“, KI [−17,2; −0,6] | Long Call: 41 gewählt, 25 geprüft (Coverage 0,61, Grund ENTRY_NOT_ARCHIVED), 1 Fenster → INSUFFICIENT_DATA; Spread: 2 gewählt → INSUFFICIENT_DATA |

Die v1-Zahlen für Optionen und Segmente sind damit **nicht haltbar**. Sie stammten aus verzerrter Auswahl bzw. aus
abhängigen Fenstern.

### Offen oder verworfen

- **Offen:** ERR-d6695229 (Forward ab 08.11.).
- **Verworfen:** keine Hypothese. Die v1-Einstufung „durch Größe erklärt“ ist kein Ergebnis mehr.
- **Prospektiver Alpha-Nachweis: nein.**
- **Optionsprofitabilität nach Kosten nachgewiesen: nein.** Die Datenlage reicht nicht (Coverage 61 %, 1 Fenster).

## Wann belastbare prospektive Bewertungen möglich sind

- **ERR-d6695229:** einzige Prüfung im Wochenlauf am **09.01.2027**. Das 4. unabhängige Fenster (21.–27.12.) ist mit
  Karenz am 04.01. fertig. Bei α = 0,00208 ist praktisch nur Falsifikation oder „nicht bestätigt“ erreichbar.
- **Neue Registrierungen (5 T):** frühestens im Lauf am **12.12.2026**, wenn 5 unabhängige Fenster vorliegen.
  Prüfungen folgen nach 4, 8 und 12 Fenstern. Die belastbaren Prüfungen nach 8 und 12 Fenstern liegen etwa
  **April bzw. Juni 2027**.
- **20 T:** Entdeckung frühestens etwa Juli 2027; Prüfungen danach im Abstand vieler Monate.

## Wo Daten oder Evidenz fehlen

- Unabhängige Beobachtungen sind die Engstelle: Ein 5-Tage-Fenster liefert alle zwei Wochen eine Beobachtung.
- Der Optionsarchiv-Start (12.10.) lässt frühe Einstiege ohne Quote: `ENTRY_NOT_ARCHIVED`.
- Pageviews und FINRA-Short-Volume liegen PIT erst ab 15.10. vor.
- Das Ensemble hat forward keinen belegten Skill.

## Offen außerhalb dieses PRs

- Fabrik-Hypothesen können weiterhin über den kontaminierten Locked-Holdout „accepted“ werden. Die Korrektur gehört
  in einen eigenen PR.
- Eine historische Replikation über das Panel gibt es nicht; nur 2 der 12 Ledger-Merkmale haben ein Gegenstück.
