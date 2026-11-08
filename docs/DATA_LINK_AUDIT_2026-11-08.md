# Verknüpfungs-Audit 08.11.2026: Erreichen die Daten ihre auswertenden Programme?

Stand: `main` 0d3cfeae (enthält #174–#178). Nur Research: keine Orders, kein LLM, keine Produktionswirkung.

## Methode

- **Artefakte:** Jedes Artefakt wurde zu seinen Konsumenten verfolgt (Grep auf Dateistamm und Verzeichnis, dynamische Pfade von Hand).
- **Quellen:** Jede der 57 registrierten Quellen (48 aktiv, `config/external_sources/*.yaml`) wurde zu den Lesern ihrer Quell-ID bzw. ihrer Metriken verfolgt, außerhalb der Ingestion (`modules/external/sources/`).
- **Ledger:** Die Abdeckung im Candidate Ledger wurde je Tag ausgezählt.
- **Meldeweg:** Der Weg vom Forschungsergebnis zur Mail wurde über alle Aufrufer von `mailer.send_mail` geprüft.

## Behoben in diesem Branch

### 1. Externer Kontext erreichte seit 28.10. fast keine Ledger-Zeile mehr

| Tag | Ledger-Zeilen | davon mit `external` |
|---|---|---|
| 22.10. | 382 | 79 |
| 27.10. | 340 | 50 |
| **28.10.** | **393** | **1** |
| 04.11. | 422 | 2 |
| 06.11. | 168 | 2 |

**Ursache:** Commit 5b1665f4 vom 28.10. („Economics-first LLM acquisition“) stellt `economics_first.select_before_haiku` vor Stufe 2b-ext. Diese hängt den externen Kontext nur an die verbliebene Shortlist (am 05.11. lauteten 363 von 415 Zeilen `reject_stage = economics_first`).

**Betroffen:** alle Quellen, die über `modules/external/context.py` laufen (Fracht, PortWatch, Wetter, Energie, Eurostat). Ihre Auswerter sind `factor_monitor` (Merkmale `ext.*`), `external/research.py`, `alpha_discovery` und `engine_monitor`.

**Fix (`pipeline.py`):**
- Der Snapshot wird einmal je Lauf in Stufe 1b-ext gebaut, vor dem Sektor-Check und vor economics-first.
- Für **jeden** angereicherten Kandidaten landet er **nur im Ledger**: `candidate_ledger.note(..., external=ctx)`.
- Der Kandidat bleibt unverändert, es wird kein `candidate["external_context"]` gesetzt. So gelangt nichts in Prescreener- oder LLM-Eingaben.
- Stufe 2b-ext nutzt denselben Snapshot. Der Ledger-Wert ist eingefroren, deshalb entsteht kein zweiter Wert.
- `stats.external_context.ledger_noted` zeigt die Zahl der notierten Kandidaten je Lauf.

Vor dem 28.10. erreichte der Kontext nur die Prescreener-Überlebenden (etwa 20 % der Zeilen). Jetzt erreicht er alle angereicherten Kandidaten. Das verringert zugleich die Auswahlverzerrung der Auswertungen.

Folgen für die Auswertungen:
- Die Population der Auswertungen über `external` wird ab dem ersten Lauf mit diesem Fix breiter. Ergebnisse vor und nach dem 28.10. bzw. nach dem Fix sind deshalb getrennt zu lesen.
- Die Shadow-Relation (`external.relation`, LLM-Shadow) gibt es weiterhin nur für die Shortlist. Alle übrigen Zeilen tragen `source: skipped`, wie bisher.
- Stufe 2b-ext hängt genau das bereits notierte Objekt an. Die Shadow-Relation kommt deshalb wie vor dem 28.10. im Ledger-Eintrag an. Das war vorher nur über eine geteilte Referenz gesichert und ist jetzt per Test abgesichert.
- Kosten:
  - Laufzeit vernachlässigbar (400 Kandidaten in 0,03 s; der Snapshot wird wie bisher einmal gebaut).
  - Ledger-Größe: Ein `external`-Eintrag hat im Schnitt etwa 3,6 KB (Oktober 2026). Bei etwa 320 zusätzlichen Zeilen je Tag sind das etwa 1,1 MB Rohdaten mehr pro Tag. In Git fällt das komprimiert deutlich kleiner aus.
  - Wird das zu groß, wäre die Abhilfe, die laufgleichen Snapshot-Primitive einmal je Lauf abzulegen und je Zeile nur `snapshot_id` und die tickerspezifischen Teile zu speichern (neue Ledger-Version).

**Tests** (`tests/test_external_context_integration.py`):
- ganze Kohorte im Ledger, Kandidaten byte-gleich;
- nur ein Snapshot-Bau;
- Fehler beim Snapshot-Bau wird abgefangen;
- `mode: off` bleibt aus;
- struktureller Guard: die Notiz steht vor Sektor-Check und economics-first.

### 2. Ergebnisse der fehlergetriebenen Forschung wurden nie gemeldet

**Befund:** `research_director --errors` schreibt `error_research.json` und das Gedächtnis (`FORWARD_VALIDATED`). Die Mails (`--notify`) bauten Aufgaben aber nur aus `option_learner`, `release_gate`, `source_health`, `workflow_status` und `autonomous_alpha/option_validation.json`. Eine forward bestätigte oder falsifizierte Hypothese wie ERR-d6695229 hätte dich nie erreicht. Dasselbe galt für eine verletzte Hash-Kette: Der Workflow-Schritt wurde rot, aber es kam keine Mail.

**Fix (`modules/research_director.py`):** Die Anbindung nutzt den bestehenden Aufgaben- und Mail-Weg, also Dedupe, `UNKNOWN`/`RESOLVED`, Wochenlimit und menschliche Entscheidungen.

| Ereignis | Aufgabe | Weg |
|---|---|---|
| `FORWARD_CONFIRMED` | `error_hypothesis_confirmed`, P1 | Ergebnis-Mail mit den 10 Abschnitten, wöchentlich (Samstag, direkt nach `--errors`) |
| `FORWARD_REJECTED` / `FORWARD_NOT_CONFIRMED` | `error_hypothesis_result`, P3 | dieselbe Ergebnis-Mail, im Wochen-Digest |
| `health = CRITICAL` (Journal, Prognosen, Registrierung) | `error_research_integrity`, P0 | täglich 21:37 UTC (`improvement_notify.yml --critical`) |

- **Keine Mail** bei laufenden Prüfungen (`PENDING_FORWARD`), `INCONCLUSIVE`, Datenlücken oder einem veralteten bzw. fehlenden Bericht (älter als 10 Tage).
- **Höchstens eine weitere Mail** je Ergebnis, und zwar nur, wenn sich der Optionsstatus ändert.
- **Einstufung „für menschliche Prüfung qualifiziert (keine Freigabe)“** nur bei bestätigender `OPTION_EDGE`. Sonst lautet sie ausdrücklich „kein Options-Alpha-Nachweis nach Kosten“.

**Tests** (`tests/test_error_driven_research.py`, synthetischer Kreislauf): Bestätigt ergibt P1, falsifiziert ergibt P3, laufende Prüfungen erzeugen keine Mail. Außerdem: alle 10 Abschnitte vorhanden, keine Wiederholung, nicht im P0-Lauf, Integrität als täglicher P0 mit RESOLVED nur nach OK-Beleg, veralteter Bericht ohne Mail.

**Echte Daten 08.11.:** keine Aufgabe, denn ERR-d6695229 ist `PENDING_FORWARD`. Die erste Prüfung fällt im Wochenlauf am 09.01.2027. Fällt sie positiv oder negativ aus, kommt die Mail am selben Tag.

## Offen (belegt, nicht in diesem Branch)

### B. Quellen ohne auswertenden Leser

Diese Quellen werden ingestiert und auf Gesundheit geprüft, aber kein Analyse-, Feature- oder Research-Modul liest ihre Werte:

| Quelle | Leser außer Ingestion |
|---|---|
| `bea_release_calendar`, `bls_release_calendar`, `fomc_calendar` | keiner |
| `clinicaltrials_sponsors`, `usaspending_contracts`, `sec_13f_holdings` | keiner |
| `fred_liquidity` (`us_on_rrp`, TGA, `us_curve_10y_2y`) | keiner; `world_model` nutzt nur `fred_regime_macro` |
| `noaa_storm_events` | nur `orchestrator` (Ingestion) |
| IMF-Familie | nur `imf_macro_research` (Bereitschaft; Alpha `NOT_EVALUATED`) |

`finra_short_volume` erreicht nur die fehlergetriebene Forschung (fester Suchraum).

**Empfehlung:**
- Diese Quellen nur über eine neue, vorab registrierte Protokollversion in einen bestehenden Auswerter aufnehmen, zum Beispiel als Ereignis-Merkmale (Termin-Nähe, Auftragsvergabe, 13F-Änderung) im Suchraum der fehlergetriebenen Forschung. Den laufenden Suchraum v2 nicht ändern.
- Oder die Ingestion bewusst stilllegen.

### C. Verwaistes Modul

`modules/research_candidate_census.py` wird von keinem Modul importiert und von keinem Workflow aufgerufen. Sein Ausgabeverzeichnis `outputs/research/candidate_census/` existiert nicht. Neun Skripte ohne Workflow sind Migrationen oder manuelle Werkzeuge; das ist in Ordnung.

## Meldewege an dich (Ist-Zustand nach diesem Branch)

| Weg | Inhalt | Zeitplan |
|---|---|---|
| `research_director --notify` | Verbesserungsaufgaben, Alpha-to-Options-Ergebnisse, **Ergebnisse der fehlergetriebenen Forschung** | Samstag (Digest P1–P3), täglich (P0) |
| `option_learner recommend --mail` | Options-Empfehlungen nur bei validiertem Modell und bestandener Freigabe | täglich (Scanner) |
| `hc_scanner --send` | Hochkonfidenz-Alerts (dedupliziert) | `ml_research` |
| `autonomous_alpha.paper_report` | Paper-Konto-Wochenbericht | wöchentlich (`ml_research`) |
| `promotion_controller --notify` | Promotion-Kandidaten | Feedback-Lauf |
| `source_health` | Datenquellen nur bei relevanter Änderung | Source-Health-Lauf |
| `ops_report --mail` | Betrieb | täglich (Scanner) |
