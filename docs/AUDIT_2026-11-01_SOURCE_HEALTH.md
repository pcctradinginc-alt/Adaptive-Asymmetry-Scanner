# Audit 2026-11-01: Source Health – auffällige Datenquellen

Grundlage: Snapshot `outputs/health/source_health_snapshot.json` (2026-11-01 18:05 UTC), Registry-Health
`outputs/external_data/health/source_health.json`, Historie `source_health_history.jsonl`, Git-Historie der
Ingest-Commits 29.10.–01.11., normalisierte Archive, `preflight_report.json`.

**Live-Prüfung:** Aus der Audit-Umgebung nicht möglich. Die Netzwerkrichtlinie blockiert (Proxy-403 bzw. DNS)
`services9.arcgis.com`, `api.e-stat.go.jp`, `www.ncei.noaa.gov`, `fred.stlouisfed.org` und `data.sec.gov`.
Es wird **keine erfolgreiche Reparatur behauptet**. Für PortWatch grenzt der reguläre Ingest-Lauf (GitHub
Actions, mit Netz) einen ArcGIS-Fehler ab jetzt selbst mit minimalen Read-only-Abfragen ein (siehe A3).

Historische Daten, Kritikalitätsklassen, Safe-Mode-Grenzen, Schwellen, Scheduler, Handelslogik, Gates,
Modelle und Forward-Journale sind unverändert.

## A – IMF PortWatch

### Befund

| Datum (Ingest) | Ports | Chokepoints | Disruptions |
|---|---|---|---|
| 28.10. | PASS | PASS | PASS |
| 29.10. | 400 „Invalid query parameters“ | PASS | PASS |
| 29.10. 17:39 | Owner-Fix `35119255`: DATE-Literal für `esriFieldTypeDateOnly` | | |
| 30.10. | 400 | 400 | 400 |
| 31.10. | 400 | `Layer-Metadaten ohne 'fields'` (Dienst ohne Schema) | 400 |
| 01.11. | 400 | **PASS** (5880 Beob., 0 ungelöst) | 400 |

- Alle drei Layer liefen am 30.10. mit demselben Fehler auf. Chokepoints kam am 31.10. ohne Feldliste zurück
  und lief am 01.11. mit **unverändertem Code** wieder. Daraus folgt: Der Anbieter hat die Dienste zwischen
  29.10. und 31.10. umgebaut (Ports: Datumsfeld auf DateOnly, so dokumentiert im Owner-Fix).
- **Disruptions:** Die Abfrage nutzt `where=1=1`, kein `orderByFields` und dieselbe Paginierung wie das
  funktionierende Chokepoints. Variabel ist nur `outFields`. Davon prüfte der Konnektor **6 von 11
  Feldnamen nie gegen die Layer-Metadaten** (`eventtype, eventname, country, n_affectedports, severitytext,
  editdate`). Zwei davon (`n_affectedports`, `editdate`) werden nirgends gelesen. Ein beim Umbau entferntes
  Feld führt bei ArcGIS Online genau zu „Invalid query parameters“.
  → Nachgewiesener Defekt: stillschweigende Schema-Annahme. Dass er die **Ursache** ist, ist plausibel, aber
  ohne Live-Abfrage **nicht bewiesen**.
  **Nachtrag 2026-11-02: durch die Produktionsmessung widerlegt.** Alle 11 Felder existieren
  (`outFields=OK`). Die Feldvalidierung bleibt als Härtung, war aber nicht die Ursache (siehe Nachtrag unten).
- **Ports:** `outFields` stammt bereits vollständig aus den Metadaten. Die gespeicherte Fehlermeldung nennt
  weder die WHERE-Klausel noch den Feldtyp noch die Seite (`resultOffset`). Offline lässt sich die Ursache nicht
  eingrenzen (Datumsliteral, Sortierung oder eine spätere Seite). **Kein Parameter geändert**, weil keiner
  nachweislich fehlerhaft ist.
- **Chokepoints RECOVERING 1/2:** Siehe E1. Mit der alten Recovery-Logik hätte eine zweite Health-Prüfung am
  selben Tag ohne neuen Ingest „HEALTHY“ gemeldet. Beleg: `finra_short_interest` am 27.10. Jetzt ist ein
  neuer erfolgreicher Ingest (frühestens 02.11.) nötig.
- Endpoint-Cache (`portwatch_endpoints.json`, entdeckt 28.10.): Daily_Ports_Data/0, Daily_Chokepoints_Data/0,
  PortWatch_ports/1, PortWatch_chokepoints_database/0. Er ist plausibel; ein Cache-Defekt ist nicht belegt.

### Korrekturen (`modules/external/sources/maritime.py`)

- **A1 Disruptions:** `outFields` = `DISRUPTION_FIELDS` ∩ tatsächliche Layer-Felder.
  - Pflichtfelder (`eventid, fromdate, todate, affectedports, alertlevel`) fehlen → wie bisher
    `SCHEMA_CHANGED` (fail-closed).
  - Ein beschreibendes Feld fehlt → `WARN` (DEGRADED) mit Feldnamen. Der Wert bleibt `None` und wird nie ergänzt.
  - Fehlende ungenutzte Felder werden in der Meldung genannt, aber nicht angefragt.
  - Bei vollständigem Schema ist die Anfrage identisch zu vorher.
- **A2 Fehlermeldung:** Sie enthält jetzt `where`, `orderByFields`, `resultOffset` und `resultRecordCount`
  (öffentliche Query-Parameter, keine Credentials).
- **A3 Isolation:** Nach einem ArcGIS-Fehlerobjekt folgen höchstens 5 Abfragen mit `resultRecordCount=1` ohne
  Geometrie. Jede übernimmt genau **einen** Parameter der fehlgeschlagenen Query (`basis`, `where`,
  `outFields`, `orderByFields`, `resultOffset`); alles andere bleibt minimal (`1=1`, Objekt-ID-Feld laut
  Metadaten `esriFieldTypeOID`).
  - Das Ergebnis steht **am Anfang** der Meldung, z. B. `[Isolation basis=OK where=FAIL(400) …]`, und bleibt
    damit auch in der auf 200 Zeichen gekürzten Snapshot-Meldung sichtbar.
  - Die Abfragen dienen nur der Diagnose. Der Status bleibt `FAIL`, und es werden keine Daten übernommen.
    Auch wenn alle Einzelabfragen OK sind, gilt die Quelle nicht als repariert.
  - Budget: je Abfrage ein Versuch ohne Retries mit 20 s Timeout. Nach 60 s folgen keine weiteren Abfragen
    (`SKIPPED(Budget)`). Damit bleibt alles innerhalb des Abrufbudgets der Quelle (Ports 600 s, sonst 300 s).
  - Lesehilfe: `FAIL` bei genau einem Parameter zeigt den fehlerhaften Teil der Query. `basis=FAIL` bedeutet,
    dass der Layer gar keine Abfrage annimmt (Dienst- oder Layer-Problem). Sind alle Einzelabfragen OK, liegt
    der Fehler in der Kombination; dann ist ein Folge-Audit nötig.
  - Das gilt für die Ports-, Chokepoints- und Disruptions-Hauptabfragen.

## B – e-Stat (`estat_jp_truck`)

- **Statistik:** `00600330` 自動車輸送統計調査, Tabelle `0003422293` (monatlich, ab 2020-04). Live bestätigt im
  Preflight (Sept. 2026) und seitdem täglich identisch gewählt.
- **Timeout:** `getStatsList` lief am 21.10. und 01.11. in einen Read-Timeout (30 s). An den übrigen Tagen
  antwortet die API in rund 2 s (Rohdaten-Metadaten 30./31.10.). `http.fetch` versucht es bereits dreimal mit
  Backoff (2 s, 4 s).
  - Der Ausfall ist anbieterseitig und sporadisch. Ein Code-Defekt ist nicht belegt.
  - Deshalb keine Änderung: Ein weiterer Retry würde nur den Lauf verlängern.
- **STALE seit 27.10.:** Jüngste Periode ist 2026-03 (Tabelle aktualisiert am 17.09.2026, 483 KB, an allen
  Tagen gleich groß). Die Registry-Grenze `max_staleness_days: 240` ist überschritten.
  - Das ist eine **echte Veröffentlichungslücke** des Anbieters (April 2026 noch nicht veröffentlicht) und
    kein technischer Fehler. Status und Grenze bleiben unverändert.
- **Paginierung:** Rund 14.800 Werte je Abruf, unter dem e-Stat-Limit von 100.000 je Antwort. Kein Fall von
  Abschneiden.
  - Latente Lücke: Ein späteres `NEXT_KEY` würde nicht ausgewertet. Nicht geändert, weil nicht eingetreten.

## C – NCEI Normals (`ncei_normals`, Missing-Rate 35 %)

Zehn Abrufe sind identisch (seit 27.09.; 30-Tage-Sperre `min_refetch_days`). Die fehlenden Werte sind
**systematisch** (31.025 von 87.600 = 35,4 %):

| Ursache | Umfang |
|---|---|
| `DLY-PRCP-NORMAL` wird an **keiner** der 40 Locations geliefert, auch nicht an der Flughafenstation USW00023169 (LAS) | 40 × 365 = 14.600 |
| 6 Stationen liefern **keinen** der 6 Datentypen: USC00090444 (ATL, ATL_METRO, GA_CONSTR), USC00315913 (CLT, NC_CONSTR), USC00411490 (DFW), USC00414321 (HOU), USC00163829 (LA_COAST), USC00121814 (SDF_UPS) | 9 Locations × 5 × 365 = 16.425 |

- Parser und Vollständigkeitsberechnung sind korrekt: Ein angefragter, aber nicht gelieferter Wert ist
  `None` und zählt als fehlend. Die Quote ist exakt nachgerechnet.
- Wahrscheinliche Ursachen:
  - Der Datentypname ist im Datensatz `normals-daily-1991-2020` nicht gültig.
  - Die Stationssuche (CDO `NORMAL_DLY`) liefert Stationen ohne Normalwerte für 1991–2020.
- Beides lässt sich ohne Live-Zugriff auf die NCEI-Dokumentation bzw. -API **nicht verifizieren**. Es gibt
  keine Ersatz-Feldnamen, keine Ersatzstationen und keine geschätzten Werte.
- Folge: `precip_in_anomaly` ist überall `None`. Temperatur-, HDD- und CDD-Anomalien fehlen an 9 Locations
  (ehrlich `None`).
- Status DEGRADED bleibt.
- Owner-Aktion: den Datentypnamen in der NCEI-Dokumentation prüfen; für die 6 Stationen eine Station mit
  1991–2020-Normalwerten verifizieren.

## D – Weitere Quellen

| Quelle | Befund |
|---|---|
| `noaa_storm_events` | HEALTHY. Bewusst als Jahresaggregat des Vorjahres (`observation_time` = 01.01.2025, Grenze 800 T). Wiederabruf erst nach 30 T (nächster ab ca. 16.11.). Kein Defekt. |
| SEC / GLEIF / TED | HEALTHY (`alt`). Laufen wöchentlich in `alt_data.yml` (Fr., zuletzt 30.10.). Im External-Ingest bewusst `DEFERRED: OWN_WORKFLOW`, also keine Doppelerfassung. |
| `vix_fred` | HEALTHY. Nur Fallback (`fallback_only`); wird nur aktiv, wenn `vix_level` nicht HEALTHY und `vix_fred` HEALTHY ist. Derzeit sind beide HEALTHY, kein Fallback aktiv. |

**UNVALIDATED (8), Ursachenklassen** – alle bewusst inaktiv, nichts automatisch eingeschaltet:

| Klasse | Quellen |
|---|---|
| Bewusst deaktiviert (dauerhaft FAIL im Preflight) | `destatis_truck_toll_download` (EXDAT ohne direkten Downloadlink; GENESIS-Quelle läuft) |
| Zurückgestellt: kein verifizierter Endpoint / Key fehlt | `nbs_cn`, `mot_cn`, `kosis_kr` (KOSIS_API_KEY) |
| Zurückgestellt: bewusst nicht laufend geladen | `fhwa_faf` (Exposure per Builder-Skript), `ecmwf_open_data` (nur Registry-Eintrag) |
| Prüfung erforderlich (Endpoint und Lizenz nicht verifiziert) | `viapass_be`, `asfinag_at` |

## E – Monitoring und Recovery

- **E1 Recovery-Doppelzählung (`modules/source_health.py`, nachgewiesen):** Registry- und Alt-Quellen lesen
  nur den Ingest-Zustand. Zwei Health-Prüfungen nach **demselben** Ingest zählten als zwei bestandene
  Prüfungen.
  - Beleg: `finra_short_interest` am 27.10. – `last_success` 12:16:40 in beiden Prüfungen (16:36 RECOVERING
    1/2, 18:14 HEALTHY).
  - Jetzt zählt eine Recovery-Prüfung nur, wenn `last_success` seit der Vorprüfung fortgeschritten ist.
    Probe-Quellen fragen in jeder Prüfung live ab und sind unverändert.
  - Ohne neuen Abruf bleibt die Quelle bei DEGRADED/RECOVERING, auch mit 0 Durchgängen. Das ist **strenger,
    nicht lockerer**: `recovery_passes_required` (2) bleibt unverändert.
  - Damit ist auch ein zweiter Pfad geschlossen: Quellen mit `min_refetch_days` werden nach einem Fehlschlag
    als PASS „übersprungen“, weil `last_success` (vor dem Fehler) jünger als die Sperre ist. Mit der alten Logik
    wären sie so ohne einen einzigen neuen Abruf wieder HEALTHY geworden.
  - Folge: Release- oder sperrgesteuerte Quellen (`is_due`, `min_refetch_days`) bestätigen ihre Recovery erst
    mit dem nächsten echten Abruf. Bis dahin bleiben sie DEGRADED, aber nutzbar (DEGRADED blockiert kein
    Feature).
- Es gibt keine neue Architektur, keinen neuen Scheduler und keine geänderten Kritikalitäten oder Schwellen.

## Datenqualität

Gewichtete Datenqualität laut Snapshot vom 01.11.: **0,90** (Safe Mode aus). Statuszählung: HEALTHY 42,
DEGRADED 2, STALE 1, BROKEN 2, UNVALIDATED 8.

Durch diesen Audit kommen **keine zusätzlichen gültigen Daten** hinzu. Deshalb wird keine Verbesserung
ausgewiesen.

Eine Verbesserung ist erst belegbar, wenn eine der folgenden Bedingungen eintritt:
- Disruptions oder Ports liefern im regulären Lauf wieder Daten.
- Chokepoints bestätigt mit einem zweiten neuen Ingest.

Die Korrektur E1 kann den angezeigten Wert kurzfristig nur senken, nie erhöhen.

## Tests

`tests/test_source_stabilization_2026_11.py` (12 Tests, alle ArcGIS-Antworten Fakes ohne Netz). 10 davon
schlagen auf dem alten Code (`8c670d1e`) nachweislich fehl:
- Recovery aus demselben Ingest, Recovery ohne neuen Abruf, Recovery über einen Neustart (persistierter
  Snapshot, wiederholte Läufe)
- Disruptions mit fehlendem Feld (2 Tests, inklusive Point-in-Time)
- Isolation (5 Tests: Eingrenzung, Seitenoffset, Netzfehler, Timeout und Budget, „Diagnose ist keine
  Reparatur“)

Die beiden übrigen sichern unverändertes Verhalten ab (Abbruch der Recovery bei Fehlschlag, identische Anfrage
bei vollständigem Schema).

## Nachweisstand

| Ebene | Inhalt |
|---|---|
| Technisch getestet | Recovery-Verschärfung, Disruptions-Feldvalidierung, Isolation (Fakes, reproduzierbar) |
| Durch Live-Abfrage bestätigt | **nichts**, da Live-Zugriff aus der Audit-Umgebung blockiert ist |
| Weiterhin offen | Ports-Ursache (Isolation im nächsten regulären Ingest-Lauf), Disruptions-Wiederherstellung, Chokepoints-Bestätigung (zweiter neuer Ingest), e-Stat-Veröffentlichung 2026-04, NCEI-Datentyp und -Stationen (Owner) |

## Nachtrag 2026-11-02 – erste Produktionsmessung (Ingest 15:59 UTC, Code aus #155)

| Quelle | Gemessen | Bewertung |
|---|---|---|
| `imf_portwatch_disruptions` | FAIL, `[Isolation basis=OK outFields=OK]`. Hauptabfrage mit `where=1=1`, alle 11 Felder, `resultOffset=0`, `resultRecordCount=2000` → 400 | Die Feld-Hypothese ist **widerlegt**. Der einzige ungeprüfte Unterschied zu den erfolgreichen Probes sind die **Paging-Parameter** `resultOffset` und `resultRecordCount`. Diese testete die Isolation bisher nicht, eine Lücke im Diagnose-Design. |
| `imf_portwatch_ports` | SCHEMA_CHANGED: `Layer-Metadaten ohne 'fields'` | Der Dienst wird beim Anbieter umgebaut (wie Chokepoints am 31.10.). Heute war keine Isolation möglich. |
| `imf_portwatch_chokepoints` | PASS, 5488 Beobachtungen, neues `last_success` 02.11. 15:59 | Die Health-Prüfung um 16:06 lief **vor** dem Ingest-Commit (16:08) und blieb korrekt bei „RECOVERING 1/2: kein neuer erfolgreicher Abruf“. Der Recovery-Fix aus #155 wirkt also im Produktivlauf; der alte Code hätte hier HEALTHY gemeldet. Die Bestätigung kommt mit der nächsten Health-Prüfung. |
| `estat_jp_truck` | PASS (4320 Beobachtungen), jüngste Periode weiter 2026-03 | Der Timeout vom 01.11. war vorübergehend. Die Veröffentlichungslücke besteht weiter. |
| `ncei_normals`, `noaa_storm_events` | übersprungen (`min_refetch_days`) | unverändert |

### Folge-Fix (eigener PR)

- **Isolation:** Die Probe `paging` prüft `resultOffset` und `resultRecordCount` wie gesendet (nur
  Objekt-ID). Sie ersetzt die bisherige Probe `resultOffset`; es bleiben höchstens 5 Probes.
- **Paginierung:** Gesteuert durch die Layer-Metadaten (`advancedQueryCapabilities.supportsPagination`).
  - Meldet der Layer ausdrücklich `false`, gibt es eine Abfrage **ohne** Paging-Parameter, wie es die
    ArcGIS-Spezifikation verlangt.
  - Kommt dann `exceededTransferLimit`, gilt das Ergebnis als unvollständig: `FAIL`, keine Teil-Daten.
  - Ist der Wert unbekannt oder `true`, bleibt das Verhalten unverändert.
  - Die Fehlermeldung nennt jetzt `supportsPagination`.
- **Nicht bewiesen:** Ob der Disruptions-Layer `supportsPagination=false` meldet, ist nicht gemessen. Der
  Fall ist Teil des Folge-Fixes, aber ausdrücklich nicht als Reparatur deklariert. Der nächste Lauf zeigt,
  ob die Paginierung aus den Metadaten greift oder ob die Probe `paging` die Ursache weiter eingrenzt.
