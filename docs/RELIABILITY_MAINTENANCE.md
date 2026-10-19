# Reliability-Maintenance (2026-10-09)

Scope: Workflow-Betrieb absichern. Unverändert: Alpha-Logik, Champion, Promotion Policy, Gate-Schwellen,
Drift-Schwellen, Forward-Starts, ROI-/Stop-Logik, Produktionsentscheidungen.

## 1. Runner
Alle Jobs in `.github/workflows/*.yml` laufen auf `ubuntu-24.04` (vorher `ubuntu-latest`, das ab 2026-10-19
auf Ubuntu 26 wechselt). `compat_ubuntu26.yml` ist ein nicht-produktiver Kompatibilitätstest auf `ubuntu-26.04`
(mittwochs + manuell): installiert Abhängigkeiten und führt pytest aus. `permissions: contents: read`, keine
Secrets, kein Commit. Der Test darf fehlschlagen (`continue-on-error`) und zeigt nur, ob ein Wechsel sicher wäre.

## 2./3. Reihenfolge und Freshness-Gate
| Kante | vorher | jetzt |
|---|---|---|
| Source Health → Scanner | nur Cron-Versatz (12:41 → 13:30) | `workflow_run` (Source Health completed) + Guard-Job `upstream_guard.py scanner`; Cron bleibt als Fallback und Startfenster |
| External Data → Source Health | Cron-Versatz | bewusst **kein** Pflicht-Upstream: External Data ist Observability (`external_context.mode != production`). Source Health meldet veraltete Quellen selbst als STALE; fehlende Läufe fängt der Watchdog |
| Scanner ↔ Feedback | Concurrency-Gruppe `history-write` | unverändert |

Der Guard (`config/workflow_schedule.yaml`, Abschnitt `gates`) liefert:
- `RUN`: Source-Health-Snapshot ist höchstens 12 h alt und hat keinen Zeitstempel in der Zukunft.
- `UPSTREAM_NOT_READY`: Snapshot fehlt, ist zu alt oder hat einen Zeitstempel in der Zukunft. Der Scanner startet nicht. Der Grund erscheint als `::warning::` und in der Step-Summary, nie still.
- `MISSED_WINDOW_CLOSED`: nach dem spätesten Start (Sitzungsschluss − 60 min, 15:00 ET; verkürzte Sitzung 12:00 ET). Kein verspäteter Produktionsscan, `::warning::` mit Grund (seit 2026-10-19).
- `SKIP_BEFORE_WINDOW`: vor 13:30 UTC. `workflow_run` darf den Scanner nicht vor seine geplante Handelszeit ziehen, sonst gelten andere Quote-Bedingungen.
- `SKIP_ALREADY_RAN`: `outputs/daily_reports/<heute>.json` existiert. Darauf beruht die Idempotenz: kein zweiter Scan am selben Tag.
- `SKIP_NOT_TRADING_DAY`: Wochenende.
- External Data wird nur gemeldet (`advisory`) und blockiert nicht.
- `workflow_dispatch` mit `force=true` übersteuert den Guard. Das wird protokolliert.

Wirkung bei heutiger GitHub-Verzögerung: Source Health läuft gegen 19:00 UTC, der Scanner direkt danach statt
gegen 20:00 UTC. Ohne Verzögerung bleibt der Scanner bei 13:30 UTC.

## 4. Missed-Run-Watchdog
`watchdog.yml` läuft alle 2 h und ruft `scripts/workflow_watchdog.py` auf. Für External Data, Source Health,
Scanner und Feedback gilt:
- **Versorgt** heißt: Das Tagesartefakt existiert, z. B. der Tagesreport oder ein Snapshot von heute. Ein grüner Run genügt nicht, denn ein Guard-Skip ist grün.
- **MISSED** heißt: Die Frist `cron + max_delay_hours` ist überschritten, der Tag ist nicht versorgt und kein Run ist aktiv. Dann folgt genau ein `workflow_dispatch`.
- **Höchstens ein Recovery-Lauf je Workflow und UTC-Tag.** Der Zähler steht in `outputs/state/workflow_status.json`; zusätzlich zählen die `workflow_dispatch`-Runs des Tages. Danach gilt `RECOVERY_EXHAUSTED` ohne weiteren Retry.
- Fehlt ein Pflicht-Upstream (Scanner → Source Health), wird der Downstream nicht angestoßen (`stale_upstream`).
- Ist die API nicht lesbar, ist der Zustand `UNKNOWN` und es wird nie blind dispatcht.
- Idempotenz:
  - Scanner: über den Guard.
  - External Data: Das Archiv dedupliziert nach Identität und Wert.
  - Source Health: überschreibt den Snapshot.
  - Feedback: läuft ohnehin zweimal täglich.
- Grenze: Der Watchdog ist selbst ein Cron. Fallen alle Schedules aus, fällt auch er aus; im Report ist das an fehlenden Tagen erkennbar.

### 4a. Recovery-Härtung nach dem Ausfall am Mo 19.10.2026
Nachgewiesen (Actions-Historie, `outputs/state/workflow_status.json`):
- Der Source-Health-Cron (12:41 UTC) kam bis 18:29 UTC nicht. Die Scanner-Crons (13:30/14:30) kamen ebenfalls nicht; selbst wenn, hätte der Guard `UPSTREAM_NOT_READY` geliefert (Snapshot vom 18.10. 16:06, > 12 h).
- Der Watchdog kam nur zweimal (04:48, 13:26 statt alle 2 h). Um 13:26 war alles `PENDING`, also `recovered=False`.
- Die Frist `12:41 + 9 h = 21:41 UTC` lag hinter dem Börsenschluss (20:00 UTC). Selbst ein pünktlicher Watchdog hätte Source Health erst nachbörslich angestoßen. Der Scanner wäre dann per `workflow_run` mit nachbörslichen Kursen gestartet, denn der Guard kannte kein Fensterende. Das ist am 05.10. (17:08 ET) und 12.10. (17:24 ET) bereits passiert.

Änderungen:
- Fristen: Source Health 9 h → 3 h (15:41 UTC), Scanner 9 h → 4 h (17:30 UTC). Die Scanner-Frist liegt nie später als der späteste Start − 1 h (`workflow_health.deadline`), z. B. 16:00 UTC an verkürzten Sitzungen.
- Spätester Start `gates.scanner.latest_start_before_close_minutes: 60`. Danach liefert der Guard `MISSED_WINDOW_CLOSED`; der Watchdog meldet `MISSED` mit `window_closed` und dispatcht nicht mehr.
- Watchdog-Auslösung: Zusätzlich zum 2-h-Cron läuft er stündlich 13–20 UTC an Werktagen und per `workflow_run` nach Options-Capture (6 Crons je Werktag) und External Data. Kein Retry-Kreis: Source Health, Scanner und Feedback lösen den Watchdog nicht aus, und jeder Workflow erhält höchstens einen Recovery-Dispatch pro Tag.
- Kette bei ausgefallenem Cron: Der Watchdog dispatcht Source Health. Dessen Abschluss startet den Scanner über `workflow_run`. Feuert das nicht, dispatcht der nächste Watchdog-Lauf den Scanner nach 17:30 UTC.
- Nicht versorgte Tage erzeugen `::warning::` mit Grund (`MISSED`, `RECOVERY_EXHAUSTED`, `UNKNOWN`) und eine Tabelle in der Step-Summary.
- Tests: `tests/test_workflow_recovery.py`.

## 5. Report
Der Montagsreport zeigt unter SYSTEM STATUS den Block **WORKFLOW STATUS** mit diesen Spalten:
- erwartet
- letzter Lauf
- Verzögerung (Median/Max)
- recovered
- verpasst
- Tage mit stale Upstream

Die Drift-Zeile zeigt Stand und Alter des Drift-Inputs.

## 6. Haiku vs. Sonnet
Abgeschnittene Antworten (`stop_reason == max_tokens`) erhalten `comparison_status = TRUNCATED`. Sie zählen
weder als Modellfehler noch in Gate-Agreement oder Routing-Entscheidung und werden getrennt berichtet
(`model_routing.comparison_report`). Ältere Paare werden über die LLM-Zeilen desselben Laufs nachträglich
gekennzeichnet.

Die Produktion bleibt bei 1600 Tokens. Nur der Challenger (Haiku) im A/B-Sample bekommt 2400 Tokens
(`max_tokens_by_model`). Begründung: gültige Haiku-Antworten haben im Mittel 1478 Output-Tokens; 8 von 18 liefen
in das Limit. Die Kostenwirkung beträgt höchstens 800 Output-Tokens × 5 $/MTok = 0,004 $ je A/B-Call, bei
6 Samples/Tag also ≤ 0,75 $/Monat. Es gibt keinen erzwungenen Modellwechsel; der Modus bleibt `auto` mit den
unveränderten Schwellen.

## 7. Wetter-Partitionierung
`nws_forecast` schreibt in Wochen-Partitionen `normalized/nws_forecast/YYYY-MM-wN.jsonl` (N = Tag 1–7 → w1, …).
Konfiguriert ist das über `normalized_partition: weekly`; alle anderen Quellen bleiben monatlich.
- Leser (`load`/`as_of`) lesen weiterhin alle `*.jsonl` einer Quelle. Das ist für Consumer transparent.
- Die bestehende Monatsdatei `2026-10.jsonl` bleibt unverändert als Altbestand und wächst nicht weiter.
- Neue Zeilen werden auch gegen sie dedupliziert.
- Offload nutzt `path.stem[:7]` als Monat.
- Größe:
  - gemessen w1 (1.–7. Okt.) = 15,7 MB;
  - erwartete Obergrenze je Wochen-Partition ≈ 16 MB, w5 (29.–31.) ≈ 7 MB;
  - Monatsdatei dagegen ≈ 65 MB.
- Option ohne Migration: gzip über den vorhandenen Object-Storage-Offload geschlossener Monate.

## 8. Drift
`drift_state` enthält zusätzlich drei Kennzeichnungsfelder:
- `drift_input_timestamp`: `generated` von `meta_learning.json`;
- `drift_input_age_days`;
- `drift_input_status`: `FRESH` / `STALE_DRIFT_INPUT` (> 8 Tage) / `UNKNOWN`.

Drift-Grenzen, Level, Konsequenzen und Promotion-Pause sind unverändert. Das Alter allein ändert den
State-Fingerprint nicht.

## 9. Alt-Data / 10. Commodity
Die Ursache des Alt-Data-Langläufers und der Timeout-Schutz liegen im Maintenance-PR (Entity-Resolution-Build:
Budget, Fortschritts-Log, atomare Writes, Status PARTIAL_*). Commodity bleibt unverändert und wird nicht beschleunigt.
