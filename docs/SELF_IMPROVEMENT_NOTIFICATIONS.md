# Self-Improvement Notifications – Claude-Code-Aufgaben per Mail

Stand 2026-10-31 (inkl. täglicher P0-Prüfung und Status UNKNOWN). Erweitert `research_director` (Erkennung, Vorlagen, Versand), `research_memory` (Einträge `dev_task`), `meta_cognition` (offene Aufgaben) und die Workflows `ml_research.yml` (Wochen-Digest) und `improvement_notify.yml` (täglich P0, manuell). Es gibt keinen zweiten Orchestrator. Versand über `modules/mailer.py` mit den bestehenden Mail-Secrets. Tests: `tests/test_improvement_notifications.py`.

## Was das System selbst erkennt

Jede Aufgabe entsteht aus einem **belegten** Befund in den vorhandenen Research-Ausgaben. Vermutungen ohne Beleg, normale Wartezeit (NEED_MORE_DATA, AWAITING_LOOK, laufende Varianten) und reine Faustregeln erzeugen keine Aufgabe.

| Art | Quelle | Bedingung | Priorität |
|---|---|---|---|
| `empty_feature` | `improvement.json` (Lücke `sparse_feature`) + Kandidaten-Ledger | Merkmal in < 50 % der geprüften Labels (ab 50 Labels); Ledger-Format und Speicherort als Beleg | P1 leer, sonst P2 |
| `outcome_capture` | Lücke `outcome_coverage` | < 80 % der gewählten Formen mit geprüftem Ergebnis (ab 10 Trades) | P1 |
| `sample_bias` | Repräsentativitätswarnung „überwiegend …“ | Evidenz zu ≥ 80 % aus einer Gruppe (die reine Zeitraum-Warnung zählt nicht) | P3 |
| `adoption` | Varianten-Status | PROPOSE/CONFIRMED_ADOPTION mit gültigem Änderungsvorschlag | P1 |
| `withdrawn_success` | Varianten-Status | Übernahmevorschlag WITHDRAWN/REVIEW_REQUIRED oder Evidenz widerspricht der Entscheidung | **P0, täglich** |
| `variant_untestable` | Varianten-Prüfung | Prüfung mit INSUFFICIENT_COVERAGE | P2 |
| `budget_exhausted` | `test_budget` | Fehlerbudget der Varianten verbraucht | P3 |
| `proven_weakness` | Lücken `no_skill`, `choice_worse_than_default` | durch Test belegt **und** alle automatischen Varianten dafür ausgeschöpft | P2 |
| `release_integrity` | `forward_sequential` (falls vorhanden) | PLAN_INVALID oder ERROR | **P0, täglich** |
| `decision_capture` | `forward_sequential` | Prüfung mit Entscheidungsabdeckung < 80 % | P1 |
| `base_not_confirmed` | `forward_sequential` | NOT_CONFIRMED oder REJECTED_NEGATIVE | P2 |
| `source_failure` | `outputs/health/source_health_*` | CRITICAL/IMPORTANT-Quelle ≥ 7 Tage durchgehend BROKEN/FAILED/STALE | P1/P2 |
| `workflow_recovery` | `outputs/state/workflow_status.json` | Workflow an ≥ 5 der letzten 7 Tage vom Watchdog nachgestartet | P2 |

Bewusst **nicht** gemeldet:
- Label-Herkunft aus den Gesamtzählern: Dort steckt Altbestand vor dem Start des Quote-Archivs (12.10.2026). Er ist nie prüfbar und kein behebbarer Defekt.
- Überanpassungs-Faustregel, nie getestete Alternativen und „Auswahl ohne Mehrwert“ (nur Punktschätzung): Die ersten beiden testen die Varianten automatisch; die letzte ist nicht belegt.
- Kurzfristige Quellen- oder Workflow-Störungen: Sie meldet bereits das Ops-Monitoring.

## Inhalt jeder Aufgabe und Mail

Gespeichert und gemeldet werden:
- Task-ID, konkreter Mangel und Belege mit Quellen
- betroffene Dateien (inklusive deterministischer Fundstellen im Code)
- vermutete Ursache, ausdrücklich als **Hypothese** markiert
- empfohlene Änderung und erwarteter Nutzen (gemessen oder „offen“, nie eine erfundene Rendite)
- Risiken, Aufwand und Zusatzkosten
- Akzeptanzkriterien, Regressionstests, Priorität und Status

Betreff: `[Scanner] Technischer Verbesserungsvorschlag – <Kurzbeschreibung>`. Gliederung:
1. Was wurde erkannt?
2. Warum ist das problematisch?
3. Wie schwerwiegend?
4. Vorgeschlagene Verbesserung
5. Datenbelege
6. Zusätzliche Infrastrukturkosten
7. Kopierfertige Claude-Code-Aufgabe: zuerst die Ursache prüfen, dann gezielt ändern, Regressionstests, volle Suite, PR ohne Auto-Merge, Grenzen.

Die Texte sind feste Vorlagen. Es gibt keinen LLM- oder Claude-API-Aufruf und keine Kosten.

## Versandregeln

- **Täglich, nur P0** (`improvement_notify.yml`, 21:37 UTC, `--notify --critical`): widerrufene Erfolge und eine defekte Freigabeprüfung, jede neue oder geänderte P0-Aufgabe sofort als eigene Mail. Den Stand der Freigabeprüfung liest der Lauf aus der jüngsten täglichen Scanner-Empfehlung (`recommendations/<Tag>_<h>d.json`), sonst aus dem Wochenstatus.
- **Wochen-Digest, nur P1–P3** (Samstag, `ml_research.yml`, `--notify`): höchstens **3** neue oder wesentlich geänderte Aufgaben je Kalenderwoche, nach Priorität. Der Rest folgt in der nächsten Woche.
- **Gemeinsame Historie, getrennte Zuständigkeit:** Beide Läufe schreiben in dieselbe Historie. Jede Aufgabe gehört aber genau einer Laufart (P0 täglich, P1–P3 wöchentlich). Ein Lauf ändert nie Status oder Versand der Aufgaben der anderen Art.
- **Keine Wiederholung** einer unveränderten Aufgabe. Erneut gemeldet wird nur bei:
  - wesentlich geändertem Befund (Fingerabdruck: Größenordnung, Status, Priorität – nicht bloß mehr Daten),
  - Wiederauftreten nach Behebung,
  - ausstehendem Versand.
  Ein behobener Befund wird still auf RESOLVED gesetzt.
- **Keine Mail ohne Handlungsbedarf.**
- **Nur Status `sent` zählt.** Fehlende Konfiguration (`not_configured`) oder ein Fehler (`failed`) wird so protokolliert:
  - Ereignis `send_failed`,
  - `last_run.mail_status` im Zustand,
  - Exit-Code 2, Schritt im Workflow als DEGRADED.
  Beim nächsten Lauf derselben Art (P0 am nächsten Tag, Digest in der nächsten Woche) wird erneut versucht; der Mailer versucht es innerhalb eines Laufs bereits dreimal.
- **Menschliche Entscheidung:** In `config/improvement_task_decisions.yaml` lässt sich je Task-ID REJECTED, IN_PROGRESS oder DONE eintragen. Danach kommt zu dieser Aufgabe keine Mail mehr.
- **Dry-Run:** `python -m modules.research_director --notify [--critical] --dry-run` zeigt die Mails und sendet oder speichert nichts. Manuell im Workflow `improvement_notify.yml` mit `dry_run=true` (Standard) und Laufart `critical` oder `weekly`.

## Status: OPEN, NOTIFIED, RESOLVED, UNKNOWN

- **OPEN / NOTIFIED:** Der Mangel ist mit aktuellen Daten nachgewiesen (noch nicht bzw. bereits gemeldet).
- **RESOLVED:** Nur wenn aktuelle, gültige Daten die Behebung **positiv** zeigen (`resolution()`), z. B.:
  - Abdeckung des Merkmals jetzt ≥ 50 % oder Merkmal bewusst entfernt,
  - Quelle wieder gesund,
  - 7 Tage ohne Nachstart,
  - Variante nicht mehr im Übernahme- bzw. Widerspruchszustand,
  - Freigabeprüfung wieder intakt.
- **UNKNOWN:** Der Mangel wird nicht mehr erkannt, aber die nötigen Daten fehlen, sind beschädigt, veraltet oder nicht aussagekräftig (z. B. zu wenige Labels für die Abdeckungsprüfung). Befund, Historie und Versandstand bleiben erhalten; es gibt keine Erfolgsmeldung und keine Mail. Wird der Mangel später mit unverändertem Befund wieder nachgewiesen, geht die Aufgabe ohne neue Mail auf ihren alten Stand zurück. Ein geänderter Befund wird gemeldet.
- **Aktualität** (`input_quality`):
  - Options-Status und Lückenanalyse ≤ 10 Tage,
  - Freigabeprüfung ≤ 10 Tage,
  - Quellen-Gesundheit und Workflow-Status ≤ 2 Tage.

  Ältere Quellen werden nicht ausgewertet; sie erzeugen keine Aufgabe und bestätigen keine Behebung.

## Koordination paralleler Läufe

1. **Getrennte Zuständigkeit** (s. o.): Keine Aufgabe kann von zwei Laufarten gemeldet werden.
2. **Concurrency:**
   - Tägliche und manuelle P0-Läufe laufen nacheinander (Gruppe `improvement-notify`).
   - Ein manueller Wochenversand ohne Dry-Run reiht sich hinter `ml_research.yml` ein (`research-write`).
   - Vorschauläufe schreiben und senden nichts und blockieren niemanden.
3. **Historie:**
   - `improvement_tasks.jsonl` wird bei parallelen Pushes per Union-Merge zusammengeführt und zeitlich sortiert ausgewertet.
   - `improvement_tasks.json` ist daraus abgeleitet; bei Konflikt gilt die Fassung des aktuellen Laufs (`scripts/ci_push.sh`).
4. Jeder Lauf checkt `main` beim Start aus und sieht die Historie des vorherigen Laufs.

## Persistenz

- `outputs/research/improvement_tasks.jsonl` (append-only): detected, changed, reopened, confirmed, unknown (Grund), resolved (Beleg), notified (Kanal, Betreff, Fingerabdruck), send_failed (Status, bereinigter Fehler).
- `outputs/research/improvement_tasks.json`: abgeleiteter Zustand je Aufgabe und letzter Lauf. Er wird gelesen von `research_memory` (`dev_task`) und `meta_cognition` (offene Aufgaben, Versandstatus).
- Idempotent: Gleiche Eingaben ergeben keine neuen Ereignisse und keine Mail.

## Sicherheitsgrenzen

- Kein automatischer Claude-Code-Aufruf, keine Codegenerierung, keine automatische Erstellung oder Übernahme von Codeänderungen. Die Mail enthält nur einen Text zum Kopieren.
- Keine Änderung an Trading-Secrets, Orders, Positionsgrößen oder Risikolimits; kein Einfluss auf Champion, Produktionsregeln oder Freigabe.
- Keine zusätzlichen kostenpflichtigen APIs oder Datenquellen, kein LLM.
- Keine Zugangsdaten in Code, Logs, Zustand oder Mails (Mailer-Fehler werden bereinigt).
- Der Schritt läuft mit `continue-on-error` nach allen Research-Schritten und beeinträchtigt sie nicht.
