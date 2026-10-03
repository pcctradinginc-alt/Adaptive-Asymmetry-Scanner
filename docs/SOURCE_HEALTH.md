# Täglicher Source Health Check

> Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten.
> Das System muss im Zweifel wissen, dass es etwas nicht weiß.

Modul: `modules/source_health.py`. Konfiguration: `config/source_health.yaml` (CODEOWNERS).
Workflow: `.github/workflows/source_health.yml`, täglich 12:41 UTC, also nach der External-Data-Ingestion (06:17) und vor dem Scanner (13:30).

## Abgedeckte Quellen

Es gibt keine neue Registry. Geprüft wird aus dem, was schon existiert:

| Gruppe | Quelle der Prüfung | Beispiele |
|---|---|---|
| Champion-Pflichtdaten | aktive Probes (`core_sources`) | Kurse (yfinance), VIX → Fallback FRED VIXCLS, Optionsketten Tradier → Fallback yfinance, Finnhub-News |
| Registry-Quellen | `outputs/external_data/health/source_health.json` (täglicher Orchestrator) und `config/external_sources/*.yaml` | PortWatch, Eurostat, FRED/ALFRED, NOAA/NWS, ENTSO-E, Lkw-Maut |
| Alternative Daten | leichte Probe, Health-Datei und Store-Statistik (`alt_sources`) | SEC Filings/Form 4/XBRL, GLEIF, TED |

## Prüfungen

- **Verbindung:** Erreichbarkeit, Authentifizierung (ein fehlender Key ergibt AUTH_MISSING, ohne Abruf), HTTP-Status und Rate Limit (429, Retry-After).
- **Antwort:** Antwortzeit, Schema (Pflichtschlüssel), leere Antwort. Die Probes nutzen begrenzten exponentiellen Backoff (3 Versuche, 2/4 s).
- **Zeitstempel:** letzte erfolgreiche Aktualisierung, observation_time, available_at und retrieved_at.
- **Datenqualität:** Missing-Rate, Duplikate, Ausreißer (robuster z, Beträge auf Log-Skala) und Plausibilität (available_at > retrieved_at oder Beobachtungen in der Zukunft ergeben **BROKEN**).
- **Freshness:** quellenabhängig über `max_staleness_days` des Vertrags, sonst die Frequenz-Tabelle der Registry. Monatsdaten gelten erst nach 150 Tagen als veraltet, nicht nach einem Tag.
- **Neue Daten:** ob seit dem letzten Check neue Daten geliefert wurden (`new_data`).
- **Nachgelagerte Nutzung:** Features, Modelle (`extra_features`), Hypothesen und Entscheidungspfade (`downstream_dependencies`).

## Status

| Status | Bedeutung |
|---|---|
| HEALTHY | alle Prüfungen bestanden |
| DEGRADED | nutzbar, aber mit reduzierter Datenqualität: Rate Limit, Timeout, langsam, teilweise fehlend, Ausreißer/Duplikate, Ingest-WARN oder **RECOVERING** |
| STALE | Daten älter als die quellenabhängige Grenze |
| BROKEN | nicht erreichbar, Auth/Schema falsch, PIT-Verletzung oder ≥ 3 Fehlschläge in Folge |
| UNVALIDATED | kein Beleg für eine erfolgreiche Lieferung, Zugangsdaten fehlen oder Quelle deaktiviert |

**Recovery:** Nach STALE oder BROKEN wird eine Quelle erst nach 2 vollständig bestandenen Prüfungen in Folge wieder HEALTHY. Bestanden heißt: erfolgreicher Abruf, Schema ok, frisch und plausibel. Bis dahin bleibt sie DEGRADED (RECOVERING).

## Kritikalität und Konsequenzen

| Kritikalität | Bei Status ≠ HEALTHY |
|---|---|
| NON_CRITICAL | Features unavailable (NaN, Verfügbarkeit 0), Pipeline läuft weiter, keine Imputation |
| IMPORTANT | zusätzlich sinkt die Datenqualität, und abhängige Signale werden deaktiviert (`disabled_signals`, Adapter verwendet sie nicht) |
| CRITICAL | betroffene Entscheidungspfade werden blockiert (`blocked_decisions`). Scanner: kein Lauf mit unbekannten Pflichtdaten |

- CRITICAL gilt nur, wenn die Quelle einen produktiven Pfad speist. Reine Research-Quellen sind höchstens IMPORTANT.
- **Globaler Safe Mode** greift bei:
  - einem blockierten kritischen Pfad,
  - ≥ 2 unabhängigen IMPORTANT/CRITICAL-Quellen mit STALE/BROKEN,
  - oder einer gewichteten Datenqualität unter 0,6.
- **Wirkung des Safe Mode** (kombiniert mit dem Modell-Safe-Mode der Meta-Cognition, `effective_safe_mode`):
  - keine neuen HC-Labels (`hc_scanner`),
  - keine positiven Boosts (Adapter),
  - keine Promotion (PromotionController; Demotion bleibt erlaubt),
  - keine Signale ohne Pflichtdaten,
  - der Champion läuft nur mit HEALTHY- oder Fallback-Pflichtdaten.
- **Unbekannt** (Snapshot fehlt oder ist älter als 30 h) wird wie aktiv behandelt (fail-closed). Der Scanner prüft dann nur die Pflichtdaten live (`scanner_preflight`).

## Fallback und Cache

- **Provider-Fallback:** nur explizit deklariert (`fallback:` in `core_sources`) und nur über einen bestehenden, getesteten Pfad:
  - VIX: `risk_gates.py` (FRED VIXCLS),
  - Optionen: `options_designer.py` (yfinance).
- **Sichtbarkeit:** `source_primary` und `source_actual` stehen in Snapshot, Feature-Availability und Optionsquote. Ein Fallback senkt die Datenqualität (0,8). Ist der Fallback selbst nicht HEALTHY, wird der Pfad blockiert, nicht improvisiert.
- **Cache:** Der letzte gültige Wert wird nur bei ausdrücklich zulässiger Datenalterung verwendet (`cache_max_age_days`). Dann gilt `stale=true`, Datenqualität 0,5, und der Feature Store setzt `<verfügbarkeit>_stale=1`.
- **Ohne Cache-Freigabe:** Der Feature Store setzt ab dem Ausfallzeitpunkt NaN und Verfügbarkeit 0. Ältere, damals gültige PIT-Zeilen bleiben unverändert.

## Ausgaben (`outputs/health/`)

| Datei | Inhalt |
|---|---|
| `source_health_snapshot.json` | je Quelle: source_id, status, checked_at, last_success, latest_observation, expected_freshness, latency, missing_rate, schema_version, error, consecutive_failures, downstream_dependencies, Fallback |
| `source_health_history.jsonl` | append-only Historie; `instability()` erkennt wiederkehrende Instabilität (Statuswechsel, Anteil ungesunder Tage über 30 T) |
| `feature_availability.json` | je Feature available, stale, data_quality, source_primary, source_actual |
| `data_safe_mode.json` | blockierte Pfade, deaktivierte Signale, nicht verfügbare oder veraltete Features, globaler Safe Mode |
| `daily_data_health.md` | Daily Data Health Report |

Der Report enthält:

- Status je Kategorie
- neue Fehler
- Fehler in Folge
- betroffene Features, Modelle und Entscheidungen
- Fallbacks
- deaktivierte Signale
- Safe Mode
- letzten Datenstand

**Mail** (`modules/mailer.py`) geht nur bei relevanter Änderung raus: Statuswechsel von oder zu HEALTHY bzw. zu STALE/BROKEN, neuer Fehler, Fallback an/aus, Safe Mode an/aus, abgeschlossene Recovery. Bei unverändert gesundem Zustand gibt es keine Mail.
