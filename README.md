# Adaptive Asymmetry-Scanner

Ein automatischer News-to-Options-Scanner für US-Aktien. Er erzeugt täglich **Paper-Vorschläge** für Long-Optionen
und Debit-Spreads – **keine realen Orders, kein Live-Trading**.

Ob der Champion Alpha hat, ist **nicht belegt**. Genau das misst das vorregistrierte Champion-Protokoll (unten),
ausschließlich an verlässlichen Netto-Optionsrenditen.

---

## Kernidee

Gesucht werden **Underreactions**: Nachrichten mit fundamentalem Gewicht, auf die der Kurs in den ersten 48 Stunden
schwach reagiert hat.

```
Mismatch = Impact (LLM, 0–10) − 5 · |48h-Bewegung| / σ
```

Gerichtete Varianten (Bewegung in Thesenrichtung, ab dem Zeitstempel der Nachricht) laufen nur im Schatten als
Challenger (`mismatch_directional_news`).

---

## Tatsächlicher Ablauf (`pipeline.py`, Mo–Fr)

```
0a Data Health        Pflichtdaten verfügbar? (sonst Abbruch mit Mail)
0  Risk Gates         VIX-Gate
1  Hard-Filter        Market Cap, Volumen, Dollar-Volumen, realisierte Vola (yfinance)
1b FinBERT/Sentiment  + News-Zeitfenster (erster/letzter Zeitstempel je Ticker, PIT)
1c Sector-Momentum    vor dem LLM (spart Calls)
EF Economics-first    echte Long-DTE-Optionsquotes vor jedem LLM-Call (config/economics_first.yaml)
2  Prescreening       Claude Haiku: Signal vs. Rauschen
3b Pre-MC-Gate        Sigma-Vorfilter
3c Bearish-Vorfilter  nur nach gepaarter Validierung (model_routing)
4  Deep Analysis      Claude Sonnet + Red Team (Impact, Surprise, Richtung, Bear Case)
4a/4b                 Bearish-Gate, Impact×Surprise-Floor
5  Mismatch-Score     + Overreaction-Cap
6–8 Monte Carlo       Quick MC, Intraday-Delta, Final MC (Target-Hit-Score, MC, unkalibriert)
9  Finale Auswahl     (RL/QuasiML am 2026-11-09 entfernt)
10 Options-Design     Laufzeit-Loop, ROI-/Edge-Gate, Trade-Score-Ranking, Korrelation,
                      Production Intelligence Adapter (nur freigegebener, gekappter Research-Einfluss)
```

Die MC-Kennzahl heißt **Target-Hit-Score (MC, unkalibriert)** – sie ist keine Gewinnwahrscheinlichkeit.
Positionsgröße: **Einheitsgröße** (1 Kontrakt/Struktur je Vorschlag, `portfolio.sizing_mode: unit`); Kelly erscheint
nur informativ, solange die MC-Wahrscheinlichkeiten unkalibriert sind.

LLM-Aufrufe senden **keine** Sampling-Parameter: die API-Version `2026-10-21` (anthropic ≥ 2.0) lehnt
`temperature/top_p/top_k` ab. Reproduzierbarkeit: Analyse-Cache für identische Eingaben und `llm_audit`
(SHA-256 von Prompt und Rohantwort) je Ergebnis.

---

## Workflows

| Workflow | Zeitplan (UTC) | Inhalt |
|---|---|---|
| `scanner.yml` | Mo–Fr 13:30/14:30 | Pipeline, Tagesreport, Mail |
| `feedback.yml` | Mo–Fr 15:30/19:30 | Exits, Outcomes (verlässlich: Entry Ask / Exit Bid, Kommission) |
| `ml_research.yml` | Sa (weekly), 3. d. M. (full) | Research, Alpha-Ledger-Zuteilung, Champion-Protokoll, Challenger |
| `weekly_report.yml` | Mo 05:47 | Wochenbericht (inkl. Champion-Protokoll-Status) |
| `monthly_report.yml` | 1. d. M. 07:00 | Monatsbericht |

Secrets: `ANTHROPIC_API_KEY` (Pflicht), `NEWS_API_KEY` (empfohlen), `TRADIER_API_KEY` (optional). Keine Keys im Code.

Lokal: `pip install -r requirements.txt`, dann `python pipeline.py` bzw. `python feedback.py`.

---

## Wie entschieden wird, ob etwas funktioniert

1. **Champion-Protokoll** (`config/champion_protocol.yaml`, `modules/champion_protocol.py`):
   * einzige Primärkennzahl: `outcome_net` verlässlicher geschlossener Champion-Trades ab `start_date`;
   * einseitiger t-Test auf Tagesmitteln bei 30/60/90 Trades;
   * Prüfberichte werden create-only eingefroren.
   * `decision_config_sha256` pinnt Gates, Filter, Risiko, Optionen, Pipeline, Modelle, Kosten, Haltedauer,
     economics-first und die Prompt-Versionen. Jede Änderung daran bricht das Protokoll (`PROTOCOL_BROKEN`) und
     braucht eine neue Protokollversion. Es gibt kein Tuning auf gesehenen Daten.
2. **Ein Fehlerbudget** (`config/alpha_ledger.yaml`, `modules/ledger_registration.py`):
   * neue Challenger, Promotion-Verträge und das Champion-Protokoll erhalten Alpha genau einmal aus dem globalen
     Ledger (idempotent, Hash-Kette, fail closed);
   * ohne Zuteilung gibt es keine Bestätigung.
   * Vor dem 2026-11-09 registrierte Tests laufen unverändert nach ihrer registrierten Regel weiter und sind als
     Altfamilie außerhalb des Budgets ausgewiesen.
3. **Challenger** (`challengers.yaml`, `modules/challenger.py`):
   * vorregistriert, mit `start_date` nach der Registrierung;
   * ausgewertet werden nur Candidate-Ledger-Zeilen nach dem Start mit verstrichenem Horizont;
   * das Verdikt ist eine reine Funktion der Daten.
   * Laufend sind u. a. `no_mc_gates` (Champion ohne die drei MC-Gates) und `mismatch_directional_news`.
4. **Promotion ist immer ein von einem Menschen gemergter PR.** Geschützte Pfade stehen in `.github/CODEOWNERS`.
5. **Research-Scope** (`config/research_scope.yaml`):
   * parallele Hypothesen-Generatoren sind eingefroren und erzeugen keine neuen Hypothesen;
   * registrierte Hypothesen werden weiter ausgewertet;
   * eingefrorener Code ist per SHA-256 gepinnt.
6. **Schatten-Messungen ohne Produktionswirkung:**
   * Ausführungskosten nach Spread-Klasse, Limit-Szenario und Haltedauer gepaart (`modules/option_learner.py`);
   * LLM-Zensus: k = 2 zufällige Tiefenanalysen ohne Gates, mit protokolliertem π (`modules/llm_census.py`).
7. **Externe Daten nur mit registriertem Nutzer** (`config/external_consumers.yaml`): Quellen ohne belegten Nutzer
   stehen auf `DEFERRED`. Archivierte Daten bleiben erhalten.

Begründung und Belege: `docs/ACTION_PLAN_2026-11-09.md`.

---

## Risk-Gates

| Gate | Bedingung (Werte in `config.yaml`) |
|---|---|
| VIX-Gate | VIX > `risk.vix_threshold` → Lauf endet |
| Earnings-Gate | Earnings < `risk.earnings_buffer_days` → Ticker blockiert |
| Bear-Case-Gate | `bear_case_severity` > `risk.max_bear_case_severity` |
| Liquidität / Spread | Open Interest < `risk.min_open_interest`, Bid-Ask-Ratio > `risk.max_bid_ask_ratio` |
| Trade-Score | < `gates.trade_score_min` → kein Vorschlag (≥ `gates.shadow_score_min` → Schatten) |

---

## External Context (Real Economy / Freight / Shipping / Weather)

Ein zusätzlicher, vollständig von der Trading-Entscheidung entkoppelter
Beobachtungs-Layer (`modules/external/`) sammelt Real-Economy-Signale
(Road-Freight, Maritime/Shipping, Chokepoints, Wetter) und ordnet sie den
Aktien-Katalysatoren des Scanners zu. Er läuft **ausschließlich im Shadow-
Modus**: er beeinflusst keine Gates, keine Scores, keine Trades.

### Architektur (ASCII)

```
config/external_sources/*.yaml          config/{industry,port,weather}_exposure.yaml
        │                                          │
        ▼                                          │
modules/external/sources/*  ──fetch──►  RawRecord  │
        │                                          │
        ▼                                          │
modules/external/pit.py  (Observation, PIT-Regeln) │
        │                                          │
        ▼                                          ▼
modules/external/archive.py  ──persist──►  outputs/external_data/
   raw/<source_id>/<YYYY-MM>/…meta.json               ▲
   normalized/<source_id>/<YYYY-MM>.jsonl              │
   manifests/<YYYY-MM-DD>/<run_id>.json                │
   health/source_health.json ◄── registry.py ──────────┘
   health/storage_telemetry.json
        │
        ▼
modules/external/features.py  (rolling/zscore/state/breadth, past-only)
        │
        ▼
modules/external/context.py / policy.py / shadow_analysis.py / governance.py
   (LLM-Relation SUPPORT|NEUTRAL|CONTRADICT, score_delta=0, Governance-Pfad)
        │
        ▼
Candidate-Ledger-Zeile: row["external"] = {snapshot_id, feature_version,
  available_at, primitives{…}, states{…}, ticker_exposure{…}, divergences{…},
  relation{…}, policy{mode, score_delta=0, veto}}
        │
        ├──► Trades: "external_context_entry" (beim Entry EINGEFROREN)
        │        │
        │        ▼
        │    feedback.py: history["feature_stats_external"]  (rein deskriptiv,
        │        NIE in einem Produktions-Lernpfad)
        │
        ├──► modules/external/research.py  (retrospektiv/in-sample, H1–H12,
        │        NIE aus der Produktions-Pipeline aufgerufen)
        │        → challengers.yaml-Snippets (Mensch fügt ein)
        │
        └──► Reports:
             - Monats-Report: "🌍 Externer Kontext (SHADOW)"
             - Status-/Trade-Mail + Daily-Markdown: kompakter Block
               ("SHADOW — NICHT in der Produktionsentscheidung verwendet")
             - modules/engine_monitor.py: externe Lern-Health-Warnungen
```

### Point-in-Time (PIT), Vintages, Archiv

- Jede `Observation` trägt `observation_time`, `available_at` und optional
  `vintage_time`/`forecast_issue_time` — Features dürfen zu Zeitpunkt `t`
  ausschließlich Werte verwenden, die zu `t` bereits **bekannt** waren
  (`modules/external/pit.py: available_as_of`).
- Revisionen überschreiben nie: derselbe Identitätsschlüssel mit neuem Wert
  wird als **neue Vintage-Zeile** angehängt (`archive.store_observations`).
  Eine Rückrevision A→B→A ist eine echte neue Vintage und bleibt erhalten.
- Rohdaten-Policy (`external_context.archive.raw_payload_policy`):
  `hash_only` (Default, nur Metadaten+Content-Hash) oder `gzip` (Payload bis
  `raw_max_bytes` zusätzlich gespeichert). Es werden **nie** Credentials/
  Header archiviert.
- Layout: `raw/<source_id>/<YYYY-MM>/…`, `normalized/<source_id>/<YYYY-MM>.jsonl`,
  `manifests/<YYYY-MM-DD>/<run_id>.json`, `health/source_health.json`,
  `health/storage_telemetry.json` (Speicher-Projektion 30d/1y/5y + Migrations-
  hinweis, wenn `external_context.archive.storage_warn_mb_1y` überschritten
  wird — GitHub-Actions-Checkout/Artefakte sind **kein** dauerhafter Speicher).

### Quellen-Registry, Status, Readiness

- `config/external_sources/*.yaml` je Datenfamilie (Road/Maritime/Weather),
  entdeckt via `modules/external/registry.py`. Jede Quelle: `license_status`
  (`OK`/`REVIEW_REQUIRED` — bei `REVIEW_REQUIRED` **nie** automatisch
  abgerufen), `requires_auth` + `auth_env_variable` (fehlt die Env-Var →
  `AUTH_MISSING`, kein Call), `criticality` (`low`/`medium`/`high`).
- `SourceHealth` (`PASS`/`FAIL`/…, `staleness` `FRESH`/`STALE`/`UNKNOWN`,
  `consecutive_failures`, `pit_integrity_failures`, …) wird nach
  `outputs/external_data/health/source_health.json` persistiert.
- `DataReadiness` (`registry.compute_readiness`) wird **ausschließlich** aus
  Schema/PIT/Provenienz/Frische/Coverage/Zeitspanne/Beobachtungszahl
  abgeleitet — **niemals** aus Returns/P&L:
  `DISABLED → COLLECTING → SCHEMA_VALIDATED → PIT_VALIDATED →
  EXPLORATORY_READY → CHALLENGER_READY → PRODUCTION_ELIGIBLE`
  (bzw. `DEGRADED`/`BLOCKED`). Mindestschwellen je Frequenz in
  `external_context.readiness` (`config.yaml`).

### Shadow-Modus: Garantien

- `external_context.mode` steuert die Stufe (`off | shadow | challenger |
  production`); aktuell **`shadow`**.
- `external_context.production.max_score_delta: 0` — im Shadow-Modus darf
  der externe Kontext **keinen** Score/Gate beeinflussen. `policy.score_delta`
  ist strukturell auf 0 begrenzt.
- Alle Reports markieren den Abschnitt explizit:
  **„SHADOW — NICHT in der Produktionsentscheidung verwendet"**.
- `learning_features.shadow` sammelt externe Feature-Namen,
  ohne dass sie in einen Produktions-Lernpfad einfließen (`learning_features.production`
  bleibt unverändert).

### LLM-Relation — was sie ist und was sie nicht ist

Die `relation`-Bewertung (`SUPPORT`/`NEUTRAL`/`CONTRADICT`, `materiality`,
`confidence`, `mechanism`) ist eine **Konsistenzprüfung**: passt der
behauptete Katalysator zur beobachteten Real-Economy-Lage? Sie ist **kein**
Kauf-/Verkaufssignal, **kein** Score-Beitrag (siehe `max_score_delta: 0`) und
**kein** Ersatz für Impact/Mismatch/EPS-Drift. Sie ist reine Zusatz-
Information für Menschen und für `modules/external/research.py`.

### Governance-Pfad (observe → … → human promotion)

Analog zum bestehenden Challenger-Prozess (siehe „Tuning-Prozess" oben):
`COLLECTING/EXPLORATORY_READY` (reine Beobachtung) → `CHALLENGER_READY`
(walk-forward validierbar, siehe `modules/challenger.py`) →
`PRODUCTION_ELIGIBLE` (nur wenn `source_id` in
`external_context.learning.promoted_external_features` steht) → **Promotion
ist ausschließlich ein von einem Menschen gemergter PR**, der `gates:`/
`learning_features.production` in `config.yaml` ändert. Keine Automatisierung
darf diesen Schritt selbst auslösen.

### Forschungs-/Hypothesen-Budget (H1–H12)

`modules/external/research.py` (aufgerufen aus `backtest_thresholds.py` /
dem Hypothesen-Generator-Pfad von `monthly_report.py`, **nie** aus der
Produktions-Pipeline) prüft mature Candidate-Ledger-Zeilen retrospektiv/
in-sample gegen den vorregistrierten Mechanismus-Katalog
`config/external_hypotheses.yaml` (H1–H12, u.a. Road-Freight-Zustand,
Maritime-Zustand, Chokepoint-Anomalien, Wetter-Disruption, Divergenzen —
**beide Richtungen** je Feature, z.B. H1A/H1B, konkurrieren unabhängig).
Reifegrad-Gate: n≥30, ≥15 unabhängige Handelstage, ≥60 Tage Kalenderspanne,
geclustertes 90%-Bootstrap-KI schließt 0 nicht ein. Budget:
`external_context.research.max_external_hypotheses_per_month` (Default 3),
max. 2 Vorschläge je Feature, keine >2-Wege-Interaktionen. Ein Treffer wird
**nie** automatisch übernommen — er erscheint nur als fertiges
`challengers.yaml`-Snippet im Monats-Report, das ein Mensch selbst einfügt
(`registered_on=heute`, `start_date=morgen`, Label „retrospektiv/in-sample,
NICHT promotion-fähig").

### Ingestion/Preflight ausführen

```bash
python -m modules.external.orchestrator ingest              # Ingestion-Lauf (alle Familien)
python -m modules.external.orchestrator ingest --family road_freight
python -m modules.external.orchestrator preflight            # Nur Konnektor-/Auth-/License-Check, kein Fetch
```

### Optionale Secrets

| Secret | Zweck |
|---|---|
| `ESTAT_APP_ID` | Eurostat-Zugriff (EU-Road-Freight) |
| `FRED_API_KEY` | FRED (US-Freight/Truck-Tonnage-Serien) |
| `DESTATIS_USER` / `DESTATIS_PASSWORD` | Destatis-Zugangsdaten (DE-LKW-Maut) |
| `NCEI_CDO_TOKEN` | NOAA NCEI Climate Data Online (Wetter-Historie) |
| `NWS_USER_AGENT_CONTACT` | Erforderlicher Kontakt-User-Agent für api.weather.gov |

Fehlt ein Secret, wird die betroffene Quelle `AUTH_MISSING`/`DISABLED` und
bleibt aus der Readiness-Kette ausgeschlossen — kein Fehler, keine
Pipeline-Unterbrechung.

### Speicher-Telemetrie & Objekt-Storage-Backend

`ExternalArchive.storage_telemetry()` misst Bytes/Zeilen je Quelle, projiziert
30d/1y/5y und flaggt, wenn die 1y-Projektion
`external_context.archive.storage_warn_mb_1y` (Default 200 MB) überschreitet
(`storage_telemetry.json` enthält zusätzlich `backend`, `bytes_in_git`,
`bytes_offloaded`). GitHub-Actions-Checkout/Artefakte sind kein dauerhafter
Speicher, git ist nur für kurz-/mittelfristiges Wachstum geeignet — dafür
gibt es ein implementiertes, optionales S3-kompatibles Object-Storage-
Backend (`modules/external/storage.py`; funktioniert mit AWS S3, Cloudflare
R2, Backblaze B2, MinIO, jedem S3-kompatiblen Store):

1. Bucket anlegen (z.B. AWS S3, Cloudflare R2, Backblaze B2 oder ein
   selbstgehostetes MinIO).
2. Secrets im Repo hinterlegen: `EXTERNAL_ARCHIVE_S3_BUCKET` (erforderlich),
   optional `EXTERNAL_ARCHIVE_S3_ENDPOINT` (R2/B2/MinIO),
   `EXTERNAL_ARCHIVE_S3_PREFIX` (Default
   `adaptive-asymmetry/external_data`), sowie `AWS_ACCESS_KEY_ID` /
   `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` — siehe `.env.example`. Der
   Workflow (`.github/workflows/external_data.yml`) reicht diese als Env
   durch und installiert `boto3` (`requirements-storage.txt`) nur, wenn
   `EXTERNAL_ARCHIVE_S3_BUCKET` gesetzt ist.
3. `external_context.archive.backend: s3` in `config.yaml` setzen (Default
   `git` — ohne diese Änderung passiert nichts, auch nicht mit gesetzten
   Secrets). `git_retention_months` (Default 3) steuert, ab wann
   abgeschlossene normalisierte Monatsdateien lokal gelöscht werden.

Ablauf je Ingestion-Run bei `backend: s3`: alle `normalized/`/`raw/`-Dateien
werden gzip-komprimiert mit sha256-Metadaten hochgeladen (idempotent),
danach per Re-Download verifiziert. **Erst nach erfolgreicher Verifikation**
werden abgeschlossene (nicht der laufende Monat) normalisierte
Monatsdateien, die älter als `git_retention_months` sind, aus dem
Git-Arbeitsverzeichnis entfernt; ein Pointer-Manifest
(`outputs/external_data/manifests/offloaded.json`) hält `s3_key`/`sha256`/
`bytes`/`offloaded_at` je ausgelagerter Datei. `ExternalArchive.load()`/
`as_of()` lesen ausgelagerte Monate transparent aus dem Object Storage
zurück (Cache in `/tmp`) — PIT-Semantik bleibt unverändert. Schlägt Upload
oder Verifikation fehl, wird **nie** gelöscht; die betroffene Quelle bekommt
Status `WARN` und die bestehende `ARCHIVE_FAILURE`-Alarmierung. Ohne
gesetztes `EXTERNAL_ARCHIVE_S3_BUCKET` bleibt `backend: s3` wirkungslos
(kein Upload, keine Löschung, wie `backend: git`).

### Grenzen

- Datenqualität/Abdeckung variiert stark je Quelle/Region — `Readiness` und
  `SourceHealth` machen das explizit, statt es zu verschleiern.
- Die LLM-Relation ist eine Heuristik, kein validiertes Kausalmodell.
- `modules/external/research.py`-Befunde sind In-Sample — ohne prospektive
  Walk-forward-Validierung (`modules/challenger.py`) bleiben sie Hypothesen.
- Der externe Kontext bleibt Shadow, bis eine Quelle `PRODUCTION_ELIGIBLE`
  erreicht UND ein Mensch sie in `learning_features.production`/`gates:`
  aufnimmt.

---

## Tests

```bash
pytest tests/ -v
```

---

## Haftungsausschluss

Dieses System generiert ** keine Anlageberatung **. Alle Vorschläge sind rein algorithmischer Natur und dienen ausschließlich zu Forschungs- und Lernzwecken. Der Einsatz von echtem Kapital auf Basis dieser Ausgaben erfolgt auf eigenes Risiko. Options-Handel kann zum vollständigen Verlust des eingesetzten Kapitals führen.

**Empfehlung:** Mindestens 6 Monate Papier-Trading (nur Logs, kein echtes Kapital) bevor reale Positionen eröffnet werden.
