# Adaptive Asymmetry-Scanner v3.5

Ein vollautomatischer News-to-Options-Scanner, der täglich Informations-Asymmetrien im US-Aktienmarkt identifiziert und konkrete Options-Vorschläge generiert.

---

## Kernkonzept

Das System sucht nicht nach "guten Nachrichten", sondern nach **Underreactions**: Fundamentale Nachrichten mit einem 3–6-Monats-Impact, auf die der Markt innerhalb der ersten 48 Stunden statistisch zu schwach reagiert hat. Dieser Mismatch zwischen fundamentaler Stärke und Preisbewegung ist der eigentliche Alpha-Hebel.

```
News-Stärke (Impact 0-10) minus Marktreaktion (Z-Score × 5) = Mismatch-Score
```

Je höher der Mismatch, desto wahrscheinlicher eine verzögerte Einpreisung.

---

## 7-Stufen-Pipeline

```
1. Daten-Ingestion      → News (NewsAPI/RSS) + yfinance Hard-Filter
2. Prescreening         → Claude Haiku: Rauschen vs. strukturelle Änderung
3. Deep Analysis        → Claude Sonnet: Asymmetry Reasoning + Bear Case
4. Mismatch-Score       → Z-Score der 48h-Bewegung vs. Impact
5. MiroFish-Simulation  → 10.000 Monte-Carlo-Pfade über 120 Tage
6. Quasi-ML Scoring     → Selbstlernende Gewichtung aus history.json
7. Options-Design       → IV-Rank-basierte Strategie via Tradier/yfinance
```

---

## Schnellstart

### 1. Repository klonen

```bash
git clone https://github.com/DEIN-USERNAME/news-mirofish.git
cd news-mirofish
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3. API-Keys konfigurieren

```bash
cp .env.example .env
# .env öffnen und Keys eintragen
```

Lokal:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export NEWS_API_KEY="..."
export TRADIER_API_KEY="..."   # optional, Fallback auf yfinance
```

### 4. Ersten Lauf starten

```bash
python pipeline.py
```

### 5. Feedback-Loop manuell ausführen

```bash
python feedback.py
```

---

## GitHub Actions Setup

### Secrets konfigurieren

Im GitHub-Repo unter **Settings → Secrets and variables → Actions**:

| Secret | Beschreibung |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API Key (Pflicht) |
| `NEWS_API_KEY` | NewsAPI Key (empfohlen) |
| `TRADIER_API_KEY` | Tradier API Key (optional) |

### Automatischer Trigger

Die Pipeline läuft automatisch **Mo–Fr um 14:30 MEZ** (13:30 UTC, cron `30 13 * * 1-5`).  
**Hinweis:** GitHub Scheduled Runs unterliegen erheblichen Verzögerungen; aktuelle Läufe starten 
oft erst gegen 17:50 UTC statt 13:30 UTC.

**Feedback-Workflow** (Exit-Checks + Lernen) läuft zusätzlich:
- **15:30 UTC** – frühe US-Session: Stop-Loss/Take-Profit nach Markt-Open
- **19:30 UTC** – 30 Min vor US-Close: letzte Exit-Chance am selben Tag

Manueller Trigger: GitHub → Actions → "Adaptive Asymmetry-Scanner" → "Run workflow".

---

## Projektstruktur

```
news-mirofish/
│
├── pipeline.py              # Haupt-Orchestrator (7-Stufen-Flow)
├── feedback.py              # Wöchentlicher Lern-Loop
├── config.yaml              # Alle Parameter zentral
├── requirements.txt
├── .env.example
│
├── modules/
│   ├── data_ingestion.py    # Stufe 1: News + Hard-Filter + EPS-Drift
│   ├── prescreener.py       # Stufe 2: Claude Haiku Batch-Filter
│   ├── deep_analysis.py     # Stufe 3: Claude Sonnet Asymmetry-Reasoning
│   ├── mismatch_scorer.py   # Stufe 4: Z-Score + Mismatch-Formel
│   ├── mirofish_simulation.py # Stufe 5: Monte-Carlo 10.000 Pfade
│   ├── quasi_ml.py          # Stufe 6: Adaptive Bin-Scoring
│   ├── options_designer.py  # Stufe 7: IV-Analyse + Kontrakt-Auswahl
│   ├── risk_gates.py        # VIX-Check, Earnings-Gate, Liquidität
│   └── reporter.py          # JSON + Markdown Report
│
├── outputs/
│   ├── history.json         # Persistente Feature-Stats + Trades (im Git)
│   └── daily_reports/
│       ├── YYYY-MM-DD.json  # Maschinenlesbar
│       └── YYYY-MM-DD.md    # Menschenlesbar
│
├── tests/
│   └── test_pipeline.py     # Pytest-Suite
│
└── .github/
    └── workflows/
        └── scanner.yml      # GitHub Actions
```

---

## Risk-Gates (Sicherheitslayer)

Alle Gates blockieren den Trade automatisch:

| Gate | Bedingung |
|---|---|
| VIX-Gate | VIX > 35 → gesamte Pipeline bricht ab |
| Earnings-Gate | Earnings < 7 Tage → Ticker blockiert |
| Bear-Case-Gate | bear_case_severity > 7 → Ticker blockiert |
| Liquiditäts-Gate | Open Interest < 100 → Kontrakt abgelehnt |
| Spread-Gate | Bid-Ask-Ratio > 10% → Kontrakt abgelehnt |

---

## Strategie-Logik

| IV-Rank | Richtung | Strategie |
|---|---|---|
| < 50 | Bullish | Long Call (DTE 120–200, Delta ~0.65) |
| ≥ 50 | Bullish | Bull Call Spread |
| < 50 | Bearish | Long Put |
| ≥ 50 | Bearish | Bear Put Spread |

---

## Quasi-ML Selbstlern-System

`history.json` speichert für jede Feature-Kombination den historischen Durchschnitts-Return:

```
FinalScore = Σ(Bin_Avg_Return_i × Current_Weight_i)
```

Nach jedem abgeschlossenen Trade (Schließung nach `learning.close_after_days`, siehe unten) werden:
1. Die Bin-Durchschnitte aktualisiert (laufender Ø)
2. Die Feature-Gewichte via Pearson-Korrelation neu kalibriert

Je mehr Trades, desto präziser das Scoring. Achtung: Sind alle Feature-Korrelationen ≤ 0,
bleiben die Gewichte auf ihren Startwerten (der Engine-Monitor warnt dann "Lern-Loop eingefroren").

**Hinweis:** Trades werden nach `learning.close_after_days` Tagen geschlossen
(aktuell: 45 Tage, siehe `config.yaml` → `learning.close_after_days`). Trades, die älter als dieser
Horizont sind, werden als „abgeschlossen" markiert und ihre tatsächlichen
Outcomes zum Training herangezogen.

**Hinweis:** Trades werden nach `learning.close_after_days` Tagen geschlossen
(aktuell: 45 Tage, siehe `config.yaml` Zeile 70). Trades, die älter als dieser
Horizont sind, werden als „abgeschlossen" markiert und ihre tatsächlichen
Outcomes zum Training herangezogen.

---

## Tuning-Prozess

Änderungen an den Produktions-Schwellen (`config.yaml` → `gates:`) folgen einem
festen Prozess, damit kein Tuning auf bereits gesehenen Daten (Post-hoc-
Overfitting) stattfindet und keine automatisierte Instanz allein die
Produktionslogik ändern kann:

1. **Vorregistrierung.** Jede Hypothese wird in `challengers.yaml` **vor**
   der Datensammlung eingetragen: eine Freitext-Hypothese, eine deklarative
   Auswahlregel für den Challenger-Arm, eine Baseline-Regel (Standard: die
   aktuelle Produktionsregel), eine Ziel-Metrik, eine Mindest-Stichprobengröße
   (`min_n`) und ein `start_date`/`max_duration_days`-Fenster.
2. **Walk-forward-Auswertung.** `modules/challenger.py` wertet ausschließlich
   Candidate-Ledger-Zeilen (`outputs/candidate_ledger/*.jsonl`) aus, deren
   Datum **nach** der Registrierung liegt und deren Outcome-Horizont bereits
   verstrichen ist. Bereits vor Registrierung gesammelte Daten fließen nie ein.
3. **Deterministisches Verdikt mit Alpha-Spending.** Für jeden Challenger wird ein
   seeded, reproduzierbares Bootstrap-Konfidenzintervall der Return-Differenz
   (Challenger − Baseline) berechnet. Da Challenger jeden Monat neu bewertet werden
   (bis sie promotiert oder expiriert sind), wird eine Alpha-Spending-Regel angewandt,
   um die Familie-weise Fehlerquote über wiederholte Looks zu kontrollieren:
   - Geplante Anzahl Looks: `n_looks = max(1, ceil(max_duration_days / 30))`
   - Effektives Signifikanzniveau: `alpha = 0.10 / (n_active × n_looks)`
   
   Dies ist eine konservative (Bonferroni-artige) Korrektur; Confidence Sequences
   sind eine künftige Verbesserung. Das Verdikt (`running`, `promote_recommended`,
   `reject`, `expired`) ist eine reine Funktion der Daten — kein manuelles Ermessen
   im laufenden Auswertungscode.
4. **Promotion ist ausschließlich ein von einem Menschen gemergter PR.** Ein
   `promote_recommended`-Verdikt erscheint informativ im Monats-Report
   (Abschnitt „🧪 Challenger (Walk-forward)”). Es ändert **nichts** automatisch.
   Die tatsächliche Übernahme in die Produktion ist ein Pull-Request, der
   `gates:` in `config.yaml` ändert und von einem Repo-Owner (siehe
   `.github/CODEOWNERS`) geprüft und gemergt wird.
5. **Grenzen für automatisierte Beiträge.** Eine KI/Automatisierung darf neue
   Challenger in `challengers.yaml` vorschlagen/registrieren (als PR, nicht
   als Direkt-Commit gegen `main`). Sie darf **nicht** `modules/challenger.py`,
   die Candidate-Ledger-Daten (`outputs/candidate_ledger/`) oder die
   Promotion-Regeln selbst verändern — diese Pfade sind über CODEOWNERS
   geschützt und benötigen menschliche Review.

```bash
python -m modules.challenger   # druckt eine Tabelle aller registrierten Challenger
```

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
        │        NIE in compute_pearson_weights/model_weights/QuasiML/RL)
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
- `learning_features.shadow` sammelt externe Feature-Namen für das RL-Schema,
  ohne dass sie ins aktive Training einfließen (`learning_features.production`
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

### Speicher-Telemetrie & Migration

`ExternalArchive.storage_telemetry()` misst Bytes/Zeilen je Quelle, projiziert
30d/1y/5y und flaggt, wenn die 1y-Projektion
`external_context.archive.storage_warn_mb_1y` (Default 200 MB) überschreitet.
GitHub-Actions-Checkout/Artefakte sind kein dauerhafter Speicher — bei
anhaltendem Wachstum: Object Storage (S3/GCS/Backblaze B2) für
`normalized/`/`raw/`, ältere `raw/`-Monate kompaktieren.

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
