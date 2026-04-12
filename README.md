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

Die Pipeline läuft automatisch **Mo–Fr um 14:30 MEZ** (12:30 UTC).  
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

Nach jedem abgeschlossenen Trade (≥ 130 Tage) werden:
1. Die Bin-Durchschnitte aktualisiert (laufender Ø)
2. Die Feature-Gewichte via Pearson-Korrelation neu kalibriert

Je mehr Trades, desto präziser das Scoring.

---

## Tests

```bash
pytest tests/ -v
```

---

## Haftungsausschluss

Dieses System generiert **keine Anlageberatung**. Alle Vorschläge sind rein algorithmischer Natur und dienen ausschließlich zu Forschungs- und Lernzwecken. Der Einsatz von echtem Kapital auf Basis dieser Ausgaben erfolgt auf eigenes Risiko. Options-Handel kann zum vollständigen Verlust des eingesetzten Kapitals führen.

**Empfehlung:** Mindestens 6 Monate Papier-Trading (nur Logs, kein echtes Kapital) bevor reale Positionen eröffnet werden.
