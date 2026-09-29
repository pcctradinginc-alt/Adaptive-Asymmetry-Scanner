# Alpha-Pipeline-Audit 2026-09-29

Unabhängiger technischer, datenbezogener und methodischer Audit des gesamten
Repositorys (Ergänzung zu `EXTERNAL_DATA_AUDIT_2026-09-27.md`, Abschnitte N1–N11).
Grundsatz: **Keine Produktionsänderung an Scores, Gates oder PPO ohne Evidenz**. Die
Korrekturen betreffen Datenfehler, stille Defaults und Sicherheitsbremsen. Neues
Lernen läuft ausschließlich im SHADOW-Modus.

## 1. Datenfluss (Ist-Zustand, verifiziert am Lauf-Log vom 2026-09-28)

```
UNIVERSUM   S&P500 + Nasdaq100 (Wikipedia, statischer Fallback) ~514 Ticker
   ↓ Stufe 1   Hard-Filter: Cap>2 Mrd, AvgVol>1M, $Vol>10M, RelVol>0.6·VIX/20 oder News-Override, News vorhanden
   ↓ Stufe 1b  FinBERT-Sentiment + Sentiment-Drift (history.sentiment_history)
   ↓ Stufe 1c  Sektor-Momentum (yfinance)
   ↓ Stufe 2   Prescreening (Claude Haiku, Batches à 20)
   ↓ Stufe 2b  Alpha-Quellen: FDA, SEC Form 4, Earnings (Finnhub), Put/Call-Skew, Dealer-OI
   ↓ Stufe 2b-ext  Externer Kontext (SHADOW, ~20 Quellen, PIT-Snapshot)
   ↓ Stufe 3/3b ROI-Precheck, Pre-MC (1k Pfade, σ-only)
   ↓ Stufe 4   Deep Analysis (Claude Sonnet + Red Team) → direction/impact/surprise/ttm
   ↓ Stufe 4a/4b Bearish-Gate (deaktiviert), Impact≥4 & Surprise≥3
   ↓ Stufe 5   Mismatch = impact − 5·|48h-Move|/σ_daily   (≤0 → verworfen)
   ↓ Stufe 6/7/8 Quick-MC (3k), Intraday-Delta, Final-MC (10k, adaptive DTE)
   ↓ Stufe 9   RL-Scoring (PPO, Arming 6/30 Trades → noch inaktiv)
   ↓ Stufe 10  Options-Design: IV-Gate, ROI-Gate (VIX-skaliert), Edge-Check, Korrelations-Check
   ↓           Fractional Kelly → Trade-Mail
LERNEN      feedback.py (2×/Werktag): Outcomes, feature_stats-Bins, Pearson-Gewichte,
            Exit-Sim, RL-Arming; candidate_ledger.update_outcomes; Challenger-Engine;
            monatlich: alpha_discovery, challenger_registrar, factor_monitor (neu)
```

Normalisierung: Es gibt keinen Querschnitts-z-Score über die Kandidaten. Die Gates
sind feste Schwellen. Der „Composite“ ist faktisch eine Kette von UND-Filtern plus
`trade_scorer`/QuasiML (Bin-Durchschnitte, Gewichte impact 0,35 / mismatch 0,45 /
eps_drift 0,20).

## 2. Befunde und Korrekturen dieses Audits

| # | Bereich | Befund | Status |
|---|---|---|---|
| A1 | Scoring | `mismatch_scorer._compute_48h_move`: Fehler bzw. < 5 Kurse → **0,0**. Damit gilt Mismatch = Impact, das **stärkste** Signal; ein Datenfehler wurde zum Top-Kandidaten | **behoben**: None → Reject `mismatch_price_data_missing` |
| A2 | Scoring | Deep Analysis: fehlende Kursdaten erschienen im Sonnet-Prompt als „48h-Preisbewegung: +0,0 %“ | **behoben**: „n/a (keine Kursdaten)“ |
| A3 | Scoring | FinBERT-Ausfall → Sentiment 0,0 „neutral“, nicht unterscheidbar; der Fallback floss in die Sentiment-Historie ein und verzerrte den Drift | **behoben**: `sentiment_status`, Fallback wird nicht gespeichert |
| A4 | Methodik | `z_score` = 2-Tages-Move / σ **täglich** (korrekt: σ·√2); `abs()` ignoriert die Richtung (ein Kursrückgang nach bullisher News zählt als „eingepreist“) | Gate **unverändert** (Evidenz nötig); Forschungsfeatures `z_score_2d_scaled`, `price_move_48h_signed` im Ledger |
| A5 | Lernen | `feedback.compute_pearson_weights`: in-sample, `max(r,0)`, keine Mindeststichprobe, 2 Läufe/Werktag × Lernrate 0,05. Ein Rausch-Feature mit r = +0,02 hätte das Zielgewicht 100 % erhalten | **Sicherheitsbremse**: nur r mit ≥ 30 Entry-Tagen und 90-%-KI > 0. Auf der aktuellen Historie unverändert (Test) |
| A6 | Outcomes | Der Feedback-Loop läuft während der Handelszeit; `ret_{h}d` wurde am Zieltag aus dem **unfertigen** Tagesbalken gefüllt und nie korrigiert | **behoben**: erst füllen, wenn der Zieltag abgeschlossen ist |
| A7 | Outcomes | Horizonte 1 d / 60 d fehlten | **ergänzt** (Underlying; Kalendertage, dokumentiert) |
| A8 | Datenqualität | Eine Quelle galt als PASS, sobald keine Exception auftrat (`nws_alerts` wochenlang PASS mit 0 Zeilen) | **behoben**: `modules/external/data_quality.py` mit EMPTY_RESULT, BELOW_MIN, HIGH_NULL_RATE, NON_FINITE, DUPLICATE_CONFLICT, FUTURE_OBSERVATION, AVAILABLE_BEFORE_PERIOD, OUT_OF_RANGE, SCALE_SHIFT. PASS → WARN, Befund in der Health unter `dq`. Gegen das echte Archiv geprüft: keine Fehlalarme |
| A9 | Lernen | Es gab keine Faktor-Performance-Datenbank, keinen Walk-Forward, keinen Decay und keine Redundanz-/Leakage-/Drift-Analyse | **neu**: `modules/factor_monitor.py` (siehe 4.) |

Bereits in N11 behoben: Universum nur A–E (Finnhub-429), falsches SEC-Insider-Signal,
NewsAPI-Firmenname, Chokepoint-z, Wetterindex, Ledger-Labels, Zeitbudget je
Quelle sowie ISO-Kadenzen/Staleness.

## 3. Datenverknüpfung und zeitliche Integrität

- **PIT extern:** Jede Beobachtung trägt `observation_time` (Periode), `available_at`,
  `retrieved_at`, `vintage_time` und `source_release_time`. Der Snapshot filtert
  `available_at ≤ as_of`. ALFRED-Vintages gibt es für FRED/BTS. Bei Eurostat gilt
  EXACT nur für die jüngste Periode, Historie ist konservativ. Neu ist das
  DQ-Gate `AVAILABLE_BEFORE_PERIOD` (eine Monatsstatistik kann nicht vor
  Monatsbeginn vorliegen).
- **Frozen Context:** Der Ledger friert den externen Kontext beim ersten `note()` ein.
  `trace_candidate.py` prüft PIT-Verletzungen.
- **Entry-Preis:** Quote-Mid in der regulären Session, sonst `next_open`. Niemals der
  letzte Schlusskurs, damit keine Vor-Signal-Bewegung eingepreist wird.
- **Walk-Forward (neu):** Trainingsdaten müssen Entry < Testmonat **und**
  Outcome-Datum < Testmonat haben (Test `test_walk_forward_never_trains_on_future…`).
- **Regime (neu):** Kurse vom **Vortag**, INDPRO nur mit `available_at ≤ Tag`
  (kein Restatement-Bias).
- **Survivorship:** Das Universum ist die *heutige* Indexliste. Für Live-Signale ist
  das unkritisch. Retrospektive Backtests über ältere Zeiträume wären verzerrt;
  deshalb bewertet der Monitor nur live aufgezeichnete Ereignisse (Ledger bzw.
  history.json).
- **Mapping:** Ticker→CIK über die offizielle SEC-Liste. Ticker→Industrie→PortWatch-
  HS-Abschnitt ist explizit konfiguriert; nicht gefundene Namen werden gemeldet.
  Chokepoints werden über den Namen zugeordnet, eindeutig laut Test. Hafen-
  Aggregate haben eine Vollständigkeitsprüfung (n_ports ≥ 90 %).
- **Kalender:** Die Horizonte zählen Kalendertage, `_price_on_or_before` nimmt den
  letzten Handelstag ≤ Ziel. Feiertage werden damit implizit behandelt.

## 4. Faktor-Monitor (`modules/factor_monitor.py`, monatlich, SHADOW)

- **Datenbasis:** Geschlossene Trades und **abgeschlossene** Schatten-Trades
  (Options-P&L inklusive Spread/Prämie = `net`), dazu Ledger-Forward-Returns 1/5/20/45/60/120 d
  (`gross_*`) und `real_strat_ret_45d` (`net_45d`).
- **Je Feature:**
  - Abdeckung, n, unabhängige Tage, IC, Rank-IC mit 90-%-KI (Fisher-z über
    unabhängige Tage);
  - Monats-IC, EWMA-IC (Halbwertszeit 3 Monate), IC-IR, Vorzeichen-Konsistenz,
    3-Monats- gegen Vorperioden-IC;
  - Terzil-Spread; Performance des Top-Terzils (Trefferquote, Mittel, Median,
    Sharpe/Sortino je Trade, Profit-Faktor, Drawdown);
  - IC je Regime, PSI-Drift über 30 Tage.
- **Tags:** dead, unstable, high_value, decaying, regime_dependent,
  potential_leakage, data_drift, redundant (|ρ| ≥ 0,8), only_profitable_before_costs.
- **Adaptive Gewichte:** EWMA-IC, bei Decay der jüngste IC; Shrinkage
  n/(n+60); 0 bei n_eff < 30 oder wenn das KI die 0 enthält. Der Walk-Forward
  vergleicht die Gewichtung out-of-sample mit Gleichgewichtung. **Keine
  Produktionswirkung** (`production_use: false`).
- **Ausgaben:** `outputs/research/factor_report.{json,md}`,
  `factor_performance.jsonl` (Datenbank), `factor_weights_shadow.json`.

### Ergebnis auf echten Daten (184 realisierte Options-Trades, 52 Entry-Tage, 04–09/2026)

- **Kein Feature** hat einen signifikanten Rank-IC; alle 90-%-KIs enthalten 0.
- Alle 184 Trades: Trefferquote 31,5 %, Mittel −1,7 %, **Median −19,9 %**,
  Profit-Faktor 0,95. Die rechtsschiefe Options-Auszahlung trägt das Mittel,
  der typische Trade verliert.
- Walk-Forward-Testfolds (68 Trades, 07–09/2026): Trefferquote 28 %, Mittel −19 %,
  Profit-Faktor 0,23. Gleichgewichteter Composite: **Rank-IC −0,19**. Trades mit
  positivem Score: Trefferquote 17 %, Mittel −28 %.
- Die adaptive Gewichtung **enthält sich** in allen Folds. Das ist korrekt, denn es
  gibt keine Evidenz, also auch keinen Score und keinen Trade.
- `mismatch`, `z_score` und `price_move_48h` sind redundant (|ρ| 0,82–0,93), die
  Kernfaktoren werden also mehrfach gewichtet. EWMA-IC von mismatch −0,35.
- Decaying: impact, surprise, quick_mc_hit_rate, sigma_30d, price_move_48h.
- Regime: alle Tage VIX < 20 und Expansion (INDPRO). Regime-Abhängigkeit ist
  **nicht messbar**, deshalb wird bewusst nichts regime-adaptiv gewichtet.

**Schluss:** Die aktuelle Signal-Logik zeigt nach Kosten keinen nachweisbaren
Informationsvorteil. Die Stichprobe ist klein (52 Tage), aber die Tendenz ist
negativ. Der Engpass ist nicht die Gewichtung, sondern der Faktor-Inhalt.

## 5. Bestehende Selbstoptimierung – exakte Bewertung

| Mechanismus | Art | Out-of-sample-Kontrolle | Status |
|---|---|---|---|
| QuasiML / `feature_stats`-Bins (feedback.update_bin) | Bin-Mittelwerte mit Prior bei n < 3 | nein | aktiv, faktisch Lookup |
| `compute_pearson_weights` | in-sample-Pearson, Lernrate 0,05 | **nein** → jetzt Signifikanzbremse | eingefroren (alle r ≤ 0) |
| PPO-RL (`rl_agent`) | Reinforcement Learning | nein | Arming 6/30, inaktiv |
| Robuster PPO (`rl_robust_shadow`) | Log-Utility-Reward | Shadow | Shadow |
| Challenger-Engine + Registrar | prospektive Hypothesen, Hash-gesperrt, Alpha-Spending | **ja**, nur Daten nach Registrierung | aktiv, wartet auf Daten |
| `alpha_discovery` | Rank-IC je Datum, BH-FDR, Holdout | ja (60/40 chronologisch) | aktiv, MIN_ROWS 300 |
| `factor_monitor` (neu) | Walk-Forward, Shrinkage, Decay, Regime | ja | SHADOW |

Fazit zu Punkt 6: Es gibt **echtes, prospektiv kontrolliertes Lernen** nur über
die Challenger-Engine und alpha_discovery, und beide sind noch datenbegrenzt. Die
Produktionsgewichte sind faktisch statisch; die einzige adaptive
Produktionsregel lief ohne OOS-Kontrolle und ist jetzt gebremst. Eine Promotion
von Shadow-Gewichten bleibt eine menschliche Entscheidung (PR).

## 6. Offene Punkte und Empfehlungen (nach Wirkung)

1. **Edge-Check** („Model“ = Volatilitätsschwelle max(8 %, 0,5·σ·√T), keine
   Prognose) verwirft fast alles. Die Schatten-Trades liefern Evidenz; eine
   Entscheidung erst mit dem Faktor-Report.
2. **Mismatch-Skalierung/Richtung (A4):** `z_score_2d_scaled` und
   `price_move_48h_signed` im Faktor-Report beobachten; bei
   Überlegenheit per Challenger testen.
3. **Regime-Daten:** Inflation (z. B. FRED CPIAUCSL via ALFRED), Credit Spreads,
   Dollar, Öl und Liquidität fehlen als PIT-Reihen. Sie wären nötig, bevor
   regime-adaptive Gewichte sinnvoll sind.
4. **Ensemble/Meta-Modell:** Erst sinnvoll, wenn mindestens ein Subsystem einen
   signifikanten IC zeigt. Die Architektur (Faktor-DB → Walk-Forward → Shrinkage-
   Gewichte) ist vorbereitet.
5. **134 `except Exception:`-Stellen:** Die Stellen im Scoring-Pfad sind geprüft
   (A1–A3). Der Rest ist überwiegend Observability, die den Lauf nicht abbrechen
   darf. Empfehlung: schrittweise `log.debug` durch strukturierte
   Status-Felder ersetzen.
6. **Kosten:** Die Options-Outcomes (`net`) enthalten Spread/Prämie. Commission
   und Slippage über den Quote-Spread hinaus sind nicht modelliert. Der
   Ledger-Vergleich brutto/netto (`cost_check`) greift, sobald `real_strat_ret_45d`
   reift.
