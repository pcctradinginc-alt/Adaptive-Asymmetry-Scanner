# Gap-Analyse: nächste Intelligenz-Stufe (Audit 2026-09-29)

Maßstab für jede neue Komponente: **Erkennt das System besser als zuvor, wann
es eine echte asymmetrische Chance hat und wann es besser nichts tut?** Mehr
Komplexität ist kein Erfolgskriterium.

## 1. Bestehende Architektur (Datenfluss)

```
Externe Daten (orchestrator, PIT-Archiv, DQ-Gates) ─┐
Kurse (yfinance/Tradier) ───────────────────────────┤
                                                    ├─ Produktion: pipeline.py (LLM-Scanner, Optionen) → Candidate-Ledger → history.json
                                                    └─ Research (SHADOW):
   ml_research: Feature-Store (Wochen-Querschnitte) → Labels → Walk-Forward je Registry-Modell → Locked → Forward-Ledger
     ├─ Unsicherheit: Quantile + konformale/isotonische Vorjahres-Korrektur, Analogien, kontrafaktische Treiber
     ├─ research_lab: Signal-DSL (Leakage-Schutz), Discovery, BH-FDR, Hypothesen-Datenbank
     ├─ meta_learning: Basis-OOS → Meta-Walk-Forward (Regime-Gewichte/Stacking) → Gate → Safe-Mode → HC-Regel
     ├─ hc_scanner: Mehrfachbedingungen → Alerts (dedupliziert) → prediction_memory (append-only)
     ├─ trade_memory: Fallgedächtnis + Failure Analyzer (echte Trades)
     └─ reports/weekly: Wochenbericht (Forward getrennt von Backtest)
Geschützt: config/research_protocol.yaml, config/meta_protocol.yaml (Hash gepinnt, CODEOWNERS)
```

## 2. Neue Fähigkeiten: vorhanden, teilweise oder fehlend

| Fähigkeit | Stand | Vorhanden | Fehlt |
|---|---|---|---|
| **World Model** | teilweise | `external/regime.py` (Inflation, Credit, Liquidität, Dollar, Öl, Strom als Labels), Markt-Regime in `factor_monitor` (Trend, Zinsen, Kurve), VIX | Einheitlicher, versionierter Zustand je Stichtag über Dimensionen; Growth, Arbeitsmarkt, Konsum, Lagerbestände, Fracht, Breite, Kredit-Risikoappetit; Unsicherheit je Dimension; Nachweis, dass er mehr erklärt als die Regime-Labels |
| **Causal Reasoning** | fehlt | nur prädiktive Walk-Forward-Tests (`research_lab`) | Lead/Lag, Granger-artig, partielle/konditionale Tests, Evidence-Level |
| **Knowledge Graph** | Ansatz | `config/industry_exposure.yaml`, `weather_exposures.yaml`, PortWatch-Industry-Map (statische Exposure-Hierarchie) | versionierte Knoten/Kanten mit Quelle, Konfidenz und Evidenztyp; Propagation Ereignis → Entitäten → Titel |
| **Active Learning** | fehlt | `scripts/source_discovery.py` (Quellen-Suche) | Informationsgewinn je fehlender Dimension; Priorisierung; Aufnahmeprüfung neuer Quellen (die DQ-/PIT-Bausteine existieren) |
| **Research Director** | Ansatz | Hypothesen manuell + Discovery (fester Suchraum) | Priorisierung aus Fehlern, Drift, Anomalien, Datenlage |
| **Research Memory** | größtenteils | `hypothesis_db` (Status, Gründe, Signal-Hash, Duplikat per Rang-Korrelation, BH über alle) | Status-Vokabular ACCEPTED/REJECTED/INCONCLUSIVE/RETEST_LATER, Begründung, Datenversion, textuelle Ähnlichkeit |
| **Counterfactual** | teilweise | Merkmals-Neutralisierung je Karte | Makro-Szenarien (Öl, Zinsen, USD, Momentum-Umkehr); Fragilitäts-Maß als HC-Bedingung |
| **Adversarial Self-Play** | teilweise | Prüfkette im Lab (Statistik, Leakage, Regime, Kosten); `red_team`/`bear_case` in der Tiefenanalyse | strukturierte Rollen mit eigenem Protokoll je Kandidat; Execution-Realismus je Signal |
| **Unknown-Unknowns** | teilweise | Failure Analyzer (echte Trades, 177 Fälle) | Clustering großer OOS-Prognosefehler über das Panel, Rückfluss in den Planner |
| **Decision Intelligence** | fehlt (Research) | `position_sizing` (Produktion, Optionen) | Portfolio-Nutzen je Kandidat: Korrelation, Konzentration, Tail-Beitrag, Shrinkage |
| **Meta-Cognition** | teilweise | `model_intelligence`, Kalibrierungsprüfung, Drift-Flags, Weekly-Warnungen | ein konsolidierter, metrisch begründeter Zustand „was wissen wir / wo irren wir“ |
| **Research-Value-Attribution** | fehlt | Zähler je Hypothese/Modell | Aufwand vs. akzeptierte OOS-Beiträge je Track |
| **Alpha Decay** | teilweise | `factor_monitor` (EWMA, Decay), Modell-Trend 13 gegen 52 Wochen | Strukturbrüche, rollende Kalibrierung/Expectancy je Feature |
| **Stress-Szenarien** | fehlt | – | Robustheitsprüfung (Crash, Zinsschock …) ohne Ersatz für OOS-Evidenz |
| **Adversarial Validation** | bewusst nicht gebaut | Drift-Band der Regime-Merkmale | begründet in `META_LEARNING_GAP_ANALYSIS.md` A10 |

## 3. Datenrisiken (gemessen, nicht vermutet)

Point-in-time-Qualität der Archiv-Quellen:

| Quelle | Historie | PIT historisch nutzbar? |
|---|---|---|
| `fred_regime_macro` (CPI, WALCL, USD, WTI; NFCI ab 2025) | 2015– | **ja** (ALFRED-Vintages, EXACT_DATE) |
| `fred_us_macro` (INDPRO, UMCSENT) | 1919– | **ja** (ALFRED) |
| `bts_freight_tsi` (US-Fracht-TSI) | 2000– | **ja** (ALFRED) |
| Destatis-Maut, Eurostat IP/ESI/Straßengüter, BTS Open Data, PortWatch | lang | **nein**: `available_at` = Abrufdatum 2026 (nur letzte Vintage, Veröffentlichungszeitpunkte unbekannt). Historisch zu nutzen hieße Revisions- und Look-ahead-Bias. |
| PortWatch Ports/Chokepoints, NWS, NHC, ENTSO-E | Wochen bis 13 Monate | nur prospektiv |

Folgen:
- Ein **historisch validierbares World Model** kann nur aus ALFRED-Reihen und Marktpreisen bestehen.
- Alternative Daten (Maut, Häfen) sind erst prospektiv testbar.
- Earnings-Revisionen, Positionierung, Lagerbestände und Arbeitsmarkt fehlen PIT.
  Arbeitsmarkt (PAYEMS, ICSA), Konsum (RSAFS) und Lager/Umsatz (ISRATIO) sind als
  ALFRED-Vintages lizenzfrei verfügbar und werden mit dem World Model ergänzt.
- Earnings-Revisionen gibt es PIT nur kommerziell (I/B/E/S etc.); das bleibt ein
  offener Punkt für Active Learning.
- Weitere Datenrisiken: Survivorship (heutige S&P-Liste), Dividendenbereinigung
  von Kursniveaus und statische Sektoren (`META_LEARNING_GAP_ANALYSIS.md` A3–A5).

## 4. Technische und Leakage-Risiken

| Risiko | Wo | Schutz |
|---|---|---|
| Normierung mit Zukunftsdaten | z-Scores von Makroreihen | nur expandierende Fenster bis t |
| Revisions-Leakage | Makro | nur ALFRED-Vintages mit `available_at < t` |
| Überlappende Ziele | 60-Tage-Drawdown bei Wochen-Stichtagen | Walk-Forward mit Purge, Block-Bootstrap über Monate |
| Mehrfachtests | 9 neue Fähigkeiten × Varianten | Präregistrierung je Phase (`config/intelligence_protocol.yaml`), BH/Bonferroni, Locked einmalig |
| Geringe effektive Stichprobe | Makro-Ziele: ~520 Wochen, 60-Tage-Überlappung → effektiv ~130 | Befunde als Evidenz-Level, keine Scheingenauigkeit |
| Parallelstrukturen | neue Module | nutzen den Feature-Store, `regime.pit_series`, Kennzahlen und Registry der bestehenden Module |
| LLM als Ground Truth | Knowledge Graph, Kausalität, Narrative | verboten; Kanten nur mit Quelle und Evidenztyp, Narrative nur aus Messwerten |

## 5. Empfohlene Reihenfolge (nach erwartetem Nutzen für „wann nichts tun“)

1. **World Model**, da es die Grundlage für Regime-Kompatibilität, Stress,
   Counterfactuals und Meta-Cognition ist. Validiert wird es direkt gegen die
   bestehende Regime-Engine: Sagt es Drawdowns, Volatilitätsspitzen,
   Sektorrotation, Aktien gegen Anleihen und die Wirksamkeit der
   Querschnittssignale (IC) besser voraus?
2. **Counterfactual-Szenarien** auf Basis des World Models: Fragilitäts-Filter
   für High-Confidence. Das ist direkt HC-relevant.
3. **Unknown-Unknown-Detektor**: Fehlercluster aus den Basis-OOS-Prognosen
   (die Daten existieren).
4. **Research Memory + Director**: Status, Ähnlichkeit und Priorisierung aus
   den Punkten 1–3.
5. **Causal Layer**: Lead/Lag und Granger-artige Tests für die ALFRED-Reihen
   gegen Sektor-Überrenditen, mit Evidenz-Leveln.
6. **Meta-Cognition + Alpha Decay + Research-Value-Attribution**: konsolidiert
   die vorhandenen Messungen.
7. **Decision Intelligence**: Portfolio-Nutzen mit Shrinkage.
8. **Knowledge Graph**: aus den vorhandenen Exposure-Configs plus gemessenen
   Lead/Lag-Kanten. Er wird erst nützlich, wenn alternative Daten PIT-Historie
   haben.
9. **Active Learning**: formalisiert die Datenlücken aus Abschnitt 3.
10. **Stress-Umgebung**, **Self-Play-Protokoll**, **Weekly-Erweiterung**,
    **Gesamtvalidierung (Challenger C–G)**, **Ablationen**, **Gate**.

Nach jedem Schritt folgen Tests, eine OOS-Messung und eine dokumentierte
Entscheidung KEEP, MODIFY oder REJECT (`docs/NEXT_INTELLIGENCE_VALIDATION.md`).
