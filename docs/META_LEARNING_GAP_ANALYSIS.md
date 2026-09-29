# Gap-Analyse vor der Meta-Learning-Schicht (Audit 2026-09-29)

## 1. Bestand

| Bereich | Vorhanden | Ort |
|---|---|---|
| Produktions-Scanner | LLM-Pipeline (Prescreen, Tiefenanalyse), Mismatch-Score, Monte Carlo, Trade-Scorer (`feature_stats`-Bins), PPO (noch nicht aktiv), robustes PPO im Schatten | `pipeline.py`, `modules/*` |
| Historischer Feature-Store | Wöchentliche PIT-Querschnitte des S&P 500 ab 2014, 24 Merkmale (Preis/Volumen als Rang, Markt, ALFRED-Makro) | `modules/ml_research.py` |
| Labels | `fwd_xs_20/60`, `mfe/mae`, `asym`, Label-Ende je Zeile; im Ledger `ret_h`, `mfe/mae`, realisierbare Optionsrenditen | `ml_research`, `candidate_ledger` |
| Walk-Forward-ML | 6 registrierte Modelle (Momentum-Regel, Elastic Net, 3× HistGBM, 2 Spezialisten), gepurgt, innere Validierung, Locked ab 2025-07 | `config/model_registry.yaml` |
| Regime | ALFRED-Makro, Energie, VIX/Trend/Zinsen | `modules/external/regime.py`, `factor_monitor` |
| Unsicherheit | Quantil-Modelle, Analogien (kNN), kontrafaktische Treiber, Walk-Forward-Kalibrierung | `ml_research.build_cards` |
| Research-Engine / Feature Factory | Signal-DSL mit Leakage-Schutz, Discovery (117 Relationen), Hypothesen-Datenbank, BH-FDR | `modules/research_lab.py` |
| Trade-Gedächtnis / Failure Analyzer | 177 Fälle, deterministische Ursachen | `modules/trade_memory.py` |
| Champion/Challenger | Regeln (`challenger.py`), ML-Registry mit Spezifikations-Hash | – |
| Performance-Attribution | Permutations-Wichtigkeit, Sektor, Regime | `ml_research` |
| Datenqualität | Data-Quality-Gates, Source-Health | `modules/external/*` |
| Versionen | Spezifikations-Hash, Code-SHA, Config-Hash, Modell-IDs im Ledger | – |
| E-Mail | Gmail-SMTP in `email_reporter._send_smtp` (Secrets `GMAIL_*`) | – |

## 2. Audit-Befunde

| # | Thema | Befund | Maßnahme |
|---|---|---|---|
| A1 | **Doppeltes Feature** | `rs_63` = `mom_3m` − SPY-Rendite. Die SPY-Rendite ist je Stichtag konstant, daher sind die Querschnittsränge **identisch**. Bei Elastic Net zählt das Merkmal doppelt, bei Bäumen ist es harmlos. | Registrierte Modelle bleiben unverändert (feste Spezifikation). Für die nächste Modellgeneration wird `rs_63` entfernt. Dokumentiert. |
| A2 | **Überkonfidente Unsicherheit** | Das 80-%-Intervall deckt im Walk-Forward nur **64,6 %** ab (2020: 50 %). P(>+10 %) ist schlechter als die Basisrate (Skill −5,7 %). Ursache sind marktweite Schocks, die Querschnittsmodelle nicht sehen. | Konformale Korrektur (CQR) und isotonische Rekalibrierung, beide nur aus dem Vorjahr. Alerts sind gesperrt, solange die korrigierte Abdeckung nicht passt. |
| A3 | **Survivorship-Bias** | Das Universum ist die heutige S&P-500-Liste. | Bewertung relativ zum Querschnittsmittel desselben Stichtags. Nicht beseitigt, nur gedämpft. |
| A4 | **Look-ahead über Dividendenbereinigung** | yfinance `auto_adjust` bereinigt historische Kurs*niveaus* rückwirkend mit späteren Dividenden. Renditen und Verhältnisse sind korrekt, `log_dollar_vol` (Niveau) ist minimal verzerrt. | Dokumentiert. Es geht nur als Rang ein, der Effekt ist klein. Nicht behoben (keine unbereinigten Volumen-Historien). |
| A5 | **Statische Sektoren** | Der Sektor wird heute abgefragt und für die Vergangenheit verwendet. | Nur für Attribution und `meta_stacking` (Sektor-Code). Dokumentiert. Die Ablation `__no_sector` misst den Einfluss. |
| A6 | **Revisionen** | Makrodaten sind ALFRED-Vintages (PIT, korrekt). Kursdaten werden bei jedem Lauf neu geladen. | `panel_hash` wird je Lauf mitgeschrieben, Seeds sind fest. Bitgenau reproduzierbar nur bei unverändertem Download. |
| A7 | **Zeitstempel** | Merkmale bis Close t, Entry Open t+1. Makrodaten sichtbar ab dem Tag nach der Veröffentlichung. Ein unfertiger heutiger Balken wird vor 21 UTC verworfen. | OK, Tests vorhanden. |
| A8 | **Prognose-Ledger nicht append-only** | `ml_predictions` schreibt `realized` in die bestehende Zeile. | Neues append-only Gedächtnis `prediction_memory`: Prognose und Outcome sind getrennte Ereignisse. |
| A9 | **Stacking-Leakage-Risiko** | Es gab noch keine Meta-Ebene. | Der Meta-Learner nutzt ausschließlich Basis-OOS-Prognosen, deren Label vor dem Meta-Testjahr endet. Automatische Leakage-Checks laufen je Fold. |
| A10 | **Adversarial Validation** | Nicht vorhanden. | Nicht gebaut: Aktienmerkmale sind je Stichtag Ränge und damit per Konstruktion verteilungsstabil. Datumsmerkmale (Makro) sind nichtstationär, ein Klassifikator trennt Zeiträume trivial (AUC ≈ 1) und sagt nichts. Ersatz: Drift-Prüfung der Regime-Merkmale gegen das 1–99-%-Band des Trainings. |
| A11 | **Alternative Daten** | Kein Modell nutzt sie (PIT-Historie < 3 Jahre). | Ablation „ohne alternative Daten“ ist n/a und wird so ausgewiesen. |
| A12 | **Forward-Trades** | Bei Shadow-Trades fehlt der Underlying-Schlusskurs (57 % der Ursachen unbekannt). 40 von 117 Outcomes sind Delta-Schätzungen. | Im Weekly Report getrennt ausgewiesen (zuverlässig / alle). |
| A13 | **Mehrfachtests** | Hypothesen laufen über BH, Modelle über Registry und Lock-Zähler. Für Meta-Varianten gab es noch nichts. | `config/meta_protocol.yaml` ist vorab registriert: Varianten fest, Bonferroni über 2 Meta-Learner, Gate vor dem Lauf fixiert, Hash gepinnt. |

## 3. Design-Entscheidungen (keine zweite Infrastruktur)

- Die Meta-Ebene nutzt den Feature-Store, die Registry, `fit`/`select_params`/`purged` und die Kennzahlen aus `ml_research`.
- Der Mailversand ist modular (`modules/mailer.py`) und nutzt dieselben Secrets wie der bestehende Report.
- Das Gedächtnis ergänzt `ml_predictions`, es ersetzt es nicht.
- High-Confidence-Schwellen stammen aus der Meta-Walk-Forward-Auswertung (Kalibrierjahre → Validierungsjahre), nicht von Hand.
