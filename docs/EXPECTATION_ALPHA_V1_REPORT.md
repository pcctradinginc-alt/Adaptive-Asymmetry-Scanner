# Expectation Alpha V1: Umsetzungsbericht (2026-10-10, SHADOW)

**Kurzfassung.** Expectation Alpha (EA) ist als reiner SHADOW-Research-Layer integriert. Je Deep-Analysis-Kandidat entsteht eine These mit:
- Erwartungs-Gap (Modell − Markt);
- Regime aus dem bestehenden World Model;
- Cross-Asset-Bestätigung;
- deterministischem Status TRADE/WAIT/ABSTAIN/ERROR;
- eingefrorenen Kill Conditions und vorregistrierten Expressions.

Die These wird append-only festgehalten und über 20/60/120/250 Handelstage ausgewertet. Sieben Verträge (EA001–EA007) messen den inkrementellen Wert prospektiv ab dem 13.10.2026. Der Champion bleibt unverändert; der Produktionseinfluss ist strukturell NONE.

**Es gibt noch keine Evidenz.** Bis heute existiert keine Forward-Beobachtung. Jede Aussage über Alpha ist offen.

Architektur: `docs/EXPECTATION_ALPHA_ARCHITECTURE.md`

## 1. Neue Dateien
| Datei | Inhalt |
|---|---|
| `config/expectation_alpha.yaml` | Modus `shadow`. Alle Schwellen sind vorab festgelegt: Gaps, Bestätigung, Kontext, Entscheidung, WAIT-Trigger, Kill Conditions, Expressions, Sektor-Mapping (NON_PIT), Horizonte, Gruppen. Der `config_hash` steht in jeder Ledger-Zeile. |
| `modules/expectation_alpha/schemas.py` | Status-Konstanten; `FeatureValue` mit Provenienz (value, unit, source, observed/published/available/retrieved, vintage, transformation, freshness, confidence) |
| `modules/expectation_alpha/config.py` | Laden und Validieren (nur `off`/`shadow`), `config_hash`, `sector_map_hash` |
| `modules/expectation_alpha/data.py` | PIT-Zugriff: Archiv-Vintages mit `available_at < Stichtag`, nur abgeschlossene Marktschlüsse, Commodity-Zeilen höchstens bis zum Entscheidungstag; Lader injizierbar |
| `modules/expectation_alpha/future_state.py` | RoC-Satz: level, Δ1M, Δ3M, velocity, acceleration, change_of_change, z, Perzentil, Regime-Zustand, Wechselwahrscheinlichkeit, Unsicherheit |
| `modules/expectation_alpha/expectation_gap.py` | Domänen-Gaps Inflation (Prozentpunkte), Zinsen, Wachstum, Öl (z). Credit und Earnings sind UNAVAILABLE. |
| `modules/expectation_alpha/regime_change.py` | Regime aus `world_model.build_world`; empirische Wechselwahrscheinlichkeit nur aus Ankern mit bekanntem Ergebnis |
| `modules/expectation_alpha/cross_asset_confirmation.py` | Signalrahmen (vektorisiert, bis Close t), Thesen-Spezifikation (Missing ≠ negativ, mehrdeutige Signale entfallen), Kennzahlen |
| `modules/expectation_alpha/thesis.py` | Kandidat → These: macro/sector alignment, context_status, Gruppen A–E/X, Evidenz für/gegen, Vertragsmerkmale |
| `modules/expectation_alpha/timing.py` | Entscheidungsregel, WAIT-Trigger, Kill Conditions mit `kill_hash` |
| `modules/expectation_alpha/expression.py` | Vorregistrierte Expressions (UNDERLYING, SECTOR_ETF, SPY, QQQ, IWM) und Auswahlregel |
| `modules/expectation_alpha/ledger.py` | Append-only Kontext, Kandidaten, Downstream, Outcomes und Läufe; Outcome-Auflösung (immediate/triggered/kill_managed); Promotion-Evidenz |
| `modules/expectation_alpha/evaluation.py` | Gruppen je Horizont, Abstinenz-, WAIT-, Expression- und Kill-Wert, Kalibrierung, Fehlerklassen, Vorschläge (nie angewendet) |
| `modules/expectation_alpha/__init__.py` | Hooks `enrich_candidates`, `record_downstream`, `resolve_outcomes`, `evaluate` |
| `tests/test_expectation_alpha.py`, `tests/test_expectation_alpha_pit.py`, `tests/test_expectation_alpha_integration.py` | Kernlogik, Pflicht-Leakage-Tests, Integration (Champion-Gleichheit, Isolation, Promotion) |
| `tests/test_ea_reporting.py`, `tests/ea_fixtures.py`, `tests/test_fred_market_expectations.py` | Bericht und Mail; synthetische Testdaten; neue FRED-Quelle |
| `docs/EXPECTATION_ALPHA_ARCHITECTURE.md`, `docs/EXPECTATION_ALPHA_V1_REPORT.md` | Architektur und dieser Bericht |

## 2. Geänderte Dateien (klein, gezielt)
| Datei | Änderung |
|---|---|
| `pipeline.py` | Stufe 4 friert nur eine tiefe Kopie der Analysen ein. Die Auswertung (`enrich_candidates`) und der Downstream-Eintrag laufen am Laufende auf jedem Exit-Pfad, nach allen Champion-Entscheidungen und innerhalb der Finalisierungsreserve. Das Ergebnis landet nur in `stats["expectation_alpha"]`; Fehler werden nur protokolliert. |
| `feedback.py` | `resolve_outcomes` und `evaluate` neben Final-MC/V2 |
| `modules/hypothesis_contract.py` | Stage `EA_NEWS_CANDIDATE`, `validate_ea`: research_only, Einfluss NONE, `population_filter` (sicherer AST), `ea_outcome_kind`, `h1` |
| `modules/promotion_controller.py` | Evidenz-Zweig für EA (nur EA-Ledger), Parameter `ea_root` |
| `modules/final_mc_ledger.py` | `observations`/`evidence` mit den Parametern `stage`, `outcome_method` und `horizons`; Defaults unverändert, FINAL_MC verhält sich gleich |
| `config/promotion_hypotheses.yaml` | EA001–EA007 (registered_at 2026-10-10 12:00Z, forward_start 2026-10-13) |
| `modules/external/sources/real_economy.py`, `config/external_sources/real_economy.yaml` | Neue Quelle `fred_market_expectations` (DGS2, DGS10, DFF, T5YIE, T10YIE, T5YIFR; ALFRED-Vintages) |
| `config/source_health.yaml` | `downstream_extra` für die EA-Nutzung |
| `reports/weekly.py`, `modules/email_reporter.py` | Abschnitt 21 und SHADOW-Block in der Mail. Die Mail zeigt nur Aggregate und keine Ticker; der Status heißt „Shadow-TRADE“. |

## 3. Wiederverwendet (keine Parallelarchitektur)
- **Daten und Normierung:**
  - `ExternalArchive`, ALFRED-Connector-Muster;
  - `world_model.pit_snapshots`, `macro_frame`, `market_frame`, `build_world`, `DIMENSIONS`, `WP.state_threshold`;
  - `commodity_intelligence.build`.
- **Ledger und Outcomes:**
  - `final_mc_ledger`: `record_downstream`, `downstream_map`, `assign_clusters`, `evidence`, `_yf_bars`, `_registered`, `stage_contracts`;
  - `atomic_io`.
- **Verträge und Promotion:**
  - `hypothesis_contract`: Registry-Hash-Kette, sicherer AST, `in_scope`, `fires`;
  - `promotion_controller`: `run`, `decide`, `insufficiency`, `promotion_checks`, `cap_population`, `family_alpha`, `group_metrics`;
  - `production_intelligence_adapter`: `regime_label`; der Adapter filtert auf CHAMPION_TRADE.
- **Bericht und Mail:** `email_reporter`, `reports/weekly.py`.

## 4. Datenstatus und Datenlücken
| Thema | Stand |
|---|---|
| Marktseite Inflation/Zinsen | Quelle `fred_market_expectations` ist angelegt, aber **noch nicht im Archiv**. Sie braucht `FRED_API_KEY` in CI; der erste External-Data-Lauf füllt die Historie ab 2015 per ALFRED. Bis dahin sind Inflations- und Zins-Gap `UNAVAILABLE` (Grund steht im Kontext). |
| Credit-Gap | `UNAVAILABLE`: keine nicht-marktbasierte PIT-Kreditreihe, ICE-Spreads aus Lizenzgründen ausgeschlossen. Credit fließt nur in Regime und Bestätigung ein. |
| Earnings-Konsens | `UNAVAILABLE`: kein PIT-Konsens, wird nie simuliert. Damit gibt es keinen Earnings-Gap. |
| Sektor-Mapping | NON_PIT (heutige Zuordnung, ökonomische Priors), versioniert und gehasht |
| Marktschlüsse | yfinance `auto_adjust`: nur Renditen und Verhältnisänderungen werden genutzt. Rückwirkende Dividendenanpassung ist dokumentiert. |
| CPI Okt. 2025 | fehlt im Archiv (Periode übersprungen); die 3M-Rate nutzt die exakte Kalenderperiode |
| Befund World Model | `macro_indicator` misst „3M“-Raten über 91 Tage. Bei Monatsreihen überspannt das je nach Monatslänge 4 Perioden (betrifft `cpi_trend`, `indpro_3m`, `payems_3m`). In EA ist es für `cpi_3m_ann` korrigiert; das World Model selbst ist unverändert, weil das außerhalb des Scopes liegt. Empfehlung: eigener PR mit Versionswechsel. |
| Netz im Entwicklungs-Container | Kein Zugriff auf Yahoo. Der E2E-Dry-Run lief mit echtem Archiv und echtem Commodity-Build, aber synthetischen Marktpreisen. Der echte Kurspfad läuft erst in CI. |

## 5. EA001–EA007 (Population EA_NEWS_CANDIDATE, research_only, Einfluss NONE)
Gemeinsame Festlegungen (Ausnahmen in der Tabelle):
- Familie `research_only@EA_NEWS_CANDIDATE` × 12 Looks, also Bonferroni α = 0,05 / (7 × 12);
- Primärhorizont 60 Handelstage, sekundär 20/120/250 (nur beschreibend);
- netto nach Kosten (2 × 10 bp Aktie, 2 × 5 bp ETF);
- forward_start 2026-10-13;
- min_n 100, unabhängige Tage 40, Kalenderspanne 180 T, Ereignis-Cluster 60, Regime ≥ 2;
- Treffer mindestens 30 / 15 Tage / 20 Cluster;
- Promotion nur bei: Δ > 0, CI-Untergrenze > 0, Robustheit gegen Ausreißer, 2 von 3 Zeitfenstern positiv, Drawdown nicht schlechter.

Bei signifikanter Gegenrichtung gilt REJECT; ein Vorzeichenwechsel ist nur als neue Hypothese möglich.

| Vertrag | H1 | Vergleich | Evidenz heute |
|---|---|---|---|
| EA001_EXPECTATION_GAP | Gap in Thesenrichtung hat inkrementelle Prognosekraft | gap-aligned vs. alle mit verfügbarem Gap | keine (0 Forward-Beobachtungen) |
| EA002_CONFIRMATION | Bei \|gap_z\| ≥ 1 verbessert confirmation_ratio ≥ 0,7 die Outcomes | große Gaps mit ≥ 3 verfügbaren Signalen | keine |
| EA003_ACCELERATION | Beschleunigung in Thesenrichtung liefert Zusatz zum Level | innerhalb gap-aligned | keine |
| EA004_WAIT | Trigger-Einstieg schlägt Sofort-Einstieg | gepaart je WAIT, gleicher Ausstiegstag (min_n 60, Tage 30, Cluster 40) | keine |
| EA005_CONTEXT_FILTER | Starke News mit Kontext +1 schlagen starke News mit Kontext ±1 | starke News mit Kontext ≠ 0 | keine |
| EA006_ABSTENTION | ABSTAIN-Fälle sind schlechter (direction −1) | gültiger Status (ohne ERROR) | keine |
| EA007_EXPRESSION | Regelbasierte Expression schlägt UNDERLYING | gepaart, nur wo die Regel ≠ Default wählt (min_n 60, Tage 30, Cluster 40) | keine |

Lokaler Registrierungstest gegen eine Kopie der echten Registry: alle sieben `VALID`, keine Ähnlichkeitskonflikte. Die echte Registrierung erfolgt im nächsten Feedback-Lauf (`promotion_controller`). Zeilen vor der Registrierung tragen keine Vertragsauswertung und zählen daher nie.

## 6. Tests
- **Neu:**
  - `test_expectation_alpha.py` (29): Gap-Mathematik, Einheiten, UNAVAILABLE, RoC, Regime-Grenzen, Confirmation, Status, Kill-Hash, Expressions, Ledger-Idempotenz, Outcomes, Fehlerklassen, Verträge.
  - `test_expectation_alpha_pit.py` (12): Pflicht-Leakage für Preise, Archiv-Vintages, Publikationsverzug, Commodity, z/Perzentil, Wechselwahrscheinlichkeit, Outcome-Trennung, Trigger-Replay, ökonomische Invalidierung, Einstieg nach der Entscheidung.
  - `test_expectation_alpha_integration.py` (6):
    - `pipeline.main()` mit EA `off` vs. `shadow` ergibt identische Kandidaten an Stufe 5, identische Stats und Rejects;
    - ein EA-Fehler bricht den Scan nie;
    - statische Isolation;
    - Promotion: höchstens FORWARD_VALIDATED mit Einfluss NONE; REJECT bei Gegenrichtung; gepaarte Evidenz; Sektor-Dominanz-Sperre greift;
    - der Adapter wendet EA-Verträge nie an.
  - `test_ea_reporting.py` (16), `test_fred_market_expectations.py` (7).
- **Volle Suite:** siehe Abschnitt 6a (Zahlen aus dem finalen Lauf).
- **E2E-Dry-Run** (echtes Archiv, echter Commodity-Build, synthetische Preise):
  - Laufzeit: Archiv 22 s + Commodity 16,5 s + Anreicherung 5 s ≈ 44 s, Budget 150 s;
  - Gaps: Wachstum OK (z 1,95), Öl OK (z −1,29); Inflation/Zinsen UNAVAILABLE (Marktseite fehlt), Credit/Earnings UNAVAILABLE;
  - 4 Kandidaten → 4 ABSTAIN (Kontext/Bestätigung);
  - Evaluation, Ledger und Laufprotokoll werden geschrieben.

### 6a. Ergebnis volle Suite
- `python -m pytest -q` auf dem finalen Code: **1798 passed, 1 skipped** (12:15 min).
- Nachlauf der EA-, Reporting- und Promotion-Tests mit der finalen Config: 106 passed.
- Lint (`pyflakes`): Alle neuen Dateien sind sauber. Die verbliebenen Warnungen in `pipeline.py`, `promotion_controller.py`, `weekly.py` und `email_reporter.py` stammen aus dem Bestand und wurden nicht geändert.

## 7. Im Audit gefundene und behobene Fehler (vor jeder Registrierung)
1. **Velocity-Einheit:** `Δ3M/3` mischte ein 4,33-Wochen- mit einem 4-Wochen-Fenster; lineare Reihen zeigten dadurch eine Scheinbeschleunigung. Jetzt gilt `Δ3M × 4/13`.
2. **z bei nahezu konstanter Historie:** Gleitkommarauschen ergab Std ≈ 1e-13 und damit z = ±4. Jetzt gilt eine relative Toleranz, darunter ist z fehlend.
3. **3M-Rate über 91 Tage:** traf je nach Monat die Periode 4 Monate zurück. Jetzt exakt 3 Kalendermonate.
4. **WAIT-Trigger:**
   - Ein Sektor-RS-Zweig war oft schon bei der Entscheidung erfüllt; WAIT wäre dann gleich dem Sofort-Einstieg gewesen. Er wurde entfernt.
   - Signal und Ausführung lagen auf demselben Schlusskurs. Jetzt erfolgt der Einstieg am Folgetag.
5. **Datenfehler:** Ein Kurs- oder Archivausfall konnte zu ABSTAIN/WAIT aus einem Teilbild führen. Jetzt ergibt er immer ERROR.
6. **Bootstrap:** Python-Schleifen hätten bei ~5000 Beobachtungen pro Jahr den Feedback-Lauf gebremst. Jetzt vektorisiert.
7. **Outcome-Kursabrufe:** wurden auch für noch nicht fällige Horizonte gestartet. Jetzt gibt es eine Werktags-Vorprüfung.
8. **Indirekter Produktionseinfluss über die Laufzeit:** EA lief zunächst nach Stufe 4 und hätte bis zu ~60 s des Scanner-Budgets verbraucht. An knappen Tagen hätten spätere Champion-Stufen dadurch Kandidaten wegen `time_budget_exceeded` verlieren können. Jetzt läuft die Auswertung am Laufende, und bei Stufe 4 wird nur eingefroren.
9. **Einstieg nach Börsenschluss:** Bei einer Entscheidung nach 16:00 New York wäre der bereits bekannte Schlusskurs der Einstieg gewesen. Jetzt gilt der erste Schlusskurs nach der Entscheidung (`entry_not_before`).

## 8. Was NICHT bewiesen ist
- Ob Erwartungs-Gaps, Bestätigung, Beschleunigung, WAIT, Kontextfilter, Abstinenz oder Expression-Wahl Alpha liefern: **null Forward-Beobachtungen**.
- Ob die vorab gesetzten Schwellen gut sind. Sie wurden nicht optimiert und werden es nicht; eine Änderung erfordert eine neue Vertragsversion.
- Ob der Kurspfad in CI stabil läuft. Lokal gab es kein Netz; getestet ist er mit synthetischen Daten.
- Ob die Inflations- und Zins-Gaps sinnvoll sind. Die Marktseite ist noch nicht archiviert.
- Ob das Sektor-Mapping die tatsächliche Makro-Sensitivität trifft (Prior, NON_PIT).
- Ob 60 Handelstage der richtige Primärhorizont sind. Die anderen Horizonte sind nur beschreibend.

## 9. Nächste Forward-Schritte
1. Merge, danach der External-Data-Lauf mit `FRED_API_KEY`. Prüfen, dass `fred_market_expectations` im Archiv und in Source Health erscheint.
2. Erster Feedback-Lauf: EA001–EA007 werden registriert (PROSPECTIVE_CHALLENGER). Ab 13.10. zählen Beobachtungen.
3. Wöchentlich Abschnitt 21 prüfen: ERROR-Quote, fehlende Domänen, Laufzeit.
4. Erste 20-Tage-Outcomes etwa Mitte November 2026, erste 60-Tage-Outcomes etwa Anfang Januar 2027.
5. Mindest-Evidenz (100 Beobachtungen, 40 Tage, 180 Tage Spanne) frühestens etwa April/Mai 2027. Vorher ist jede Effektaussage NEED_MORE_DATA.
6. Champion-Wirkung nur über einen neuen Champion-Vertrag per menschlichem PR, nach FORWARD_VALIDATED.
7. Separat empfohlen: World-Model-PR für exakte 3M-Raten (Versionswechsel, Validierung neu).

## 10. Delegation (SPEC → SUBAGENT → OPUS REVIEW → TEST → FIX → INTEGRATE)
| Aufgabe | Modell | Review-Ergebnis |
|---|---|---|
| FRED-Quelle `fred_market_expectations` + 7 Tests | Sonnet | übernommen; nur neue Klasse und Eintrag; keine ICE-Reihen |
| Testdaten `tests/ea_fixtures.py` | Haiku | übernommen; deterministisch, Vintages und Revision korrekt |
| Abschnitt 21 und Mail-Block + 16 Tests | Sonnet | übernommen; nur Aggregate, „Shadow-TRADE“, rückwärtskompatible Signaturen |
| Gap, Regime, Confirmation, Status, Kill, Expression, Ledger, Outcomes, Evidenz, Verträge, Promotion, Hooks, PIT- und Integrationstests | Opus | selbst implementiert, Audit-Fixes siehe Abschnitt 7 |
