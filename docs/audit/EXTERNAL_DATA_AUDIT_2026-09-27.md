# External Data + Learning + Trade Quality Audit — 2026-09-27

Stand: `main` nach PR #40. Methode: Code-Pfade verfolgt (nicht README/Kommentare),
Live-Abrufe über GitHub Actions (Preflight + Ingestion), Probe-Skripte gegen das
Archiv, Auswertung der committeten Outputs. Alle Aussagen unterscheiden strikt:
DATEN EXISTIEREN → FUNKTIONIEREN → PIT-SICHER → FEATURE → ERREICHT MODELL →
KORRELIERT → INKREMENTELL → PROSPEKTIV VALIDIERT → BESSERE TRADE-ÖKONOMIE.

---

## 1. Executive Summary

- **Datenebene: funktioniert.** 16 aktive Quellen liefern live (15 PASS, NWS-Vorhersage WARN mit 3/45 Orten). Während des Audits wurden 11 Korrektheitsfehler gefunden und behoben (u. a. falsch etikettierte Bahndaten als „Japan-Lkw“, Stale-Quellen mit Konfidenz 1.0, doppelt archivierte ALFRED-Vintages, verlorenes `catalyst_relevance`-Feature).
- **PIT: sicher, aber meist nicht rückwirkend nutzbar.** Absichtliche Leakage-Proben (Zukunftswerte, spätere Revisionen, nachträglich ausgegebene Prognosen) wurden alle korrekt ausgeschlossen. Nur ALFRED-Quellen (BTS TSI via FRED, UMCSENT, INDPRO) sind echt PIT-sicher rückwirkend; alle anderen sind `CURRENT_VALUE_BACKFILL_ONLY` oder `FORWARD_ARCHIVE_ONLY`.
- **Modellintegration: SHADOW-only, korrekt abgeschottet.** Externe Features beeinflussen weder Score, Gates, Quasi-ML, PPO noch den Deep-Analysis-Prompt. `policy.mode = shadow` erzwingt `score_delta = 0`, `veto = False`.
- **Evidenz für inkrementellen Wert: keine.** Noch keine einzige Ledger-Zeile und kein Trade trägt externen Kontext (Candidate Ledger seit 2026-09-26, letzter Scanner-Lauf 2026-09-25). Die einzige PIT-sichere Rückwärtsprobe (US-Frachtindex vs. 117 geschlossene Trades) hat effektiv **2 Makro-Regime**, die exakt mit den Monaten April/Mai zusammenfallen → nicht auswertbar.
- **Bestehende Lernschleife ist das größere Problem:** Quasi-ML-Gewichte seit Beginn auf Startwerten eingefroren (alle Feature-Korrelationen ≤ 0; Alarm existiert). PPO ist degeneriert (aktuell 100 % BOOST, früher 100 % SKIP). Trade-Ökonomie: Mittel +9,5 %, Median −41,8 %, Trefferquote 33 %; die 5 größten Gewinner erklären 194 % des Gesamtergebnisses.
- **Empfehlung:** nichts befördern; weiter sammeln; ab 2026-09-28 füllt sich der Ledger. Drei eng spezifizierte Challenger (§28) nach ausreichender Ledger-Historie registrieren.

## 2. Current Architecture (tatsächlicher Laufzeitpfad)

| Pfeil | Implementiert | Getestet | Tatsächlich genutzt | Status |
|---|---|---|---|---|
| Quelle → HTTP (`modules/external/http.py`) | ja | ja | ja (Actions, täglich 06:17 UTC) | produktiv (Ingestion) |
| HTTP → Raw (`archive.store_raw`, Policy `hash_only`/`gzip ≤ 2 MB`) | ja | ja | ja | Hash immer, Payload nur klein |
| Raw → Normalisierung (Konnektoren je Familie) | ja | ja | ja | — |
| Normalisierung → PIT (`pit.available_as_of`) | ja | ja + Proben | ja | korrekt |
| PIT → Features (`features.py`, `*_features.py`) | ja | ja | ja | kausal |
| Features → Snapshot (`build_external_context`) | ja | ja | ja (einmal pro Scanner-Lauf) | SHADOW |
| Snapshot → Kandidat (`pipeline.attach_external_context_stage`) | ja | ja | **noch nie in Produktion gelaufen** (erster Lauf 2026-09-28) | SHADOW |
| Kandidat → Ledger (`candidate_ledger.note(external=…)`) | ja | ja | **0 Zeilen bisher** | eingefroren beim ersten Setzen |
| Ledger → Shadow/Outcome (`feature_stats_external`) | ja | ja | leer | Observability |
| Outcome → Feedback → Quasi-ML / Pearson / PPO | ja | ja | ja (nur 3 Basisfeatures) | **externe Features ausgeschlossen** (Governance-Guard) |
| → Challenger (`challenger.py`, `challengers.yaml`) | ja | ja | 2 aktive, 0 Daten | human promotion |
| → Report (`reporter.py`, `external/reporting.py`) | ja | ja | globaler Block, nicht pro Kandidat | SHADOW gekennzeichnet |
| → Trade-Empfehlung | — | — | externe Daten fließen nicht ein | korrekt |

## 3. Data Source Live Verification (Ingestion 2026-09-27 19:42 UTC)

| source_id | Herausgeber / Zugang | Auth | Ergebnis | Letzte Beobachtung | Eignung |
|---|---|---|---|---|---|
| destatis_truck_toll | Destatis GENESIS REST 2020 (ZIP, klassisches CSV), Tabelle 42191-0001 | GENESIS-Konto | PASS, 1.120 Obs, monatlich ab 2008 | 2026-08 | PRODUCTION INGESTION |
| bts_freight_tsi | FRED/ALFRED TSIFRGHT (offizielle API, Vintages) | FRED_API_KEY | PASS, 7.789 Vintage-Obs | 2026-06 | PRODUCTION, **PIT-sicher** |
| bts_open_data_tsi | data.bts.gov Socrata `bw6n-ddqk` | – | PASS, 638 | 2026-07 | Fallback (nur aktueller Wert) |
| eurostat_road_freight | Eurostat Dissemination API `road_go_ta_tott` (jährlich) | – | PASS, 15.128 | 2025 | PRODUCTION, langsam |
| eurostat_road_freight_quarterly | Eurostat `road_go_tq_tott` | – | PASS, 60.282 | 2025-Q4 | PRODUCTION |
| estat_jp_truck | e-Stat API v3, 自動車輸送統計調査 (Tabelle 0003422293, 千トン) | ESTAT_APP_ID | Abruf OK, 4.320 Obs; Erstimport nach Entfernen der Altserie 2010–2020 | aktuelle Tabelle | FORWARD/RESEARCH bis Archivierung bestätigt |
| imf_portwatch_ports / _chokepoints | IMF PortWatch ArcGIS FeatureServer | – | PASS, 9.282 / 5.880; 0 unaufgelöst | 2026-09-18 / 09-20 | FORWARD ARCHIVAL (Lizenz: privat, nicht kommerziell) |
| eurostat_sentiment | Eurostat `ei_bssi_m_r2` | – | PASS, 7.162 | 2026-08 | PRODUCTION |
| eurostat_industrial_production | Eurostat `sts_inpr_m` (B-D, I21, SCA) | – | PASS, 2.785 | 2026-07 | PRODUCTION |
| fred_us_macro | ALFRED UMCSENT + INDPRO | FRED_API_KEY | PASS, 40.445 Vintage-Obs | 2026-08 | PRODUCTION, **PIT-sicher** |
| nws_forecast / nws_alerts | api.weather.gov | UA-Kontakt | WARN (3/45) / PASS | laufend | FORWARD ONLY |
| ncei_normals | NCEI Access Data Service, normals-daily-1991-2020 | NCEI_CDO_TOKEN | PASS, 87.600 | statisch | Referenz |
| nhc_storms | NHC CurrentStorms.json + TCM-Advisory-Text | – | PASS | laufend | FORWARD ONLY |
| fhwa_faf | faf.ornl.gov | – | DEFERRED: Server setzt Cloud-IP-Verbindungen zurück | – | MANUAL (`--zip`) |
| viapass_be, asfinag_at | – | – | DEFERRED: Katalogsuche (data.europa.eu, data.gv.at) ohne maschinenlesbaren Feed; über Eurostat BE/AT abgedeckt | – | DEFERRED |
| nbs_cn, mot_cn | – | – | DEFERRED: keine offizielle offene API; Scraping ausgeschlossen | – | DEFERRED |
| kosis_kr | KOSIS OpenAPI | KOSIS_API_KEY (fehlt) | DEFERRED: Key + Tabellen-IDs fehlen | – | DEFERRED |
| noaa_storm_events, ecmwf_open_data | – | – | deaktiviert | – | DEFERRED |

Keine undokumentierten/privaten Endpoints. ArcGIS (PortWatch) ist ein offizieller Dienst ohne Versionsgarantie → Schemaänderungen werden als `SCHEMA_CHANGED` gemeldet.

## 4. Point-in-Time Audit

Leakage-Proben (Temp-Archiv, `as_of(T)` und `build_external_context(T)`):

| Fall | Erwartung | Ergebnis |
|---|---|---|
| Beobachtung mit `available_at > T` | ausgeschlossen | ✅ |
| Revision nach T | nur Version bis T | ✅ (1.5 statt 2.0) |
| Prognose mit `forecast_issue_time > T` | ausgeschlossen | ✅ |
| Revisions-Tripel (original, revidiert, aktuell) | alle unterscheidbar | ✅ `vintage_history` |
| Kandidat T1, Revision T2, Feedback T3 | T1-Werte bleiben | ✅ `feedback.py` liest nur `external_context_entry`; kein `build_external_context`/`as_of` in feedback, monthly_report, challenger |
| NHC: aufgelöster Sturm aus altem Abruf | nicht aktiv | ✅ seit PR #19 (Abruf-Marker) |

PIT-Klassen: **PIT_SAFE_BACKFILL** = bts_freight_tsi, fred_us_macro (ALFRED `realtime_start`). **CURRENT_VALUE_BACKFILL_ONLY** = Destatis, Eurostat (alle), e-Stat, bts_open_data_tsi, PortWatch (`available_at` = Datensatz-Update- bzw. Abrufzeit; Historie konservativ, aber ohne echte Vintages). **FORWARD_ARCHIVE_ONLY** = NWS, NHC. NCEI-Normals = statische Referenz.

## 5. Data Quality / Revision Audit

- Missing ≠ 0: überall `None` statt 0 (Eurostat, Destatis „…/-“, e-Stat „-/***“, PortWatch).
- Einheiten: Eurostat THS_T/MIO_TKM getrennt; e-Stat 千トン; NCEI °F; Destatis Index 2021 = 100.
- Saisonbereinigung: Destatis X13 SA als `index_sa`, BV4.1 separat; Eurostat SA/SCA gefiltert; e-Stat unbereinigt → YoY-z (PR #40).
- Partielle Perioden: PortWatch-Tage mit < 90 % der Häfen ausgeschlossen; GENESIS-Platzhaltermonate übersprungen (PR #26).
- Revisionen: ALFRED-Vintages vollständig; Doppelarchivierung erneut gelieferter Vintages behoben (PR #28).
- Reproduzierbarkeit: `payload_hash`, `parser_version`, `feature_versions`, unveränderliche Snapshots mit Inhalts-Hash. Große Payloads (> 2 MB) nur als Hash → Originalantwort dort nicht reproduzierbar, nur per Neuabruf verifizierbar.

## 6. Feature Engineering Audit

Alle Rolling-Fenster kausal (`j ≤ i`), keine zentrierten Fenster, keine Vollstichproben-Normalisierung. `rolling_zscore` schließt den aktuellen Wert in Mittel/Std ein (dämpft z, kein Leak). Hard-Data-z seit PR #20 auf Vorjahresrate (Trendproblem). `drop_incomplete_last` nutzt Wanduhr-`now` (aktuell nicht im Kontextpfad aufgerufen, P2).

Zustände: US = BTS TSI (FRED bevorzugt, sonst Socrata) + Trucking-Komponente; EU = Destatis + Eurostat je Land (quartalsweise bevorzugt); ASIA = e-Stat (seit PR #40 überhaupt befüllt); GLOBAL_FREIGHT = Combine der drei Regionen; GLOBAL_MARITIME = PortWatch GLOBAL-Aggregat. Konfidenz ≤ 0.3 bei ≤ 1 frischer Quelle — **seit PR #36 mit echtem Alter** (vorher war jede Quelle mit z „frisch“; ein 400 Tage alter Wert ergab Konfidenz 1.0). Offener P1: `eu_freight_state` kann nur Deutschland sein; die Einschränkung steht nur in `quality.eu_sources`.

## 7. Candidate Integration Audit

Snapshot einmal pro Lauf (`now` = Laufzeit), Anhang nach Ingress/Schema-Gates, vor ROI/Pre-MC/Deep-Analysis. Prescreen-Rejects erhalten keinen Kontext (P2). Alle späteren Stufen erhalten das Feld über `{**candidate}`-Spreads. `catalyst_relevance` war immer `None`, weil der Katalysator erst nach dem Anhängen existiert → behoben (PR #37). Deep-Analysis-Prompt enthält **keinen** externen Kontext; separater Shadow-LLM-Call bekommt nur strukturierte Zustände + Deep-Analysis-Felder.

## 8. Candidate Ledger Audit

Pro Signal: `signal_timestamp`, `event_id`, Status/Reject-Grund, Features, `external` (Snapshot-ID, `feature_version`, `available_at`, Zustände, Exposure, Relation, Policy mit `mode/score_delta/veto`), RL-Modell-Hash. Das reicht, um später zu wissen, was wann mit welcher Version bekannt war. Einfrieren beim ersten Setzen; innerhalb des Laufs per Referenz (Relation wird bewusst ergänzt), beim Flush serialisiert (P2: tiefe Kopie + explizites Relation-Update wäre robuster). **0 Zeilen bis heute.**

## 9.–11. Existing Learning Loop, Quasi-ML, PPO

| Komponente | Input | Min N | Aktiv | Produktion | Befund |
|---|---|---|---|---|---|
| feature_stats (Bins) | closed_trades | – | ja | ja | nur impact/mismatch/eps_drift |
| Pearson-Gewichte | closed_trades | 5 | ja | ja | **eingefroren** auf 0.35/0.45/0.20: alle r ≤ 0 → `max(r, 0) = 0` → EMA + Renormierung heben sich auf. Alarm „Lern-Loop eingefroren“ existiert. |
| Quasi-ML | Gewichte + Bins | – | ja | ja (Ranking) | Korrelationen zum Outcome: impact +0.01, mismatch +0.01, eps_drift −0.07, surprise −0.11, z_score −0.01 |
| PPO | 117 closed trades, 11-dim Obs v1 | 5 | Training ja | **nein** (Veto aus) | **degeneriert: 100 % BOOST** (deterministisch, in-sample); früher 100 % SKIP. Belohnung = Mittelwert, von Ausreißern dominiert. Neuer Kollaps-Alarm (PR #39), Dimensions-Guard (PR #38). |
| Challenger | Ledger | min_n je Arm | 2 aktiv | nie automatisch | start 2026-09-28, 0 Daten |
| backtest_thresholds | closed + shadow | – | manuell | nein | ausdrücklich nicht promotion-fähig |

PPO vs. Baseline (in-sample, nur deskriptiv): Baseline N = 117, Mittel +9,5 %, Median −41,8 %, Trefferquote 33 %, Anteil großer Verluste (< −80 %) 29 %, Profit Factor 1,22. PPO wählt dieselben 117 → Selektionsrate 100 %, Regret 0, kein Informationsgewinn. Echte Out-of-Sample-Bewertung unmöglich (Modell auf allen Trades trainiert, keine protokollierten PPO-Entscheidungen).

## 12. External Feature Learning Audit (Matrix)

| Familie | Gesammelt | Ledger | Outcomes | Shadow-Analyse | Challenger-fähig | Quasi-ML-fähig | PPO-fähig | Produktiv |
|---|---|---|---|---|---|---|---|---|
| Road Freight | ✅ | ✅ Code / 0 Zeilen | ✅ Code / leer | ✅ | ✅ (Regeln `external.*`) | ❌ Guard | ❌ | ❌ |
| Maritime | ✅ | ✅ / 0 | ✅ / leer | ✅ | ✅ | ❌ | ❌ | ❌ |
| Weather | ✅ | ✅ / 0 | ✅ / leer | ✅ | ✅ | ❌ | ❌ | ❌ |
| Supply-Chain-Divergenzen | ✅ | ✅ / 0 | ✅ / leer | ✅ | ✅ | ❌ | ❌ | ❌ |
| Real Economy | ✅ | ✅ / 0 | ✅ / leer | ✅ | ✅ | ❌ | ❌ | ❌ |

Status: B (nur Shadow). Nichts ist versehentlich in Produktion; nichts ist befördert (`promoted_rules = []`).

## 13.–18. Incremental Alpha, Ablation, Sektoren, Interaktionen, FP/FN

**Nicht durchführbar — ehrlich begründet:**
- Ledger/Trades mit externem Kontext: **0**. Ablationen BASE vs. BASE + Familie brauchen Kandidaten mit eingefrorenem Kontext und reifen Outcomes.
- Einzige strikt PIT-sichere Probe: US-Frachtindex (ALFRED) zum Einstiegszeitpunkt der 117 Trades. Nur **4 verschiedene Zustände**, praktisch 2 Regime:

| TSI-z bekannt am Einstieg | N | Einstiegstage | Mittel | Median | Trefferquote |
|---|---|---|---|---|---|
| −1,12 (Vintage Jan) | 71 + 3 | 17 | +26,4 % | −32,6 % | 39 % |
| +2,74 / +2,36 (Vintage Feb/Apr) | 40 + 3 | 17 | −19,5 % | −52,6 % | 23 % |

  Diese Aufteilung ist **identisch mit April vs. Mai** (April: 71 Trades, Mittel +29 %; Mai: 40 Trades, Mittel −22 %). Fracht-Regime und Kalendermonat sind vollständig vermengt; effektives N ≈ 2. **EXPLORATORY ONLY, keine Aussage möglich.**
- UMCSENT-z und INDPRO-z waren über die gesamte Spanne konstant im selben Bucket → keine Variation.
- Sektoren: geschlossene Trades haben kein Sektorfeld → keine Sektoranalyse möglich (Ledger speichert Sektor künftig).
- Road + Maritime, Wetter-Interaktion, Prognose-Revision, SUPPORT/CONTRADICT, vermiedene Verlierer/verlorene Gewinner: **keine Daten**. Maritime und Wetter sind Forward-only bzw. Current-Value; eine Rückrechnung wäre Look-ahead.

## 19. Effective Sample / Multiple Testing

117 Trades, 34 Einstiegstage, Kalender 2026-04-11 bis 2026-08-28 (95 % in April/Mai), 4 Makro-Zustände. Ein Makro-Wert, der 30 Aktien am selben Tag zugeordnet ist, zählt als 1 Beobachtung. Getestete Hypothesen in diesem Audit: 3 (TSI-z, Survey-z, Hard-z), alle explorativ. Keine Schwellen abgeleitet. Alpha-Spending (`ALPHA_BASE 0.10 / (n_active · n_looks)`) bleibt unverändert.

## 20. Trade Output Audit

Tagesbericht: globaler externer Block mit SHADOW-Kennzeichnung, Quellenstatus, Frachtzustände, Seefracht, Engpässe, Wetter, Quellenangaben. **Fehlt** (P2): Aufschlüsselung pro Kandidat (Road/Maritime/Weather/Exposure/Relation/Modelleffekt/Feature-Status) und Score-Tripel (Basis / externe Anpassung / final + Feature- und Modellversion). Solange SHADOW gilt, ist der Modelleffekt 0 — das sollte pro Kandidat explizit stehen.

## 21. Bugs Found

| # | Prio | Befund | Status |
|---|---|---|---|
| 1 | P0 | Stale-Quellen galten als frisch (`is_fresh = z is not None`) → Konfidenz 1.0 aus veralteten Daten | behoben #36 |
| 2 | P0 | „Japan-Lkw“ waren Güterbahn-Daten (statsCode 鉄道輸送統計) | behoben + Archiv gelöscht #35 |
| 3 | P0 | Erneut gelieferte ALFRED-Vintages wurden jedes Mal als Revision angehängt | behoben #28 |
| 4 | P0 | Aufgelöste NHC-Stürme blieben dauerhaft „aktiv“ | behoben #19 |
| 5 | P1 | `catalyst_relevance` immer `None` (Kontext vor Deep-Analyse) | behoben #37 |
| 6 | P1 | PPO-Loader ohne Dimensionsprüfung des geladenen Modells | behoben #38 |
| 7 | P1 | Kein Alarm bei PPO-Policy-Kollaps | behoben #39 |
| 8 | P1 | e-Stat: 10-stellige Zeitcodes unbekannt; Dimensionen kollabierten; Altserie gewählt; ASIA-Zustand nie befüllt; Saisonniveau-z | behoben #34/#40 |
| 9 | P1 | Eurostat: Jahres-Fallback mit 3-Jahres-Fenster → immer None; Quartals-Datensatz falsch (Cross-Trade) | behoben #21/#23 |
| 10 | P1 | Destatis/NCEI/Eurostat-IP-Parser (ZIP, Monatstabelle, Stations-ID, 413) | behoben #23–#26 |
| 11 | P1 | Hafenzuordnung (Long Beach doppelt, 4 Häfen unaufgelöst) | behoben #31–#33 |
| 12 | P1 | Quasi-ML-Gewichte eingefroren (alle r ≤ 0) | **nicht geändert** (wäre Alpha-Tuning); Alarm existiert |
| 13 | P1 | PPO-Belohnung vom Mittelwert/Ausreißern dominiert → degenerierte Policies | offen (Designfrage, kein Fix ohne Tuning) |
| 14 | P1 | `eu_freight_state` kann nur Deutschland sein, ohne Kennzeichnung im Zustand | offen |
| 15 | P2 | Prescreen-Rejects ohne externen Kontext | offen |
| 16 | P2 | Ledger-Freeze per Referenz statt tiefer Kopie | offen |
| 17 | P2 | Report ohne Aufschlüsselung pro Kandidat / Score-Tripel | offen |
| 18 | P2 | Große Raw-Payloads nur als Hash | dokumentiert |
| 19 | P2 | `drop_incomplete_last` nutzt Wanduhr | offen (nicht im Kontextpfad) |
| 20 | P3 | Exposure-Regel „keine HQ-Exposure, Quelle Pflicht“ nur Konvention, nicht erzwungen | offen |

## 22. Bugs Fixed

PRs #19–#40 (siehe Tabelle). Jeder Fix mit Regressionstest.

## 23. Test Results

`pytest`: **809 passed, 1 skipped** (Stand PR #40). Enthalten: PIT-Tests (Zukunft, Revision, Prognose, Vintage-Duplikate), Freshness, Ledger-Freeze, Governance-Guard, Challenger-Walk-Forward, PPO-Guard/Kollaps, Reporting.

## 24. Live End-to-End Run

Datenebene live verifiziert (Preflight + Ingestion, §3). Einen Scanner-Lauf habe ich **bewusst nicht manuell ausgelöst**: Er schreibt Trades in `history.json`. Der erste Produktionslauf mit externem Kontext ist Montag 2026-09-28; eine Prüfung (Trigger 14:20 UTC) verfolgt dann einen Kandidaten durch Ledger und Report.

## 25. Data Source Matrix

| Quelle | Live | Stabil | PIT-sicher | Historie | Vintages | Features | Ledger | Shadow | Challenger-bereit | Produktiv | Empfehlung |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTS TSI (ALFRED) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Code ✅ / 0 | ✅ | nach Daten | ❌ | sammeln |
| UMCSENT/INDPRO (ALFRED) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅/0 | ✅ | nach Daten | ❌ | sammeln |
| Destatis Lkw-Maut | ✅ | ✅ | aktueller Wert | ✅ 2008+ | ❌ | ✅ | ✅/0 | ✅ | forward | ❌ | sammeln |
| Eurostat Road (Q/A) | ✅ | ✅ | aktueller Wert | ✅ | ❌ | ✅ | ✅/0 | ✅ | forward | ❌ | sammeln (langsam) |
| Eurostat ESI/IP | ✅ | ✅ | aktueller Wert | ✅ | ❌ | ✅ | ✅/0 | ✅ | forward | ❌ | sammeln |
| e-Stat Japan | ✅ | neu | aktueller Wert | ✅ | ❌ | ✅ (seit #40) | ✅/0 | ✅ | forward | ❌ | Tabelle live bestätigen |
| PortWatch | ✅ | ✅ | forward | begrenzt | ❌ | ✅ | ✅/0 | ✅ | forward | ❌ | sammeln (nicht kommerziell) |
| NWS/NHC | ✅ | ✅ (NWS 3/45 WARN) | forward | ❌ | Prognose-Vintages | ✅ | ✅/0 | ✅ | forward | ❌ | sammeln |
| NCEI Normals | ✅ | ✅ | statisch | – | – | ✅ | – | – | – | ❌ | Referenz |
| FAF / BE / AT / CN / KR | ❌ | – | – | – | – | – | – | – | – | – | DEFERRED |

## 26. Feature Value Matrix

| Feature | Ökonomischer Mechanismus | Quelle | Frequenz | Effektives N | PIT | Inkrementeller Wert | Stabilität | Sektor | Status |
|---|---|---|---|---|---|---|---|---|---|
| us_freight_tsi_z | physische Güterbewegung → Industrie/Transport-Umsatz | BTS/ALFRED | monatlich | ≈ 2 Regime | sicher | unbekannt | unbekannt | Industrie, Transport | INSUFFICIENT_DATA |
| eu/de truck z | Lkw-Fahrleistung ↔ Produktion | Destatis, Eurostat | monatl./quartalsw. | 0 | aktueller Wert | unbekannt | – | Industrie, Chemie, Autos | KEEP_COLLECTING |
| global_maritime_z, Engpässe | Welthandel, Lieferketten | PortWatch | täglich | 0 | forward | unbekannt | – | Logistik, Einzelhandel, Halbleiter | KEEP_COLLECTING |
| road_shipping_divergence | Inland vs. Seehandel | abgeleitet | – | 0 | gemischt | unbekannt | – | Industrie | EXPLORATORY |
| hard_vs_survey_divergence | Stimmung vs. reale Produktion | Eurostat/ALFRED | monatlich | ≈ 0 Variation | EU aktuell / US sicher | unbekannt | – | zyklisch | INSUFFICIENT_DATA |
| HDD/CDD-Anomalie | Energienachfrage | NWS + NCEI | täglich | 0 | forward | unbekannt | – | Versorger, Energie | KEEP_COLLECTING |
| Sturm-/Wetterstörung | Schadenlast, Flugausfälle | NHC/NWS | Ereignis | 0 Ereignisse | forward | unbekannt | – | P&C, Airlines, Logistik | KEEP_COLLECTING |
| external_relation (SUPPORT/CONTRADICT) | Kontext bestätigt/widerspricht Katalysator | LLM-Shadow | pro Kandidat | 0 | pro Lauf | unbekannt | – | alle | KEEP_COLLECTING |

## 27. Current Learning State

Geschlossene Trades 117 · reife Shadow-Trades 65 von 232 (Mittel −20,8 %, Median −9,1 %, Trefferquote 29 %) · Ledger-Zeilen 0 · unabhängige Kandidatentage im Ledger 0 · PPO-Trainingsstichprobe 117 · PPO degeneriert (100 % BOOST) · PPO-Veto **aus** · aktive Challenger 2 (`final_mc_dte_shadow`, `trade_score_61`, Start 2026-09-28) · externe Challenger 0 · befördert 0 · unbeförderte externe Hypothesen 12 (H1–H12) · Quasi-ML-Gewichte impact 0.35 / mismatch 0.45 / eps_drift 0.20 (eingefroren).

## 28. Highest-Priority Future Challengers (Spezifikation, NICHT registriert)

Registrierung erst, wenn der Ledger ≥ 8 Wochen Daten hat; `start_date` strikt nach `registered_on`; Ausführung ausschließlich per Challenger-Mechanismus mit menschlicher Beförderung.

**C1 — Gemeinsame Fracht-Kontraktion (H5-Familie)**
feature `external.states.global_freight_state` ∈ {CONTRACTION, STRONG_CONTRACTION} UND `global_maritime_state` kontrahierend, jeweils confidence ≥ 0.5 · Richtung: BULLISH-Kandidaten in exponierten Sektoren schlechter · Rationale: gleichzeitige Schwäche von Straße und See = echte Nachfrageschwäche, nicht ein Kanal · Sektoren: Industrials, Transportation, Materials, Autos, Chemicals (Exposure ≥ MEDIUM laut `industry_exposure.yaml`) · Challenger-Regel: diese Kandidaten nicht empfehlen · Baseline: Produktion · Primärmetrik: mittlere reale Optionsrendite pro Trade; sekundär Anteil Verluste < −80 % und verlorene Gewinner · min N 40 je Arm · min 15 unabhängige Signaltage und ≥ 2 Zustandswechsel · Dauer max. 9 Monate.

**C2 — CONTRADICT-Relation (H12)**
feature `external.relation.relation == CONTRADICT` mit materiality ≥ MEDIUM · beide Richtungen zulässig (Widerspruch kann starke idiosynkratische Firmen markieren) · Rationale: Kontext widerspricht dem Katalysator · Sektoren: alle mit relevance ≥ LOW · Regel: nur markieren, Vergleich CONTRADICT vs. NEUTRAL/SUPPORT · Primärmetrik: mittlere Rendite, zweiseitig · min N 30 CONTRADICT · min 12 Signaltage · max. 9 Monate.

**C3 — Wetterstörung bei wetterexponierten Sektoren (H4)**
feature Wetterstörung (NWS-Warnungen/NHC-Track in kuratierter Exposure-Region) · Richtung: kurzfristig negativ für Airlines/Logistik, positiv für Versorger bei HDD/CDD-Extremen · Rationale: operative Störung/Nachfrage · Sektoren: Airlines, Logistics, Trucking, Utilities, P&C · Regel: nur markieren · Primärmetrik: mittlere Rendite; sekundär Prognose-Revision vs. Absolutwert · min 15 unabhängige Wetterereignisse · max. 12 Monate.

## 29. Remaining Limitations (nach Evidenz und Wirkung)

1. **Zu wenig Outcomes und unabhängige Tage** (117 Trades, 34 Tage, 2 Makro-Regime; 5 Gewinner = 194 % des Ergebnisses).
2. **Ledger leer** — keine externe Evidenz vor Oktober 2026.
3. **Trade-Ökonomie/Exit/Optionsstruktur** — Median −42 %, Shadow-Median −9 %: Selektion und Exit sind der größte Hebel, nicht zusätzliche Daten.
4. **Schwache Basis-Features** — |r| ≤ 0.11; Quasi-ML lernt nichts.
5. **PPO-Belohnung** — mittelwert- und ausreißergetrieben, degeneriert.
6. **PIT-Historie externer Daten** — nur ALFRED rückwirkend sicher.
7. Exposure-Mapping, LLM-Kontext: derzeit kein messbarer Engpass.

## 30. Recommended Next Actions

1. Laufen lassen: ab 2026-09-28 Ledger mit eingefrorenem Kontext füllen; nichts befördern.
2. Montag: einen Kandidaten End-to-End im Ledger/Report verifizieren.
3. Per-Kandidat-Block im Report mit „Modelleffekt 0 / SHADOW ONLY“ (P2).
4. PPO: Belohnung robust machen (z. B. Median/geclippte Rendite) — nur als Challenger, Veto aus lassen.
5. Quasi-ML: Ursache der fehlenden Vorhersagekraft der Basis-Features untersuchen, bevor externe Features hinzukommen.
6. Nach 8–12 Wochen Ledger: C1–C3 registrieren.

---

## Antworten Q1–Q22

- **Q1** Ja für alle 16 aktiven Quellen (NWS 3/45 WARN). FAF, BE, AT, CN, KR sind bewusst DEFERRED.
- **Q2** Automatische Produktions-Ingestion: ALFRED (BTS TSI, UMCSENT, INDPRO), Eurostat, Destatis GENESIS, NCEI. Forward-Archiv: PortWatch, NWS, NHC. e-Stat erst nach Live-Bestätigung der neuen Tabelle.
- **Q3** Deaktiviert bzw. forward-only: FAF (manuell), Viapass, ASFINAG, NBS, MOT, KOSIS, NOAA Storm Events, ECMWF; PortWatch, NWS und NHC nur Forward-Archiv.
- **Q4** Ja im Sinne von „kein Leak“; rückwirkend echt PIT-sicher nur ALFRED.
- **Q5** Nein. Proben für Zukunft, Revision und Prognose sind bestanden, und der Feedback-Pfad rechnet nie neu.
- **Q6** Code ja; tatsächlich noch 0 Zeilen (erster Lauf 2026-09-28).
- **Q7** Nein (SHADOW, `score_delta = 0`, `veto = False`, Guard gegen Lernpfade).
- **Q8** Nein — keinerlei Evidenz.
- **Q9** Derzeit keine nachweisbar; Daten fehlen.
- **Q10** Nicht bestimmbar; Korrelation zu bestehenden Features erst mit Ledger-Daten.
- **Q11** Unbekannt (keine gemeinsame Historie). Ist Challenger C1.
- **Q12** Unbekannt.
- **Q13** Unbekannt (Wetter forward-only, 0 Ereignisse im Ledger).
- **Q14** Unbekannt; die Hypothese betrifft Airlines, Versorger, Logistik und P&C.
- **Q15** Unbekannt (0 Relationen). Ist Challenger C2.
- **Q16** Nein: Die Gewichte sind eingefroren, weil alle Korrelationen ≤ 0 liegen. Die Lernschleife läuft technisch, lernt aber nichts.
- **Q17** Degeneriert (100 % BOOST, vorher 100 % SKIP).
- **Q18** Nach 8–12 Wochen Ledger: gemeinsame Fracht-Kontraktion, CONTRADICT-Relation, Wetterstörung in exponierten Sektoren.
- **Q19** Hard-vs-Survey (keine Variation), Engpass-Anomalien (zu selten), Prognose-Revisionen (zu wenige Ereignisse), Japan (neu).
- **Q20** Nein — nicht messbar; keine Integration empfohlen.
- **Q21** Weiter wie bisher; zusätzlich Sektor und Optionsdaten pro Trade vollständig im Ledger, protokollierte PPO-Scores, KOSIS nur bei vorhandenem Key.
- **Q22** Keine weiteren Makro- oder Fracht-Quellen, bevor die bestehenden Evidenz liefern; kein Scraping von CN/BE/AT-Seiten; keine weiteren Wetter-Variablen.

---

# Nachtrag (2026-09-27 abends): Nachweise, PIT je Quelle, Inkrementalanalyse, Challenger

## N1. e-Stat Japan: final

- Geladen wird ausschließlich **自動車輸送統計調査** (STAT_NAME `@code 00600360`), Tabelle **0003422293**, Einheit 千トン (Tausend Tonnen), gegliedert nach gewerblichem und privatem Verkehr sowie Fahrzeugklasse. Die Auswahl verlangt 自動車輸送統計 in `STAT_NAME` oder `STATISTICS_NAME`; bevorzugt wird die jüngste Monatstabelle.
- Die Güterbahn-Daten (00600350 鉄道輸送統計調査) und die Altserie 2010–2020 sind aus raw/ und normalized/ gelöscht. Die neue Tabelle wurde als Erstimport archiviert (4.320 Werte, bis 2026-03). Manifeste und Health-Logs vergangener Läufe bleiben als Protokoll bestehen.
- Regressionstests prüfen explizit:
  - Tabellen fremder Statistiken werden verworfen.
  - `stat_code` 00600360 wird akzeptiert.
  - 3 Werte mit unterschiedlichen `@tab`/`@cat01`/`@cat02` ergeben 3 getrennte Identitäten.
  - 10-stellige Zeitcodes werden korrekt gelesen.
- Frische: e-Stat erscheint mit etwa 6 Monaten Verzug, daher eigene Klasse `monthly_lagged` (240 Tage).

## N2. „fabcity/awesome-fabcity-data#35“

Kommt weder im Repository noch in der Git-Historie vor. Keine Aktion dieser Sitzung hat ein anderes Repository berührt; alle PRs liegen in pcctradinginc-alt/Adaptive-Asymmetry-Scanner. Mit hoher Wahrscheinlichkeit ist es ein Anzeige- oder Logging-Artefakt außerhalb des Scanners.

## N3. Source- und PIT-Tabelle (Archiv, Stand 2026-09-27 20:00 UTC)

| Quelle | Endpoint | Letzte Beob. | Frequenz | Revisionen im Archiv | available_at | PIT-Qualität | Lizenz | Status |
|---|---|---|---|---|---|---|---|---|
| bts_freight_tsi | FRED/ALFRED `series/observations` (realtime) | 2026-06 | monatlich | **317 IDs mit mehreren Vintages** | ALFRED `realtime_start` | **PIT_SAFE** | FRED ToU | PASS |
| fred_us_macro | ALFRED UMCSENT, INDPRO | 2026-08 | monatlich | **1.395 IDs mit Vintages** | `realtime_start` | **PIT_SAFE** | FRED ToU | PASS |
| bts_open_data_tsi | data.bts.gov Socrata `bw6n-ddqk` | 2026-07 | monatlich | keine | Datensatz-Update 2026-09-14 | CURRENT_VALUE | Public Domain | PASS (Fallback) |
| destatis_truck_toll | GENESIS REST 2020 `data/tablefile` 42191-0001 | 2026-08 | monatlich | ab jetzt forward (Wertänderung = neue Vintage) | Abrufzeit | CURRENT_VALUE / forward | DL-DE-BY-2.0 | PASS |
| eurostat_road_freight (+ quarterly) | Dissemination API `road_go_ta_tott` / `road_go_tq_tott` | 2025 / 2025-Q4 | jährlich / quartalsweise | forward | Datensatz-Update | CURRENT_VALUE; **EXACT nur jüngste Periode** (N4) | Eurostat reuse | PASS |
| eurostat_sentiment / _industrial_production | `ei_bssi_m_r2` / `sts_inpr_m` | 2026-08 / 2026-07 | monatlich | forward | Datensatz-Update | wie oben | Eurostat reuse | PASS |
| estat_jp_truck | e-Stat API v3 `getStatsData` 0003422293 | 2026-03 | monatlich (Verzug ~6 Monate) | forward | Abrufzeit (kein Release-Datum in der API) | CURRENT_VALUE / forward | Gov. Standard Terms 2.0 | PASS |
| imf_portwatch_ports / _chokepoints | ArcGIS FeatureServer Daily Ports / Chokepoints | 2026-09-18 / 09-20 | täglich | **435 IDs mit Vintages** (spätere Korrekturen, z. B. unvollständige Tage) | **Abrufzeit, nie activity_date** | FORWARD (Historie nur konservativ) | privat, nicht kommerziell | PASS |
| nws_forecast / _alerts | api.weather.gov | laufend | stündlich | Prognose-Vintages je `forecast_issue_time` | Abrufzeit | FORWARD | Public Domain | WARN (3/45) / PASS |
| nhc_storms | CurrentStorms.json + TCM-Text | laufend | Ereignis | Advisory-Vintages | Abrufzeit | FORWARD | Public Domain | PASS |
| ncei_normals | Access Data Service normals-daily-1991-2020 | statisch | – | – | Abrufzeit | Referenz | Public Domain | PASS |
| Viapass, ASFINAG, NBS, MOT, KOSIS, FAF | – | – | – | – | – | – | – | DEFERRED (§3) |

**PIT-Einzelnachweise:**
- **BTS:** Rückwärtsanalysen nutzen `archive.as_of(T)` über ALFRED-Vintages, nicht die heute revidierte Historie.
- **PortWatch:** `activity_date` wird nie als `available_at` verwendet.
- **Wetter:** `forecast_issue_time` ist von `forecast_valid_time` getrennt, realisiertes Wetter fließt nie als Prognose ein (Proben §4).
- **Ledger T1/T2/T3:** Test `test_ledger_keeps_t1_context_after_t2_revision`. Nach einer Revision bei T2 behält der Ledger den T1-Wert (130), `as_of(T1)` liefert weiter 130 und `as_of(T2)` den revidierten Wert 120. Beide Vintages bleiben im Archiv.

## N4. Korrigiertes PIT-Etikett (Eurostat)

`available_at` = Update-Zeit des Datensatzes ist nur für die jüngste Periode exakt. Ältere Perioden waren früher verfügbar, der Wert ist dort also konservativ und kein Leck. Bisher trug die gesamte Historie das Etikett `EXACT_TIMESTAMP`.
- **Fix im Parser:** ältere Perioden erhalten `CONSERVATIVE_DATE`.
- **Einmal-Migration** (`scripts/migrate_precision_2026_09_27.py`): 24.358 Archivzeilen, nachweislich nur das Präzisions-Etikett geändert (0 Abweichungen in allen anderen Feldern), idempotent.

## N5. Beweise: SHADOW = OFF, keine externen Daten im Produktionslernen

`tests/test_external_shadow_proofs.py`:
- Kein Entscheidungsmodul (Trade-Score, Quasi-ML, Risk-Gates, Mismatch, RL, Deep-Analysis) enthält `external_context`.
- Quasi-ML-Ranking, Trade-Score und RL-Scoring sind mit und ohne maximal negativen externen Kontext (STRONG_CONTRACTION, CONTRADICT, Wetterstörung 0.9) **identisch**.
- Die Pearson-Gewichte sind mit und ohne `external_context_entry` in den geschlossenen Trades identisch.
- PPO-Beobachtungen und -Belohnungen sind mit und ohne externen Kontext identisch.

Damit gilt: Freight/Shipping/Weather → Ledger → Outcomes → Research, aber **nicht** → Pearson-Gewicht oder PPO-Input.

## N6. 117 geschlossene Trades: PIT-saubere Inkrementalanalyse (`scripts/incremental_analysis.py`, EXPLORATIV)

**Stichprobenstruktur:**
- 117 Trades, **34 unabhängige Signaltage**, 4 Monate (2026-04-11 bis 2026-08-28).
- Sektoren: nicht erfasst (die geschlossenen Trades haben kein Sektorfeld; der Ledger speichert es künftig).
- **2 Fracht-Regime**, deckungsgleich mit den Monaten: CONTRACTION = April (71) + August (3), STRONG_EXPANSION = Mai (40) + Juli (3).

**Vorab festgelegter Filter „BULLISH entfernen, wenn US-Fracht kontrahiert“** (Schwelle aus `classify_state`, nicht getunt):

| | Base | + US-Fracht-Filter | Δ |
|---|---|---|---|
| N Trades | 117 | 52 | −65 |
| Mittel | +9,5 % | −11,2 % | −20,7 pp |
| Median | −41,8 % | −51,9 % | −10,0 pp |
| Anteil großer Verluste (≤ −80 %) | 29,1 % | 26,9 % | −2,2 pp |
| Erwartete Rendite je Signal | +9,5 % | −5,0 % | **−14,5 pp** (Cluster-CI 95 %: −36 % bis +11 %) |
| Vermiedene Verlierer / verlorene Gewinner | | 38 / 27, davon **15 große Gewinner ≥ +100 %** | |

Der Filter hätte mehr Wert vernichtet als vermieden: Die großen asymmetrischen Gewinner lagen im „Kontraktions“-Monat. Wegen der Deckung von Regime und Monat (effektiv 2 Beobachtungen) ist das weder für noch gegen Fracht eine belastbare Aussage.

**Orthogonalität von us_freight_z zu den bestehenden Features** (je Tag / je Trade):

| Feature | je Tag | je Trade |
|---|---|---|
| mismatch | −0,60 | −0,49 |
| z_score | +0,58 | +0,44 |
| impact | −0,28 | – |
| surprise | −0,18 | – |
| eps_drift | +0,13 | – |

Das ist ein Kalender-Artefakt (andere Kandidatenmischung im April und im Mai), kein Beleg für Redundanz oder Unabhängigkeit. VIX, Sektor-Momentum und Preis-Momentum sind in den geschlossenen Trades nicht gespeichert und damit nicht prüfbar. EU-Fracht, Shipping und Wetter sind für diese Trades nicht rückwirkend PIT-sicher, also N/A.

**Ledger-Teil:** Er wertet ab Oktober automatisch aus (Familien US / EU / ASIA / Shipping / Road+Shipping / Wetter, gleiche Kennzahlen) und läuft monatlich im Workflow. Branchenspezifische Auswertungen (Industrials: Fracht + Shipping, Airlines: Wetter, Versorger: HDD/CDD, P&C: NHC …) ergeben sich über die Exposure-Relevanz im eingefrorenen Kontext.

## N7. Robuster PPO-Challenger (`modules/rl_robust_shadow.py`)

**Aufbau:**
- Belohnung = Log-Depotwachstum bei realem Positionsanteil: log(1 + f·s·r)/f mit f = `portfolio.max_position_pct` (0,10), s = 1,5 bei BOOST.
- Chronologisches Neu-Training mit festem Seed; Walk-forward auf 80 % / 20 %.
- Kollaps-Kennzeichnung; Aktionen nur als `features.rl_robust_action` im Ledger.

**Ergebnis heute:**
- log(1 + r) mit vollem Depot pro Trade führt zu 100 % SKIP. Der geometrische Mittelwert pro Trade liegt bei −54 %, weil 25 % der Trades Totalverluste sind.
- Mit realem Positionsanteil wählt das Modell 100 % NORMAL, in-sample wie walk-forward.
- Das pathologische „immer BOOST“ ist beseitigt. Eine Trennung der Trades gelingt mit den 11 Merkmalen und 117 Trades aber nicht, passend zu den Feature-Korrelationen |r| ≤ 0,11.
- Bewusst nicht weiter justiert (das wäre Tuning).

**Registrierung:** Challenger `ppo_robust_shadow` ist registriert: Start 2026-09-28, min_n 30, 12 unabhängige Tage, Metrik `real_strat_ret_45d`.

## N8. Automatische Registrierung der drei Challenger nach 10 Wochen

**Vorschläge (eingefroren am 2026-09-27 in `config/challenger_proposals.yaml`, gesichert per SHA-256-Lock):**
- `ext_joint_freight_contraction` (H5)
- `ext_relation_contradict` (H12)
- `ext_weather_disruption_exposed` (H4)

Alle drei sind als Filter formuliert: Produktion ohne die ausgelösten Kandidaten. Primärmetrik ist die reale Strategierendite; die Arm-Differenz misst verlorene Gewinner mit.

**Ablauf (`modules/challenger_registrar.py`, täglich im Feedback-Workflow):**
- Sobald der Ledger ≥ 70 Tage und ≥ 100 Zeilen umfasst, wird jeder Vorschlag mit `registered_on` = Tag und `start_date` = Folgetag eingetragen.
- Die ersten 10 Wochen werden **nie** ausgewertet, nur gezählt.
- Eine nachträgliche Änderung der Spezifikation (Hash) verhindert die Registrierung.
- `min_clusters` (12–15 unabhängige Tage) verschärft das Minimum pro Challenger.

Getestet: nicht vor 70 Tagen, danach genau einmal, Hash-Schutz, nur Zeilen nach der Registrierung, Filterlogik. Promotion bleibt ein menschlicher PR; kein Produktionsgewicht ändert sich automatisch.

## N9. Pfad-Nachweis für echte Kandidaten

`scripts/trace_candidate.py` gibt für Ledger-Kandidaten den Pfad aus:
- Quellen des Snapshots mit `available_at_max` ≤ Signalzeit
- Neuaufbau des Kontexts zum Signalzeitpunkt aus dem **heutigen** Archiv mit Vergleich zu den eingefrorenen Werten (Abweichung = PIT-Leck oder Versionswechsel)
- Zustände, Exposure, Relation
- Ledger-Status und Richtung
- Real- oder Shadow-Trade, Outcome, Feedback

Probe mit dem heutigen echten Kontext: keine PIT-Verletzung, exakt reproduzierbar. Der erste echte Lauf ist 2026-09-28; die geplante Prüfung um 14:20 UTC wendet das Skript auf reale Kandidaten an.
