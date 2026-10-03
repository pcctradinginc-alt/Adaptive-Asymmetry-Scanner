# Datenquellen-Programm: Audit, Reihenfolge, Bewertung

Stand: 2026-10-03 · Alle Quellen sind **RESEARCH/SHADOW**.

Leitfrage je Quelle:

> Liefert diese Quelle Information, die das System zum Entscheidungszeitpunkt noch nicht
> besitzt? Verbessert sie nach Kosten und Leakage-Kontrolle Entscheidungen auf neuen Daten?
> Unter welchen Bedingungen ist sie nützlich?

## 1. Bestehende Architektur (wird genutzt, nicht dupliziert)

| Baustein | Ort | Rolle im Programm |
|---|---|---|
| SourceContract | `config/external_sources/*.yaml`; Pflichtfelder in `modules/external/registry.py`, Alt-Data-Felder in `modules/alt_data/registry.ALT_CONTRACT_FIELDS` | Anbieter, Endpoint, Lizenz, Auth (nur Env/Secret), Rate Limits, Frequenz, Historie, Revisionspolitik, PIT-Fähigkeit, **revision_risk, maintenance_cost, semantics** (neu) |
| RAW → NORMALIZED | `modules/external/http.py` (Retry, Backoff, **Retry-After bei 429** neu, Secret-Redaktion, Hash, Fingerprint); `modules/external/pit.py` `Observation` | source_id, observation_time, available_at, retrieved_at, vintage_time, parser_version, payload_hash |
| ENTITY-MAPPED | `modules/entity_resolution/` (SEC + GLEIF, append-only, valid_from/valid_to, Konfidenz) | Ticker ↔ CIK ↔ LEI ↔ Mutter ↔ **Töchter** (neu), Land |
| FEATURE STORE | `modules/alt_data/registry.py` + `feature_store.attach` (merge_asof, nie 0 statt NaN) | nur registrierte Features in Modell und Hypothesen |
| Inkrementeller Wert | `modules/alt_data/evaluate.py` (Protokoll `alt-v1`, gepinnt) | Baseline gegen Baseline + Quelle. Gemessen: ΔMonatsrendite (Bootstrap, Bonferroni über alle Quellen), ΔIC, Brier, ECE, LogLoss, Precision@K, Sharpe, MaxDD. Redundanz-Screen, Sektor/Regime → KEEP/MODIFY/REJECT |
| Hypothesen | Factory, Research-Lab, `config/alt_hypotheses.yaml` (vorab registriert) | Einfrieren, Walk-Forward, BH, Placebo/Lag, Replikation, Ablation, Prospective Challenger |
| Meta-Learning über Quellen | `evaluate.conditional_value` → `outputs/research/source_conditions.json` (**neu**) | IC je Quelle × Sektor × Regime × Horizont (20/60 T), Jahres-IC, Alpha Decay |
| Produktion | PromotionController → ProductionIntelligenceAdapter | einziger Weg; keine Quelle schaltet sich selbst produktiv |

## 2. Audit der 21 Quellen (+ optionale)

Status:

- **VOLL:** produktiv im Research-Kreislauf.
- **TEIL:** vorhanden, aber unzureichend.
- **REGIME:** nur Makro/Regime, nicht firmenspezifisch.
- **FEHLT:** noch nicht angebunden.

| # | Quelle | Bestand vorher | Lücke | Diese Runde |
|---|---|---|---|---|
| 1 | GLEIF | TEIL: LEI und Eltern; Live nur 24/150 HIGH, 126 LOW, 0 Eltern | `filter[entity.legalName]` ist exakt, SEC-Kurzformen („NVIDIA CORP“) treffen nicht; LOW-Fälle wurden jeden Lauf neu versucht; keine Töchter | **ausgebaut** (siehe 3.1) |
| 2 | SEC EDGAR | TEIL: Form 3/4/5, Submissions (8-K-Items, Verzögerungen, NT). Bewertung **REJECT** | keine XBRL-Fundamentaldaten, keine Text-Diffs | **XBRL companyfacts neu** (3.2); Text-Diffs offen |
| 3 | USAspending | FEHLT | – | nächster Schritt |
| 4 | TED | VOLL ab 2025: 11 042 Zuschläge, Feld-Probe 2016–2024 = 0 % | ältere Bekanntmachungen nutzen vermutlich andere Felder; Töchter mit anderem Namen fehlen | Tochter-Matching über GLEIF (PIT); Legacy-Felder offen |
| 5 | UN Comtrade | FEHLT | Länder-/Produkt-Exposure nötig (GLEIF-Land) | nach USAspending |
| 6 | EIA | FEHLT (ENTSO-E für Strom EU vorhanden, REGIME) | – | später |
| 7 | ALFRED/FRED | VOLL für Regime: cpi_yoy, Fed-Bilanz, USD, WTI über ALFRED-Vintages (`real_economy.py`, `ml_research.macro_features` nur bis Vortag veröffentlicht) | weitere Serien (Arbeitsmarkt, Industrieproduktion) | später |
| 8 | Federal Register | FEHLT | Entity-Mapping auf Branchen nötig | später |
| 9 | ClinicalTrials.gov + openFDA | FEHLT | Sponsor → Emittent über GLEIF-Töchter | später |
| 10 | GDELT | FEHLT | Rauschen/Deduplizierung | später |
| 11 | PEGELONLINE | FEHLT (Flusspegel als DATA_GAP in der Factory) | Exposure „Rhein-Chemie“ nicht PIT | später |
| 12 | GIE AGSI/ALSI, ENTSO-E | ENTSO-E VOLL (REGIME); GIE FEHLT | – | später |
| 13 | NOAA Storm Events, OpenFEMA | NOAA TEIL (`noaa_storm_events`, enabled=false, Backfill-Modus); OpenFEMA FEHLT | Standort-Exposure fehlt | später |
| 14 | Census Building Permits | FEHLT | – | später |
| 15 | CFTC COT | FEHLT | – | später |
| 16 | NASA FIRMS | FEHLT | Standort-Exposure fehlt | später |
| 17 | Wikimedia Pageviews | FEHLT | – | später |
| 18 | PatentsView/EPO | FEHLT | Assignee → Emittent (GLEIF-Töchter) | später |
| 19 | PyPI/npm/GitHub | FEHLT (DATA_GAP) | Paket → Emittent-Mapping | später |
| 20 | OpenSky | FEHLT | – | später |
| 21 | DWD/Copernicus | NWS/NCEI/ECMWF vorhanden (REGIME); DWD FEHLT | – | später |
| opt. | IMF PortWatch, Lkw-Maut, Eurostat | VOLL (REGIME) | – | – |
| opt. | Certificate Transparency, Wayback, OSM, Abwasser, USDA, Cloudflare/RIPE | FEHLT | nur RESEARCH, nie produktiv (fragil) | – |

**Befund Survivorship:** 147 von 767 Tickern des PIT-Universums haben keine CIK, weil die SEC-Tickerliste nur aktive Firmen führt (z. B. ABMD, AGN, CELG). Alt-Features für diese Titel sind NaN, nie 0. Ein Name→CIK-Abgleich über die EDGAR-Gesamtliste ist als eigener Schritt offen.

## 3. Diese Runde

### 3.1 GLEIF (Schritt 1)

- **Suche:** Volltext auf den Kernnamen (ohne Rechtsform), dann exakter Name, dann erweiterte Form (CORP → CORPORATION). Abbruch, sobald ein identischer Normname gefunden ist. Die Zuordnung bleibt streng (`match_lei`: identischer Normname + Land = HIGH).
- **Abklingzeit:** Erfolglose Suchen werden erst nach 28 Tagen wiederholt, damit Budget für neue Firmen frei bleibt. Gründe zählt `entity_report.gleif.low_reasons`; die Historie liegt in `outputs/entity/gleif_attempts.json`.
- **Tochtergesellschaften:** `direct-child-relationships` plus Namen über `filter[lei]`. Sie werden als `usage="exposure_subsidiary"` gespeichert.
  - Gültig ab offiziellem Beziehungsbeginn, sonst ab Abruf; nie rückwirkend.
  - Die Konfidenz ist nie höher als die der LEI-Zuordnung des Emittenten.
  - Budget: 60 Firmen je Lauf, Auffrischung nach 90 Tagen.
- **TED:** Gewinnernamen dürfen jetzt auch exakt auf GLEIF-Töchter passen, aber nur wenn die Beziehung am Veröffentlichungstag bestand. Konfidenz MEDIUM, `attrs.via = gleif_subsidiary`.
- **Bewertung:** GLEIF liefert keine Features und ist daher Infrastruktur. Bewertet wird die Mapping-Qualität (HIGH/MEDIUM-Anteil, Töchter) im Montagsbericht.

### 3.2 SEC XBRL companyfacts (Schritt 2)

Modul: `modules/external/sources/sec_xbrl.py`, Vertrag `sec_companyfacts`, Quelle `sec_xbrl_fundamentals` (eigene Quelle → eigene Ablation).

**Point-in-Time:**

- Jede Zahl ist eine eigene Beobachtung je Filing: `available_at = filed + 1 T`, `vintage_time = filed`.
- Zum Stichtag t gilt je Periode das **jüngste Filing ≤ t**. Restatements wirken erst ab ihrer Einreichung; ein Test belegt das.
- Unveränderte Vergleichszahlen späterer Filings werden verworfen, echte Revisionen bleiben.

**Features** (je maximal 5, keine Rohwerte):

| Feature | Definition |
|---|---|
| `xbrl_rev_yoy` | Umsatz gegen Vorjahresquartal (Q4 = GJ − 9M bzw. − Q1..Q3) |
| `xbrl_sue` | EPS-Überraschung gegen saisonalen Random Walk, standardisiert |
| `xbrl_accruals` | (NI − CFO) / Bilanzsumme |
| `xbrl_asset_growth` | Wachstum der Bilanzsumme |
| `xbrl_share_change` | Nettoemission |

Veraltete Werte (Periodenende älter als 200 T bzw. 450 T bei Accruals) werden zu NaN.

**Betrieb:**

- Inkrementell: ein Abruf nur, wenn seit dem letzten Abruf ein neues 10-K/10-Q im Submissions-Store steht.
- Budget 800 je Lauf, 10 Anfragen/s.
- Ein 404 wird als „kein XBRL-Filer“ vermerkt. Schema-Drift wird gemeldet, nie geraten.

**Vorab registrierte Hypothesen** (vor jedem Datenabruf): ALT-XBRL-001 (PEAD), -002 (Accruals), -003 (Nettoemission), -004 (Umsatzwachstum bei schwachem Kurs).

**Verdikt:** offen. Es folgt aus dem nächsten `alt_data` + `ml_research` (full) Lauf.

### 3.3 Meta-Learning, Scoreboard, Factory, Bericht

- **`conditional_value`:** IC je Feature × Sektor × Regime (vol, trend) × Horizont (20/60 T) plus Jahres-IC.
  - Alpha Decay wird klassifiziert: stable, decaying, reversed oder no_early_effect.
  - Zwei Sichten: **selection** (2016–2018) erzeugt Hypothesen, **dev** (2019–2025) beschreibt nur.
- **Factory:**
  - Quelle × Sektor/Regime-Zellen mit |t| ≥ 2 **aus den Auswahljahren** werden zu skopierten Hypothesen (höchstens 4 je Lauf). Getestet wird im Walk-Forward der späteren Jahre mit BH.
  - Neue Familien erben die Erfahrung der Quelle (Posterior von `research:alt_data:<quelle>`).
  - Neue Divergenz: SUE↑ bei Kurs↓.
- **Source Value Scoreboard:** Coverage, Freshness, Reliability, Mapping Quality, Revision Risk, Maintenance Cost, OOS Value, Forward Value (gemessen aus dem Forward-Ledger der Quellverträge), Alpha Decay, Score.
  - Status ist RESEARCH, SHADOW, CHALLENGER oder REJECTED.
  - PROMOTED setzt nie die Quelle selbst, sondern nur der PromotionController per menschlichem PR.
- **Montagsbericht:**
  - Abschnitt 17 zeigt Data Source Health, Entity Resolution, Quellen mit Forward-Mehrwert, Quellen ohne Mehrwert, Alpha Decay und bedingte Befunde.
  - Abschnitt 19 zeigt die Factory mit Herkunft der Hypothesen, Datenlücken und Challengern.

## 4. Reihenfolge und Abbruchregel

Audit → **GLEIF** → **SEC (XBRL)** → USAspending/TED → Comtrade → EIA → ALFRED → Federal Register → ClinicalTrials/openFDA → GDELT → PEGELONLINE.

Nach jeder Quelle folgen:

1. Tests und PIT-Audit
2. Live-Ingest (GitHub Actions)
3. `ml_research full`: Walk-Forward, Ablation, bedingte Werte
4. KEEP/MODIFY/REJECT

Erst danach kommt die nächste Quelle. Quellen mit REJECT bleiben im Memory und senken die Priorität ähnlicher Ideen. Ihre Rohdaten bleiben erhalten, damit sie später erneut geprüft werden können.

## 5. Ergebnisse (wird nach jedem Lauf ergänzt)

| Quelle | Abdeckung Dev | Verdikt | Forward | Notiz |
|---|---|---|---|---|
| sec_deep_events | 1.0 | REJECT | – | Insider/8-K ohne Zusatznutzen gegenüber der Baseline |
| ted_procurement | < 0.3 (erst ab 2025) | – (Abdeckung) | ab 2026-10-05 | nur prospektiv bewertbar |
| sec_xbrl_fundamentals | offen | offen | ab 2026-10-12 | erster Lauf nach Merge |
