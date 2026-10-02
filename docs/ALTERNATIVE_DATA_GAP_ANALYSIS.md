# Alternative Data: Bestandsaufnahme und Gap-Analyse (2026-10-02)

Grundlage: vollständige Durchsicht von `modules/external/` (Framework mit
Registry, Archiv, PIT-Modell, Data Quality, Source Health),
`config/external_sources/*.yaml` (30 registrierte Quellen, 19 aktiv),
`modules/alpha_sources.py`, `modules/data_validator.py`,
`modules/data_ingestion.py`, `modules/knowledge_graph.py` und
`config/data_catalog.yaml`.

**Wichtig für die Architektur:** Ein PIT-Framework existiert bereits und wird
**erweitert, nicht dupliziert**:
- `modules/external/pit.Observation` mit `observation_time`,
  `source_release_time`, `available_at`, `retrieved_at` und `vintage_time`;
- Konnektor-Basis `sources/base.Connector`;
- versioniertes Rohdaten-Archiv `external/archive.py` mit `content_hash`;
- Source Health `external/data_quality.py`;
- Registry-Einträge mit Lizenz-, PIT- und Revisionsangaben.

Neue Quellen werden dort als Konnektor und Registry-Eintrag angelegt. Neu
hinzu kommen nur die Ebenen, die fehlen: Entity Resolution, Exposure und
firmenbezogene Features.

**Netzwerk:** In der Entwicklungs-Sandbox sind alle externen APIs gesperrt
(Organisationsrichtlinie), auch SEC, GLEIF, TED, Comtrade, EIA, GDELT und
PEGELONLINE. In GitHub Actions sind sie erreichbar. Konnektoren werden deshalb
gegen das offizielle Schema mit Fixtures getestet und **live nur in CI**
verifiziert (Muster wie `external_preflight.yml`).

## Bestandsmatrix

Legende Nutzung:
- **PROD:** beeinflusst die tägliche Mail.
- **SHADOW:** nur Kontext oder Ledger, ohne Entscheidungswirkung.
- **RESEARCH:** nur Research-Stack.
- **–:** nicht vorhanden.

| Informationsart | Status | Modul / API | Felder | Frequenz | Historie / PIT | Feature Store | Nutzung |
|---|---|---|---|---|---|---|---|
| **SEC EDGAR – EPS** | PARTIAL | `data_validator.fetch_eps_sec_edgar` / `data.sec.gov/api/xbrl/companyfacts` | EPS Basic/Diluted, Summe der letzten 4 Einträge | live je Kandidat | **nicht PIT** (heutiger Stand; `filed`-Datum wird ignoriert; 10-K-Jahreswerte werden mitsummiert) | nein | PROD (Datenqualitäts-Hinweis im Deep-Analysis-Prompt) |
| **Insider (Form 4)** | PARTIAL | `alpha_sources.fetch_sec_insider_trades` / `data.sec.gov/submissions` + Form-4-XML | Code P/S, Shares, Preis, Insider, Datum | live, 14 Tage | **keine Historie**, nicht archiviert, kein PIT-Feature | nur Candidate-Ledger (`insider_*`) | PROD (Cluster-Schlagzeile im Prompt) |
| SEC 8-K / Filing-Events | **MISSING** | – | – | – | – | – | – |
| SEC 10-K/10-Q-Textänderungen | **MISSING** | – | – | – | – | – | – |
| **PortWatch** (Häfen, Chokepoints, Disruptions) | EXISTS | `external/sources/maritime.py` | Port Calls, Handelsvolumen, Transits | täglich/Event | CONSERVATIVE_DATE, keine Vintages | ja (`maritime_features`) | SHADOW (Mail-Kontext) |
| Shipping | EXISTS | wie PortWatch | – | – | – | – | SHADOW |
| **Wetter** (NWS, NHC, NCEI) | EXISTS | `external/sources/weather.py` | Warnungen, Stürme, HDD/CDD | stündlich | EXACT_TIMESTAMP, nur vorwärts | ja | SHADOW |
| **Truck/Freight** (Destatis-Maut, BTS-TSI, Eurostat, e-Stat JP) | EXISTS | `external/sources/road_freight.py` | Indizes | monatlich | teils Vintages (ALFRED/BTS) | ja | SHADOW |
| **e-Stat** (Japan) | EXISTS | dto. `estat_jp_truck` | Lkw-Fracht | monatlich | CONSERVATIVE_DATE | ja | SHADOW |
| **Makro** (FRED/ALFRED, Eurostat) | EXISTS | `external/sources/real_economy.py`, `external/regime.py` | CPI, NFCI, Fed-Bilanz, USD, WTI, IP, Sentiment | täglich bis monatlich | ALFRED-Vintages (PIT) | ja, auch `ml_research.macro_features` | RESEARCH + SHADOW |
| **Energie** | PARTIAL | `external/sources/energy.py` (ENTSO-E Strom DE); WTI über FRED | Strompreis, Last | täglich | CONSERVATIVE_DATE | ja | SHADOW |
| EIA (Lager, Raffinerie, Gas, LNG) | **MISSING** | – | – | – | – | – | – |
| **News** | EXISTS | `data_ingestion` (Finnhub, NewsAPI, yfinance) | Schlagzeilen | live | keine Historie | nein | PROD (LLM-Prescreen und Deep Analysis) |
| GDELT | **MISSING** | – | – | – | – | – | – |
| Government Procurement (TED) | **MISSING** | – | – | – | – | – | – |
| Trade Flows (Comtrade) | **MISSING** | – | – | – | – | – | – |
| River Logistics (PEGELONLINE) | **MISSING** | – | – | – | – | – | – |
| **Entity Graph / LEI (GLEIF)** | **MISSING** | – | – | – | – | – | – |
| Ticker ↔ CIK | PARTIAL / REDUNDANT | zweimal: `alpha_sources.sec_cik_for_ticker` **und** `data_validator._load_ticker_map` | heutige Zuordnung | live | nicht PIT | nein | PROD |
| **Knowledge Graph** | EXISTS | `knowledge_graph.py` | Ticker/Sektor/Branche/Themen/Häfen | je Lauf | Referenzkanten nicht PIT (Audit P2-3) | – | Erklärung; HC-Check inaktiv (0 Messkanten) |
| Exposure | PARTIAL | `config/industry_exposure.yaml`, `weather_exposures.yaml`, `faf_exposure.yaml` | Branche/Firma → Thema, Gewicht | statisch | nicht PIT, kuratiert | – | SHADOW (KG, Kontext) |

**Redundant:** Die Ticker→CIK-Zuordnung ist doppelt implementiert.
`modules/entity_resolution/` wird zur gemeinsamen Quelle; die beiden
bestehenden Aufrufer bleiben bis zur Umstellung funktionsfähig.

## Lücken nach Priorität (Auftrag)

| Prio | Lücke | Bewertung |
|---|---|---|
| Infrastruktur | **Entity Resolution (Ticker↔CIK↔LEI↔Eltern, PIT)** | Voraussetzung für TED, Comtrade, GDELT und SEC-Firmenbezug. Zuerst. |
| 1A | **SEC Deep Events** | Höchster erwarteter Wert und beste PIT-Qualität. Jede Filing-Annahme trägt einen offiziellen `acceptanceDateTime` (EXACT_TIMESTAMP). Filings sind unveränderlich (Änderungen = eigene /A-Filings). Historie ab 2003 frei verfügbar (Form-3/4/5-Datensätze der SEC, submissions JSON). |
| 1B | TED Procurement | Erst nach SEC (Auftrag). EU-Fokus, Firmenbezug nur über GLEIF/Namensabgleich; für S&P-500-Firmen wenige direkte Gewinner zu erwarten. |
| 1C | UN Comtrade | Länder×Produkt; Firmenbezug erfordert Produkt-/Länder-Exposure, das kaum frei verfügbar ist. Monatlich, mit Revisionen. |
| 1D | EIA | Hohe PIT-Qualität (Wochen-Releases, Termine bekannt). Sektor-Exposure (Energie, Chemie, Versorger) vorhanden. |
| 1E | GDELT | Event-Intensität; Deduplikation und Entity-Linking sind schwierig. |
| 1F | Flusspegel | Nur vorwärts sinnvoll (Pegel-Historie über PEGELONLINE begrenzt); wenige US-Firmen mit Rhein-Exposure. |

## Reihenfolge und Abbruchkriterien

1. Entity Resolution / GLEIF (Infrastruktur, kein Alpha).
2. SEC Deep Events: Form 4 (historisch, Bulk-Datensätze), 8-K-Items,
   Filing-Timing; Textänderungen (10-K/10-Q) als eigener, später Schritt.
3. **Bewertung:** Baseline gegen Baseline + SEC im Walk-Forward (PIT-Universum)
   und Prospective Challenger. KEEP/MODIFY/REJECT.
4. Erst danach TED, dann Comtrade, EIA, GDELT, Fluss, jeweils mit
   Abschlussbewertung.

**Optionale Quellen** (Polymarket, Wikipedia, FIRMS, OpenSky, CT-Logs,
PyPI/npm) und **ausgeschlossene Quellen** (Scraping, Jet-Tracking u.ä.) werden
nur als Research-Ideen in `config/data_catalog.yaml` geführt, nicht gebaut.
