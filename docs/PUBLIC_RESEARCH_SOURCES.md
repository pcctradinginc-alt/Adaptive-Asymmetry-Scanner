# Öffentliche Research-Quellen (2026-10-15)

Alle Quellen hier sind **RESEARCH_SHADOW**: Sie werden nur archiviert bzw. als Research-Features berechnet. Sie wirken
nicht auf Champion, Risk Gates, Kill-Switch, EA-Shadow oder Promotion. Es gibt keine kostenpflichtigen Dienste.
Zugangsdaten stehen nie im Code.

| ID | Quelle | Kennzahlen | PIT (`available_at`) | Zugang |
|---|---|---|---|---|
| A4 `nasdaq100_members` | nasdaq.com, offizielle Nasdaq-100-Liste | Mitgliederzahl, Symbole (attrs) je Tag | Abrufzeit | ohne Key |
| B1 `alt_sec_earnings` (Feature) | SEC 8-K Item 2.02 aus dem bestehenden Submissions-Archiv | Tage seit Release, Sitzung (BMO/RTH/AMC), Releases in 365 T | `acceptanceDateTime` (exakt) | ohne Key |
| B3 `finra_short_volume` | FINRA Reg SHO Daily (CNMS) | Leerverkaufsanteil am Volumen je Ticker/Tag | Abrufzeit | ohne Key |
| B3 `finra_short_interest` | FINRA Query API, Consolidated Short Interest | Short-Interest-Stücke, Days-to-Cover | Abrufzeit | ohne Key (live geprüft 2026-10-15); optional `FINRA_API_CLIENT_ID`/`FINRA_API_CLIENT_SECRET`, falls FINRA künftig Zugangsdaten verlangt (dann `AUTH_MISSING`) |
| B4 `fred_liquidity` | FRED/ALFRED: RRPONTSYD, WTREGEN, T10Y2Y | ON-RRP, TGA, Kurve 10J−2J | ALFRED `realtime_start` | vorhandenes `FRED_API_KEY` |
| B5 `wikimedia_pageviews` | Wikimedia REST (nur menschliche Nutzer) | Seitenaufrufe je Ticker/Tag | Abrufzeit | ohne Key |
| B7 `fomc_calendar`, `bls_release_calendar`, `bea_release_calendar` | Federal Reserve (HTML), BLS (iCalendar), BEA (release_dates.json, sonst Plan-Seite) | geplante Termine (UTC), FOMC mit SEP markiert | Abrufzeit | ohne Key |
| B8 `clinicaltrials_sponsors` | ClinicalTrials.gov API v2 | aktive Studien, aktive Phase 3, Primärabschluss in 90 T, Updates in 7 T | Abrufzeit | ohne Key |
| B9 `usaspending_contracts` | USAspending.gov API v2 | Bundesaufträge mit Aktivität in 7 T (USD, Anzahl) | Abrufzeit | ohne Key |
| B10 `sec_13f_holdings` | SEC Form-13F-Datensätze (DERA) | institutionelle Halter, gehaltene Stücke je Quartal | Tag nach dem letzten einbezogenen Filing | ohne Key |

## Regeln

- Fehlendes bleibt fehlend: Es gibt keine Nullen als Ersatz und keine geschätzten Werte.
- Schemaänderungen setzen den Status `SCHEMA_CHANGED` und lösen den bestehenden Alarm aus. Sie werden nie still
  umgedeutet.
- Namenszuordnung (B8–B10) erfolgt über `normalize_name` der Entity-Map: exakter Name oder GLEIF-Tochter (B9),
  Sponsor-Präfix (B8). Mehrdeutige Namen werden verworfen. Die Abdeckung steht in der Meldung jedes Laufs.
- 13F: Es zählen nur fristgerechte 13F-HR (höchstens 45 Tage nach Quartalsende). Amendments, Optionen und
  Nicht-Aktien sind ausgeschlossen. Pro Lauf wird ein Datensatz verarbeitet; ein Marker im Archiv verhindert eine
  doppelte Verarbeitung. Der Preflight archiviert nichts und zählt daher nie als verarbeitet.
- Die Nasdaq-100-Liste von nasdaq.com wird nur archiviert (PIT-Mitgliederhistorie). `get_universe` gehört zur
  eingefrorenen UNIVERSE_V1-Definition (Code-Hash). Eine Umstellung der Produktionsquelle wäre deshalb eine
  eigene V1-Änderung und braucht eine ausdrückliche Freigabe.
- Die Live-Formate werden mit dem Workflow „External Data Preflight“ geprüft. Der Eingang `family` steht dafür auf
  `public_research,real_economy`.

## Archivwachstum (A5)

Kalte, abgeschlossene Partitionen, die seit `compress_cold_after_days` (21) Tagen unberührt sind, werden verlustfrei
nach `<key>.jsonl.gz` komprimiert. Alle Leser lesen `.jsonl` und `.jsonl.gz` transparent.
