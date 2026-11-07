# Datenzuverlässigkeit und Quellensuche, 2026-11-06

Es gibt **keinen positiven Alpha-Nachweis**, und keine neue Quelle ist angebunden. Dieser Bericht beschreibt Defekte,
Reparaturen und Kandidaten. Statusangaben: VERIFIED, FIXED, OPEN, HYPOTHESIS, DATA_GAP, NEED_MORE_DATA, REJECTED.

## A Datenqualitätsaudit

Grundlage ist der Snapshot `outputs/health/source_health_snapshot.json` vom 2026-11-06T15:58Z mit 63 Quellen.

| Zustand | n | Quellen / Grund |
|---|---|---|
| HEALTHY | 49 | darunter alle 5 CRITICAL |
| UNVALIDATED | 9 | bewusst deaktiviert oder zurückgestellt (`destatis_truck_toll_download`, `asfinag_at`, `viapass_be`, `ecmwf_open_data`, `fhwa_faf`, `kosis_kr`, `mot_cn`, `nbs_cn`, `imf_aipi`). Sie werden nicht automatisch eingeschaltet. |
| BROKEN | 2 | `imf_portwatch_ports`, `imf_portwatch_chokepoints` (IMPORTANT): ArcGIS-Kontingent erschöpft (429) |
| DEGRADED | 2 | `imf_portwatch_disruptions` (429), `ncei_normals` (Missing-Rate 35 %, Owner-Prüfung der Stationen offen) |
| STALE | 1 | `estat_jp_truck` (Veröffentlichungslücke beim Anbieter, neueste Periode 2026-03) |

**Safe Mode** ist aktiv: Zwei unabhängige IMPORTANT-Quellen sind ausgefallen. Die Datenqualität liegt bei 0,888 und
ist abhängigkeitsgewichtet (`CRIT_WEIGHT`), also kein einfacher Durchschnitt. Die Status `SCHEMA_CHANGED` und
`AUTH_MISSING` gibt es bereits auf Abrufebene (`SchemaError`, `AuthError`); ein neues Statussystem ist nicht nötig.

| ID | Befund | Beleg | Status |
|---|---|---|---|
| R1 | ArcGIS meldet das Kontingent im JSON-Körper (HTTP 200, `{"error": {"code": 429, "API calls quota exceeded"}}`). `http.fetch` sieht keinen 429-Status. Je Layer folgten bis zu 5 Isolations-Probes (alle wieder 429), danach dieselbe Kette für die übrigen Layer, also für ein erschöpftes Kontingent bis zu 18 Abrufe je Lauf. | Snapshot-Gründe aller drei PortWatch-Layer | **FIXED** (dieser PR) |
| R2 | Long-Exit-Marks aus dem Archiv verlangten Bid > 0, der Live-Pfad akzeptierte Bid 0. Wertlose Optionen fielen im Archivpfad aus der Stichprobe, mit positiver Verzerrung. | `option_quotes.usable_rth_quote` gegen `candidate_ledger._fill_real_option_marks` | **FIXED** (dieser PR); Realdaten: 0 von 1.053 fälligen Labels betroffen (1.051 gefüllt, 2 NO_QUOTE wegen CTVA-Kontraktanpassung) |
| R3 | Die Aufnahmeprüfung (`research_director.onboarding_checklist`) wertete eine fehlende Lizenz als „ok“ und prüfte den Zugang nie | Code | **FIXED** (dieser PR) |
| R4 | Frühere Reparaturen (PortWatch-Paging #156, Neuentdeckung #157, IMF-Vintages #162, Recovery-Regel #155) | – | nicht erneut angefasst |
| R5 | Wochenend-Versatz Option-Mark gegen Underlying-Label (33 % der 5-T-Options-Labels) | `docs/ALPHA_ASSESSMENT_2026-11-06.md` A8 | OPEN (Labeldefinition, Freigabe) |
| R6 | „IV-Rank“ ohne IV-Niveau | ebd. A7 | OPEN (Strategiewahl, Freigabe) |

## B Zuverlässigkeit

**R1 – ArcGIS-Kontingent:**
- `arcgis_rate_limited()` erkennt Rate-Limits über Code 429, „too many requests“ oder „quota exceeded“.
- Bei einem Treffer gibt es keine Isolations-Probe mehr. Die Meldung trägt den Präfix `[UPSTREAM_RATE_LIMITED]`.
- Ein Kreisunterbrecher je Host sperrt für den restlichen Lauf alle weiteren Abrufe: Layer-Auflösung, Metadaten und
  Abfrage.
- Der Fehler bleibt sichtbar, Status BROKEN. Alte Daten werden nie verwendet.
- Tests: `tests/test_arcgis_rate_limit.py` (9).
  - Vorher: 1 Abruf plus bis zu 5 Probes je Layer.
  - Nachher: 1 Abruf, danach 0 weitere für den Host.

**R2 – Long-Exit zu Bid 0:**
- Zählt bei Ask ≤ 0,25 USD (`ZERO_BID_EXIT_MAX_ASK`) als beobachteter Totalverlust.
- Bid 0 bei größerem Ask bleibt offen (fehlende Geldseite), im Archiv- wie im Live-Pfad.
- Die Entry-Regel ist unverändert.
- Gilt für Ledger-Marks, Provenienz (VERIFIED) und den ROI-Abgleich.
- Tests in `tests/test_short_leg_exit_audit.py` angepasst bzw. ergänzt; dort stand das alte Verhalten bewusst fest.

**R3 – Aufnahmeprüfung:** Der Status `license` kann jetzt `missing` sein. Neu sind `access` (ok nur bei
`access_verified: true`) und `blocked_by` in `active_learning`.

**Tests:**
- 1.446 Tests in den betroffenen Modulen grün (Optionen, Ledger, Provenienz, Alpha-Link, PortWatch).
- 265 Tests für Research Director, Autonomie und Lernmodell grün.

## C Neue Datenquellen

**Zugang:**
- Aus der Entwicklungsumgebung sind alle 15 geprüften Hosts gesperrt (Proxy, HTTP 000). Der Zugang ist deshalb für
  keine Quelle verifiziert.
- Lizenz und Limits sind nur aus Sekundärquellen belegt. Die Websuche fand keine offizielle Lizenzseite.
- Nach den Vorgaben gilt: **keine Integration**. Die besten fünf Kandidaten stehen in `config/data_catalog.yaml` mit
  `access_verified: false` und REVIEW_REQUIRED.

**Was schon bekannt war:** `docs/DATA_SOURCES_PROGRAM.md` führt die meisten Kandidaten bereits als „FEHLT“, meist
wegen fehlender Unternehmenszuordnung. SEC Deep Events und TED wurden historisch geprüft: REJECT bzw. erst ab 2025
bewertbar.

| Rang | Quelle | Datenart | Kosten | Lizenz | Historie | PIT | Unternehmensbezug | Hypothese | Bewertung |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wikipedia-Pageviews | Aufmerksamkeit je Artikel, täglich | frei, User-Agent Pflicht, Rate-Limits seit 2026 | freie Lizenz, Typ ungeprüft | ab 07/2015 (Sekundärquelle) | Tageswert ab Folgetag; Bot-Bereinigung als Revisionsrisiko | direkt (Artikel ↔ Emittent) | D1 | höchster erwarteter Informationsgewinn: ticker-genau, historisch auf dem ML-Panel 2015–2026 testbar (das einzige Panel mit Power), neu im System |
| 2 | openFDA Enforcement | Rückrufe (food/drug/device) | frei, Schlüssel optional | CC0 laut Sekundärquellen | ab 2004 | `report_date` = Veröffentlichung | `recalling_firm` → GLEIF (bestehendes Entity-Resolution) | D2 | ereignisgenau, historisch testbar; die Zuordnung ist der Engpass |
| 3 | Federal Register | Regeln, Bekanntmachungen | frei, kein Schlüssel | gemeinfrei laut Sekundärquellen | ab 1994 | Veröffentlichungsdatum exakt | Agentur/CFR → Branche (kein Emittent) | D3 | Branchenebene; Paging-Fallstrick (Seite 1 bei zu hoher Seitenzahl) |
| 4 | GDELT 2.0 | Ereignisse, Ton | frei | Projektbedingungen prüfen | ab 2015 | 15-Minuten-Zeitstempel | über Namen, verrauscht | – | hohe Redundanz zu News und FinBERT, großer Aufwand |
| 5 | USGS NWIS | Flusspegel, Abfluss | frei | US-Behörde | Jahrzehnte | **nicht PIT**: vorläufige Werte werden revidiert | Branche/Regime | D5 | nur Forward-Archiv |
| – | PEGELONLINE | Pegel (Rhein u. a.) | frei | DL-DE | REST nur etwa 30 Tage | Forward-Archiv | Regime | – | Historie fehlt (DATA_GAP) |
| – | NASA FIRMS | Brände/Hotspots | Registrierung (MAP_KEY) | NASA | ab 2000 | ja | Werksstandorte fehlen | – | Zuordnung fehlt (DATA_GAP) |
| – | USDA Export Sales | Exportverkäufe, wöchentlich | api.data.gov-Schlüssel | US-Behörde | lang | Veröffentlichungstag | Agrar-Branche | – | Regime-Ebene, Überschneidung mit Rohstoffdaten |
| – | FDIC | Bankkennzahlen, quartalsweise | frei | US-Behörde | lang | Meldetermin prüfen | Regionalbanken (Zert.-Nr. ↔ Ticker) | – | langsam; Sektor schmal |
| – | npm/PyPI/GitHub | Downloads, Aktivität | frei | je Plattform | npm ab 2015, PyPI-Stats etwa 180 T | ja | Paket → Emittent fehlt | – | Zuordnung fehlt (DATA_GAP) |
| – | crt.sh (Certificate Transparency) | Zertifikate | frei | ungeklärt | lang | ja | Domain → Emittent | – | hoher Aufwand, unklarer Mechanismus (REJECTED für jetzt) |
| – | Stellenanzeigen | Nachfrage nach Arbeit | meist kommerziell oder ToS-beschränkt | ungeklärt | – | – | – | – | **REJECTED** (Lizenz/ToS) |

## D Unorthodoxe Hypothesen (alle HYPOTHESIS, nicht getestet)

**D1 – Aufmerksamkeitslücke.** Nachrichten mit geringer Pageview-Reaktion, gemessen an ihrer News-Intensität, werden
langsamer eingepreist.
- Mechanismus: begrenzte Aufmerksamkeit.
- Daten: Pageviews, News-Zähler.
- Kontrollgruppe: gleiche News-Intensität, hohe Aufmerksamkeit.
- Test: Querschnitt je Tag, sektorneutral, 2015–2026 Walk-Forward auf dem ML-Panel, BH über alle Varianten.
- Option: Census-Call gegen Nicht-Handeln, netto nach Spread.
- Falsifiziert, wenn der Interaktions-IC ≤ 0 oder nach BH nicht signifikant ist.

**D2 – Rückruf-Diffusion.** Ein FDA-Rückruf belastet den Hersteller verzögert; Wettbewerber profitieren verzögert.
- Kontrollgruppe: gematchte Unternehmen derselben Branche am selben Tag.
- Test: Event-Study 1–20 T.
- Option: Put auf den Hersteller gegen Nicht-Handeln, netto.
- Falsifiziert, wenn der CAR nach Kosten nicht von 0 verschieden ist.

**D3 – Regulatorische Drift.** Finale Regeln wirken auf Branchen stärker als die Entwürfe, die der Markt schon kennt.
- Kontrollgruppe: Entwurf gegen finale Regel derselben Agentur; Placebo-Tage.
- Test: Branchen-Überrendite 5–20 T.

**D4 – Varianzrisikoprämie im Census.** Das Verhältnis IV zu RV beim Einstieg sagt die ausführbare Optionsrendite
voraus. Dafür ist keine neue Quelle nötig, nur realisierte Vola je Zeile.
- Kontrollgruppe: Terzile innerhalb desselben Tages.
- Test: gepaart, netto nach Spread.
- Am besten belegt in der Literatur; ein Hinweis im Census liegt vor (IC −0,13 bei 20 T, n_eff ≈ 1).
- **Explorativer Vorabtest (06.11., NEED_MORE_DATA):**
  - Realisierte Vola aus den Einstiegskursen des Ledgers, nur Tage vor dem Signal, mindestens 8 Tagesrenditen.
  - Basis: 298 Census-Optionen an 10 Signaltagen (14.–30.10.), ausführbare 5-T-Rendite.
  - Rang-IC von IV/RV: −0,046 (positiv an 40 % der Tage).
  - Unteres minus oberes IV/RV-Terzil: +1,3 pp (t = 0,26).
  - Die Richtung entspricht der Hypothese, die Größe ist nicht von 0 unterscheidbar. Die RV-Schätzung ist verrauscht.
    Belastbar wird der Test erst mit der RV-Erfassung je Zeile (G1) und mehr unabhängigen Perioden.

**D5 – Wasserstraßen-Engpass.** Niedrige Pegel an Mississippi und Ohio erhöhen die Kosten Barge-abhängiger Agrar- und
Chemieunternehmen und werden verzögert eingepreist.
- Nur prospektiv testbar (Revisionen), Placebo-Perioden.

## E Pilotintegrationen

**Keine.** Gründe:
- Der Zugang ist aus der Entwicklungsumgebung nicht verifizierbar.
- Lizenz und Limits sind nicht offiziell bestätigt.
- Laut Auftrag ist eine Integration bei ungeklärten Bedingungen ausgeschlossen.

Vorbereitet ist Folgendes: Katalogeinträge mit Aufnahmeprüfung (`access`, `license`) und klare Hypothesen.

**Voraussetzungen für einen Piloten mit Wikipedia-Pageviews:**
1. Ein Mensch bestätigt Lizenz und Rate-Limits auf den offiziellen Seiten.
2. Ein read-only-Probe-Abruf aus GitHub Actions gelingt.
3. Dann folgen SourceContract, Rohdaten-Archiv mit Hash, Artikel-Mapping über Entity-Resolution und der Panel-Test
   gegen die Baseline (`alt_data_validation`).

## F Autonomes Lernen

- **Kreislauf vorhanden:** Datenkatalog → `active_learning` (Informationsgewinn je Weltmodell-Dimension) bzw.
  Datenquellen-Vorschläge bei einer `no_skill`-Lücke → Aufnahmeprüfung → `alt_data_validation` → Research Memory.
- **Neu:** Kandidaten ohne Lizenz- oder Zugangsnachweis sind nicht verfügbar (`blocked_by`).
- **Offen:**
  - `active_learning` bewertet nur makroökonomische Unsicherheit. Quellen für unerklärte Querschnittsfehler wie
    `attention` oder `product_quality` haben dort keinen Informationsgewinn; sie werden nur über `no_skill` vorgeschlagen.
  - Abgelehnte Quellen (z. B. `sec_deep_events`) werden noch nicht automatisch zurückgestuft.

## G Nächste Schritte

1. D4: realisierte Vola je Census-Zeile erfassen. Das kostet keine neue Quelle und hat den höchsten erwarteten Nutzen.
   **Umgesetzt in diesem PR:** `iv_rank_components.rv_21_level`, das Niveau der realisierten 21-T-Vola bis zum Signal.
   Damit ist IV/RV = `real_option.iv / rv_21_level` je Census-Kontrakt ab dem Merge prospektiv messbar. Reine
   Erfassung: Strategiewahl, IV-Rank und Lernmodell sind unverändert.
2. Lizenz und Zugang von Wikipedia-Pageviews durch einen Menschen prüfen lassen, danach Pilot D1 mit historischem
   Panel-Test.
3. Die PortWatch-Kontingent-Fixes beobachten. Nach dem Merge sollte die Zahl der Abrufe pro Lauf bei einem 429 sinken.
4. `active_learning` um einen Informationsgewinn aus unerklärten Querschnittsfehlern erweitern, z. B. Residuen des
   Lernmodells je Merkmalsgruppe.
5. Abgelehnte Quellen in Vorschlägen zurückstufen; eine Neubewertung nur bei neuen Daten.
