# TED Procurement Intelligence (Alternative Data, Phase 3)

Status: **SHADOW / RESEARCH**. Module: `ted_events.py` (API, Parser, Abgleich), `ted_ingest.py`
(Probe, Backfill, Speicher, Health), `ted_features.py` (Features `ted-f1`). Vertrag:
`config/external_sources/ted.yaml`. Hypothesen: ALT-TED-001/002 in `config/alt_hypotheses.yaml`.

## Quelle und PIT

| | |
|---|---|
| Endpunkt | `POST https://api.ted.europa.eu/v3/notices/search` (JSON: query, fields, page, limit) |
| Inhalt | Zuschlagsbekanntmachungen (Gewinnername, Datum, Wert/Währung, CPV, Käuferland) |
| available_at | Veröffentlichungstag + 1 Tag, 00:00 UTC (CONSERVATIVE_DATE) |
| Lizenz | Wiederverwendung erlaubt (Beschluss 2011/833/EU), Quellenangabe „TED – Amt für Veröffentlichungen der EU“ |

## Abdeckung wird gemessen, nicht angenommen

Ältere TED-Formate liefern Gewinnernamen nicht durchgängig. Eine **Feld-Probe** je Jahr misst den
Anteil der Ergebnis-Bekanntmachungen mit Gewinnernamen; erst ab dem ersten Jahr, ab dem dieser
Anteil bis heute durchgehend ≥ 50 % ist (`field_start_year`), gilt „kein Treffer“ als echte Null.
Davor, bei unvollständigen (abgeschnittenen/fehlgeschlagenen) Entitäts-Jahren und nach dem
Abrufzeitpunkt sind alle Features **NaN** (`alt_ted_available = 0`).

## Entitäten und Abgleich

Suchname = normalisierter Firmenname aus der Entity-Map (`research_ticker_identity`), Abgleich gegen
Firmen- und frühere Namen:

* HIGH – normalisierter Gewinnername identisch
* MEDIUM – Gewinnername beginnt mit dem Firmennamen + Leerzeichen, Firmenname unterscheidbar
  (≥ 2 Wörter oder ≥ 8 Zeichen)
* sonst keine Zuordnung

Bekannte Lücken: Töchter mit abweichendem Namen fehlen (keine GLEIF-Kind-Beziehungen); delistete
Firmen ohne SEC-Zuordnung (147 von 767 PIT-Tickern) haben keine TED-Features.

## Backfill

Je Entität und Jahr eine Suche, alle Seiten (max. 10 × 100; mehr → Jahr unvollständig). Budget
1500 Anfragen je Lauf, inkrementell: vollständige Vorjahre werden nie erneut abgefragt, das laufende
Jahr jedes Mal. Speicher `outputs/external_data/ted/awards/<jahr>.csv.gz` (append-only),
Zustand `state.json`, Health `health.json`.

## Features (`ted-f1`)

| Feature | Definition | Fenster-Abdeckung |
|---|---|---|
| ted_awards_90d | Zuschläge, 90 T | 365 T |
| ted_any_award_365d | 1 wenn ≥ 1 Zuschlag in 365 T | 365 T |
| ted_award_value_365d | log1p(Summe EUR-Werte, 365 T); NaN wenn Zuschläge ohne EUR-Wert | 365 T |
| ted_awards_z | 90 T gegen die vier vorangehenden 90-T-Fenster, z | 450 T |

Beträge nur bei ausgewiesener Währung EUR – nie umgerechnet oder geschätzt.

## Bewertung

Gleiches Protokoll wie SEC (`config/alt_data_protocol.yaml`, alt-v1): Redundanz-Screen, Selektion
2016–2018, Walk-Forward 2019–2025 Baseline vs. Baseline+TED, Bootstrap mit Bonferroni über
Baselines × Quellen, KEEP/MODIFY/REJECT; Hypothesen im Research-Lab, Forward ab 2026-10-05.
Erwartung vorab: Liegt `field_start_year` nach 2019, ist die Abdeckung der Dev-Jahre < 30 % →
Protokoll-Regel REJECT (zu wenig Abdeckung), bis genug Forward-Daten vorliegen.
