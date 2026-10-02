# Entity Resolution (Ticker ↔ CIK ↔ LEI ↔ Konzern)

Modul: `modules/entity_resolution/` · Speicher: `outputs/entity/entity_map.jsonl` · Bericht: `outputs/entity/entity_report.json`

## Datensatz (`EntityRecord`)

`entity_id, ticker, cik (10-stellig), lei, canonical_name, aliases, parent_entity, ultimate_parent,
jurisdiction, country, industry, domains, mapping_source, mapping_confidence (HIGH|MEDIUM|LOW),
mapping_score, valid_from, valid_to, usage, evidence`

## Quellen

| Quelle | Inhalt | Lizenz | valid_from |
|---|---|---|---|
| SEC `company_tickers.json` | Ticker → CIK, Name | US-Gov, gemeinfrei | erster Abruf |
| SEC `submissions/CIK##########.json` | Name, frühere Namen, SIC, Staat, Ticker | gemeinfrei | `formerNames.from` bzw. erster Abruf |
| GLEIF `lei-records` | LEI, Rechtsname, Sitzland, Status | CC0 | `initialRegistrationDate` |
| GLEIF `direct-parent` / `ultimate-parent` | Konzernbeziehungen | CC0 | Beziehungsbeginn (`relationship.startDate`) |

## Zuordnungsregeln

* **SEC Ticker→CIK** – zwei Datensätze:
  * `usage = "pit"`, HIGH, gültig ab erstem Abruf (heutige Zuordnung gilt nur ab heute).
  * `usage = "research_ticker_identity"`, MEDIUM: für die historische Verknüpfung mit Kursdaten, die
    ebenfalls nach heutigem Ticker geschlüsselt sind. Die CIK ist dauerhaft; Risiko = Ticker-Wiederverwendung,
    daher MEDIUM und gekennzeichnet.
* **LEI-Match** (`match_lei`): HIGH = exakter normalisierter Name + gleiches Land; MEDIUM = exakter Name,
  anderes Land; LOW = mehrdeutig oder kein Treffer (wird nicht für Features verwendet).
* **Konzern**: Eltern-Beziehung als eigener Datensatz ab Beziehungsbeginn – nie rückwirkend auf den LEI-Datensatz.

## Store-Semantik

* Append-only: `upsert(rec, as_of)` → `unchanged` | `opened` | `replaced` (alter Datensatz bekommt `valid_to`).
* Keine Rückdatierung: `as_of` vor dem letzten `valid_from` → `ValueError`.
* `resolve(..., as_of)` liefert nur zum Stichtag gültige Datensätze, sonst `None` (unbekannt ≠ falsch).
* `profile(...)` führt Felder aus allen gültigen Datensätzen zusammen, mit Quellenangabe je Feld.

## Qualitätsmetriken (Bericht)

`wanted, mapped, unmapped, lei_high/medium/low, parents, errors` – die Mapping-Quote geht als
`mapping_quality` in den Source Value Score ein.

## Bekannte Grenzen

* Ticker-Wiederverwendung (z. B. neue Firma unter altem Ticker) bleibt MEDIUM-Risiko, bis ein historischer
  Ticker-Datensatz (CRSP-ähnlich) verfügbar ist – keiner ist frei verfügbar.
* GLEIF-Namensabgleich: Budget 150 Abfragen/Lauf (inkrementell), Fair Use.
* Redundanz im Altbestand: Ticker→CIK existiert zusätzlich in `alpha_sources.sec_cik_for_ticker` und
  `data_validator._load_ticker_map` (siehe Gap-Analyse); Konsolidierung ist ein eigener Schritt.
