# SEC Deep Event Intelligence

Module: `sec_events.py` (Parser), `sec_ingest.py` (Ingestion/Speicher), `sec_features.py` (Features `sec-f1`).

## Quellen und PIT-Zeitstempel

| Quelle | Endpunkt | available_at | Präzision |
|---|---|---|---|
| Form 3/4/5 Bulk-Datensätze | `sec.gov/files/structureddata/data/form-345-data-sets/{YYYY}q{Q}_form345.zip` | FILING_DATE + 1 Tag | CONSERVATIVE_DATE |
| Submissions (8-K, 10-K, 10-Q, NT 10-K/Q) | `data.sec.gov/submissions/CIK##########.json` (+ Folgeseiten) | `acceptanceDateTime` | EXACT_TIMESTAMP |
| Live Form 4 nach Bulk-Ende | Filing-Text `{accession}.txt` (über `alpha_sources.parse_form4_xml`) | `acceptanceDateTime` | EXACT_TIMESTAMP |

Quartale werden erst 30 Tage nach Quartalsende abgerufen (Veröffentlichungsverzug).
Fair Access: User-Agent mit Kontakt, ≤ 10 Anfragen/s.

## Normalisierung

* Form 4: nur Original-Meldungen (keine /A), nur Transaktionscodes **P** (Open-Market-Kauf) und **S** (Verkauf);
  USD = Stückzahl × Preis; Eigentümer-CIKs als Attribut.
* 8-K: Item-Nummern; 10-K/10-Q: Verzögerung = Einreichung − `reportDate` (Tage).
* Speicher: `outputs/external_data/sec/{insider,filings}/<jahr>.csv.gz`, append-only, Dedupe über `series_id`.

## Features (`sec-f1`)

| Feature | Definition | verfügbar |
|---|---|---|
| sec_insider_buy_value_90d | log1p(USD Käufe, 90 T) | innerhalb Form-345-Abdeckung |
| sec_insider_buyers_90d | verschiedene Käufer, 90 T | dto. |
| sec_insider_net_value_90d | sign·log1p(|Käufe − Verkäufe|), 90 T | dto. |
| sec_insider_cluster_30d | 1 wenn ≥ 2 Käufer in 30 T | dto. |
| sec_8k_count_30d_z | 8-K-Anzahl 30 T vs. 24 Vormonatsfenster, z | ≥ 25 Monate Historie |
| sec_8k_negative_90d | 8-K mit Items 1.02, 1.03, 2.04, 2.06, 3.01, 4.01, 4.02 | ≥ 1 J. Historie |
| sec_exec_change_90d | 8-K Item 5.02 | ≥ 1 J. Historie |
| sec_filing_delay_z | Verzögerung letzter 10-K/Q vs. bis zu 12 vorherige, z | ≥ 5 Berichte |
| sec_late_filing_365d | NT 10-K/NT 10-Q in 365 T | ≥ 1 J. Historie |

"Keine Insider-Käufe" ist eine **echte Null nur innerhalb belegter Abdeckung**; außerhalb NaN.
Vektorisierte Berechnung (`features_cik`), auf Gleichheit mit der Referenz `features_for` getestet.

## Hypothesen-Verträge

ALT-SEC-001 (Insider-Cluster, +), ALT-SEC-002 (Insider-Käufe nach Kursschwäche, +),
ALT-SEC-003 (negative 8-K-Items, −), ALT-SEC-004 (Einreichungsverzögerung, −).
Registriert 2026-10-02, Forward ab 2026-10-05. Siehe `config/alt_hypotheses.yaml`.

## Zurückgestellt

* 10-K/10-Q-Textänderungen (Lazy Prices): erfordert Volltext-Download je Filing; erst nach Bewertung
  der strukturierten Events, eigener Vertrag.
* 13D/13G, S-1/S-3 (Verwässerung): nächste Iteration, wenn sec-f1 bewertet ist.
