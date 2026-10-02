# Knowledge Graph

Modul: `modules/knowledge_graph.py`. Protokoll:
`config/intelligence_protocol.yaml` → `knowledge_graph`.
Ausgaben:
- `outputs/research/knowledge_graph.json`;
- Versionen in `knowledge_graph_versions.jsonl`;
- Validierung in `knowledge_graph_validation.json`.

## Aufbau
- **Knoten:** Ticker, Sektoren, Branchen, Sektor-ETFs, Makro-Treiber,
  Lieferketten-Routen.
- **Kanten**, jeweils mit `source`, `confidence`, `evidence_type` und
  `valid_from`:
  - `belongs_to_sector`, `maps_to_industry`, `represents_sector`;
  - `exposed_to` (Branche/Firma → Thema). Quelle: kuratierte Configs
    `industry_exposure.yaml` und `weather_exposures.yaml`, `curated_config`,
    ohne Vorzeichen; LOW gilt als uncertain;
  - gemessene Indikator → Sektor-ETF-Kanten (`measured_oos`), nur aus
    Causal Research ab `predictive_relationship`; derzeit 0;
  - `part_of_route`.
- **Versionierung:** Der Inhalts-Hash ist die Version. Eine neue Version
  entsteht nur bei Änderungen. Das Log ist append-only.
- **Keine LLM-Kanten:** Unbelegte Beziehungen erscheinen nicht.

Stand (Lauf 2026-09-29): 603 Knoten, 612 Kanten, Version `7e9edfcf500a10f3`.

## Abfragen
- `ticker_evidence(t)`: alle belegten Beziehungen eines Titels inklusive
  Evidenz.
- `propagate(driver, shock)`: Schock über `exposed_to` und Sektor auf Titel
  verteilen; Gewicht = Konfidenz × Vorzeichen.
- `event_impact(event)`: betroffene Titel eines Ereignisses (z.B. Route,
  Sektor).
- Im HC-Scanner: Widerspruch zwischen KG-Exposure und Trade-Richtung führt zur
  Ablehnung.

## Validierung: **REJECT** (als Signalquelle)
`validate_propagation` prüft, ob propagierte Makro-Schocks die Sektorrenditen
OOS vorhersagen. Da Causal Research keine BH-signifikante Beziehung liefert,
gibt es keine Tilts (n = 0).

Der KG bleibt **Dokumentation, Evidenzspeicher und Erklärung** (Weekly Report,
HC-Begründungen), ohne Einfluss auf Scores.
