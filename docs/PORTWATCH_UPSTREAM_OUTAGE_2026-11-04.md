# IMF PortWatch — upstream service outage, 2026-11-04

## Evidence and causal classification

The two daily ArcGIS endpoints are presently **not healthy**. The official service returns a JSON API error even though the HTML ArcGIS service directory still lists its previously known fields:

- `.../Daily_Ports_Data/FeatureServer/0?f=pjson` returns `{"error":{"code":400,"message":"Layer 'Daily_Ports_Data' does not exist or is inaccessible.", ...}}`.
- `.../Daily_Chokepoints_Data/FeatureServer/0?f=pjson` returns `{"error":{"code":400,"message":"Invalid URL", ...}}`.
- The 2026-11-04 ingest observed `SCHEMA_CHANGED` for ports (because the 400 payload contained no `fields`) and `FAIL` for chokepoints (its query returned `Invalid URL` even for all five read-only isolation probes).
- The service is hosted by the **external IMF/Esri publisher**. We cannot restart its hosted dataset, repair its backing table, or assure a publication date from this repository.
- Daily trade/country-regional aggregates, downstream mirrors and static port reference data are **not** equivalent to daily port calls / chokepoint transit observations. Do not silently substitute them or backdate their availability.

## Fixed in the connector

1. Explicit `PortWatchUpstreamUnavailable(FetchError)` class for ArcGIS JSON error envelopes. Metadata service error is `FAIL / UPSTREAM_UNAVAILABLE`, never `SCHEMA_CHANGED`. Genuine valid JSON without required fields continues to raise `SchemaError`.
2. FeatureServer root resolves real map-layer **or table** IDs. Service-level 400 error cannot silently become a guessed layer 0, and discovery cannot cache a known failed root as a valid endpoint.
3. The fallback for an inaccessible *reference* service is limited to ID/name extraction from the already verified **same daily** layer. No other organization is accepted.
4. If even the minimal `ObjectID` query returns `400 Invalid URL`, isolation stops after this one probe and reports `UPSTREAM_UNAVAILABLE` instead of wasting further queries.
5. Existing strict publisher identity checks, provenance, `PASS/WARN/FAIL` semantics and Source Health two-independent-success recovery gate remain unchanged. No invented quote/metric, no data-safe-mode override, no production influence.

## Acceptance checks

Offline (simulated provider responses; proves only the code path):

```bash
python -m pytest -q tests/test_portwatch_upstream_outage_20261104.py tests/test_maritime_source.py tests/test_source_stabilization_2026_11.py tests/test_p0_p1_reliability_20261104.py
python -m pytest -q -p no:cacheprovider
```

Live (read-only preflight, no observation imports):

```bash
python -m modules.external.orchestrator preflight --family maritime --json outputs/external_data/health/portwatch_preflight.json
```

GitHub: run `External Data Preflight` / `External Data Ingestion` and then `Source Health Check` on `main`. Evaluate each source separately:

- `imf_portwatch_ports`: successful metadata (field list), read-only daily query, non-empty verified curated port observations, plausible date/lag; source is not `SCHEMA_CHANGED` or `FAIL`.
- `imf_portwatch_chokepoints`: successful metadata and dated read-only transit query, reference ID match, non-empty verified observations. Neither an `Invalid URL` nor an empty response is proof of restored transit data.
- Confirm **two independent successful ingestion cycles** according to existing Source Health recovery rules before the service may become `HEALTHY`.
- If the official API still reports 400, leave both **BROKEN/FAIL** and Safe Mode unchanged, escalate to PortWatch/ArcGIS publisher with the HTTP error samples. A green GitHub workflow or passing unit tests do not prove remote recovery.

Official service endpoint bases:
- https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0
- https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0
