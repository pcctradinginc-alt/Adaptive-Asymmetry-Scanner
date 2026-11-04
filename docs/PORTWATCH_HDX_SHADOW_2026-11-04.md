# IMF PortWatch — OCHA/HDX independent **shadow** data path

## Why this is NOT an active data fallback

OCHA-DAP publishes country-level PortWatch port-call CSV files (Djibouti, Kenya,
Lebanon, Mozambique, Oman, Somalia, Sudan, Yemen) and one global daily chokepoint
transit CSV on [HDX](https://data.humdata.org/organization/portwatch). Its
[downloader](https://github.com/OCHA-DAP/hdx-portwatch-viz/blob/main/download_data.py)
pins the nine resource IDs recorded in
`modules/external/sources/portwatch_hdx_shadow.py`. The alternative has **less
geographical port coverage** than the direct IMF layer: 8 country files do not
reproduce the 2,065-port global ArcGIS time series. Do not interpret a green
country sample as a global substitute.

On **2026-11-04** OCHA's own workflow downloaded all nine CSVs as **0 KB**,
then crashed on `pandas.errors.EmptyDataError` when generating its dashboard.
See [failed OCHA action](https://github.com/OCHA-DAP/hdx-portwatch-viz/actions/runs/41874548155).
The last successful dashboard refresh was 2026-10-28. We cannot claim
restoration based on these mirrors.

## Wiring / non-promotion contract

`modules/external/sources/portwatch_hdx_shadow.py` is a **separate,
non-registered, read-only connector probe**. It cannot emit a `ConnectorResult`,
`Observation`, `SourceStatus.PASS`, or write an archive. The primary ingestion
`PortWatchPortsConnector`, `PortWatchChokepointsConnector` and existing
Source Health logic are **unchanged**. No downgrade of data quality, no stale
data inserted into a point-in-time model, no bypass of Safe Mode.

`.github/workflows/portwatch_hdx_shadow.yml` runs Wednesday at 19:45 UTC or
manually. It performs `--all` checks and writes exactly one atomic JSON
report to `outputs/external_data/health/portwatch_hdx_shadow.json`, pushed
using the repository's existing `scripts/ci_push.sh` guard. An invalid mirror
fails the job *after* the failure evidence is persisted, rather than reporting
false success. It never imports CSV rows into the primary data store.

## Gates for every individual resource

1. HDX CKAN `resource_show?id=<pinned-ID>` resolves metadata; the returned ID,
   CSV format, expected resource filename/name and exact PortWatch
   `package_show.organization.name` are checked. The package must actually
   include the resource ID.
2. Direct download is `https://data.humdata.org/` only, with each redirect hop
   explicitly validated (protocol, host, credentials, port). No arbitrary
   third-party mirror, no plain HTTP, no user-supplied URL.
3. Request timeouts, streaming read and a 96 MiB per-resource hard cap.
   Zero-byte payloads and HTML error pages with HTTP 200 fail.
4. Strict UTF-8 CSV, required column names `date,portname,portcalls` for ports
   or `date,portname,n_total` for chokepoints. All rows checked for valid date,
   finite nonnegative count and a nonblank name.
5. At least 7 distinct dates of real data. Latest observation must be at
   most **21 days** old (consistent with the primary source's data freshness
   guard), not future-dated. A weekly dashboard refresh is **not** evidence
   of recent observations.
6. Report contains resource ID, verified package, retrieved timestamp,
   record count, latest observation date, data lag, byte count and SHA-256,
   with `status: PASS_SHADOW/FAIL_SHADOW`. Always
   `production_fallback_enabled: false` and `safe_mode_override: false`.

Only the **Djibouti canary and global chokepoint** are fetched by default for
manual local checks. The scheduled workflow uses `--all` to audit the eight
available country files + global chokepoints.

## Operator checks

```bash
python -m pytest -q tests/test_portwatch_hdx_shadow_20261104.py
python scripts/portwatch_hdx_shadow.py --all
```

The CLI's nonzero exit on invalid/stale data is intentional. A green CI check
proves offline behavior only; compare live resource metadata and actual
dated CSV rows before claiming the mirror is recovered.

## Promotion design (NOT in this PR)

Before any future production use: verify actual HDX licensing for private
research, confirm field equivalence per metric and geographic scope, establish
observation `available_at` from *first observed retrieval* rather than
backdating CSV history, store immutable raw vintages with checksums, compare
dual-feed overlaps against canonical IMF ArcGIS values, and obtain explicit
review approval. Keep original source-down flag until the appropriate
independent recovery gate has passed. Do not automatically merge country data
with global series or promote it in response to a source outage.
