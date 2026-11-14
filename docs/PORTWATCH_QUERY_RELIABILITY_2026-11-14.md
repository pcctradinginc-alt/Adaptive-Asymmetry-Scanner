# PortWatch reliability and NCEI diagnostics — 2026-11-14

## Evidence
On 2026-11-14 the upstream ArcGIS JSON payload returned 429: 6018 request units used against a quota of 6000 per minute, retry after 60 seconds. The three IMF PortWatch feeds share services9.arcgis.com. These are **publisher-side total units**, not a per-client measurement. A public shared host can fail despite correct client behavior.

## Changes
1. Historical `returnDistinctValues` fallback now uses the same bounded 429 retries, shared per-host circuit breaker and 130-second cumulative wait budget as the standard daily query. Previously this code path called the transport directly. After exhausting the limit, subsequent requests to the same host are blocked, with FAIL exposed.
2. If the provider's reference layer is unavailable, the complete, just-retrieved daily window supplies IDs **only when every configured curated entity can be uniquely matched** to actually observed name and ID. Records from different days are deduplicated. If any name is missing or ambiguous, the existing full historical DISTINCT query remains mandatory.
3. Orchestrator logs and summary record actual ArcGIS HTTP request attempts and rate-limited responses by host. Counts reset for each ingestion run. These are not ArcGIS request units.
4. Source Health gains documented downstream SHADOW/Research dependencies for Ports, Chokepoints, Disruptions, and NCEI normals. No fake trading production dependency, no cache of failed feeds.
5. NCEI read-only audit reports missing fraction by real station, metric, and station-metric from PIT-visible archive; runs weekly under the research stage's DEGRADED accounting; no imputation.

## Required acceptance
- New focused offline CI must pass rate-limit/breaker, existing schema and upstream, source health, maritime reference coverage and NCEI regression tests.
- Complete repository tests and required check-runs must pass for exact PR HEAD before merge.
- A live ArcGIS recovery cannot be proven from offline tests. Confirm new successful upstream ingestion and complete, fresh, verified observations in subsequent scheduled runs, with source health recovery rules. No signal from a 429 failure is treated as fresh.
- Do not change model weights, live trading rules, PIT records, historical labels or risk limits.

Manual NCEI diagnostic: `python -m modules.ncei_missingness_audit --as-of 2026-11-14` (uses the local archive, no network).
