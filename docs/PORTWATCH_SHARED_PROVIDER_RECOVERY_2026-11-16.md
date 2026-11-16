# IMF PortWatch shared-ArcGIS recovery – 2026-11-16

## Observed incident
The public `services9.arcgis.com` quota is 6000 request units/minute.
The 2026-11-16 ingestion reports 7083 units used, ArcGIS JSON 429 even
after the bounded waits, and all three datasets FAIL. The quota reflects
the whole public publisher, not just this repository. No local retry
algorithm can guarantee the publisher will become available.

## Minimal recovery
- The small important chokepoint dataset runs before event disruptions and
  the much larger daily ports query. This prioritizes useful verified data
  before the shared provider rate limit is exhausted.
- A separate daily off-window `PortWatch Recovery` workflow invokes the
  existing `--family maritime` CLI. It shares `external-data-write`
  concurrency with the normal orchestrator and uses the same fail-closed
  rules and bounded 429 retries. No invented observations or stale-as-fresh
  fallback; if the provider is still over quota, status stays FAIL.
- The Source Health report distinguishes **affected important datasets**
  from **independent upstream failure domains**, but retains the existing
  conservative safe-mode threshold and model safe-mode triggers.
- The OCHA HDX shadow can follow a redirect from a validated HDX resource to
  the observed public S3 regional storage host. Initial S3 URLs, other hosts,
  unverified publishers and chained untrusted redirects remain blocked.
  This mirror remains SHADOW only: its ports cover eight countries, not the
  global reference universe, and its CSV must independently pass schema,
  freshness and provenance tests.

## Evidence gates after merge
1. Test code and workflow syntax with CI. Inspect the exact PR HEAD checks.
2. Observe GitHub Actions `PortWatch Recovery`, then verify actual fresh
   primary ArcGIS observations with two independent successful ingestion
   checks before declaring source HEALTHY; do not equate action success with
   upstream success.
3. Shadow HDX PASS is diagnostic **only**, never production fallback or
   reason to override Research Safe Mode.
4. Models, hypotheses, signal thresholds, risk gates and trading rules are
   unchanged. Proven Alpha remains unconfirmed.
