# Eight IMF statistical feeds — research integration, not a production alpha release

## Coverage and responsibility

| source_id | IMF dataset | Machine transport | Selected research questions |
|---|---|---|---|
| imf_fsi | FSIC Financial Soundness Indicators | IMF SDMX 2.1 | Bank capitalization and non-performing loans |
| imf_weo | World Economic Outlook | IMF DataMapper v1 | Forecast GDP/inflation/unemployment **as retrieved** |
| imf_mfs | MFS Monetary Aggregates | IMF SDMX 2.1 | Monetary aggregates and credit proxies |
| imf_gdd | Global Debt Database | IMF DataMapper v1 | Households and corporate/private debt |
| imf_fiscal_monitor | Fiscal Monitor | IMF DataMapper v1 | Fiscal balance and government gross debt |
| imf_climate | CCI Climate Change Indicators | IMF SDMX 2.1 | Emissions and fossil-fuel subsidy candidates |
| imf_fas | Financial Access Survey | IMF SDMX 2.1 | ATM/branch access candidates |
| imf_aipi | AI Preparedness Index | IMF DataMapper v1 | 2023 structural country readiness; not short-horizon predictive alpha |

All eight source IDs are registered in `config/external_sources/imf_macro.yaml`
and use existing `ConnectorResult`, PIT `Observation`, `ExternalArchive`,
normalization, source-health, provenance and DQ infrastructure.

Official IMF API documentation:
https://data.imf.org/en/Resource-Pages/IMF-API
https://www.imf.org/external/datamapper/api/

## Strict data handling

- **Do not backfill release timestamps.** Every historical year/quarter
  retrieved today has `available_at = retrieved_at` and conservative
  availability precision. WEO future-year projections are stamped with
  `forecast_issue_time=retrieved_at` and the explicit future valid date.
  This archive becomes useful for prospective learning; older historical
  values are **NOT** valid 2015-2025 model inputs.
- DataMapper indicator metadata must affirm the expected dataset and unit.
  Missing entire selected indicators, malformed JSON, data without numbers,
  non-finite values, unexpected country/time schema and oversized returns fail.
  Only approved ISO3 countries are accepted for observed values.
- SDMX must be from an IMF publisher, DSD from IMF, with country, indicator and
  frequency keys discovered from official referenced *codelists*. Indicator
  matching requires an **unambiguous** official label. No guessed code,
  auto-mapping to another metric, unspecified unlimited wildcard scan, or
  ingestion of error HTML/JSON as successful XML.
- Data scope is tiny and bounded (a handful of countries, two or three
  matched indicators, max four observations per SDMX series). It is **not**
  a complete copy of each IMF database or a bulk-download.
- Unknown official SDMX dataflow schema or absent selected indicator means
  **SCHEMA_CHANGED**, not a made-up value. Upstream failures are **FAIL**
  and are visible to the source-health dashboard.
- Statistical source rights: <https://www.imf.org/en/about/copyright-and-terms>.
  Credit IMF and indicate transformations. Noncommercial research only
  without additional permission; contact copyright@imf.org about commercial
  trading/product use. No LLM training. Research candidates are strictly
  disabled for automated trading/model adoption.

## Runtime isolation

The ordinary `external_data.yml` daily job excludes `run_only_by_family`
sources. This prevents 8 potentially slow IMF SDMX jobs from blocking or
starving PortWatch and all other existing feeds.

`.github/workflows/imf_macro.yml` performs:
1. **On pull request:** new offline regression suite + read-only
   `python -m modules.external.orchestrator preflight --family imf_macro`.
   Live preflight fails if even one dataset has no verified observations.
2. **On merge to main and weekly on Sunday:** isolated
   `ingest --family imf_macro` under the SAME `external-data-write`
   concurrency group as the daily ingestion, followed by
   `python -m modules.imf_macro_research`. Commit archive/health/readiness
   evidence using `scripts/ci_push.sh` (which prevents feature branch pushes).
3. Job fails after evidence persistence unless **all 8** research sources
   are ready with valid data. An upstream IMF outage is not disguised as a
   green run. Check GitHub Actions logs and the generated diagnostics.

The `outputs/research/imf_macro/status.json` report contains per-feed
sample size, source health, country coverage, latest observation and
up to 100 latest PIT feature candidates per source. These are *inputs
for subsequent experiments*, not proven alpha: `model_input_enabled=false`,
`production_promotion_allowed=false`, `alpha_validation=NOT_EVALUATED`,
`historical_walk_forward_eligible=false`.

## Required next gate for model training

No direct wiring into `option_learner` or `world_model`. First obtain
enough **forward-collected** historical vintages, validate
`available_at <= signal_time`, measure coverage by day and asset sector,
and introduce paired baseline-vs-IMF challenger experiments with time embargo
and multiple-testing correction. Require independent positive net executable
returns across sufficient regime diversity. Without evidence, no model
feature promotion and absolutely no relaxation of Safe Mode or risk limits.

## Local checks

```bash
python -m pytest -q tests/test_imf_macro_eight_feeds_20261104.py
python -m modules.external.orchestrator preflight --family imf_macro
python -m modules.external.orchestrator ingest --family imf_macro --dry-run
python -m modules.imf_macro_research
```

*Approval criterion*: all offline+full CI green, all **live** preflights
with actual data at current head green, then merge. After merge rerun
production-host workflow and audit normalized data and candidate evaluation.
Do not claim valid new predictive features merely because downloads work.
