# IMF statistical feeds (7 active, AIPI disabled) — research integration, not a production alpha release

## Coverage and responsibility (revised 2026-11-05)

All active feeds use ONE reader, `modules/external/sources/imf_sdmx3.py`, against
the official IMF SDMX 3.0 API. Agency, dataflow, DSD dimension order and every
code were verified live from a GitHub runner on 2026-11-05.

| source_id | IMF dataflow | Key (COUNTRY x ...) | Content |
|---|---|---|---|
| imf_weo | IMF.RES/WEO (+ official `WEO_*_VINTAGE` editions) | NGDP_RPCH, PCPIPCH, LUR, GGXWDG_NGDP, BCA_NGDPD, A | GDP growth, inflation, unemployment, debt, current account incl. projections |
| imf_fiscal_monitor | IMF.FAD/FM (+ `FM_*_VINTAGE`) | GNLB/GPB/G63G_S13_POGDP_PT, A | Fiscal balance, primary balance, gross debt |
| imf_gdd | IMF.FAD/GDD | FL_PS/FL_S14/FL_S11/FL_S13_POGDP_PT, A | Private, household, corporate, government debt |
| imf_fsi | IMF.STA/FSIC | S12CFSI x AQ12/FSI688/AQ14/ROE_CFSI_PT, Q | NPL ratio, capital ratio, provisioning, ROE of deposit takers |
| imf_mfs | IMF.STA/MFS_MA (+ `MFS_MA_*_VINTAGE`) | BM_MAI x XDC/USD, M | Broad money (non-euro countries; euro members do not report it nationally) |
| imf_climate | IMF.STA/CCI | AMGST_dC, NDT_NUM, A | Temperature anomaly, climate-related disasters |
| imf_fas | IMF.STA/FAS | FA26, FA18, OUTL_A, OUTD_L x PHTADLT_NUM/POGDP, A | Bank branches/ATMs per 100k adults, loans/deposits % GDP |
| imf_aipi | – | – | **Disabled**: not on SDMX 3.0; DataMapper returns 403 (Akamai) to CI. Never bypassed. |

Countries: USA, DEU, JPN, GBR, FRA, CHN (MFS: USA, JPN, GBR, CHN, CAN, AUS, CHE, IND).

Root causes of the original failures (all eight feeds red):
- WEO/FM: `Accept: ...json;version=2.0` (missing patch level) -> HTTP 500; `2.0.0` works.
- FSIC: invented code `NPLR_PT`, ISO-2 country `DE` -> 500. Real: `USA.S12CFSI.AQ12_CFSI_PT.Q`.
- MFS/CCI/FAS: SDMX 2.1 path with label-guessing found no indicator axis.
- GDD/AIPI: DataMapper is Akamai-blocked (403) for GitHub runners.
- `c[TIME_PERIOD]` filters are ignored/empty on this API; startPeriod/endPeriod are used.

## Strict data handling

- **Current editions** are stamped `available_at = retrieved_at` (prospective PIT).
- **Official IMF vintage dataflows** (frozen editions such as
  `WEO_2025_OCT_VINTAGE`) are stamped with the first day of the month after
  the edition month, which is always later than the real publication date.
  These are genuine historical vintages and give real revision history
  (`as_observed_revision` / `as_observed_forecast_revision`) from day one.
- WEO/FM values for the current and future years are forecasts:
  `forecast_issue_time = available_at`, `forecast_valid_time = period`.
- Every series key is checked against the requested codes; dimension order
  must equal the verified DSD; non-numeric/non-finite values, unknown time
  formats, duplicate identities, HTML or empty bodies -> SCHEMA_CHANGED with
  zero partial observations. A missing requested indicator -> SCHEMA_CHANGED.
  HTTP/network errors -> FAIL. Official missing values are skipped, never 0.
- A failing vintage lookup never discards current data (status WARN).
- Statistical source rights: <https://www.imf.org/en/about/copyright-and-terms>.
  Credit IMF and indicate transformations. Noncommercial research only
  without additional permission; contact copyright@imf.org about commercial
  trading/product use. No LLM training. Research candidates are strictly
  disabled for automated trading/model adoption.

## Runtime isolation

The ordinary `external_data.yml` daily job excludes `run_only_by_family`
sources. This prevents the IMF SDMX jobs from blocking or
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
3. Job fails after evidence persistence unless **all enabled** research sources
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
python -m pytest -q tests/test_imf_sdmx3_feeds_20261105.py
python -m modules.external.orchestrator preflight --family imf_macro
python -m modules.external.orchestrator ingest --family imf_macro --dry-run
python -m modules.imf_macro_research
```

*Approval criterion*: all offline+full CI green, all **live** preflights
with actual data at current head green, then merge. After merge rerun
production-host workflow and audit normalized data and candidate evaluation.
Do not claim valid new predictive features merely because downloads work.
