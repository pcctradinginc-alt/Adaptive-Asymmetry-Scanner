# FD3 – Autonomous PIT Feature Discovery (Research / Shadow)

This is a separately versioned experiment extending unused-source inventory
PR #180. It does not change production scanner, option learner, RL, risk
limits, or the existing error_research v2 experiment.

## Inputs and as-of guarantees

Data are read from the existing ExternalArchive only:
- fred_regime_macro: WALCL, published ALFRED vintages.
- fred_liquidity: RRPONTSYD, WTREGEN, T10Y2Y.
- fomc_calendar / bls_release_calendar / bea_release_calendar:
  ONLY schedules archived at or before the simulated decision point.

The decision timestamp is midnight UTC after the US session of the stock
research panel date, before the next opening. It is not an intraday scanner
timestamp. ALFRED DATE-only release vintages get a full additional calendar
day of conservatism. Missing, late or stale values are not replaced with zero.

Net liquidity USD billions = WALCL / 1000 - TGA / 1000 - RRP. Both WALCL
and TGA are in USD millions; RRP is in USD billions. A 20-calendar-day
change is calculated from two independent PIT snapshots. Maximum age:
WALCL/TGA 14 days; RRP and 10y-2y curve 7 days.

For calendars, the archived plan may include future event dates if the
schedule was actually known at the decision time. A revised plan invalidates
the old scheduled date only after the new plan became available. Without any
archived plan the event distance is unknown, not "no event".

## Frozen hypotheses

The immutable protocol is config/research_feature_discovery_v3.yaml,
identified by SHA256 (editing the same version is rejected).

Four versioned shadow studies:
1. Net liquidity delta times stock beta.
2. Within two days of announced FOMC event times stock beta.
3. Within two days of announced BLS event times stock beta.
4. Within two days of announced BEA event times stock beta.

Global macro features are combined with a stock's existing PIT beta to
create cross-sectional differentiation; global levels alone do not rank
stocks within a day.

## Learning and audit trail

The research panel is the already-frozen ML_PANEL_CACHE; no extra market
download. Deterministic ridge baseline uses momentum, relative strength,
realized volatility, beta and dollar-volume factor. Challenger adds exactly
one FD3 feature, with identical tickers, dates, training cutoffs and costs.

Minimum 26 distinct training dates and 30 test stocks. A training label may
enter only when its 20-trading-day label_end strictly precedes test_date.
Predictions are frozen before a future outcome and never overwritten.
At maturity the first observed yfinance stock label is frozen once in
append-only outcomes. Yfinance stock labels are NOT executable option fills.

Historical diagnostics are exploratory only. Later forward results compare
paired daily rank-IC and top-decile stock excess returns, selecting windows
at least 28 calendar days apart. Before 8 independent forward windows,
status is INSUFFICIENT_INDEPENDENT_DATES. Even after this FD3 only suggests
research review, never declares statistically confirmed options alpha.

Research-output paths:
  outputs/research/feature_discovery_v3/predictions.jsonl
  outputs/research/feature_discovery_v3/outcomes.jsonl
  outputs/research/feature_discovery_v3/status.json

## Execution

  python -m modules.research_feature_discovery --panel-cache /path/to/panel.pkl
  python -m pytest -q tests/test_research_feature_discovery_v3.py

The existing ml_research.yml runs the feature study after ML research,
writes outputs with the normal research commit, and marks a failed stage
DEGRADED. It does not run new network requests or issue orders.

## Explicit exclusions / next phase

ClinicalTrials, USAspending, SEC 13F and NOAA need independently reviewed
issuer matching or location exposure, reliable publication timestamps and
PIT coverage before validation. Source-health PASS is not alpha evidence.

No change to the champion, locked forward tests, option learner or RL policy
may come from FD3. A separately preregistered and owner-reviewed study
must show prospective incremental predictive skill AND option-economic
improvement using archived RTH bid/ask, spreads, fills and commissions.
