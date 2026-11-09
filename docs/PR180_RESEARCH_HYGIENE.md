# PR #180 — Research Hygiene / Candidate Coverage and Unused Source Inventory

## Purpose and no-production-change contract

Main goal remains **discover independent, economic alpha**. The goal of this
small PR is to measure **where the research pipeline does not use data**,
not to start more models or falsely improve historical backtests.

- The frozen discovery protocol, multiple-testing denominators, existing
  research/quote API budgets, paper-trading logic, production rules, risk limits
  and pre-registered hypothesis hashes are unchanged.
- No source is automatically enabled or disabled, no new indicator becomes an
  input to a live predictive model, and no strategy gets promoted.
- No paid datasets, key requests, extra option-chain reads or LLM calls.
- PR does not claim alpha or imply successful quote exits without provenance.

## Weekly audit in the existing workflow

Every Saturday ML research run (MODE=weekly), **after the existing quote
provenance stage**, two read-only tools generate dated JSON outputs under
`outputs/research/`. These are included in the existing research commit.

### Candidate census

`python -m modules.research_candidate_census --as-of YYYY-MM-DD`

- Counts **all** candidate-ledger rows in the preceding seven calendar days,
  including those filtered out by economics-first. These rows cannot be
  mistaken for actual executed orders.
- Measures unique tickers, observed underlying marks, observed but
  **not verified** option-return fields, reason for rejection, directional vs
  research-only option sampling and a deterministic 40-ticker audit sample.
- The separate historical verification summary uses ONLY
  `outputs/research/evidence_discovery/<as_of>.json`, produced by the existing
  `modules.research_evidence_discovery`. Its genuinely verified labels are
  checked against actual entry and exit option quotes, appropriate timestamps
  and contract identities. The census never verifies a quote itself or
  recomputes returns.
- The **verified** figure explicitly covers **all eligible historical
  candidates**, NOT the seven-day census cohort. A seven-day marked return is
  not inferred to be one of the verified outcomes.
- Missing monthly ledger partitions, a missing/future/malformed evidence
  report, or mismatched as-of dates give `DATA_GAP` and a nonzero process exit.
  No fake zero outcomes; no automatic backfill or unregistered hypotheses.
- Existing `--date` command remains backward compatible for manual checks;
  it never triggers additional quote capture.

### Source-utilization inventory

`python -m modules.research_source_utilization --as-of YYYY-MM-DD`

- Reads **only existing** `outputs/external_data/health/source_health.json`
  and `outputs/external_data/health/storage_telemetry.json` on disk.
- Reports PASS / UNKNOWN source health and recorded archive-row count,
  retrieval days, estimated raw and normalized bytes/day and one-off
  backfill bytes, **when all data are available and recent**. Missing/stale
  fields are null, not zero, and the report is marked `DATA_GAP`.
- The "unused by analysis" status was verified in the code review on
  **2026-11-08**. It is a **dated manual assessment**, not a continuously
  recomputed dependency graph. An analyst must refresh this classification
  whenever source consumers change.
- Both new weekly stages are `DEGRADED` in the existing stage check if they
  fail. A failure is visible and **does not block the earlier alpha forward
  evaluation** or cause a spurious success.

## Baseline evidence (archive snapshot 2026-11-08 13:40 UTC)

| Feed | Measured status | Existing history / footprint | Alpha-directed next action |
| --- | --- | --- | --- |
| ALFRED / FRED liquidity | PASS | 3,613 normalized rows, 3.58 MB one-off backfill | High: consider PIT-vintage liquidity shock and regime hypotheses |
| BEA calendar | PASS | 74 rows, 70,996 backfill bytes | High: event-window ex ante exposure |
| BLS calendar | PASS | 34 rows, 186,207 backfill bytes | High: event-window ex ante exposure |
| FOMC calendar | PASS | 12 rows, 342,144 backfill bytes | High: event-window ex ante exposure |
| ClinicalTrials | PASS | 2,536 rows; currently 26/48 healthcare issuers matched | Medium: verify mapping and actual public registration timestamps |
| USAspending | PASS | 678 rows; 13 issuers matched | Medium: verify award date vs. public disclosure, map entities |
| SEC 13F | PASS | 7,590 rows, ~202 MB one-off backfill measured | Low: historical publication timing + current snapshots, evaluate cost first |
| NOAA Storm Events | PASS, backfill only | 1,849 rows, ~13.2 MB one-off backfill | Low: no PIT firm exposure; already throttled, not every day |

These counts come from the repository's
`outputs/external_data/health/{source_health,storage_telemetry}.json`
and may change. "PASS" is only **source health**, not economic utility or
quote verifiability. The daily footprint fields use storage telemetry estimates,
not monetary invoices. NOAA is already in a low-frequency backfill regime;
"unused" does not mean that it is downloaded every day.

## Candidate for a future, **separately approved and preregistered** study

The first candidates worth considering are FRED liquidity and public economic
release calendars. Do not change today's `error_research v2` search space or
train on these new fields inside this PR.

A proposed **new** version must explicitly fix, before the first data-driven
selection or test:
1. **Research questions**: do changes in overnight RRP/TGA from available
   ALFRED vintages explain stock/sector relative returns *beyond* existing
   PIT factors? Is pre-announced proximity to FOMC/BLS/BEA releases associated
   with **incremental** predictive skill or option costs in different
   regimes? An event calendar contains no surprise value by itself.
2. **PIT timing**: snapshot is available only after archived publication
   timestamp, not simply the observation date. Calendar future event dates
   are known at the announcement time only; no benefit from later revisions.
3. **Immutable trial design**: input cutoffs, universe, horizons, economic
   baseline, turnover, transaction cost assumptions, independence windows,
   feature and hypothesis budget, correction for the complete multiple-test
   search, negative controls and falsification conditions.
4. **Incremental economics**: stock baseline versus baseline + source,
   then frozen option structures with real bid/ask entry/exit archives,
   spread, slippage, commissions and independent forward periods.
5. **Decision**: retain source only if prospective incremental evidence
   justifies its data collection and engineering cost; otherwise recommend
   disabling low-value ingestion in a separate reviewed PR.

There is **no current proven alpha** in these feeds; they are research
proposals, not new production features.

## Change inventory

- `modules/research_candidate_census.py` — weekly as-of report reusing
  pre-existing quote provenance, old daily CLI retained
- `modules/research_source_utilization.py` — new source health/cost inventory
- `.github/workflows/ml_research.yml` — two bounded weekly steps, degraded
  checks; **no new workflow schedule**
- `tests/test_pr180_research_utilization.py` — offline synthetic evidence,
  leak/fail-closed and CI wiring checks
- This document and `docs/RESEARCH_CANDIDATE_CENSUS.md`

No generated historic research output is committed as evidence of new alpha.
