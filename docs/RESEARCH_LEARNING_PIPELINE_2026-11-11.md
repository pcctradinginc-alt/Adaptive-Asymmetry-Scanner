# Research learning pipeline v1 – gated rollout (2026-11-11)

## Objective
Grow trustworthy, prospective **executable-option** evidence, not trade count.
The frozen Champion v2 (first measurement 2026-11-11) is unchanged:
no modifications to `config.yaml`, `config/economics_first.yaml`,
`config/champion_protocol.yaml`, `modules/champion_protocol.py`,
`config/research_protocol.yaml` or live trading gates.
Research observation must never be called proven alpha.

## Implemented in this PR

### 1. Forensic ENTRY_NOT_ARCHIVED audit
`python -m modules.label_recovery_audit --horizon 5` and `--horizon 20`.
Compares existing signal/strategy IDs, exact contracts, event dates and observed
bid/ask archive. Records per-signal missing IDs and usable same-day quotes in
`outputs/research/label_recovery_audit/YYYY-MM-DD_5d.json`.
**No automatic historical label recovery**: quotes from before the 2026-10-12
archive start cannot be fabricated. The reported 308 missing 5-day entries
require independently verifiable original raw observations; mathematical
backfill is prohibited. A report counts 0 labels restored until externally
validated provenance actually exists.

### 2. Independent LLM census, pilot five candidates/day
During FinBERT stage, before the sector filter, `llm_census_offline.freeze`
writes a reproducible random `k=5` cohort of hard-filter survivors
and inclusion probability `k/N` to
`outputs/research/llm_census_inputs/YYYY-MM-DD.json`.
The main pipeline's old k=2 LLM census is disabled in
`config/llm_census.yaml` (its settings are research-only).
Nightly `.github/workflows/llm_census_offline.yml` reads **only frozen
candidate identities, news and basic features**, and submits Sonnet via the
existing DeepAnalysis with duplicate model-routing A/B disabled.
There are NO writes to Candidate Ledger or production directions, orders,
gates, or scoring. Output is independent, append-once and namespaced under
`outputs/research/llm_census_offline/`.
Fail closed if missing Anthropic key, malformed frozen cohort, unknown monthly
projection, research budget pressure, or less than $1 projected headroom to
the $10/month API target. Hard cap 10; configured pilot k=5 and 600s runtime.
The 10–20/day expansion is **NOT** approved by this PR; measure actual usage
and processing completion before lifting caps.
**Methodological limitation**: DeepAnalysis still queries 48h price context
and macro data at the later evaluation time. Its outputs must NOT be treated
as strict as-of-scan forward trading signals; this is a descriptive LLM
information-value audit, not proof of incremental options alpha.

### 3. Provider-approved +40 real option snapshots
`modules/research_snapshot_expansion.additional_slots` adds a maximum of
40 research-only snapshots beyond the existing 40, with a hard 80 total.
The production snapshot selection remains first and unchanged.
Additional requests are **disabled until** someone verifies Tradier's plan,
quotes endpoint rate limits, total chains/term structure/alternative contracts
costs and explicitly sets the GitHub Actions variable
`TRADIER_RESEARCH_80_APPROVED=1`.
Without that variable, the total is unchanged at 40.
The extra random/bandit cohort retains selection propensities in the ledger.
The extra cap is entirely independent of `config.yaml`.
**Costs**: 80 symbols may require substantially more than 80 calls once
alternative expiries, option legs and IV are included. No paid plan activated.

### 4. No-LLM prospective options challenger
`modules/no_llm_challenger.py` is invoked in the Scanner workflow **before**
the day's Option Learner updates its acquisition model.
It uses only the pre-existing fitted `acquisition_model.json` with strictly
prior label cutoff (< signal day). If unavailable, stale, or unapproved,
ABSTAIN. No backtesting/re-ranking after label arrival.
It **only considers genuine non-directional research-census BULLISH calls**,
never Sonnet-chosen Champion candidates, Puts, or invalid entry quotes.
Outputs one frozen, append-once day-of-paper ranking with up to 5 watchlist
names at `outputs/research/no_llm_challenger/YYYY-MM-DD.json`.
This is an unvalidated acquisition score, not a calibrated option EV or
a trade recommendation. **No orders, no production ranking influence.**

### 5. Honest prospective no-LLM results
Weekly `.github/workflows/options_learning_evidence.yml` evaluates
pre-frozen watchlists using `option_learner.candidates()`: the same
VERIFIED provenance of real observed Entry Ask / Exit Bid minus fees.
Unknown/late labels remain UNVERIFIED, never 0 or an imputed quote.
Results contain verified coverage, daily mean returns and comparison
against 0% never-trade baseline. They do NOT claim a formal matched
Champion outperformance test (that requires more independent days and
identical capital/entry policy).

## NOT implemented / further experiments
- Filling all 308 missing historical labels: evidence may simply not exist.
- Automatically raising third-party quote/LLM budgets: provider and token
  costs are not yet proven.
- Adding k=10/20 live LLM calls: capped at 5 until real outcomes/costs.
- New production ranking, direct promotion, filters removal, Bearish trading,
  threshold loosening or orders: all explicitly deferred.
- Fully matched triple-arm study Champion versus no-LLM versus LLM-assisted:
  needs an independent prospective research protocol and common
  contract/entry timestamps. The existing outcome file is a pilot only.
- New autonomous retraining of strategies from LLM-specific features:
  deferred pending strictly time-stamped, unbiased and independently
  verified data.

## Quality gates and rollback
Tests `tests/test_research_learning_pipeline.py` establish missing evidence
stays missing, no prior-model leakage, frozen cohort immutability, no orders,
price identity, budget hard caps, provider-approval gate, and no change to
pinned Champion configuration.
CI full suite + leakage/Research protocol tests required prior to merging.
Rollback: disable the independent workflow, set
`config/llm_census_offline.yaml enabled: false`, and unset the approval
variable. The original `config/llm_census.yaml` can be re-enabled to
restore the old two-per-day research census without changing Champion trades.
