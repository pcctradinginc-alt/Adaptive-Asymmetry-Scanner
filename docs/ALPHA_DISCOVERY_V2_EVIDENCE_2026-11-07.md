# Autonomous Alpha Discovery 2.0 — evidence audit (2026-11-07)

**Research only.** Based on stored repository outputs at branch point
`a6f5377`. No order, no API purchase, no claim of validated alpha.
P0 historical-data integrity is a safety prerequisite to P2–P7.

## What is independently proven today?

**Zero confirmed alpha hypotheses.** The source
`outputs/research/autonomous_alpha/summary.json` shows 120 historical
experiments, `n_prospectively_confirmed=0`, and the reused locked holdout
`CONTAMINATED`. Historical ACCEPTED, ROBUST or VALIDATED cannot be counted as
out-of-sample success. Immutable source labels remain for forensic auditing;
consumer status is now EXPLORATORY or PENDING_FORWARD.

The factory is still allowed to screen on historical data, but before any
independent confirmation it must create a frozen prospective challenger
with registration date, spec hash and a future start date. Subsequent verdicts
must derive solely from registered forward data and the existing permanent
multiple-testing budget. No model/production change is authorized by this work.

## Reproducible, authentic repository data review (no synthetic fills)

| Existing file | Observed count or state | What it means |
|---|---:|---|
| `autonomous_alpha/option_link.json` | 94 signals; 46 decisions (29 TRADE, 17 ABSTAIN); 0 outcomes | Ex ante paper decisions exist, but there are no verified settled options outcomes for this new link |
| `autonomous_alpha/option_validation.json` | NEED_MORE_DATA, 0 validation plans, test budget 20 slots unused | No admissible independent alpha-to-options confirmation yet |
| `autonomous_alpha/option_economics.json` | 29 PENDING, 17 ABSTAIN | No observed post-cost portfolio alpha from those pending positions |
| `option_learner/status_5d.json` | 495 VERIFIED; 308 ENTRY_NOT_ARCHIVED | A separate 5-day population, not additional independent observations for the alpha-to-options 20-day link |
| `option_learner/status_20d.json` | 130 VERIFIED; 308 ENTRY_NOT_ARCHIVED; 1 EXIT_NOT_ARCHIVED | This population has large historical entry-quote provenance gaps |
| `evidence_discovery/2026-11-07.json` | 8,995 rows; 119 VERIFIED; 272 ENTRY_NOT_ARCHIVED; 1 EXIT_NOT_ARCHIVED | Research evidence availability; not the same denominator as the option learner |
| `error_research.json` | 5d: 2 independent windows; 20d: 0 | Less than the frozen prospective testing requirement |
| `option_learner/status_5d.json`, `status_20d.json` | `news_novelty` feature coverage 0 | Existing feature missing for training; variant extraction may have separate coverage |

Full file paths for the above research outputs start with `outputs/research/`.
The first three include `autonomous_alpha/` under that root.

**Do not pool these populations** or average their counters. The historical
provenance ratios are not independent 20-day forward-test coverage. Never use
stale, inferred or model-derived bids/asks as observed quotes. Historical
missing entry quotes, particularly before capture began in October 2026,
are irrecoverable as point-in-time observations.

## P0–P7 completion boundaries

- **P0:** The new safety gate marks contaminated historical results
  EXPLORATORY for all sources. Preexisting cached canonical labels, factory
  robustness screens and experiment registry snapshots cannot independently
  authorize promotion. Unit tests include a deliberately enormous synthetic
  effect; the stored historical source evidence is preserved.
- **P1:** Read-only data-quality telemetry reports source provenance, quote
  failure reasons, below-80% feature coverage, date of first recorded gap in
  monitoring, blockers and independent-period uncertainty. Repeated daily
  gap receipts are create-only; the historical onset is explicitly unknown.
  Existing RTH capture stays unchanged (already collects Tradier quotes using
  a read-only secret).
- **P2/P3:** Reuses hypothesis_factory, research_lab, research_director and
  research_memory. The Research Director enforces the safe PIT expression
  language, maximum tests per planning run, similar prior tested-hypothesis
  suppression and measurable feature deferrals. New dossiers fix economic
  mechanism, ex ante expectations, 20-day horizon, potential call/spread
  structures, controls, falsification and forward-registration intent. These
  dossiers are *proposed tests*, not evidence.
- **P4:** Existing option_link, option_economics and option_validation
  already compare Long Call and Bull Call Spread with observed quotes,
  commissions and stress costs. Existing error budgets remain immutable.
  As the link has no verified settled outcomes, statistical superiority is
  currently **NEED_MORE_DATA**.
- **P5:** Research Memory records observations and quote/feature gaps;
  the Director consumes memory before selecting the next hypothesis.
  The GitHub paper account is an economic monitor, never a second
  independent training label or production strategy.
- **P6:** Synthetic flow tested from historical-screen gate through future
  frozen challenger and memory/Director follow-up; authentic repository
  outputs are read by a separate no-network test. A positive alpha finding
  is NOT expected or fabricated.
- **P7:** The existing weekly email now has “Autonomous Alpha Discovery
  2.0”: EXPLORATORY, PENDING_FORWARD, CONFIRMED, FALSIFIED,
  NEED_MORE_DATA, verified quotes, hypothesis plans, missing data, and
  post-cost conclusions only when sufficient.

## Release and scientific caveats

The first real weekly run and full CI of the **new PR** must pass before this
change could be merged. No new production strategy, no automatically modified
risk gates, no new broker account or cost-bearing API, and **NO AUTO MERGE**
for this research PR. An alpha result is not confirmed just because a
synthetic test exercises the pipeline.

Remaining priority: capture **future** real entry and exit quotes, accumulate
nonoverlapping independent forward periods, and only after the registered
evidence budget is satisfied assess whether the system has discovered a
repeatable economically positive options alpha.
