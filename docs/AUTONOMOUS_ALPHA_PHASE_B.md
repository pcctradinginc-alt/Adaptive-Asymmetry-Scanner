# Autonomous Alpha – Phase B: Bounded self-improving PIT factor discovery

## Scheduled, no manual start

The existing GitHub Actions Research workflow schedules:
- **Full research monthly on the 3rd at 07:41 UTC**: PIT panel -> Research Director -> Hypothesis Factory -> bounded genetic Explorer -> Research Lab -> Factory robustness -> Experiment Registry -> commit output artifacts.
- **Weekly research Saturdays at 08:23 UTC**: ML forward predictions, existing research checks, and experiment registry ingest. No new hypotheses are generated weekly.
- Existing scanner and feedback workflows continue independently.

No cron/service on a desktop is required. The original scheduled workflow is the runner; each execution can be examined in the GitHub Actions UI.

## Exploration engine

Module: `modules/autonomous_alpha/explorer.py`

- Recombines **seven preregistered point-in-time features** using additive blends, interactions, and a predeclared high-VIX gate.
- Parent directions are ex-ante economic assumptions, not signs optimized using future returns.
- Every expression must pass the exact existing Research Lab AST feature allowlist; future returns, labels, arbitrary Python and external code are impossible in the generated signal language.
- Allocates tests with deterministic seeded Thompson sampling (historical successes and failures are an **exploration priority**, NEVER independent confirmation) and a reserved novelty/random slot.
- Fixed safety limits: at most **2 new hypotheses per monthly full run**, at most 36 lifetime by configuration; hard code cap 3/month and 40 lifetime; max 2 variants per family.
- The registered hypothesis ID and SHA256 cover its immutable signal, parents, direction, family and horizon. Journal stored at `outputs/research/autonomous_alpha/proposals.jsonl`.
- Retries in the same UTC calendar month add **zero further trials**, preventing multiple-testing inflation. Duplicated identical Git-union JSONL rows are tolerated, mismatched ids/content fail closed.
- The existing Research Lab tests **every registered hypothesis** with its fixed cost/stress, PIT, walk-forward and multiple-testing machinery. Old failures remain in the denominator.
- With the documented **CONTAMINATED historical locked holdout**, every selected adaptive hypothesis that passes historical BH becomes `historical_exploratory_pending_forward`, NOT accepted. The locked interval is not re-used to confirm it.
- No automatic trading, new position sizing or production gate modifications; no new LLM calls. Economic options-chain verification and independent realized-option P&L still require a later *prospective* stage.

## Monitoring

The scheduled workflow commits:
- `outputs/research/autonomous_alpha/proposals.jsonl` (immutable proposals)
- `outputs/research/autonomous_alpha/exploration_plan.json` (latest run status)
- `outputs/research/hypothesis_db.json` (Lab OOS outcomes)
- `outputs/research/autonomous_alpha/experiments.jsonl` (Phase A evidence)
- `outputs/research/autonomous_alpha/summary.json` (historical vs prospective)

No manual start is necessary, but commands exist for diagnostics:

    python -m modules.autonomous_alpha explore
    python -m modules.autonomous_alpha status
    python -m modules.autonomous_alpha leaderboard
    python -m modules.autonomous_alpha verify

## Explicit limitations

Phase B delivers an adaptive, deterministic research hypothesis generation
loop, not self-editing Python modules or proof of profitable trading alpha.
The Explorer's use of historical outcome-dependent allocation creates
data-snooping risk, so even an apparently strong historical walk-forward
result only proposes a prospective challenger; it cannot approve production.

Any future code-generation agent requires a separate isolated, network-limited
sandbox, reviewable diffs, fixed experiment limits, and an independent forward
promotion stage. All existing forward history and production scanner mechanics
remain unchanged.
