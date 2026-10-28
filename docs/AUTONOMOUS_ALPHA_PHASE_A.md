# Autonomous Alpha Engine – Phase A (2026-10-28)

## Purpose and limits

Phase A unifies the existing research workflow under one CLI and an auditable
experiment registry. It does not yet generate self-modifying Python code,
perform Bayesian optimization or automatically promote trading rules.

Production scanner, options/ROI logic, risk, position sizing, protected
research and promotion policies, shadow ledgers and prior forward outcomes
are unchanged.

The historical locked OOS interval starting 2025-07-01 is explicitly marked
CONTAMINATED. It cannot confirm independent alpha or authorize promotion.

## Architecture

Existing reusable modules:
- modules.ml_research: PIT panel, purged walk-forward, models.
- modules.research_director: diagnoses research questions.
- modules.hypothesis_factory: preregistered candidates; optional LLM.
- modules.research_lab: PIT DSL, repeated-testing control, evidence.
- modules.research_memory: failures and successes, prioritized future topics.
- modules.promotion_controller: independent promotion contracts and guards.

New:
- modules.autonomous_alpha.orchestrator: fixed allowlisted stage sequence,
  one shared panel, fail on error; no arbitrary code evaluation.
- modules.autonomous_alpha.registry: content-addressed append-only journal,
  frozen spec hashes, historical result leaderboard, rebuildable SQLite index.
- modules.autonomous_alpha.__main__: local command-line entry point.

Data flow:

    PIT Panel
      -> Research Director
      -> Hypothesis Factory (LLM disabled by default)
      -> Research Lab (existing protected validation)
      -> Robustness Battery
      -> Research Memory
      -> Immutable experiment journal
      -> next Research Director plan (via existing Research Memory)

The new registry is an evidence index; it does not rerank real trades or
modify the original research thresholds. Genetic mutation, Bayesian search
and sandboxed code-writing agents are planned later, not yet implemented.

## Persistence and reproducibility

Source research remains authoritative in:
- outputs/research/hypothesis_db.json
- outputs/research/factory_plan.json
- outputs/research/factory_results.json
- outputs/research/ml_research.json

New journal:
outputs/research/autonomous_alpha/experiments.jsonl

Derived rollup:
outputs/research/autonomous_alpha/summary.json

Each journal event records a stable experiment key, immutable specification
fingerprint, source record SHA256, known historical net OOS results, Sharpe,
drawdown, stress results and source timestamps. Missing metrics are null.
Positive historical results do not imply prospective options profitability.

Concurrent git union merge may duplicate identical journal rows; reading
deduplicates by content hash. Altered or corrupted journal records cause
a hard error. SQLite is local and reconstructible, never authoritative.

## Usage from repo root (dependencies required)

    python -m modules.autonomous_alpha run
    python -m modules.autonomous_alpha status
    python -m modules.autonomous_alpha leaderboard --limit 10
    python -m modules.autonomous_alpha verify

Full run may require substantial external downloads and CPU.

Reuse an already trusted, locally produced research panel:

    python -m modules.autonomous_alpha run --panel-cache /path/to/panel.pkl

Do not import untrusted pickle files.

Ingest existing research outputs without retesting or network calls:

    python -m modules.autonomous_alpha ingest --strict

A nonzero exit code means the run or evidence integrity check failed.

## CI integration

The existing monthly/weekly ml_research workflow runs the strict ingest
after the prior research stages and before its existing commit step.
No parallel research pipeline, no new LLM calls, and no additional option
quotes are introduced by this CI integration. The registry step is critical:
a broken journal does not silently pass.

## Planned separately

Phase B: systematic PIT-DSL exploration, genetic and Bayesian search,
fixed multiple-testing families, deterministic ablation, research budgets.

Phase C: isolated code-generation sandbox, capability restrictions,
prospective paper outcomes and independent promotion/demotion controls.

An Alpha discovery or favorable historical Sharpe is never sufficient
authorization for a change to production trading behavior.
