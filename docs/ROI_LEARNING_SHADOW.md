# ROI Learning Gate – isolated research-only counterfactuals

The production OptionsDesigner remains unchanged: **no lowered production ROI, median, edge, earnings or risk threshold**.

This research path runs using the already constructed and rejected real option contracts from `OptionsDesigner.roi_reject_log`. It does NOT call Sonnet, Haiku, an agent or any new price discovery service to propose a trade.

## Research thresholds

Compare the original (pre-Monte-Carlo) estimated options ROI to frozen research-only labels **0%, 5%, 7%**. These are descriptive threshold buckets, NOT a live-ROI approval, and do not bypass the negative-median or option-break-even rules. All cohorts are explicitly flagged `production_eligible: false`.

## Lifecycle

1. Each production scan registers **at most 5** unique rejected tickers per New York trading day (in practice often fewer, because the existing Sonnet budget restricts the final set). Deterministic hash ordering with failure-gate stratification avoids selectively logging only the best-looking near misses.
2. The original source candidate, selected OCC contract legs, direction, original ROI, real production hurdle, rejection gates and actual registration time are stored in an immutable, SHA-256 journal: `outputs/research/roi_learning/cohorts/YYYY-MM.jsonl`. Idempotent under scanner restarts.
3. Existing independent RTH Options-Capture workflow automatically monitors the registered long and short legs for entry and 5/20/45-day marks. Capture targets have a **hard 30-symbol ceiling per run** and are combined with the ordinary quote batch. No second scanner or extra LLM job.
4. Existing Feedback workflow invokes `python -m modules.roi_learning_shadow evaluate` without network/LLM. It matches **observed, RTH-executable bid/ask quotes** in `outputs/options_archive/quotes`. Entry requires quote retrieval strictly AFTER registration and no later than 5 calendar days. Exit requires a real bid/ask quote at/after each frozen horizon, no later than 8 calendar days beyond the target. Spreads require two legs from the same NY session within 15 minutes.
5. The worker records conservative buy-at-ask/sell-at-bid (or spread buy/sell crossing), plus $0.65 per leg per side (explicit configurable assumption, not a guarantee of actual broker fees) to calculate a paper-only net return. Any missing, stale or unusable quote means **UNKNOWN / not a training outcome**. No Black–Scholes fallback.
6. Frozen outcomes are appended to `outputs/research/roi_learning/marks/YYYY-MM.jsonl`. A compact diagnostics report is available at `outputs/research/roi_learning/summary.json`. Neither source is consumed by the live production decision path.

## Commands for diagnostics only

The production, options capture and feedback cron schedules remain unchanged. **No manual start is needed**.

    python -m modules.roi_learning_shadow evaluate
    python -m modules.roi_learning_shadow status
    python -m modules.roi_learning_shadow due

## Evidence warnings and next milestones

- Shadow fills from archived RTH NBBO-like quotes are **not broker fills**. Commissions are assumed, and market impact, fill probability, assignment and slippage may differ.
- Because selection is after the expensive analysis, the existing hard cap of **3 Sonnet candidates** can yield fewer than five registered samples daily. We do NOT increase LLM budget to inflate the count.
- Existing earlier 21–45-DTE ROI precheck and the final 150–365-DTE options path still disagree. This change intentionally does not relax the production precheck. Its economic impact requires an independent directional-neutral shadow comparison.
- Do not mix these paper marks with real close outcomes, the previous generic stock-level shadow returns, or PPO's currently unreliable small-sample rewards.
- A larger shadow dataset can improve diagnostic calibration; **more rejected candidates do not automatically establish positive options alpha**.
