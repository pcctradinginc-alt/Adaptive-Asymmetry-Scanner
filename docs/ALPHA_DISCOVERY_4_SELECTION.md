# Alpha Discovery 4.0 — intelligent hypothesis selection without more models

## Why this change

Previously the Scientific Hypothesis Factory sorted the eligible candidates by
a largely static priority (existing Research Memory posterior, economic
relevance, novelty, PIT readiness, computational cost) and used the highest
scoring ideas, with an exploratory reservation. In a crowded research pipeline
this could spend multiple protected test slots on essentially the same economic
family or signal. Research Director had a separate top-N allocation.

PR #178 adds ONE shared, deterministic **marginal research information**
selector to the existing two planning modules. It is a capacity-allocation
algorithm, **not a predictive model, new training loop or statistical proof**.

## Inputs and outputs

- Inputs: existing Factory or Director research candidate dossier, established
  research-priority score, existing Research Memory uncertainty, available
  signal feature names and family/source metadata.
- For replay safety, Research Memory entries known after the decision timestamp
  are ignored during planning. Legacy entries without timestamps are
  distinguishable; they do not independently prove forward economic alpha.
- Unchanged prerequisites: available PIT data, existing safe-expression
  allowlist, similarity block against previously tested research, permanent
  lifetime family cap, historical Research Lab/robustness and
  preregistered independent forward evaluation. Neither raw future outcomes
  nor locked holdout results are used in allocation.
- The Factory **retains its protected factory-v1 protocol file byte for byte**:
  6 research tests max per run; 20% (ceil = 2) exploratory places; no increase
  in global alpha budget, model numbers or family variants. The autonomous
  genetic explorer's independent cap of max 2 new hypotheses/month remains
  unchanged.
- Director retains the existing MAX_NEW_PER_RUN bound of 5, with at most two
  regime variants from the same measured source within one batch.

## Selection logic

1. Existing readiness and Research Memory disqualification stages exclude
   missing PIT features, previously tested hypotheses and disallowed
   near-duplicates before this algorithm even receives candidates.
2. Within each existing budget, the algorithm reserves exploratory places
   first, then greedily selects remaining candidates by **marginal research
   value**, recalculating after each selected hypothesis.
3. The candidate's earlier priority score is adjusted only by (a) its
   already known posterior uncertainty (a bounded information-gain proxy),
   (b) overlap of its signal's required PIT features with already selected
   ideas, and (c) repetition of a research-source class in the batch.
   No realized strategy return, optimized threshold or hidden label is
   consulted. This score is explicitly **not forecast profit**.
4. Near-identical expressions under different titles and overrepresented
   economic families cannot consume multiple slots in one batch. A slot
   is permitted to remain empty rather than being filled with a duplicate.
5. Stable tie-breaking by content-addressed hypothesis ID makes the selected
   IDs reproducible even when input iteration order changes.

## Audit trail and independent verification

Both Factory and Director persist a structured selection report with the
policy version, slots used, exploratory coverage, distinct idea families,
selection order and quantified reasons why nonselected candidates were
deferred. Each candidate records base priority, uncertainty source,
overlap penalties, marginal score and selection decision. Registered
hypothesis specification hashes, signatures, later Lab checks, FDR/BH
multiplicity denominator, option economics and locked-forward contracts
are not revised.

Review:
- modules/hypothesis_selection.py (new pure selector)
- modules/hypothesis_factory.py (same existing six-slot plan, new selector)
- modules/research_director.py (same Director cap, new selector)
- tests/test_hypothesis_selection.py (synthetic, no external feed)
- Existing Factory and Director regression, autonomous anomaly tests and
  full repository CI.

## What this does NOT establish

Better diversity and more transparent research allocation can reduce wasted
tests. **This change alone cannot prove alpha, increase expected profits or
guarantee discovery**. Only the existing independent prospective alpha-option
validation, with actual archived RTH entry/exit quotes, adverse execution,
correctly registered tests and statistical error budget, can establish
evidence for improved performance. Production and risk-policy changes
still require explicit human approval.

No new API keys, broker connectivity, paid model, real order execution,
automatic merge or feature fabrication.
