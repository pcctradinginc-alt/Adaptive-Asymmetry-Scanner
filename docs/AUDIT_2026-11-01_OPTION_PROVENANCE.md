# Audit 2026-11-01: 20-day option provenance and research candidate sample

## Starting point (status 2026-10-31)

| Horizon | VERIFIED | ENTRY_NOT_ARCHIVED |
|---|---|---|
| 5 days | 359 | 308 |
| 20 days | 0 | 308 |

## 20-day provenance

| Question | Finding | Evidence |
|---|---|---|
| Are missing entry quotes historical gaps? | **Yes.** The quote archive starts on 2026-10-12. All 309 expressions with entries from 2026-09-28 to 2026-10-09 are `ENTRY_NOT_ARCHIVED` (308 of them labelled). They stay flagged as missing; nothing is reconstructed. | Entry provenance per entry day |
| Are new entry quotes archived completely? | **Yes.** 495 of 495 expressions with entries from 2026-10-14 to 2026-10-30 have archived, RTH-usable and price-matching entry quotes (timing-v2 record IDs). | Entry part of `research_evidence_discovery.provenance` |
| Does exit capture work at 20 days? | **Yes.** All 308 existing 20-day marks (on the pre-archive entries) reproduce exactly from archived RTH quotes, and none is late. | Exit check on archive data |
| Why 0 VERIFIED at 20 days? | **Not mature yet.** The first 20-day mark of an archived entry is due on 2026-11-03 (entry 2026-10-14 + 20 calendar days). It is not a defect. | Ledger entry days |
| Holding period, quote times, trading hours | Consistent: calendar days in the ledger and in provenance, the mark window of `option_quotes.mark_window`, and the first trading day on or after the due day. | Code |
| Only reproducible results counted as VERIFIED? | Yes. 359 of 359 5-day labels after the archive start are VERIFIED. Claims the archive does not support are rejected (existing tests). | Provenance counts |
| Robust after restarts? | The archive has 15 of 15 trading days since 2026-10-12. Union merge and append-only storage are in place. Scanner, feedback, source health and external data were OK every trading day. | `workflow_status.json` |

### Defect found and fixed: spread short legs at exit

**Cause.** `option_quotes.usable_rth_quote` requires bid > 0 for every leg. A spread's short leg, however, is bought back at the ask.

**Observed case.** On the 20-day due day, 2026-10-28, `PTC270416C00210000` (the short leg of a bull call spread) quoted bid 0 / ask 0.20. The spread mark slipped to 2026-10-29. The long-only mark of the same candidate stayed on 2026-10-28.

**Impact.**
* Spread labels get shifted horizons.
* If the zero bid persists for more than 7 or 14 days, the label becomes `LATE_MARK` or `NO_QUOTE`.
* This hits exactly the losing spreads, so verified evidence carries a systematic survivorship bias.

**Fix.** `option_quotes.usable_rth_exit_quote(rec, side)`:
* Long legs follow the unchanged rule.
* Short legs need ask > 0; a bid of 0 is allowed.

It is used only where a short leg is valued or checked at exit:
* candidate ledger, spread marks from the archive;
* provenance, short-leg exit;
* alpha-option link and economics.

What stays unchanged:
* every default, including `timing_policy`, trade fills and the reconciliation;
* entry (a short leg is sold at the bid, so bid > 0 is still required);
* long legs.

**Effect on existing evidence: none.** Provenance counts are identical before and after the fix:
* 5 days: 308 / 359;
* 20 days: 308;
* discovery: 272.

Marks that already exist are never rewritten.

### Documented only, not changed (not demonstrated)

* **Long leg with bid 0 at exit.** Its true value would be 0 (−100 %), but the quote is not usable, so such total losses would be missing from verified labels. No case exists in the archive: 0 of about 2,300 ledger quotes.
  * This is a design decision for a human; it is not changed here.
  * Monitoring: `real_*_mark_status_* = NO_QUOTE` together with `ZERO_BID` quotes.

## Research candidate sample (`candidate_ledger.flush`)

| Check | Finding |
|---|---|
| **Reproducibility** | Exact. On 2026-10-30 the 38 of 38 selected names reproduce from the ledger using `sha1(date:research:signal_id)`. The selection does not depend on outcomes. |
| **Budget** | At most 40 snapshots per day on every day. No day exceeds the budget. |
| **Directional signals first** | Yes: 2 directional names plus 38 from the research sample = 40. |
| **Candidates without an LLM direction** | Contract direction is `RESEARCH_DIRECTION = BULLISH`. Row direction stays `None`, so scorecards are unaffected. |
| **Budget exclusions** | Directional rows record them per row (`budget_random_exclusion`). Research rows record pool size and budget on the selected rows, so exclusions can be reconstructed. There is no per-row marker. |
| **Representativeness (2026-10-30: sample 38, pool 323)** | Market cap, dollar volume, relative volume, day return and short momentum show no significant difference (KS p = 0.54–0.92). Sector and filter reason are within random variation (37 of 38 economics_first versus 34 expected). |

**Remaining uncertainties:**
* Only one day of research-sample data exists so far (2026-10-30).
* On 2026-10-28 and 2026-10-29 there were only 1 and 3 snapshots, before the research sample existed. This is a historical gap.
* The research contract variant (`real_option_alt`) was merged after the scanner run of 2026-10-30. Its first run is 2026-11-02.
* Calls only, so no statement about bearish signals.
* **Verdict:** no technical defect, so the procedure is unchanged.

## Economic evidence

A complete economic proof is still missing. There are not yet any verified 20-day option labels after the archive start. The first is due on 2026-11-03, and statistical validation needs many more matured, independent periods.
