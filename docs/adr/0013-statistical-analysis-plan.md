# ADR-0013: Statistical analysis plan

Date: 2026-09-07. Status: accepted.

## Decision

1. **Unit and pairing.** The unit is the benchmark item. Every comparison is paired on identical item sets with identical cached inputs.
2. **Confidence intervals.** 95% paired bootstrap over items with 10,000 resamples, resampling within task so that task proportions are preserved. Macro averages are recomputed inside each resample.
3. **Pre-registered subset.** 1,000 items, 125 per task, sampled proportionally by label within task with seed 20260907. The id list and its hash are committed at pre-registration. Because sampling is proportional, subset scores are unbiased estimates of full-set scores.
4. **Detectable effects** for a paired accuracy difference with about 30% discordant items: full seen (3,900 items) about 2.5 points; full held-out (2,600) about 3 points; subset overall (1,000) about 5 points; subset per task (125) about 14 points. Subset per-task numbers are therefore descriptive only, and every claim names the item set it rests on.
5. **Judge noise floor.** The official judge test-retest (ADR-0007) gives a per-metric noise floor; differences below it are reported without interpretation.
6. **Multiple comparisons.** The four pre-registered tests (H1, H2, H-B, H3) are reported with unadjusted CIs and with Holm-adjusted bootstrap p-values. Ablations and ladder rungs are descriptive with CIs.
7. **Multi-label tasks.** Official C/K Recognition plus precision, recall, and F1 from emitted versus gold label sets after normalization; the cap sensitivity analysis (ADR-0010) on the subset.
8. **Breakdowns.** Per task; per held-out distance factor (ADR-0001 table); per language from Whisper detection; per clip-length bucket; per parse-failure status.
9. **Construction variability.** Three Stage 2 replicates: D_val and D_ext-test Rec-S mean and standard deviation; round-by-round curves for all three; the validation-overfitting estimate is D_val minus D_ext-test for the frozen harness.
10. **Cost and latency.** Mean and p95 Target calls, tokens, wall-clock per item, and dollar cost per 1,000 items for every row, so that budget-matched comparisons can be checked by the reader.

## Consequences

- `src/evaluation/bootstrap.py` implements the paired bootstrap once and every table is generated from raw predictions by one script.
- No number in the paper is computed by hand.
