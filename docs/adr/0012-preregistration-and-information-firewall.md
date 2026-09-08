# ADR-0012: Pre-registration, hypotheses, and the information firewall

Date: 2026-09-07. Status: accepted.

## Context

The paper's claims are about transfer to held-out capabilities and about a Builder's contribution. Both are easy to inflate through post-hoc choices: which metric, which threshold, which construction run, which items. The original H3, "retain at least 80% of Gemini's performance", is already satisfied by the plain Qwen2.5-VL-7B baseline on the CoT score (46.6 / 56.0 = 83%) and would be met without any harness. The hypotheses are therefore re-stated as pre-registered tests with a primary metric, and the procedure that keeps benchmark information away from construction is written down as rules with tags and tests.

## Decision

### Pre-registration

Before the first Builder round, `docs/preregistration.md` is committed and tagged `prereg-v1`. It contains: the hypotheses below with metric and threshold; the ladder (ADR-0009); the SHA-256 of the pre-registered subset id list; the SHA-256 of the construction, validation, and external-test manifests and of the leakage report; the frozen L2 prompt, Stage 1 scaffold, emotion lexicon, and describer prompts; the analysis plan (ADR-0013). Any later change is an amendment with date and reason, never a silent edit.

### Primary metric

Recognition score (Rec-S), macro-averaged over tasks within a group: seen (ER-Lab, ER-Wild, SA) and held-out (Noise-ER, FG-ER, ML-ER, FG-SA, IR). CoT-S, Rea-S, Avg Step, and multi-label precision, recall, and F1 are secondary and descriptive.

### Hypotheses as tests

| Id | Statement | Rows | Items | Test |
|---|---|---|---|---|
| H1 seen improvement | Rec-S(L5) − Rec-S(L0) > 0 on seen macro | L0, L5 | full seen (3,900) | 95% paired bootstrap CI excludes 0 |
| H2 held-out transfer | Rec-S(L5) − Rec-S(L0) > 0 on held-out macro | L0, L5 | full held-out (2,600) | CI excludes 0; per-task point estimates reported, negative ones included |
| H-B Builder gain | Rec-S(L5) − Rec-S(L3) > 0 on seen macro and on held-out macro | L3, L5 | full | CI excludes 0 for each group separately |
| H3 gap closed | Gap Closed = (Rec-S(L5) − Rec-S(L0)) / (Rec-S(L6) − Rec-S(L0)) on overall macro | L0, L5, L6 | subset (1,000), paired | tier 1 ≥ 0.25, tier 2 ≥ 0.5; the tier reached is reported with its CI |
| H4 generality | exploratory, no threshold | Qwen2.5-VL-7B ± L5, InternVL if P2 | subset | descriptive |

Gemini Performance Retained is still reported, as a descriptive number with its baseline value next to it, so that readers see how much of it the baseline already had.

### Information firewall

1. **Tags.** `prereg-v1` before the first Builder round; `harness-frozen` after Stage 3 converges; `results-v1` after the final judge runs. The paper cites all three hashes.
2. **No benchmark inspection before the freeze.** Until `harness-frozen`, no person opens a benchmark item, a benchmark-derived cache, or a model output on a benchmark item. Benchmark preprocessing runs as a script whose outputs are not inspected.
3. **Benchmark Target runs happen only after the freeze**, with one exception: the M1 calibration run of Qwen2.5-VL-7B Direct, whose per-item outputs are not inspected and whose scores are viewed only as per-task aggregates for calibration. That run uses a different model and no harness.
4. **Construction artifacts contain no benchmark ids.** A test greps `results/construction/` and the harness directory for every benchmark video id and fails on any hit.
5. **Post-freeze changes.** After `harness-frozen`, harness files change only to fix crashes in I/O or serving. Each such patch is recorded with its diff under `results/post_freeze_patches/` and listed in the paper. No prompt, threshold, schema, or verification function changes after the freeze.
6. **Negative results.** Every pre-registered test is reported whatever its outcome, including per-task negative transfer.

## Consequences

- A failed H3 tier or a null H-B is a reportable result, not a reason to iterate.
- The construction replicate that reaches the benchmark is the first completed one (ADR-0006), fixed in the pre-registration.
