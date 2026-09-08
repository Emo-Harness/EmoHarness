# ADR-0005: Builder permissions, objective, and the Stage 1 scaffold

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

The Builder edits the harness between rounds. Three things had to be fixed: what it may touch, what score it optimizes, and what the human-written starting point contains. The review found that the starting point matters for the paper's central claim: if the human scaffold already contains the key ideas, the Builder's measured contribution shrinks and the AI4AI claim weakens. The review also found that the README's failure taxonomy names held-out task types (multi-label, intent, fine-grained), so it must not be shown to the Builder.

## Decision

### Permissions, unlocked in stages

| Stage | Builder may edit | Humans own |
|---|---|---|
| 1 | Nothing. Humans write the minimal scaffold below. | Everything |
| 2 | `harness/prompts/*.md`, `harness/*.yaml` (sampling, routing, thresholds) | All Python, schemas, output controller |
| 3 | Pure functions under `harness/verification/` that pass the unit tests and a static allowlist (no I/O, no network, no model calls, no per-task branches keyed on label strings) | Output controller, schemas, tools |

A stage is unlocked only after the previous stage's loop has run to convergence and its result is tagged.

### The Stage 1 scaffold is deliberately minimal

It contains exactly:

- the evidence schema (visual, audio observable cues, audio-model opinion, text);
- four prompts: extract visual evidence; generate coarse and fine hypotheses from the evidence record and the candidate list; verify from an alternative evidence ordering (audio first); synthesize the five-step rationale;
- one deterministic check: a cross-modal agreement flag that records whether the per-modality hypotheses agree, with no conflict-resolution rule;
- the output controller (ADR-0010).

It contains no label priors, no per-task branches, no salient-moment retrieval, no calibration rules, and no few-shot examples. Anything beyond this list is the Builder's to discover, so that the Stage 3 minus Stage 1 difference is a fair measure of the Builder's contribution.

### What the Builder is told

- The Builder's system prompt describes the seen task wording only ("identify the most prominent emotion / sentiment") and the evidence schema.
- The README failure taxonomy is **not** given to the Builder. The Builder clusters failures from D_build only; the taxonomy is used after the freeze for analysis.
- The Builder never sees D_val, D_ext-test, benchmark items, or held-out prompt wordings.

### Objective

- Primary: Recognition score on D_val, computed locally with the label normalizer.
- Gate 1: Reasoning score estimated with the proxy judge on a fixed 200-item sample of D_val must not fall more than the calibrated noise margin below the incumbent (ADR-0007).
- Gate 2: regression-set failure rate at or below 5%.
- Gate 3: mean Target calls per item and p95 latency inside the round's budget.
- The output controller fixes the rationale structure at five steps. The Builder cannot change the step count, the step order, or the final-step template.

## Consequences

- The Builder cannot improve its score by writing vaguer or longer reasoning, which keeps the Reasoning gate honest.
- "Builder Prompt-Only" in the results tables means the Stage 2 harness; "Full EmoHarness" means the Stage 3 harness; "Human-Designed Harness" means the Stage 1 scaffold.
- If the Builder adds nothing measurable over Stage 1, the paper reports that; the ladder in ADR-0009 is designed to make that outcome visible rather than hidden.
