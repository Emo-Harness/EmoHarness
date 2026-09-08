# ADR-0010: Output controller and multi-label policy

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

The official judge extracts steps from free text and scores the last step against the gold label. For ML-ER and FG-ER the last step is scored C/K: correct predicted labels over gold label count, with no penalty for extra predictions. Both official judge prompts contain examples that reward over-prediction: ML-ER gives 3/3 to eight predicted labels against three gold labels, and FG-ER gives 1/1 to seven predicted labels against one gold label. Predicting every candidate would therefore score 100% Recognition on those tasks.

The review raised a stricter point: the controller's cardinality handling is where knowledge of held-out task formats could enter the harness through its human authors. The policy must therefore be derived from the public prompt grammar, not from a task list, and its one free parameter must be disclosed and tested for sensitivity.

## Decision

1. **The output controller is human-written and frozen for the Builder.** It emits exactly five steps in the paper's extraction style, then the answer tag:
   `Step 1` setting and context; `Step 2` facial and bodily cues; `Step 3` vocal and verbal cues; `Step 4` cross-modal synthesis and the strongest alternative; `Step 5` "Based on the analysis, the predicted {noun} is X." followed by `<answer>X</answer>`.
2. **The controller parses the public prompt template, not a task list.** It reads the quantifier and the task noun from the pattern `identify (the most prominent | one or several | multiple) <noun> displayed in the video` and the candidate list from `Candidate labels: ...`. Cardinality follows the quantifier alone: "the most prominent" emits exactly one label; "one or several" emits one to `cap`; "multiple" emits two to `cap`. The noun is copied into Step 5. No branch anywhere in the harness is keyed on a task name or on specific label strings.
3. **Label strings are copied verbatim from the prompt's candidate list.** The local normalizer maps synonyms only for local scoring, never for the emitted answer.
4. **Multi-label policy is evidence-gated with a cap.** A label is emitted only if at least one evidence item supports it and the cross-modal check does not contradict it; labels are ranked by support and truncated at `cap = 3`. The cap is a design prior (people rarely display more than three simultaneous discrete emotions), chosen by authors who had read the benchmark's public task descriptions. The paper says so, and reports a sensitivity analysis over cap ∈ {2, 3, 4, no cap} on the subset in the appendix; the main table uses the pre-registered value.
5. **Every reasoning step must cite an item that exists in the evidence record.** A post-check removes uncited claims before emission so that Reasoning scores reflect grounded statements.
6. **Refusals are never emitted.** If the Target fails to commit, the controller falls back to the highest-scoring hypothesis from the verification stage.
7. **Multi-label reporting.** Next to the official C/K Recognition score the paper reports precision, recall, and F1 computed locally from emitted versus gold label sets, and describes the C/K weakness in the evaluation-hygiene discussion.

## Consequences

- Official Recognition on ML-ER and FG-ER may trail methods that over-predict. That is accepted and stated.
- The Builder cannot change step count, step order, the final-step template, or the cap (ADR-0005).
