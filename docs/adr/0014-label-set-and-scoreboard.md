# ADR-0014: The Label set and the Scoreboard as deep modules

Date: 2026-09-07. Status: accepted.

## Context

An architecture review (`/improve-codebase-architecture`, 2026-09-07) found that the glossary's *Label set* was spread over four modules with two same-named `canonical_label` functions of different contracts, and that per-item scores reached the ladder table through fourteen modules with two implementations of the task macro that disagreed on empty groups (0.0 in `evaluation/metrics.py`, `None` in `runner/results.py`). The Builder's candidate evaluation built throwaway `PredictionRecord`s only to score them. Every number in the paper passes through this path.

## Decision

1. **`emoharness.data.label_set.LabelSet`** owns everything about a candidate list: canonical lookup, answer parsing, cardinality from the prompt quantifier, single-label and C/K Recognition, precision/recall/F1, overlap, rendering. `data/labels.py` and `data/answers.py` are deleted; normalization is private except for the cross-item key `normalize_label` and the gold-string splitter `split_labels`. `ParsedPrompt` remains the parse result carried in the evidence record; `LabelSet` is the behaviour on top of it.
2. **`emoharness.evaluation.scoreboard.Scoreboard`** is an immutable value with three constructors, `from_predictions`, `from_outputs`, and `from_step_scores`, and a `source` of `local` or `official`. It is the only interface through which the ladder table, the pre-registered hypothesis tests, the Builder gate, the proxy-judge calibration, and the fake judge read scores. `evaluation/metrics.py` is deleted; the aggregate functions in `evaluation/official.py` and `runner/results.py` are deleted.
3. **One empty-set rule:** `macro(group)` and `mean()` return `None` when nothing qualifies. "No data" is never rendered as a score of zero; the ladder shows `-`, a hypothesis test reports that it could not be computed, and a Builder candidate without a value fails the gate.
4. **The Builder gate objective is the flat mean of Recognition over D_val** (`Scoreboard.mean()`), the literal reading of ADR-0005. Per-kind macros are available for reporting only.
5. **Paired bootstrap stays an internal seam.** `Scoreboard.compare` and `Scoreboard.ci` call `evaluation/bootstrap.paired_bootstrap`, generalized to arbitrary strata (task, kind, or pooled); consumers never call it directly.
6. **Replaced tests are deleted, not layered.** The interface is the test surface: `test_label_set.py` and `test_scoreboard.py` replace the helper tests of the removed modules.

## Consequences

- The candidate-list rule lives in one place; eight call sites and three scoring sites share it.
- The Builder gate scores `HarnessOutput`s directly.
- `runner/results.py` is I/O only: it loads rows and hands back Scoreboards.
- Callers that need a number where none exists must decide explicitly; the type system (`float | None`) makes them.
