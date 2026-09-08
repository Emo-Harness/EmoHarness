# ADR-0001: Project goal and evaluation protocol

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

MME-Emotion ships 6,500 items, all of them test items, with no train or validation split. The original README proposed carving 10% build and 10% validation out of the three seen task families. That would leave seen-task scores computed on an 80% subset, which cannot be placed on the official leaderboard.

The project has two goals at once: a defensible research claim about harness transfer from seen to held-out capabilities, and a leaderboard-comparable number for Qwen3-VL-8B with the harness. The paper's central claim is that a Builder model adds value on top of a human-written scaffold, and that this value transfers to capabilities the Builder never saw.

## Decision

1. Harness construction and validation data come from **outside MME-Emotion**. No MME-Emotion item, label, or video is read by the Builder or used for any selection, threshold, or early-stopping decision.
2. Construction data covers only the **seen capability kinds**: lab emotion recognition, in-the-wild emotion recognition, and 3-class sentiment. It contains no fine-grained, multi-label, noise-injected, fine-grained-sentiment, or intent data.
3. Final evaluation runs on **all 6,500 items** for the rows that carry the main claims, with the official prompt and the official judge, so those scores are leaderboard-comparable. Other rows run on a pre-registered 1,000-item subset (ADR-0009).
4. The seen/held-out distinction is defined at the capability level: "seen" means the harness saw that kind of task during construction, never that it saw those items.
5. Construction data is deduplicated against **all eight** benchmark task families by media hash, perceptual hash, and transcript similarity before use (ADR-0003).
6. The primary metric for every hypothesis is the official **Recognition score**. CoT and Reasoning scores are reported for leaderboard comparability but never carry a claim, because Reasoning rewards cautious wording and can be raised by output formatting alone (ADR-0012).

## What "held-out" excludes, and what it does not

Excluded from every human and Builder decision before the `harness-frozen` tag:

- benchmark items: video, audio, and every derived cache computed from them;
- item labels and any per-item or per-task statistic computed from the benchmark;
- any model output on a benchmark item.

Not excluded, because no method and no baseline could avoid them:

- the benchmark paper's public task descriptions;
- the public prompt template, including its quantifier words ("the most prominent", "one or several", "multiple") and the task noun ("emotion", "sentiment", "fine-grained sentiment", "intent");
- the candidate label strings that arrive inside each prompt at inference time.

The authors read the benchmark paper and the official scripts before designing the harness. Design choices that could encode task-format knowledge are confined to the output controller and are listed with sensitivity analyses in ADR-0010. The paper states this explicitly rather than claiming the designers were blind to the task list.

## Held-out distance

Held-out tasks differ from the construction data along different factors. The paper reports this table so that readers can see which held-out results are near-transfer and which are far-transfer. Source-dataset identifications marked "likely" are confirmed during M1 from file naming and label sets.

| Held-out task | Media domain | Label space | Output cardinality | Task noun | Language | Acoustic condition |
|---|---|---|---|---|---|---|
| Noise-ER | new (likely MER-style Chinese clips) | new (includes "worried") | same | same | Chinese, seen via CH-SIMS | new (injected noise) |
| FG-ER | new (likely BoLD-style body-language clips) | new (26 EMOTIC categories) | new ("one or several") | same | mixed | same |
| ML-ER | new (likely MAFW-style clips) | new (10 classes) | new ("multiple") | same | mixed | same |
| FG-SA | seen (MOSI/MOSEI-style monologues) | new (9-level intensity) | same | new ("fine-grained sentiment") | English | same |
| IR | partly seen (dialogue clips, likely MC-EIU) | new (8 intents) | same | new ("intent") | mixed | same |

FG-SA is the nearest held-out task and FG-ER, ML-ER, and IR the farthest. Transfer results are discussed against this table, not only as one macro average.

## Consequences

- An external construction corpus must be assembled, licensed, and deduplicated against the whole benchmark (ADR-0003).
- The information firewall that keeps benchmark data away from construction is a procedure with tags and tests, not a promise (ADR-0012).
- Every claim in the paper names the rows, the item set, and the metric it rests on (ADR-0013).
