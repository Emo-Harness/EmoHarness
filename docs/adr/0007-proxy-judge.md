# ADR-0007: Proxy judge, judge noise, and audio-clue-source effect

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

ADR-0005 makes Reasoning score a non-regression gate during construction. The official judge is GPT-4o with frames and the official audio clue. Construction items are external (ADR-0003), so no official audio clue exists for them, and running GPT-4o on every candidate every round is outside the budget. The review added two measurements: the official judge's own test-retest noise, and the effect of the judge reading a Qwen2-Audio clue while the harness reasoned from a different audio description.

## Decision

1. The proxy judge is **Gemini 3 Flash** through the API, prompted with the official judge prompt verbatim, given the same 1 fps frames capped at 10, the harness's own observable-cue audio text in the audio-clue slot, and the gold label.
2. The gate is evaluated on a fixed, pre-registered 200-item sample of D_val, identical for every candidate and every round.
3. **Calibration.** Before the first construction round, the proxy is compared with GPT-4o-2024-11-20 on the same 200 items with the same inputs. The report records per-step agreement and the mean Rea-S offset. The Gate 1 margin in ADR-0005 is set to two standard deviations of the per-item disagreement, with a floor of 2 points. If agreement is below 70%, the proxy is replaced by GPT-4o on a 100-item sample for the gate only.
4. **Official judge test-retest (P1, about $10).** The official judge is run twice on 300 items of the Qwen Direct full run. Per-step agreement and the Rec-S and Rea-S differences between the two passes are the judge noise floor; differences between methods smaller than that floor are reported but not interpreted.
5. **Audio-clue-source effect (analysis only, after the freeze).** A 300-item sample of Full EmoHarness outputs is re-judged with the harness's own audio text substituted for the official clue. The Rea-S difference quantifies how much Reasoning score is lost because the judge and the harness read different audio descriptions. This is an appendix analysis and changes nothing in the harness.
6. The proxy judge is never used for any number that appears in a results table.

## Consequences

- Proxy cost for a full construction run stays under $50 including replicates.
- Builder and proxy judge share a model family. This is acceptable because the proxy only gates and never drives the primary objective; the same-family audio row (ADR-0004) and the official judge on final runs catch any family bias.
