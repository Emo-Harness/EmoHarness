# Paper outline

Working title candidates:

- EmoHarness: Can a Stronger Model Build a Transferable Emotion-Understanding Harness for a Frozen Smaller Model?
- Building Harnesses, Not Weights: Builder-to-Target Transfer of Multimodal Emotion Reasoning

## Contributions (as they will be claimed)

1. A construction protocol in which a Builder model edits an executable, tool-augmented harness around a frozen Target under staged permissions, with construction data disjoint from the benchmark and held-out capabilities never exposed.
2. A ladder decomposition (modality, format, scaffold, Builder) that separates "adding an audio tool" from "the Builder built something", with the Builder gain measured on all 6,500 benchmark items.
3. Results on MME-Emotion for Qwen3-VL-8B: seen and held-out gains, gap closed toward Gemini 3.1 Pro, budget-matched comparison against test-time scaling, and a held-out distance analysis.
4. Evaluation-hygiene findings about the official protocol: multi-label C/K scoring without a precision penalty, contradictory FG-SA judge examples, the judge seeing only the first ten seconds, and the dependence of the Reasoning score on output format.

## Claims mapped to evidence

| Claim | Rows | Items | Metric | Where |
|---|---|---|---|---|
| The harness improves the Target on seen tasks (H1) | L0, L5 | full seen | Rec-S | Table 1 |
| The improvement transfers to held-out tasks (H2) | L0, L5 | full held-out | Rec-S per task and macro | Table 1, Table 2 |
| The Builder adds value beyond the human scaffold (H-B) | L3, L5 | full | Rec-S | Table 1 |
| The gain is not only test-time scaling (E3) | L2, Best-of-4, L3 | subset | Rec-S | Table 3 |
| Part of the Qwen-to-Gemini gap is closed (H3) | L0, L5, L6 | subset | Rec-S, Gap Closed | Table 1 |
| Which modules matter (E5) | L5 and five ablations | subset | Rec-S | Table 4 |
| The construction is reproducible | three Stage 2 replicates | D_val, D_ext-test | Rec-S mean ± sd | Figure 2 |
| Transfer depends on held-out distance | L5 − L0 per task | full held-out | Rec-S gain | Figure 3 |

## Tables and figures

- Table 1: the ladder L0–L6 on seen macro, held-out macro, and overall; Rec-S with CI, Rea-S, CoT-S, Avg Step, Target calls, cost per 1,000 items.
- Table 2: per-task Rec-S for L0, L3, L5, with precision, recall, and F1 on ML-ER and FG-ER.
- Table 3: equal-call comparison: L2, Best-of-4 over L2, L3, L5, sequential refinement if run.
- Table 4: ablations of L5 on the subset.
- Table 5: cross-Target: Qwen2.5-VL-7B Direct and + L5 (and InternVL3.5-8B if run), plus the calibration against the published numbers.
- Table 6: judge noise floor, proxy calibration, audio-clue-source effect, protocol gap.
- Figure 1: architecture (to be drawn: perception, the four harness stages, the Builder loop).
- Figure 2: round-by-round D_val and D_ext-test Rec-S for the three Stage 2 replicates and the Stage 3 run.
- Figure 3: held-out gain against the distance factors from ADR-0001.
- Figure 4: confusion-matrix shifts from L0 to L5 on ER-Wild and IR.
- Figure 5: qualitative examples with evidence records, including one failure.
- Appendix: cap sensitivity; per-language breakdown; parse-failure rates; construction transcript summary; post-freeze patches; full prompts and schemas.

## Related work to position against

- Automated prompt and pipeline optimization: OPRO; PromptBreeder; DSPy and MIPROv2; TextGrad; Trace; GEPA; ADAS; AFlow. The difference claimed: a multimodal, tool-augmented harness with code edits, evaluated on held-out capabilities rather than on the optimized distribution, with the Builder's gain isolated from tool and format gains.
- Test-time scaling and self-improvement: self-consistency; best-of-N; Self-Refine; Reflexion. Compared under equal Target calls.
- Weak-to-strong and capability transfer between models: weak-to-strong generalization (the reverse direction); distillation without weight updates; scaffolding and agent harness literature.
- Multimodal emotion understanding: Emotion-LLaMA and Emotion-LLaMAv2/MMEVerse; HumanOmni; R1-Omni; Audio-Reasoner; HumanSense; EmoBench-M; EmoTrans; MER challenge series; source datasets IEMOCAP, MELD, DFEW, MAFW, CMU-MOSEI, CH-SIMS, MC-EIU, BoLD/EMOTIC.
- LLM-as-judge evaluation and its failure modes, as context for the evaluation-hygiene section.

Citation keys for 2025–2026 papers are verified with the citation-check tooling before submission; none is cited from memory.

## Limitations to state

- One benchmark, one Builder, one primary Target; H4 is exploratory.
- Seen-task gains may include distribution-specific priors learned from source-dataset train splits; held-out tasks are the clean test.
- The harness authors had read the public task descriptions; the controller's cardinality cap is a disclosed prior.
- Subset rows have wider intervals; per-task subset numbers are descriptive.
- The Reasoning score rewards cautious wording; claims rest on Recognition.
- Gemini API non-determinism; judge model versions may be retired.

## Ethics and responsible use

Emotion labels are ambiguous and culturally dependent; outputs are not measurements of internal states. Data statement with every license and access route (ADR-0003). No media redistributed. Per-language results reported to expose disparities.

## Reproducibility package

Frozen harness at `harness-frozen`; pre-registration at `prereg-v1`; raw predictions and judge outputs for every row; subset id list; manifests and leakage report; construction transcripts; cost ledger; one script that regenerates every table and figure.

## Reviewer objections and where they are answered

| Objection | Answer |
|---|---|
| "The gain is just adding an audio model." | Ladder rung L1 isolates modality gain; H-B is L5 − L3 with both rows using the same audio. |
| "The Reasoning score is format engineering." | L2 is the format-only control; claims rest on Rec-S. |
| "Held-out tasks were known to the authors." | ADR-0001 states what is and is not excluded; controller uses the prompt grammar; cap sensitivity reported. |
| "The 80% retention target is trivial." | Replaced by Gap Closed with pre-registered tiers; retention reported next to the baseline's value. |
| "Why not GEPA / DSPy?" | Stage 2 is the prompt-only comparison; an established optimizer is a P2 row or a stated limitation. |
| "One construction run could be luck." | Three Stage 2 replicates; first completed run proceeds by pre-registration. |
| "Validation overfitting." | D_ext-test gap reported. |
| "The judge is noisy." | Test-retest noise floor; differences below it are not interpreted. |
| "You exploited the multi-label metric." | Evidence gate with cap; precision and F1 reported; weakness documented. |
| "Only one Target." | Stated as a limitation; H4 exploratory with one or two extra points. |
