# Architecture decision records

| ADR | Decision |
|---|---|
| [0001](0001-goal-and-evaluation-protocol.md) | Dual goal; construction data from outside the benchmark; what "held-out" excludes; held-out distance table; Rec-S primary |
| [0002](0002-compute-and-budget.md) | Local vLLM serving; official judge reserved for final runs; cost ledger |
| [0003](0003-construction-corpus.md) | MELD, CMU-MOSEI, CH-SIMS, IEMOCAP or CREMA-D; 450 build / 450 val / 300 external test; leakage checks against all eight tasks |
| [0004](0004-audio-evidence-pipeline.md) | Whisper + eGeMAPS + Qwen3-Omni Captioner in two channels; no-speech and multi-speaker handling; official audio clues never read |
| [0005](0005-builder-permissions-and-objective.md) | Staged Builder permissions; minimal Stage 1 scaffold; taxonomy hidden from the Builder; Rec-S primary, Rea-S gate |
| [0006](0006-builder-loop-mechanics.md) | Controlled JSON patch loop; text-first packets; content-addressed cache; three replicates, first completed run proceeds |
| [0007](0007-proxy-judge.md) | Gemini 3 Flash proxy calibrated against GPT-4o; judge test-retest noise floor; audio-clue-source analysis |
| [0008](0008-gemini-direct-protocol.md) | Gemini Direct on the subset (P0), matched protocol (P1), full run only with leftover budget (P2) |
| [0009](0009-experiment-scope-and-judge-allocation.md) | The ladder L0–L6; Builder gain = L5 − L3; P0/P1/P2 rows with a cost model and cut line |
| [0010](0010-output-controller-and-multilabel-policy.md) | Controller parses the public prompt grammar; evidence-gated multi-label, cap 3 with sensitivity analysis |
| [0011](0011-target-variant-and-input-protocol.md) | Qwen3-VL-8B-Instruct; 2 fps native video; four expected calls, six hard cap including retries |
| [0012](0012-preregistration-and-information-firewall.md) | Pre-registered hypotheses with Rec-S thresholds; tags; no benchmark inspection before the freeze |
| [0013](0013-statistical-analysis-plan.md) | Paired bootstrap; detectable effect sizes; judge noise floor; Holm adjustment; breakdowns |
| [0014](0014-label-set-and-scoreboard.md) | LabelSet and Scoreboard deep modules; one empty-set rule; flat-mean gate objective; bootstrap as internal seam |
| [0015](0015-teacher-gpt-luna-qwen3-only.md) | Teacher is GPT 5.6 Luna (OpenAI); upper bound runs the matched protocol; Qwen3-only auxiliaries; Qwen2.5-VL-7B calibration optional |
