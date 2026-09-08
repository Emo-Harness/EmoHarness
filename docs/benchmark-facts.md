# MME-Emotion: verified facts

Source of truth: the official repo (`QwenAudio/MME-Emotion`, MIT), the HF dataset (`Karl28/MME-Emotion`, 6.32 GB), and arXiv 2508.09210. Verified on 2026-09-06 by reading the scripts and annotation files, not the abstract.

## Data

| Task | Items | Label set size | Notes |
|---|---:|---:|---|
| ER_Lab | 500 | 9 | IEMOCAP session 5 (`Ses05*`). Labels include Excited, Frustration, Other. |
| ER_Wild | 1800 | 7–11 | Six 300-item sub-sources with different label sets (DFEW-style `test_*.avi`, MELD `dia_*`, an 11-class MAFW-style set, an 8-class set with sarcastic/shy). |
| SA | 1600 | 3 | positive / negative / neutral. Sources include MELD `test_dia_*`, CH-SIMS-style `video_00xx`/`aqgy*`, and YouTube-id clips (MOSI/MOSEI-style). |
| Noise_ER | 500 | 6 | happy, neutral, angry, surprise, sad, worried (`samplenew3_*`, MER2024-noise style). |
| FG_ER | 500 | 26 | EMOTIC 26-category set. About 20% of items carry `;`-separated multiple labels. |
| ML_ER | 600 | 10 | Comma-separated multi-label answers, 2–3 labels per item. |
| FG_SA | 500 | 9 | very strong negative ... very strong positive. |
| IR | 500 | 8 | Intent labels: neutral, questioning, consoling, suggesting, encouraging, agreeing, wishing, acknowledging. |

Annotation schema per item: `video_id`, `label`, `label_set`. There is no question field. Videos are `.mp4`, `.avi`, or `.mkv` inside per-task zips. All 6,500 items are test items. There is no official train or validation split.

Label strings are inconsistent across sub-sources (`happy` vs `happiness`, `angry` vs `anger`). The official metric script normalizes only five synonyms; the judge handles the rest.

## Official prompts (paper Appendix C.2)

Single-label tasks:

```
As an emotion recognition expert, identify the most prominent emotion displayed in the video. Candidate labels: {label_set}. Please think step by step. Enclose your final answer within <answer></answer> tags.
```

Variants replace "emotion" with "one or several emotions" (FG-ER), "multiple emotions" (ML-ER), "sentiment" (SA), "fine-grained sentiment" (FG-SA), "intent" (IR). Audio-only models get "audio" in place of "video".

## Official scoring pipeline

1. `extract_step.py`: GPT-4.1 (`gpt-4.1-2025-04-14`, temperature 0) rewrites the free-text response into `<step>Step 1: ... Step K: the predicted emotion is X</step>`. A bare label becomes a single step.
2. `eval_cot_gpt4o.py`: GPT-4o (`gpt-4o-2024-11-20`, temperature 0) receives up to 10 frames sampled at 1 fps, the official audio clue, the gold label, and the steps. It rates every step 0/1. The last step is 1 only if the predicted emotion matches the label. Other steps are 0 only when they clearly contradict visual or audio cues.
3. `cal_metrics.py`: Rec-S = last-step rating. Rea-S = mean of the other ratings (0 when there is one step). CoT-S = 0.5 Rec-S + 0.5 Rea-S. Avg Step is also reported.

Task-specific judge differences, verified by diffing the eight scripts:

- **ML-ER and FG-ER** score the last step C/K: correctly predicted labels over gold label count, no penalty for extra predictions. The ML-ER prompt's own example gives 3/3 to eight predicted labels against three gold labels; the FG-ER example gives 1/1 to seven predicted labels against one gold label.
- **FG-SA** requires an exact fine-grained match, but its prompt contains contradictory examples: "predicted positive, label strong positive" is rated 0/1 in Example 2 and 1/1 in Example 3. Near-miss scoring on FG-SA therefore depends on the judge's reading.
- **SA, IR, Noise-ER, ER-Lab** differ from ER-Wild only in the noun (sentiment, intent) used in the prompt.
- Every script samples frames at 1 fps and keeps the first ten, so the judge sees only the first ten seconds of a clip.

Implications for harness design:

- A label-only answer caps CoT-S at half of Rec-S.
- Rea-S rewards cautious, evidence-grounded steps and punishes only clear contradictions.
- Rec-S can be computed locally without any judge once label normalization is handled.

## Leaderboard (overall, from the paper)

| Model | Rec-S | Rea-S | CoT-S | Avg steps |
|---|---:|---:|---:|---:|
| Gemini-2.5-Pro | 39.3 | 72.7 | 56.0 | 5.1 |
| Audio-Reasoner | 38.1 | 71.6 | 54.8 | 5.0 |
| GPT-4o | 27.8 | 79.8 | 53.8 | 4.4 |
| Qwen2.5-VL-72B | 31.3 | 75.7 | 53.5 | 4.8 |
| Qwen2.5-VL-7B | 28.4 | 64.8 | 46.6 | 4.8 |
| Qwen2.5-Omni | 17.4 | 59.3 | 38.4 | 3.7 |

No Qwen3-VL or Gemini 3.x result is published yet. Both baselines must be run in this project.

## Cost drivers

- Official judge per item: one GPT-4.1 text call (about 1k tokens) and one GPT-4o call carrying up to 10 frames at default detail. A 720p frame costs about 1.1k tokens, so an item costs about 6–12k input tokens depending on clip resolution, roughly $0.02–0.035. A full run is about $150–250 and must be measured on the first 100 items before the estimate is trusted. IEMOCAP clips in the zip are small (60–430 KB), so ER-Lab frames are cheaper than MELD or DFEW frames.
- Gemini Direct: video tokens per clip plus output; about $20 for the 1,000-item subset, about $100–150 for the full set.
- Target calls per item under the harness: expected 4, cap 6.

## Why "retain 80% of Gemini" was dropped

Qwen2.5-VL-7B already retains 46.6 / 56.0 = 83% of Gemini-2.5-Pro's CoT score and 28.4 / 39.3 = 72% of its Recognition score with no harness. The project's success criterion is therefore Gap Closed on Recognition (ADR-0012).
