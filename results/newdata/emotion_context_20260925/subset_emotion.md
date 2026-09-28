# Emotion questions only: does the harness emotion context help there?

Subset rule fixed from question text before any subset result was computed (see `scripts/emoctx_subset.py`). Δ in pp with 95% video-clustered CIs; exploratory, no multiplicity correction.

## social_iq_val

795 questions: core 262, broad 353.

- e.g. (core): Describe the overall tone and mood of the video | Describe the tone of the woman's speech that starts at 0:20 and what this suggests about her mood | How can one characterize the woman at the end's emotions?
- e.g. (broad_only): Why does the woman with the big earrings smile? | How does the man in the beige jacket react when the woman in the red jacket says that she would love to meet his family? | Why is the woman smiling?

### core subset (n=262)

| student | acc none / asr / emoctx (subset) | emoctx−asr subset | emoctx−asr rest | interaction | emoctx−none subset |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 63.4 / 71.0 / 71.8 | +0.76 [-3.3, +4.7] | -6.00 [-9.5, -2.7] | +6.77 [+1.8, +11.7] | +8.40 [+2.5, +14.3] |
| Qwen3-VL-4B | 61.1 / 67.6 / 66.0 | -1.53 [-6.9, +3.9] | -3.94 [-7.0, -0.8] | +2.41 [-4.2, +8.8] | +4.96 [-2.1, +11.9] |
| Qwen3.5-4B | 69.1 / 74.0 / 73.7 | -0.38 [-4.8, +3.8] | +1.50 [-1.7, +4.6] | -1.88 [-7.5, +3.7] | +4.58 [-1.1, +10.2] |
| Gemma-4-E4B | 58.4 / 65.6 / 61.1 | -4.58 [-9.7, +0.4] | -3.38 [-7.2, +0.7] | -1.20 [-7.4, +5.0] | +2.67 [-4.1, +9.6] |

### broad subset (n=353)

| student | acc none / asr / emoctx (subset) | emoctx−asr subset | emoctx−asr rest | interaction | emoctx−none subset |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 61.2 / 70.3 / 69.1 | -1.13 [-5.2, +2.9] | -5.88 [-9.6, -2.4] | +4.75 [-0.2, +10.0] | +7.93 [+3.3, +12.8] |
| Qwen3-VL-4B | 60.9 / 66.9 / 66.9 | +0.00 [-4.7, +4.8] | -5.66 [-8.9, -2.4] | +5.66 [-0.4, +11.9] | +5.95 [+0.5, +11.3] |
| Qwen3.5-4B | 66.6 / 72.5 / 72.8 | +0.28 [-3.3, +3.8] | +1.36 [-1.9, +4.5] | -1.07 [-5.6, +3.5] | +6.23 [+1.1, +11.0] |
| Gemma-4-E4B | 57.8 / 66.3 / 60.6 | -5.67 [-10.0, -1.5] | -2.26 [-6.5, +2.0] | -3.40 [-8.7, +1.9] | +2.83 [-3.2, +8.9] |

## intentqa_test

2134 questions: core 17, broad 90.

- e.g. (core): how did the child use his hands to show his excitement at the start | how did the man feel after crawling through the small play hole | how does the lady show her happiness at the end of the video
- e.g. (broad_only): how did the two women react after the woman in stripes released their hand at the start of the video | what does the woman in the train do after smiling near the end | why did the woman react after the girl in white blew out the candles

core: too few questions (17); skipped.

### broad subset (n=90)

| student | acc none / asr / emoctx (subset) | emoctx−asr subset | emoctx−asr rest | interaction | emoctx−none subset |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 72.2 / 71.1 / 70.0 | -1.11 [-9.1, +6.8] | +0.54 [-1.1, +2.2] | -1.65 [-9.8, +6.4] | -2.22 [-11.5, +7.3] |
| Qwen3-VL-4B | 73.3 / 64.4 / 65.6 | +1.11 [-7.5, +9.5] | +1.91 [+0.5, +3.3] | -0.80 [-9.3, +7.7] | -7.78 [-18.6, +3.0] |
| Qwen3.5-4B | 74.4 / 65.6 / 70.0 | +4.44 [-2.9, +12.0] | -1.76 [-3.3, -0.1] | +6.21 [-1.3, +14.0] | -4.44 [-12.0, +3.1] |
| Gemma-4-E4B | 67.8 / 62.2 / 61.1 | -1.11 [-10.4, +7.7] | +1.13 [-0.5, +2.8] | -2.24 [-11.5, +6.6] | -6.67 [-15.3, +1.2] |

