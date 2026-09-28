# Does the harness emotion context help the bare model?

Each student's own harness writes one question-independent emotion context per video (its native MME-Emotion task, 7 basic emotions). The same bare student then answers the lettered multiple-choice question from the same cached frames, with no text (**none**), with the transcript + prosody text (**asr**), or with that text plus the harness emotion context (**emoctx**).

## IntentQA test

2134 questions on 567 videos; chance 20%.

| student | none | asr | emoctx | unparsed (n/a/e) |
|---|---|---|---|---|
| Qwen3-VL-8B | 70.99% | 72.63% | 73.10% | 0/0/1 |
| Qwen3-VL-4B | 67.48% | 68.42% | 70.29% | 0/0/0 |
| Qwen3.5-4B | 79.90% | 77.84% | 76.34% | 1/1/3 |
| Gemma-4-E4B | 57.31% | 60.59% | 61.62% | 3/3/10 |

| student | comparison | Δ (pp) | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| Qwen3-VL-8B | asr−none | +1.64 | [+0.09, +3.19] | 0.0414 | 0.2070 |
| Qwen3-VL-8B | emoctx−asr | +0.47 | [-1.15, +2.05] | 0.5782 | 0.6948 |
| Qwen3-VL-8B | emoctx−none | +2.11 | [+0.47, +3.76] | 0.0144 | 0.0864 |
| Qwen3-VL-4B | asr−none | +0.94 | [-0.67, +2.55] | 0.2666 | 0.6948 |
| Qwen3-VL-4B | emoctx−asr | +1.87 | [+0.47, +3.27] | 0.0106 | 0.0848 |
| Qwen3-VL-4B | emoctx−none | +2.81 | [+1.03, +4.58] | 0.0020 | 0.0180 |
| Qwen3.5-4B | asr−none | -2.06 | [-3.66, -0.47] | 0.0116 | 0.0848 |
| Qwen3.5-4B | emoctx−asr | -1.50 | [-3.05, +0.09] | 0.0658 | 0.2632 |
| Qwen3.5-4B | emoctx−none | -3.56 | [-5.27, -1.82] | <0.0002 | 0.0024 |
| Gemma-4-E4B | asr−none | +3.28 | [+1.41, +5.11] | 0.0004 | 0.0040 |
| Gemma-4-E4B | emoctx−asr | +1.03 | [-0.63, +2.66] | 0.2316 | 0.6948 |
| Gemma-4-E4B | emoctx−none | +4.31 | [+2.21, +6.29] | <0.0002 | 0.0024 |

Sensitivity, questions answered in all three arms (no Holm):

| student | n | none | asr | emoctx | emoctx−asr [95% CI] | emoctx−none [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 2133 | 70.98% | 72.62% | 73.14% | +0.52 [-1.11, +2.09] | +2.16 [+0.51, +3.81] |
| Qwen3-VL-4B | 2134 | 67.48% | 68.42% | 70.29% | +1.87 [+0.47, +3.27] | +2.81 [+1.03, +4.58] |
| Qwen3.5-4B | 2129 | 79.94% | 77.88% | 76.47% | -1.41 [-2.95, +0.15] | -3.48 [-5.17, -1.77] |
| Gemma-4-E4B | 2121 | 57.47% | 60.68% | 62.00% | +1.32 [-0.29, +2.99] | +4.53 [+2.54, +6.55] |

| student | contexts | harness fallbacks | predicted emotions |
|---|---|---|---|
| Qwen3-VL-8B | 567 | 2 | happy 265, neutral 256, angry 26, sad 9, surprise 8, fear 3 |
| Qwen3-VL-4B | 567 | 1 | happy 297, neutral 149, angry 53, sad 25, surprise 23, fear 20 |
| Qwen3.5-4B | 567 | 6 | neutral 261, happy 222, surprise 25, fear 25, sad 18, angry 15, disgust 1 |
| Gemma-4-E4B | 567 | 15 | happy 234, neutral 210, angry 47, surprise 39, sad 23, fear 13, disgust 1 |

## Social-IQ 2.0 val

795 questions on 122 videos; chance 25%.

| student | none | asr | emoctx | unparsed (n/a/e) |
|---|---|---|---|---|
| Qwen3-VL-8B | 61.89% | 70.19% | 66.42% | 0/0/0 |
| Qwen3-VL-4B | 59.12% | 68.43% | 65.28% | 0/3/0 |
| Qwen3.5-4B | 65.91% | 71.82% | 72.70% | 4/11/4 |
| Gemma-4-E4B | 57.74% | 64.91% | 61.13% | 4/9/23 |

| student | comparison | Δ (pp) | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| Qwen3-VL-8B | asr−none | +8.30 | [+4.95, +11.77] | <0.0002 | 0.0024 |
| Qwen3-VL-8B | emoctx−asr | -3.77 | [-6.68, -0.93] | 0.0094 | 0.0564 |
| Qwen3-VL-8B | emoctx−none | +4.53 | [+1.04, +8.12] | 0.0138 | 0.0690 |
| Qwen3-VL-4B | asr−none | +9.31 | [+5.43, +13.07] | <0.0002 | 0.0024 |
| Qwen3-VL-4B | emoctx−asr | -3.14 | [-5.78, -0.52] | 0.0222 | 0.0888 |
| Qwen3-VL-4B | emoctx−none | +6.16 | [+2.33, +9.84] | 0.0020 | 0.0140 |
| Qwen3.5-4B | asr−none | +5.91 | [+2.67, +9.32] | 0.0002 | 0.0024 |
| Qwen3.5-4B | emoctx−asr | +0.88 | [-1.61, +3.35] | 0.5276 | 0.5276 |
| Qwen3.5-4B | emoctx−none | +6.79 | [+3.73, +9.82] | 0.0002 | 0.0024 |
| Gemma-4-E4B | asr−none | +7.17 | [+2.67, +11.58] | 0.0010 | 0.0080 |
| Gemma-4-E4B | emoctx−asr | -3.77 | [-7.04, -0.49] | 0.0244 | 0.0888 |
| Gemma-4-E4B | emoctx−none | +3.40 | [-1.22, +7.74] | 0.1570 | 0.3140 |

Sensitivity, questions answered in all three arms (no Holm):

| student | n | none | asr | emoctx | emoctx−asr [95% CI] | emoctx−none [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 795 | 61.89% | 70.19% | 66.42% | -3.77 [-6.68, -0.93] | +4.53 [+1.04, +8.12] |
| Qwen3-VL-4B | 792 | 59.34% | 68.69% | 65.28% | -3.41 [-6.07, -0.77] | +5.93 [+2.07, +9.67] |
| Qwen3.5-4B | 777 | 66.28% | 72.97% | 73.23% | +0.26 [-2.20, +2.64] | +6.95 [+3.97, +9.88] |
| Gemma-4-E4B | 768 | 58.20% | 65.49% | 62.76% | -2.73 [-5.85, +0.28] | +4.56 [+0.24, +8.79] |

| student | contexts | harness fallbacks | predicted emotions |
|---|---|---|---|
| Qwen3-VL-8B | 122 | 6 | neutral 46, happy 34, angry 24, sad 11, surprise 4, fear 3 |
| Qwen3-VL-4B | 122 | 2 | happy 47, angry 27, sad 19, neutral 18, surprise 5, fear 5, disgust 1 |
| Qwen3.5-4B | 122 | 2 | neutral 43, happy 27, angry 16, sad 15, fear 13, surprise 6, disgust 2 |
| Gemma-4-E4B | 122 | 6 | neutral 48, happy 28, angry 21, sad 11, surprise 6, fear 5, disgust 3 |

