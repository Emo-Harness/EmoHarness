# New-dataset transfer of the 2026-09-19 per-student harnesses

L5 row, each student on its own published bundle, question carried in the frozen MME-Emotion prompt grammar. Accuracy with a 95% CI resampled over videos.

## IntentQA test

2134 items, 567 videos, 0 excluded; chance 20%.

| student | accuracy | 95% CI | no answer | mean calls |
|---|---|---|---|---|
| Qwen3-VL-8B | 63.45% | [61.21, 65.69] | 0 | 4.01 |
| Qwen3-VL-4B | 57.50% | [55.08, 59.82] | 0 | 4.00 |
| Qwen3.5-4B | 59.93% | [57.71, 62.13] | 0 | 4.06 |
| Gemma-4-E4B | 52.30% | [50.02, 54.52] | 0 | 4.04 |

## Social-IQ 2.0 val

795 items, 122 videos, 148 excluded; chance 25%.

| student | accuracy | 95% CI | no answer | mean calls |
|---|---|---|---|---|
| Qwen3-VL-8B | 61.13% | [57.21, 65.06] | 0 | 4.06 |
| Qwen3-VL-4B | 53.71% | [50.22, 57.35] | 0 | 4.09 |
| Qwen3.5-4B | 61.26% | [57.48, 65.16] | 0 | 4.06 |
| Gemma-4-E4B | 59.37% | [55.31, 63.50] | 0 | 4.08 |

