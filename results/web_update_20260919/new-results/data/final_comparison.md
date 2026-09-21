# Student-specific optimization + full evaluation

Generated 2026-09-19T17:34:46Z. Plan: coordination/full_run_plan_20260918.md. Resamples: 10000. Mode: **final**.

> Publication gate passed.

## Item set

Scored on the 5507 rows of `results/v3/student_specific_full_20260918/evaluation_items.jsonl` (sha256 70be1bacef4a06b3...). `data/manifests/mme_emotion.jsonl` has 5577 rows; 70 were removed for sharing a media hash with a build, val, or reserved ext_test item.

The development strata are **old_986** and **new_4521**. 14 of the 1,000 pre-registered subset ids fall inside the excluded set, so the intersection with the clean set is 986 and the remainder is 4521 -- matching the amended plan and `execution_manifest.json`. Labels carry the counts so that no reader takes them for a round 1,000 and 4,577.

| view | n |
|---|---|
| full | 5507 |
| seen | 2963 |
| held_out | 2544 |
| old_986 | 986 |
| new_4521 | 4521 |

Reserved-media re-check (fails closed): `{'build': 0, 'val': 0, 'ext_test': 0}`, ok=True. Item-set hash check against execution_manifest.json: ok=True.

## Coverage

16 of 16 arms present. Identical item set across every arm: True.

Arms covering the exact frozen set: 16 / 16. Ready for a final comparison: True.

## qwen3vl_8b

| slot | n | full macro | seen macro | held_out macro | old_986 macro | new_4521 macro | logical | actual | parse fail | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 5507 | 0.2995 | 0.2869 | 0.3071 | 0.2939 | 0.3016 | 5507 | 4491 | 6 | 11679 |
| L1 | 5507 | 0.3915 | 0.4150 | 0.3774 | 0.4038 | 0.3902 | 5507 | 4492 | 1 | 12519 |
| L3_scaffold | 5507 | 0.4047 | 0.4141 | 0.3991 | 0.4066 | 0.4079 | 22044 | 17978 | 21 | 9554 |
| L5_own | 5507 | 0.4068 | 0.4167 | 0.4008 | 0.4113 | 0.4088 | 22046 | 6151 | 26 | 3152 |

### Per-task Recognition

| slot | ER_Lab | ER_Wild | FG_ER | FG_SA | IR | ML_ER | Noise_ER | SA |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.085 | 0.362 | 0.236 | 0.132 | 0.215 | 0.388 | 0.564 | 0.414 |
| L1 | 0.232 | 0.380 | 0.192 | 0.308 | 0.321 | 0.417 | 0.650 | 0.633 |
| L3_scaffold | 0.288 | 0.355 | 0.262 | 0.267 | 0.412 | 0.489 | 0.566 | 0.599 |
| L5_own | 0.294 | 0.350 | 0.262 | 0.278 | 0.406 | 0.487 | 0.572 | 0.606 |

### Paired differences (task-macro primary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.0920 [+0.0778, +0.1065] p=0.000 | +0.1281 [+0.1036, +0.1532] p=0.000 | +0.0703 [+0.0527, +0.0884] p=0.000 | +0.1099 [+0.0797, +0.1400] p=0.000 | +0.0886 [+0.0708, +0.1074] p=0.000 |
| L3 - L1 | +0.0132 [-0.0026, +0.0289] p=0.104 | -0.0010 [-0.0282, +0.0265] p=0.933 | +0.0217 [+0.0024, +0.0407] p=0.027 | +0.0028 [-0.0254, +0.0319] p=0.854 | +0.0177 [-0.0050, +0.0401] p=0.131 |
| L5 - L3 | +0.0021 [-0.0018, +0.0059] p=0.292 | +0.0026 [-0.0047, +0.0104] p=0.488 | +0.0018 [-0.0023, +0.0059] p=0.406 | +0.0047 [-0.0018, +0.0113] p=0.162 | +0.0009 [-0.0067, +0.0085] p=0.800 |
| L5 - L0 | +0.1072 [+0.0907, +0.1238] p=0.000 | +0.1297 [+0.1009, +0.1582] p=0.000 | +0.0937 [+0.0734, +0.1144] p=0.000 | +0.1174 [+0.0837, +0.1500] p=0.000 | +0.1072 [+0.0848, +0.1296] p=0.000 |

### Same differences, sample-weighted (secondary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.0922 [+0.0795, +0.1051] p=0.000 | +0.1137 [+0.0955, +0.1320] p=0.000 | +0.0671 [+0.0499, +0.0847] p=0.000 | +0.1092 [+0.0791, +0.1389] p=0.000 | +0.0885 [+0.0743, +0.1024] p=0.000 |
| L3 - L1 | -0.0019 [-0.0148, +0.0112] p=0.771 | -0.0243 [-0.0432, -0.0054] p=0.009 | +0.0242 [+0.0055, +0.0427] p=0.013 | +0.0033 [-0.0250, +0.0325] p=0.837 | -0.0030 [-0.0176, +0.0117] p=0.685 |
| L5 - L3 | +0.0012 [-0.0024, +0.0048] p=0.501 | +0.0010 [-0.0047, +0.0067] p=0.762 | +0.0015 [-0.0025, +0.0055] p=0.472 | +0.0047 [-0.0018, +0.0114] p=0.165 | +0.0005 [-0.0036, +0.0046] p=0.822 |
| L5 - L0 | +0.0916 [+0.0766, +0.1066] p=0.000 | +0.0904 [+0.0685, +0.1120] p=0.000 | +0.0929 [+0.0729, +0.1131] p=0.000 | +0.1172 [+0.0836, +0.1499] p=0.000 | +0.0860 [+0.0691, +0.1025] p=0.000 |

### Media-clustered diagnostic, full view

Whole media clusters drawn instead of items, with joint weights across tasks and the macro over a task set fixed before resampling. Not the pre-registered interval: it measures what the within-clip dependence actually does to these intervals, in whichever direction it goes.

| comparison | clustered estimate [95% CI] | clusters |
|---|---|---|
| L1 - L0 | +0.0920 [+0.0776, +0.1067] | 5083 |
| L3 - L1 | +0.0132 [-0.0028, +0.0288] | 5083 |
| L5 - L3 | +0.0021 [-0.0016, +0.0060] | 5083 |
| L5 - L0 | +0.1072 [+0.0901, +0.1241] | 5083 |

## qwen3vl_4b

| slot | n | full macro | seen macro | held_out macro | old_986 macro | new_4521 macro | logical | actual | parse fail | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 5507 | 0.2939 | 0.2861 | 0.2985 | 0.2992 | 0.2892 | 5507 | 4492 | 6 | 7033 |
| L1 | 5507 | 0.3839 | 0.3979 | 0.3755 | 0.3817 | 0.3793 | 5507 | 4491 | 3 | 7294 |
| L3_scaffold | 5507 | 0.3760 | 0.4126 | 0.3541 | 0.3867 | 0.3690 | 22075 | 17968 | 67 | 6114 |
| L5_own | 5507 | 0.3856 | 0.4173 | 0.3665 | 0.3928 | 0.3811 | 22031 | 16374 | 3 | 5215 |

### Per-task Recognition

| slot | ER_Lab | ER_Wild | FG_ER | FG_SA | IR | ML_ER | Noise_ER | SA |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.085 | 0.347 | 0.219 | 0.111 | 0.276 | 0.369 | 0.518 | 0.426 |
| L1 | 0.186 | 0.370 | 0.208 | 0.351 | 0.310 | 0.394 | 0.614 | 0.638 |
| L3_scaffold | 0.226 | 0.356 | 0.177 | 0.208 | 0.357 | 0.454 | 0.574 | 0.656 |
| L5_own | 0.237 | 0.357 | 0.184 | 0.219 | 0.379 | 0.464 | 0.586 | 0.657 |

### Paired differences (task-macro primary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.0900 [+0.0762, +0.1038] p=0.000 | +0.1118 [+0.0906, +0.1340] p=0.000 | +0.0770 [+0.0588, +0.0945] p=0.000 | +0.0825 [+0.0533, +0.1113] p=0.000 | +0.0901 [+0.0736, +0.1071] p=0.000 |
| L3 - L1 | -0.0079 [-0.0221, +0.0064] p=0.278 | +0.0146 [-0.0084, +0.0373] p=0.214 | -0.0214 [-0.0396, -0.0031] p=0.022 | +0.0050 [-0.0234, +0.0329] p=0.724 | -0.0103 [-0.0278, +0.0082] p=0.267 |
| L5 - L3 | +0.0095 [+0.0012, +0.0182] p=0.024 | +0.0047 [-0.0096, +0.0191] p=0.523 | +0.0124 [+0.0018, +0.0231] p=0.022 | +0.0061 [-0.0104, +0.0224] p=0.478 | +0.0121 [+0.0009, +0.0240] p=0.031 |
| L5 - L0 | +0.0917 [+0.0763, +0.1073] p=0.000 | +0.1312 [+0.1049, +0.1581] p=0.000 | +0.0680 [+0.0486, +0.0871] p=0.000 | +0.0937 [+0.0619, +0.1253] p=0.000 | +0.0919 [+0.0727, +0.1113] p=0.000 |

### Same differences, sample-weighted (secondary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.0926 [+0.0802, +0.1053] p=0.000 | +0.1097 [+0.0918, +0.1276] p=0.000 | +0.0727 [+0.0550, +0.0898] p=0.000 | +0.0813 [+0.0522, +0.1100] p=0.000 | +0.0951 [+0.0812, +0.1092] p=0.000 |
| L3 - L1 | -0.0060 [-0.0181, +0.0065] p=0.325 | +0.0034 [-0.0135, +0.0199] p=0.717 | -0.0169 [-0.0345, +0.0010] p=0.065 | +0.0063 [-0.0221, +0.0341] p=0.663 | -0.0086 [-0.0224, +0.0049] p=0.210 |
| L5 - L3 | +0.0068 [+0.0000, +0.0138] p=0.050 | +0.0020 [-0.0071, +0.0111] p=0.688 | +0.0124 [+0.0020, +0.0229] p=0.021 | +0.0063 [-0.0103, +0.0223] p=0.469 | +0.0069 [-0.0006, +0.0147] p=0.073 |
| L5 - L0 | +0.0934 [+0.0794, +0.1075] p=0.000 | +0.1151 [+0.0938, +0.1360] p=0.000 | +0.0682 [+0.0491, +0.0871] p=0.000 | +0.0938 [+0.0618, +0.1254] p=0.000 | +0.0934 [+0.0771, +0.1093] p=0.000 |

### Media-clustered diagnostic, full view

Whole media clusters drawn instead of items, with joint weights across tasks and the macro over a task set fixed before resampling. Not the pre-registered interval: it measures what the within-clip dependence actually does to these intervals, in whichever direction it goes.

| comparison | clustered estimate [95% CI] | clusters |
|---|---|---|
| L1 - L0 | +0.0900 [+0.0764, +0.1040] | 5083 |
| L3 - L1 | -0.0079 [-0.0219, +0.0064] | 5083 |
| L5 - L3 | +0.0095 [+0.0008, +0.0181] | 5083 |
| L5 - L0 | +0.0917 [+0.0760, +0.1070] | 5083 |

## qwen3_5_4b

| slot | n | full macro | seen macro | held_out macro | old_986 macro | new_4521 macro | logical | actual | parse fail | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 5507 | 0.2962 | 0.3017 | 0.2928 | 0.2926 | 0.2980 | 5507 | 4491 | 98 | 4053 |
| L1 | 5507 | 0.3499 | 0.3821 | 0.3307 | 0.3430 | 0.3495 | 5507 | 4491 | 56 | 4526 |
| L3_scaffold | 5507 | 0.3723 | 0.4132 | 0.3477 | 0.3824 | 0.3684 | 22092 | 18014 | 76 | 7201 |
| L5_own | 5507 | 0.3723 | 0.4132 | 0.3477 | 0.3824 | 0.3684 | 22092 | 0 | 76 | 131 |

### Per-task Recognition

| slot | ER_Lab | ER_Wild | FG_ER | FG_SA | IR | ML_ER | Noise_ER | SA |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.141 | 0.381 | 0.207 | 0.061 | 0.262 | 0.467 | 0.468 | 0.383 |
| L1 | 0.203 | 0.382 | 0.208 | 0.217 | 0.325 | 0.392 | 0.512 | 0.561 |
| L3_scaffold | 0.299 | 0.334 | 0.238 | 0.249 | 0.339 | 0.457 | 0.456 | 0.606 |
| L5_own | 0.299 | 0.334 | 0.238 | 0.249 | 0.339 | 0.457 | 0.456 | 0.606 |

### Paired differences (task-macro primary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.0538 [+0.0405, +0.0678] p=0.000 | +0.0804 [+0.0580, +0.1021] p=0.000 | +0.0378 [+0.0203, +0.0554] p=0.000 | +0.0504 [+0.0218, +0.0784] p=0.001 | +0.0515 [+0.0336, +0.0695] p=0.000 |
| L3 - L1 | +0.0223 [+0.0074, +0.0371] p=0.003 | +0.0311 [+0.0067, +0.0555] p=0.013 | +0.0170 [-0.0017, +0.0359] p=0.075 | +0.0394 [+0.0095, +0.0690] p=0.010 | +0.0190 [-0.0014, +0.0397] p=0.067 |
| L5 - L3 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| L5 - L0 | +0.0761 [+0.0607, +0.0916] p=0.000 | +0.1115 [+0.0854, +0.1373] p=0.000 | +0.0549 [+0.0356, +0.0748] p=0.000 | +0.0898 [+0.0577, +0.1212] p=0.000 | +0.0705 [+0.0494, +0.0914] p=0.000 |

### Same differences, sample-weighted (secondary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.0589 [+0.0460, +0.0720] p=0.000 | +0.0820 [+0.0628, +0.1006] p=0.000 | +0.0319 [+0.0145, +0.0494] p=0.000 | +0.0497 [+0.0211, +0.0777] p=0.001 | +0.0609 [+0.0466, +0.0752] p=0.000 |
| L3 - L1 | +0.0093 [-0.0045, +0.0230] p=0.188 | +0.0013 [-0.0182, +0.0213] p=0.899 | +0.0185 [+0.0001, +0.0371] p=0.048 | +0.0388 [+0.0088, +0.0686] p=0.011 | +0.0028 [-0.0123, +0.0177] p=0.718 |
| L5 - L3 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| L5 - L0 | +0.0682 [+0.0533, +0.0829] p=0.000 | +0.0834 [+0.0611, +0.1050] p=0.000 | +0.0504 [+0.0313, +0.0700] p=0.000 | +0.0886 [+0.0565, +0.1199] p=0.000 | +0.0637 [+0.0474, +0.0801] p=0.000 |

### Media-clustered diagnostic, full view

Whole media clusters drawn instead of items, with joint weights across tasks and the macro over a task set fixed before resampling. Not the pre-registered interval: it measures what the within-clip dependence actually does to these intervals, in whichever direction it goes.

| comparison | clustered estimate [95% CI] | clusters |
|---|---|---|
| L1 - L0 | +0.0538 [+0.0401, +0.0674] | 5083 |
| L3 - L1 | +0.0223 [+0.0077, +0.0370] | 5083 |
| L5 - L3 | +0.0000 [+0.0000, +0.0000] | 5083 |
| L5 - L0 | +0.0761 [+0.0601, +0.0918] | 5083 |

## gemma4_e4b

| slot | n | full macro | seen macro | held_out macro | old_986 macro | new_4521 macro | logical | actual | parse fail | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 5507 | 0.2703 | 0.2656 | 0.2731 | 0.2788 | 0.2704 | 5507 | 4491 | 0 | 14498 |
| L1 | 5507 | 0.3774 | 0.4056 | 0.3605 | 0.3836 | 0.3753 | 5507 | 4491 | 0 | 14603 |
| L3_scaffold | 5507 | 0.4028 | 0.4174 | 0.3940 | 0.4221 | 0.3952 | 22191 | 18099 | 221 | 8292 |
| L5_own | 5507 | 0.4021 | 0.4179 | 0.3926 | 0.4165 | 0.3952 | 22184 | 8758 | 209 | 2458 |

### Per-task Recognition

| slot | ER_Lab | ER_Wild | FG_ER | FG_SA | IR | ML_ER | Noise_ER | SA |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.090 | 0.308 | 0.251 | 0.091 | 0.254 | 0.360 | 0.410 | 0.399 |
| L1 | 0.271 | 0.334 | 0.202 | 0.328 | 0.345 | 0.404 | 0.524 | 0.612 |
| L3_scaffold | 0.294 | 0.310 | 0.260 | 0.388 | 0.375 | 0.472 | 0.474 | 0.649 |
| L5_own | 0.294 | 0.315 | 0.260 | 0.386 | 0.369 | 0.476 | 0.472 | 0.645 |

### Paired differences (task-macro primary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.1071 [+0.0916, +0.1229] p=0.000 | +0.1400 [+0.1112, +0.1686] p=0.000 | +0.0874 [+0.0693, +0.1057] p=0.000 | +0.1048 [+0.0740, +0.1353] p=0.000 | +0.1049 [+0.0832, +0.1272] p=0.000 |
| L3 - L1 | +0.0254 [+0.0108, +0.0395] p=0.001 | +0.0118 [-0.0119, +0.0352] p=0.323 | +0.0335 [+0.0154, +0.0515] p=0.001 | +0.0385 [+0.0099, +0.0673] p=0.010 | +0.0199 [+0.0006, +0.0393] p=0.043 |
| L5 - L3 | -0.0007 [-0.0033, +0.0020] p=0.610 | +0.0005 [-0.0030, +0.0042] p=0.793 | -0.0014 [-0.0051, +0.0022] p=0.458 | -0.0056 [-0.0132, +0.0015] p=0.124 | -0.0000 [-0.0030, +0.0028] p=0.985 |
| L5 - L0 | +0.1318 [+0.1151, +0.1486] p=0.000 | +0.1524 [+0.1235, +0.1815] p=0.000 | +0.1195 [+0.0979, +0.1411] p=0.000 | +0.1377 [+0.1027, +0.1739] p=0.000 | +0.1248 [+0.1028, +0.1473] p=0.000 |

### Same differences, sample-weighted (secondary)

| comparison | full | seen | held_out | old_986 | new_4521 |
|---|---|---|---|---|---|
| L1 - L0 | +0.1016 [+0.0881, +0.1152] p=0.000 | +0.1171 [+0.0972, +0.1374] p=0.000 | +0.0836 [+0.0658, +0.1014] p=0.000 | +0.1036 [+0.0729, +0.1341] p=0.000 | +0.1012 [+0.0860, +0.1163] p=0.000 |
| L3 - L1 | +0.0188 [+0.0066, +0.0306] p=0.002 | +0.0054 [-0.0111, +0.0216] p=0.536 | +0.0343 [+0.0166, +0.0520] p=0.001 | +0.0383 [+0.0099, +0.0668] p=0.011 | +0.0145 [+0.0014, +0.0275] p=0.029 |
| L5 - L3 | -0.0000 [-0.0032, +0.0032] p=0.995 | +0.0010 [-0.0040, +0.0061] p=0.761 | -0.0012 [-0.0048, +0.0023] p=0.515 | -0.0056 [-0.0132, +0.0015] p=0.161 | +0.0012 [-0.0024, +0.0048] p=0.518 |
| L5 - L0 | +0.1204 [+0.1050, +0.1359] p=0.000 | +0.1235 [+0.1009, +0.1458] p=0.000 | +0.1167 [+0.0953, +0.1378] p=0.000 | +0.1363 [+0.1013, +0.1723] p=0.000 | +0.1169 [+0.0997, +0.1341] p=0.000 |

### Media-clustered diagnostic, full view

Whole media clusters drawn instead of items, with joint weights across tasks and the macro over a task set fixed before resampling. Not the pre-registered interval: it measures what the within-clip dependence actually does to these intervals, in whichever direction it goes.

| comparison | clustered estimate [95% CI] | clusters |
|---|---|---|
| L1 - L0 | +0.1071 [+0.0914, +0.1232] | 5083 |
| L3 - L1 | +0.0254 [+0.0110, +0.0397] | 5083 |
| L5 - L3 | -0.0007 [-0.0033, +0.0019] | 5083 |
| L5 - L0 | +0.1318 [+0.1144, +0.1490] | 5083 |

## Holm-adjusted p, full view, across the whole arm family

Family size 16, counting only comparisons that were actually computed. On a partial run that is a partial family, so these adjusted values tighten as more arms land and only the final 16-arm report should be read as the corrected result.

| comparison | adjusted p |
|---|---|
| gemma4_e4b L1 - L0 | 0.000 |
| gemma4_e4b L5 - L0 | 0.000 |
| qwen3_5_4b L1 - L0 | 0.000 |
| qwen3_5_4b L5 - L0 | 0.000 |
| qwen3vl_4b L1 - L0 | 0.000 |
| qwen3vl_4b L5 - L0 | 0.000 |
| qwen3vl_8b L1 - L0 | 0.000 |
| qwen3vl_8b L5 - L0 | 0.000 |
| gemma4_e4b L3 - L1 | 0.005 |
| qwen3_5_4b L3 - L1 | 0.018 |
| qwen3vl_4b L5 - L3 | 0.144 |
| qwen3vl_8b L3 - L1 | 0.521 |
| qwen3vl_4b L3 - L1 | 1.000 |
| qwen3vl_8b L5 - L3 | 1.000 |
| gemma4_e4b L5 - L3 | 1.000 |
| qwen3_5_4b L5 - L3 | 1.000 |

## Reading these numbers

Development evaluation. The old stratum has been inspected repeatedly across this project, and each L5 bundle was selected on D_val by the promotion gate that produced it, so an L5 - L3 interval is not a held-out estimate of construction value. D_ext-test (129 items) remains unevaluated. Overlapping intervals are not evidence of equality.

The clean item set holds more items than media: one clip can appear as several items, and mostly under different tasks -- FG_SA most of all, where 260 of 461 items share a clip with another item. The primary and secondary intervals resample items independently, so that dependence can miscalibrate them. Which direction is an empirical question, not something to assert: correlated errors within a clip usually widen the true interval, but the effect is not guaranteed to go one way. The clustered column measures the actual difference on these predictions rather than predicting its sign.

