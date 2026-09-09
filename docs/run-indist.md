# In-distribution effect check, 2026-09-08

Purpose: a same-distribution check of the Builder loop with GPT 5.6 Luna as
Builder and Qwen3-VL-8B-Instruct as Target, comparable in kind to the earlier
Gemini 3.1 Pro run that was reported as "about 20 points over five to six
rounds". It follows the original README's split (build 10% / val 10% of the
three seen task families, test on the rest), **not** the pre-registered
protocol of ADR-0001/ADR-0003. Numbers from this run are not
leaderboard-comparable and are labelled in-distribution wherever reported.

## What runs

| Step | Command (via `scripts/run_indist.sh`) | Notes |
|---|---|---|
| split | `examples/split_benchmark_indist.py --per-kind 130,130,40` | build 400 / val 394 / ext_test 129; 5,577 test items remain |
| preprocess | `examples/preprocess.py --items external`, then `--items subset` | frames at 2 fps (4 to 32, 640 px wide), whisper large-v3, eGeMAPS, Qwen2.5-Omni-7B cue and opinion channels, no diarization |
| subset | `examples/preregister.py --size-per-task 125` | 1,000 test items, 125 per task |
| baselines | `run_ladder.py --rows L0,L3 --items subset` | Qwen Direct; Stage 1 human scaffold |
| Builder | `build_harness.py --stage 2 --replicate 0 --rounds 6` | Luna, `reasoning_effort: high`; gate scored by Luna at effort none |
| result | `run_ladder.py --rows L4 --items subset`, `--rows L6` | Stage 2 harness; Luna Direct as the teacher reference |
| tables | `analyze.py --resamples 2000` | local Recognition only; no official judge |

The numbers to read: D_val Recognition from round 0 to the accepted final
bundle in `results/construction/stage2/0/loop_result.json`, and L4 minus L0
(and minus L3) per task on the subset in `results/metrics/ladder.md` and
`per_task.json`.

## Deviations from the ADRs, all deliberate

- Construction data is MME-Emotion's own seen-task items (README v1 split).
- No diarizer: `pyannote.audio` does not import against the torchaudio on
  this box; MME clips are single-utterance, so the full transcript stands in.
- Audio describer is Qwen2.5-Omni-7B, not the 30B captioner: the 8 GB host
  RAM cgroup cannot hold the 30B loader beside whisper.
- Proxy judge is Luna (`provider: openai`), not Gemini 3 Flash: no Gemini key.
- Official judge not run; Recognition is computed locally (exact match after
  the official synonym normalisation; C/K on the multi-label tasks).
- Sub-0.14 s clips (3 media files, 4 items across D_val and the subset) have
  fewer real frames than `min_frames: 4`; ffmpeg writes nothing for the fourth
  seek, so `extract_frames` repeats the previous frame. The Target therefore
  sees one duplicated frame on those clips. Their first cache records listed a
  frame that did not exist and were repaired in place (frames re-extracted,
  audio fields kept) on 2026-09-08.
- Clips wider than 640 px are transcoded once to 640-wide mp4 (video stream
  only) before they reach vLLM, and mkv/avi containers are remuxed to mp4;
  the frontend otherwise tensorises 1080p frames in host RAM (0.8 GB per
  request) and OpenCV over-reports Matroska frame counts. The processor's own
  total-pixel budget shrinks 32 frames below 640 px anyway. Cache keys hash the
  original file, so this changes bytes sent, not identity.
- vLLM's loader samples at most 32 frames uniformly per clip (`--media-io-kwargs`);
  the 2 fps target of ADR-0011 is not enforced at the loader because clips
  under one second then yield a single frame, which the processor rejects.

## Amendments made after seeing results (on the record)

- 2026-09-08 21:41, after round 0 of replicate 0: `gates.max_mean_calls`
  raised from the implicit 4.000 (the *expected* call count used as an exact
  ceiling) to 4.5. Round 0 had rejected all four candidates, two of them solely
  for "mean calls 4.005 exceeds 4.000", i.e. one retry among 394 D_val items;
  the best candidate (+1.3 points on D_val, regression 4.3%) would otherwise
  have been promoted. ADR-0011's hard cap is 6 calls with one retry per stage,
  so a ceiling of exactly the expected value was a defect, not a chosen
  threshold. Both replicates were restarted from round 0 with the new value.
  The regression-rate limit (5%), the Reasoning margin, and the strict
  Recognition improvement requirement are unchanged and will not be loosened
  after the fact; replicate 1's round 0 would still have accepted nothing
  under the new ceiling (all four candidates failed regression or Recognition).

## Results (2026-09-09 01:02, local Recognition, 1,000-item subset, 125 per task)

| Row | ER_Lab | ER_Wild | SA | Noise_ER | FG_ER | ML_ER | FG_SA | IR | seen | held-out | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 Qwen3-VL-8B Direct | 8.0 | 36.8 | 38.4 | 52.8 | 28.8 | 34.7 | 14.4 | 20.0 | 27.7 | 30.2 | 29.2 |
| L3 Stage-1 scaffold (human) | 28.0 | 34.4 | 60.0 | 59.2 | 24.5 | 45.9 | 28.0 | 50.4 | 40.8 | 41.6 | 41.3 |
| L4 Stage-2 (Luna, replicate 0) | 28.8 | 32.8 | 60.8 | 60.8 | 25.7 | 44.9 | 28.0 | 50.4 | 40.8 | 42.0 | 41.5 |
| L6 Luna Direct (frames + audio text) | 28.0 | 45.6 | 62.4 | 64.0 | 37.2 | 47.9 | 30.4 | 40.0 | 45.3 | 43.9 | 44.4 |

95% bootstrap intervals on the overall score are about ±2.7 points (`results/metrics/ladder.md`).

- Harness gain, L3 − L0: **+12.1** overall (+13.1 seen, +11.4 held-out).
- Builder gain, L4 − L3: **no measurable improvement**. The point difference is +0.2 (a net of 2 items: 9 better, 7 worse out of 1,000), well inside the paired bootstrap interval, and the single accepted patch moved D_val by only 0.5 while two of three replicates accepted nothing; the honest reading is that Stage-2 construction with Luna added nothing to the scaffold.
- Teacher, L6 − L0: +15.2; the scaffold alone closes 80% of that gap.
- Excluding the 14 subset items the leakage report flags as sharing media with construction items changes no number by more than 0.2.

Stage-2 construction (Luna, `reasoning_effort: high`, gates as preregistered except the
mean-calls ceiling amendment above):

| Replicate | Rounds | Candidates | Accepted | D_val before → after |
|---|---|---|---|---|
| 0 (machine 1) | 4, early stop | 16 | 1 (round 1: "promote explicit smiles over neutral audio", +0.5, regression 4.3%) | 47.72 → 48.22 |
| 1 (machine 2) | 2, early stop | 7 | 0 | 47.72 |
| 2 (machine 2) | 2, early stop | 5 | 0 | 47.72 |

Across 28 candidates the two gates fail in complementary ways: every candidate that
raised D_val Recognition (best +1.8 at 50.0) regressed more than 5% of the items the
incumbent already solved (5.1% to 15%), and every candidate inside the regression
limit failed to raise Recognition, with the single exception above. Luna's patches
buy Recognition by assigning affect labels more readily and pay for it in items the
scaffold had right.

Cost: Luna $5.42 across both boxes (Builder $0.47 for 14 calls, gate proxy judge $4.36
for 10,000 calls, teacher Direct $0.59). Student calls: 43,780, local.

Reading against the earlier "about 20 points with Gemini 3.1 Pro": on this protocol
and metric the whole harness gives +12 and the Luna Builder adds +0.2. Whether the
earlier figure counted the scaffold, was measured on D_val, or reflects a stronger
Builder cannot be settled from the record; the Terra config
(`EMOHARNESS_BUILDER_CONFIG=teacher_gpt_5_6_terra.yaml`) is prepared for that test.

## Box-specific setup

- cgroup: 2 CPUs, 8 GB RAM, one A100-80GB (nproc/free lie; read
  `/sys/fs/cgroup/cpu.max` and `memory.max`).
- Project venv: `.venv` on `/usr/bin/python3.12` with `pip install -e .` plus
  test deps, faster-whisper, opensmile, openai, and `nvidia-cublas-cu12` /
  `nvidia-cudnn-cu12` for ctranslate2 (`LD_LIBRARY_PATH` set by the runner).
- vLLM venv: `~/.venvs/emoharness-vllm011`, `vllm[audio]==0.11.2` with torch
  2.9.0+cu128 (the driver is 570 / CUDA 12.8; vLLM 0.28's cu130 torch fails).
- Triton needs `Python.h`: `libpython3.12-dev` was extracted (no root) to
  `~/.local/python3.12-headers`; the serve scripts export `CPATH` to it.
- Servers never run together (host RAM): the describer during preprocessing,
  the Target afterwards. `scripts/run_indist.sh` switches them.
