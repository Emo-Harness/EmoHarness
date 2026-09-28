# In-distribution effect check

Purpose: a same-distribution check of the Builder loop with GPT 5.6 Luna as
Builder and Qwen3-VL-8B-Instruct as Target, comparable in kind to the earlier
Gemini 3.1 Pro run that was reported as "about 20 points over five to six
rounds". It follows the original README's split (build 10% / val 10% of the
three seen task families, test on the rest), **not** the pre-registered
protocol of ADR-0001/ADR-0003. Numbers from this run are not
leaderboard-comparable and are labelled in-distribution wherever reported.

Two runs exist. **v2 (2026-09-11, current)** uses deterministic perception
tools only, a same-model verifier in every Builder round, and per-round cost
records; its artifacts are under `results/`. **v1 (2026-09-08/09)** used a
Qwen2.5-Omni-7B audio describer; its artifacts are under
`results/archive/v1-omni-describer/` and its numbers are kept at the end of
this document.

## v2: what changed and why (2026-09-11)

1. **No audio-language model.** The v1 harness asked Qwen2.5-Omni-7B for a
   free-text description of the audio and for a valence/arousal opinion, both
   of which entered the evidence record. That is a second model's judgment,
   not a tool, so any gain could have been that model's rather than the
   Builder's. Perception is now ffmpeg frames, whisper large-v3 transcript and
   language, and openSMILE eGeMAPS phrases; the evidence ids are `v*`, `a*`
   (transcript sentences, then prosody phrases), and `t1` (ADR-0004 amendment).
   The scaffold prompts lost the two sentences about opinion items; nothing
   else in the Stage 1 bundle changed.
2. **Verifier per round.** After the deterministic checks, the round's valid
   candidates go to Luna once more, in a separate role with its own prompt,
   which sees the current files, the cluster keys and sizes, and each
   candidate's rationale with a unified diff. It rejects overfitting to the
   packet's items, label forcing, contract breaks, evidence-discipline
   violations, out-of-scope edits, and YAML damage. Unapproved candidates are
   never evaluated; `round_XX/verification.json`, `review.txt`, and the
   `vetoed` list in `candidates.json` record why (ADR-0006 §8). A written
   candidate the loader cannot load is recorded as rejected instead of
   crashing the run.
3. **Cost per round.** `round_XX/cost.json` and the round records in
   `loop_result.json` carry calls, tokens, and dollars by role (`builder`,
   `verifier`, `judge_proxy`, `target`), plus the cost of scoring the parent
   bundle and the stage total (ADR-0006 §9).
4. **Cache projection instead of re-transcription.** The preprocessing cache
   is keyed by the config hash, and the describer block leaving the config
   changes that hash. Whisper and openSMILE are deterministic on the same
   `audio.wav`, so a one-off migration (since removed; it is in the git history) wrote every v1 entry
   under the new hash with the description and opinion fields dropped and
   `speech_present` recomputed as "transcript non-empty" (36 of 1,903 clips
   had been marked speech-present on the strength of the description alone).
   Frames and audio are byte-identical to v1. Both caches verify
   (`preprocess.py --verify`: 923/923 external, 1000/1000 subset).
5. **One machine.** The second box was not available, so machine 1 runs
   every step; L0 is unchanged (native video, no audio text) and comes from
   the call cache.

Everything else (splits, subset, gates including the 4.5 mean-calls ceiling,
Luna at `reasoning_effort: high`, local Recognition only) is as in v1.

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

## v2 results (2026-09-11 07:16, local Recognition, 1,000-item subset, 125 per task)

| Row | ER_Lab | ER_Wild | SA | Noise_ER | FG_ER | ML_ER | FG_SA | IR | seen | held-out | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 Qwen3-VL-8B Direct | 8.0 | 36.8 | 38.4 | 52.8 | 28.8 | 34.7 | 14.4 | 20.0 | 27.7 | 30.2 | 29.2 |
| L3 Stage-1 scaffold (human) | 28.0 | 37.6 | 63.2 | 52.0 | 26.6 | 49.0 | 23.2 | 47.2 | 42.9 | 39.6 | 40.9 |
| L4 Stage-2 (Luna Builder, replicate 0) | 28.0 | 37.6 | 63.2 | 52.0 | 26.6 | 49.0 | 23.2 | 47.2 | 42.9 | 39.6 | 40.9 |
| L6 Luna Direct (frames + audio text) | 28.8 | 43.2 | 64.0 | 63.2 | 33.0 | 50.2 | 31.2 | 37.6 | 45.3 | 43.0 | 43.9 |

Paired stratified bootstrap on the overall score, 2,000 resamples:

| Delta | estimate | 95% CI | p | items better / worse |
|---|---|---|---|---|
| L3 - L0 | **+11.6** | [+8.2, +14.9] | <0.001 | 246 / 123 |
| L4 - L3 | **+0.0** | [+0.0, +0.0] | 1.000 | 0 / 0 |
| L6 - L0 | +14.7 | [+11.4, +17.6] | <0.001 | 258 / 89 |

**L4 equals L3 exactly**, and not by coincidence: the ladder's Stage-2 row is the
*first completed* replicate (ADR-0006 §6, declared before the runs), replicate 0
accepted nothing in either of its two rounds, so its final bundle is the Stage 1
scaffold byte for byte apart from `manifest.json`. The Builder's measured
contribution in v2 is therefore zero, not a small number.

Stage-2 construction (Luna Builder and Luna verifier, gates as preregistered
with the v1 mean-calls ceiling of 4.5):

| Replicate | Rounds | Candidates evaluated | Vetoed | Accepted | D_val start -> end | Luna cost |
|---|---|---|---|---|---|---|
| 0 (the declared run) | 2, early stop | 8 | 0 | 0 | 45.43 -> 45.43 | $0.84 |
| 1 | 2, early stop | 1 | 3 | 0 | 45.43 -> 45.43 | $0.24 |
| 2 | 4, early stop | 16 | 0 | 1 (round 1, "Conflict-side affect resolution", +2.0, regression 2.8%) | 45.43 -> **47.46** | $1.60 |

The regression gate is what stops almost everything, as in v1: of the 25
candidates evaluated across the three replicates, 19 raised D_val Recognition
and 17 of those regressed more than 5% of the items the incumbent already
solved. The two that passed both gates were both in replicate 2 round 1
(regression 4.5% and 2.8%); the loop took the higher-Recognition one. Replicate
2 then spent three more rounds without beating 47.46, and replicate 0, whose
result the protocol reports, never cleared the bar at all. Spread across
replicates is therefore the dominant effect: the same Builder, the same packet
format, and the same gates give 0, 0, and +2.0 points on D_val depending only
on sampling.

**The one accepted patch does not transfer.** Replicate 2's bundle was scored on
the same subset as a clearly-labelled side row
(`results/raw_predictions/L4_replicate2/`, not part of the ladder, a secondary
measurement of a best-of-three outcome):

| Row | seen | held-out | overall | vs L3 |
|---|---|---|---|---|
| L3 Stage-1 scaffold | 42.9 | 39.6 | 40.9 | - |
| replicate 2 (1 accepted patch, D_val +2.0) | 42.4 | 39.7 | **40.7** | **-0.1** [-1.0, +0.8], p=0.80, 13 better / 15 worse |

So the +2.0 points the patch won on D_val (394 items) are worth -0.1 on the
1,000 test items, and the 28 items it moves are a coin flip. This is the
cleanest statement the run produces: the loop's objective can be improved and
the improvement is selection noise on the construction split, not a capability
the harness gained. Taking the best of three replicates instead of the declared
one would therefore not have changed the conclusion either, which is worth
saying because it is the obvious objection to reporting replicate 0.

### The verifier

Ten verifier calls, 108,663 input and 21,708 output tokens, $0.05 in total: 1.4%
of the run's Luna spend, about half a cent per round. It vetoed 3 of the 28
candidates it reviewed, all three in replicate 1 round 1 and all three under
rule 2 (label forcing), quoting the offending text:

- "treat those as evidence for Frustration when that exact candidate is present"
  and "do not rank Neutral above it merely because the visible expression is
  restrained";
- "Favor the exact positive entry" and "rank it above neutral";
- "support the exact negative entry" and "rank it above neutral".

Those are exactly the "buy Recognition by asserting a label more readily" moves
that v1 identified as the reason candidates regress, so the verifier is
catching the failure mode it was written for. It is not a bottleneck: the gates
rejected 24 of the 25 candidates it approved. Its one measurable side effect is
that replicate 1 evaluated 1 candidate instead of 4 and reached its early stop
sooner.

A defect found and fixed during the run: Luna's first review copied the whole
heading (`Candidate 2: <name>`) into each verdict's `name`, no verdict matched
its candidate, and all four candidates of a round were vetoed as "unreviewed";
two such rounds hit the early-stop rule. Commit `5a498fb` makes the name its own
line in the request, ignores a leading `Candidate N` when matching, and falls
back to position when a report carries exactly one verdict per candidate. That
void replicate cost $0.07 and was deleted before the rerun.

### Cost

| Role | Calls | Input tokens | Output tokens | USD |
|---|---|---|---|---|
| builder | 11 | 1,137,757 | 70,421 | 0.31 |
| verifier | 10 | 108,663 | 21,708 | 0.05 |
| judge_proxy (Reasoning gate) | 6,800 | 12,495,243 | 360,400 | 2.93 |
| teacher_direct (L6) | 999 | 2,770,683 | 16,802 | 0.57 |
| target (student, local) | 37,279 | 50,393,827 | 7,779,351 | 0.00 |
| **Luna total** | | | | **3.87** |

The Reasoning gate is 76% of the bill and the Builder itself 8%. Wall clock:
4 h 39 min on one box (02:37 to 07:16), plus 21 minutes of L0/L3 before that.

### v2 against v1

| | v1 (Omni describer, no verifier) | v2 (tools only, verifier) |
|---|---|---|
| L0 | 29.2 | 29.2 |
| L3 | 41.3 | 40.9 |
| L4 | 41.5 | 40.9 |
| L6 | 44.4 | 43.9 |
| L3 - L0 | +12.1 | +11.6 |
| L4 - L3 | +0.2 (9 better / 7 worse) | +0.0 (0 / 0) |
| Accepted patches / replicates | 1 / 3 | 1 / 3 (not in the declared replicate) |
| Luna spend | $5.42 | $3.87 |

Removing the audio-language model cost 0.4 points on the scaffold (41.3 to 40.9)
and 0.5 on the teacher row, both well inside the ±2.7-point interval on a row's
overall score. The reading of the experiment does not change: the human
scaffold is worth about +12 points over Direct and closes roughly 80% of the
gap to the Luna teacher, and Stage-2 Builder construction adds nothing
measurable on top of it, with or without a verifier.

## Stage 3: what the Builder does with code, and whether it transfers (2026-09-12)

ADR-0009 puts the AI4AI claim on `Builder gain = L5 - L3`, and L5 had never
been run: every earlier Builder result was Stage 2, where only prompts and YAML
are editable. Stage 3 also unlocks `harness/verification/*.py`, the two pure
functions that decide whether the seen and heard cues agree and how the
hypotheses are ordered. They run locally on the stage JSON and issue no model
calls, so a candidate that edits only them reuses every cached Target call and
has a different failure profile from a prompt edit: it can change the emitted
label without changing a single word the Target was asked.

Two replicates ran, differing in one thing. Replicate 0 got the packet as it
was: the verification paths listed under "Editable paths" and nothing about what
they do. It proposed 32 candidates over 8 rounds and **edited a prompt every
single time**, never once the functions the stage had unlocked. Replicate 1 got
the block added in ADR-0006 §11, which gives each function's name, signature,
effect and static-allowlist contract, and names no direction to take.

| | Replicate 0, seams not described | Replicate 1, seams described |
|---|---|---|
| Rounds / candidates | 8 / 32 | 8 / 32 |
| **Candidates editing code** | **0** | **3**, all `alternative_hypotheses.py` |
| `cross_modal.py` edits | 0 | 0 |
| Other files reached | prompts only | prompts, plus `sampling.yaml` twice |
| Accepted patches | 2 | 5 |
| D_val | 45.43 → 47.97 | 45.43 → 47.72 |
| Cost | $3.05 | $2.70 |

Both replicates ran all 8 rounds because the Builder, which now ends its own
stage (§10), never once asked to stop.

### The result: three constructions, three zeros on test

| Run | Action space | D_val | Subset, vs L3 |
|---|---|---|---|
| Stage 2, replicate 2 | prompts, YAML | **+2.0** | **−0.1** [−1.0, +0.8] |
| Stage 3, replicate 0 | plus code, seams not described | **+2.5** | **−0.1** [−1.4, +1.2] |
| Stage 3, replicate 1 | plus code, seams described | **+2.3** | **−0.3** [−1.5, +0.8] |

Overall local Recognition: L3 40.9, L4 40.9, L5 40.8 (replicate 0), L5 40.5
(replicate 1), teacher 43.9. **Builder gain, L5 − L3, is −0.1**, the number
ADR-0009 rests the AI4AI claim on.

The third run had every advantage: the largest action space, the Builder setting
its own round budget and using all of it, five patches through gates that
required a strict Recognition improvement and at most 5% regression (the
accepted ones ran as low as 0.6%), and a packet that described the code seams.
The objective moved 2 to 2.5 points and none of it transferred. Two independent
stages and three independent replicates give the same answer, so this is a
property of the loop rather than a run of bad luck: 394 validation items cannot
separate a reading rule that generalises from one that happens to suit them, and
a gate on those same items cannot either.

### What the Builder does when code is unlocked

Wording was part of it and not all of it. Describing the seams moved code edits
from 0 to 3 of 32 and widened the action space (`sampling.yaml` appeared for the
first time), so the packet had indeed been naming paths without saying what they
were. But 29 of 32 candidates still edited prose, and `cross_modal.py` was never
touched in 64 candidates across both replicates. Given a Python seam that
decides the answer, a described contract, and eight rounds, this Builder prefers
to write the rule in English in a prompt and let the frozen code apply it.

That preference is visible in what passes, too. Both replicates' accepted
patches are adjudication rules — "resolve conflicts by affective strength",
"anger over frustration for reproach", "separate appraisal from delivery" —
which is exactly the job of the two functions it declined to edit. It is not
that it failed to see where the decision is made; it saw, and chose prose.

### Recorded for this pass

- Replicate 0 is the uninformed condition and must stay that way: the §11 block
  is rendered only when a `verification/*.py` path is editable, so Stage 2
  packets and replicate 0's provenance are unchanged.
- The block deliberately names no direction. A test asserts it contains none of
  "should", "prefer", "instead of", "try", "better", because a code edit made
  after we suggested one would be our idea, not the Builder's.
- Replicate 1's L5 row is under `results/replicates/stage3-rep1/`
  (`EMOHARNESS_RESULTS_DIR`) so it does not overwrite replicate 0's. That
  override also relocates `construction/`, so the row was run with an explicit
  `--bundle results/construction/stage3/1/final`.

## Ladder decomposition and the student size sweep (2026-09-12)

The v2 run above measured only L0 and L3, so its "+11.6 harness gain" pooled
three ingredients ADR-0009 defines separately and had no equal-compute control.
This pass fills every hole: L1, L2 and best-of-4 on the 8B student, and the whole
single-student ladder on Qwen3-VL-2B and 4B. No paid API; every row is the same
1,000-item subset, the same preprocessing cache, and the same frozen Stage 1
bundle. Rows for 2B and 4B live under `results/students/qwen3vl-{2b,4b}/`
(`EMOHARNESS_RESULTS_DIR`); their Target-call keys already carry the model id, so
all three share one cache.

Overall local Recognition:

| Student | L0 | L1 | L2 | best-of-4 | L3 |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 28.4 | 33.7 | 32.9 | 33.9 | **31.8** |
| Qwen3-VL-4B | 30.0 | 38.0 | 34.4 | 35.9 | **38.9** |
| Qwen3-VL-8B | 29.2 | 40.2 | 36.9 | 37.5 | **40.9** |

Paired stratified bootstrap, 3,000 resamples, bold where p < 0.05:

| Layer | 2B | 4B | 8B |
|---|---|---|---|
| L1 − L0, modality | **+5.3** [+2.5, +8.2] | **+8.0** [+5.0, +10.8] | **+11.0** [+8.0, +13.8] |
| L2 − L1, format | −0.8 (p=0.52) | **−3.5** [−6.1, −1.0] | **−3.3** [−5.8, −0.8] |
| L3 − L2, scaffold | −1.1 (p=0.43) | **+4.5** [+1.8, +7.3] | **+4.0** [+1.2, +6.8] |
| best-of-4 − L2, resampling at equal calls | +1.0 (p=0.41) | +1.5 (p=0.16) | +0.6 (p=0.58) |
| L3 − best-of-4, structure vs resampling | −2.0 (p=0.16) | **+3.0** [+0.3, +5.8] | **+3.4** [+0.6, +6.1] |
| L3 − L0, all three | **+3.5** [+0.4, +6.4] | **+8.9** [+5.6, +12.0] | **+11.6** [+8.3, +14.8] |

Five things this settles.

1. **Most of the "harness gain" is the transcript.** For the 8B, +11.0 of the
   +11.6 is L1 − L0: appending the whisper transcript, its language, and the
   openSMILE prosody phrases to the official prompt, still one call, still
   native video. The Target has no audio input, so this is the harness handing
   a deaf model what it could not hear. It is not a reasoning scaffold.
2. **The human-written five-step format is a significant negative**, −3.3 on the
   8B and −3.5 on the 4B, two independent models agreeing. It costs most on the
   multi-label tasks (8B ML_ER 40.6 → 30.7, FG_ER 21.1 → 14.9): the checklist
   walks the model through cues one at a time and it then commits to fewer
   labels. L2's prompt is frozen at pre-registration and was not edited.
3. **Four calls spent on structure beat four calls spent on resampling**, but
   only above a capacity threshold. best-of-4 over L2 is worth nothing anywhere
   (+0.6 to +1.5, never significant), while L3 beats it by +3.4 on the 8B and
   +3.0 on the 4B. This is the one clean positive result for the scaffold: the
   gain is from structure, not from the extra compute.
4. **The scaffold has a capacity threshold between 2B and 4B.** L3 − L2 is
   −1.1 on the 2B (no effect), then +4.5 and +4.0. Below the threshold the
   scaffold is a net burden: the 2B's L3 (31.8) is *worse* than its best-of-4
   (33.9) and its FG_ER falls from 14 at L0 to 7 at L3. The modality layer, by
   contrast, rises smoothly (+5.3, +8.0, +11.0) with no threshold.
5. **Direct scores do not rank these models; harness gains do.** L0 is 28.4,
   30.0 and 29.2 across a fourfold parameter range, while L3 − L0 is +3.5, +8.9
   and +11.6. On this benchmark a Direct number says little about a model; what
   separates them is how well they exploit a harness.

This also disposes of the headroom explanation for the Builder's zero. The 2B
sits 12.1 points below the Luna teacher row, four times the 8B's 3.0, and still
cannot use the human scaffold at all. Room is not what the Builder lacked;
Stage-2 edits prompts and YAML, and the 2B's limit is that it cannot follow a
structured evidence protocol, which no prompt edit reaches.

### Deviations recorded for this pass

- L2's checklist still asks for non-speech sounds (laughter, sighs, music). The
  v2 amendment removed the audio describer and the Target has no audio input, so
  that bullet is unsatisfiable; it was left alone because L2 is frozen at
  pre-registration, and it biases L2 downward. best-of-4 shares L2's prompt and
  therefore the same bias, so L3 − best-of-4 is the unbiased comparison.
- Serving moved mid-pass. Rows up to the 8B's L2 ran on an A100-80GB in a
  2 CPU / 8 GB cgroup; from the 8B's best-of-4 onward on an H200-144GB in a
  10 CPU / 32 GB cgroup. Throughput went from 21 to 105 Target calls per minute
  because the old box's video-decode semaphore, set to 2 to survive its 8 GB
  limit, left the GPU at 2% KV utilisation and 85% of each call's 36 s in a
  queue. Decoding parameters, prompts, media and the call cache are unchanged,
  and cached responses carry across, so the move changes timing only.
- The serving multimodal processor cache was switched on (1 GB, later 4 GB)
  because best-of-4 sends one clip four times and decoding it four times was
  what pushed host memory past the watchdog. It is a serving cache over a
  deterministic processor: identical inputs, identical outputs.

## v1 results (2026-09-09 01:02, local Recognition, 1,000-item subset, 125 per task; with the Omni describer)

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
- v2 serves one model only (the Target); whisper runs in-process on the GPU
  during preprocessing. v1 additionally served the Omni describer on port
  8001 and never both at once (host RAM).
