# IntentQA and Social-IQ 2.0: two tests of the 2026-09-19 per-student harnesses

Two out-of-distribution experiments with the four per-student bundles that
`docs/PROVENANCE-student-specific-full-20260919.md` pins (Qwen3-VL-8B,
Qwen3-VL-4B, Qwen3.5-4B, Gemma-4-E4B). Both use IntentQA **test** (2134
questions, 567 videos) and Social-IQ 2.0 **val** (795 questions, 122 videos;
the release's test split has no answers). Every model is served locally by vLLM
at temperature 0. No judge, and no paid API call.

| | full report |
|---|---|
| 1. transfer: the harness answers the question | `results/newdata/intentqa_socialiq_20260923/report.md` |
| 2. emotion context: the harness informs the bare model | `results/newdata/emotion_context_20260925/report.md` |
| 2b. emotion questions only | `results/newdata/emotion_context_20260925/subset_emotion.md` |

## 1. Transfer (2026-09-23): the harness as the QA solver

`scripts/newdata_manifests.py` renders each question into the MME-Emotion
prompt grammar, so the frozen harness can run unchanged. The question goes in
the noun slot (`identify the most prominent answer to the question "..."`), the
options become the candidate labels, and commas are stripped from options.
Ladder rows L0/L1/L3/L5 run through `newdata_row.py` and `run_newdata.py`, and
`newdata_report.py` scores them.

Result: no significant L5−L0 gain on either dataset. IntentQA drops
significantly for Qwen3.5 (−11.86 pp) and Gemma (−9.61 pp). L1 (video plus the
audio text) is the best row in 7 of 8 student×dataset cells.

This protocol asks an emotion harness to answer non-emotion questions. It was
replaced by experiment 2, which tests what the harness is for.

## 2. Emotion context (2026-09-25): does the harness's emotion context help the bare model?

Each student's own harness runs on its native task, once per video: the
MME-Emotion ER_Wild prompt with the labels `neutral, happy, angry, sad, fear,
surprise, disgust`. No question is involved. The context it writes has three
parts:

- the predicted emotion with its confidence;
- up to three ranked alternatives;
- its step 1–4 analysis, with the `[v#]`/`[a#]` evidence citations removed.

The same bare student then answers a lettered multiple-choice prompt. It sees
the same cached frames (2 fps, up to 32) in every condition. Only the text
block between the frames and the question differs:

| condition | text block |
|---|---|
| `none` | nothing |
| `asr` | the Whisper transcript plus openSMILE prosody block, the same text L1 appends |
| `emoctx` | the `asr` block, followed by the harness emotion context |

`emoctx − asr` is what the harness's own analysis adds beyond the raw
perception it was computed from. `emoctx − none` is the total effect.

Reproduce:

```bash
.venv/bin/python scripts/emoctx_manifests.py
setsid nohup .venv/bin/python -u scripts/run_emoctx.py --take-over \
    >> logs/emoctx/driver.$(hostname).log 2>&1 < /dev/null &
.venv/bin/python -u scripts/emoctx_topup.py      # after the driver exits
.venv/bin/python scripts/emoctx_report.py
.venv/bin/python scripts/emoctx_subset.py
```

### Accuracy (%)

| | Social-IQ none / asr / emoctx | IntentQA none / asr / emoctx |
|---|---|---|
| Qwen3-VL-8B | 61.89 / 70.19 / 66.42 | 70.99 / 72.63 / 73.10 |
| Qwen3-VL-4B | 59.12 / 68.43 / 65.28 | 67.48 / 68.42 / 70.29 |
| Qwen3.5-4B | 65.91 / 71.82 / 72.70 | 79.90 / 77.84 / 76.34 |
| Gemma-4-E4B | 57.74 / 64.91 / 61.13 | 57.31 / 60.59 / 61.62 |

### What each part adds (pp, Holm over 12 comparisons per dataset)

| | Social-IQ asr − none | Social-IQ emoctx − asr | IntentQA asr − none | IntentQA emoctx − asr |
|---|---|---|---|---|
| Qwen3-VL-8B | +8.30 (0.002) | −3.77 (0.056) | +1.64 (0.21) | +0.47 (0.69) |
| Qwen3-VL-4B | +9.31 (0.002) | −3.14 (0.089) | +0.94 (0.69) | +1.87 (0.085) |
| Qwen3.5-4B | +5.91 (0.002) | +0.88 (0.53) | −2.06 (0.085) | −1.50 (0.26) |
| Gemma-4-E4B | +7.17 (0.008) | −3.77 (0.089) | +3.28 (0.004) | +1.03 (0.69) |

Findings:

- `emoctx` beats `none` in 7 of 8 cells. Most of that gain is the transcript
  alone (`asr − none`).
- The harness's own analysis (`emoctx − asr`) survives Holm nowhere. On
  Social-IQ it is negative for 3 of 4 students.
- Restricting to questions every arm answered does not change these
  conclusions.

### 2b. Emotion questions only

`scripts/emoctx_subset.py` defines emotion questions by a keyword rule on the
question text. The rule was written before any subset result was computed.
Social-IQ has 262 core and 353 broad matches. IntentQA has 17 core and 90
broad, too few to analyse.

On Social-IQ core emotion questions, `emoctx − asr` is:

| student | emoctx − asr (pp) |
|---|---|
| Qwen3-VL-8B | +0.76 |
| Qwen3-VL-4B | −1.53 |
| Qwen3.5-4B | −0.38 |
| Gemma-4-E4B | −4.58 |

No interval lies above zero. For the two Qwen3-VL students, the overall
Social-IQ drop sits on the non-emotion questions (−6.0 and −3.9 pp). For
Qwen3-VL-8B the interaction is +6.8 [+1.8, +11.7]. The context distracts more
than it informs. This analysis is exploratory, with no multiplicity
correction.

## Protocol details that affect the numbers

- **Answer parsing.** A reply counts only if it commits to a letter: a
  leading letter, an `<answer>` tag, an explicit "answer is X", or a last line
  that is only the letter (optionally followed by its option text). A letter
  merely mentioned in the analysis never counts. Unparsed replies score wrong
  and are counted per arm. `tests/unit/scripts/test_emoctx.py` pins these
  rules. The report re-reads every reply with the final parser.
- **Qwen3.5-4B output budget.** With thinking off, Qwen3.5-4B still writes an
  analysis before answering. At 64 output tokens, 12 of 13 smoke replies ended
  before any letter. It therefore gets the 1024-token budget from its target
  config. Its prompt is unchanged, so its three conditions remain identical to
  each other. The other students reply with a bare letter within 64 tokens.
- **Context length.** For the Qwen models, 32 frames of 25 IntentQA videos
  (100 questions) exceed `max_model_len` 16384, and vLLM rejects the call
  (HTTP 400). The same 100 questions are rejected in every arm. The harness
  sends fewer images per call and is unaffected. `emoctx_topup.py` re-sends
  only the rejected calls on a server with `max_model_len` 32768. Errors are
  never cached, so every call that already succeeded is served unchanged. The
  16384 files are kept as `*.maxlen16384.*`.
- **Refusals.** With more text, Gemma more often answers "none of the options
  fit" without a letter: 4, 9 and 23 unparsed replies across none/asr/emoctx on
  Social-IQ. The sensitivity table restricted to questions answered in every
  arm isolates this.
- **Two boxes, one results tree.** The NFS home is shared, and containers were
  replaced twice during the run. `owner.json` names the only host allowed to
  write. `emoctx_row.py` refuses every other host. Status and logs are per
  host.

## Correction to experiment 1 as published

The 2026-09-24 website addendum states "0 unparsable answers". That is wrong.
The single-call rows fall back to the first option when nothing parses, and
the count used did not see those fallbacks. The real counts are:

| rows | items per arm that fell back |
|---|---|
| L0/L1 | 0–81 (up to 3.8%) |
| L3/L5 harness rows | 2–39 used a fallback output |

The accuracies themselves are correct as computed.

## What is and is not in git

Committed:

- the scripts and tests;
- `report.{md,json}`;
- `subset_emotion.{md,json}`;
- `manifest_record.json` with manifest hashes;
- per-step `done.json` provenance;
- the eight harness-generated `contexts.jsonl` files.

Not committed (regenerable from the scripts and the shared caches):

- raw predictions and harness outputs;
- ledgers;
- the rendered manifests;
- logs.
