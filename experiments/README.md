# Experiments

Three evaluations of the four per-student harnesses in [`../bundles/`](../bundles/):

1. **Main evaluation**, on MME-Emotion: the ladder rows L0, L1, L3 and L5 for
   every student, run with the generic CLI in `examples/`.
2. **Transfer** (`experiments/transfer/`), on IntentQA and Social-IQ 2.0: the
   harness itself answers the question.
3. **Emotion context** (`experiments/emotion_context/`), on the same datasets:
   the harness writes an emotion context and the bare model answers with it.

Every model is served locally with vLLM at temperature 0. The rows are:

| row | what the student sees |
|---|---|
| L0 | the native video |
| L1 | the video plus a text block with the Whisper transcript and openSMILE prosody |
| L3 | the shared scaffold in `harness/` |
| L5 | the student's own bundle |

Intervals come from paired bootstrap resampling. Holm correction runs over each
comparison family.

## 1. Main evaluation: MME-Emotion, 5507 items

The item set is `clean`, which is every benchmark item minus the 70 whose clip
also appears in the construction split. It is computed from the two
manifests.

```bash
# one student at a time
EMOHARNESS_STUDENT_MODEL=Qwen/Qwen3-VL-4B-Instruct \
EMOHARNESS_VLLM_VENV=~/.venvs/emoharness-vllm011 scripts/serve_student.sh &

export EMOHARNESS_RESULTS_DIR=results/main/qwen3vl_4b
T=configs/target/qwen3vl_4b_instruct.yaml
python examples/run_ladder.py --rows L0,L1,L3 --items clean --target $T
python examples/run_ladder.py --rows L5 --items clean --target $T --bundle bundles/qwen3vl_4b
python examples/analyze.py --rows L0,L1,L3,L5 --items clean
```

Task-macro Recognition (%), 5507 items:

| student | L0 | L1 | L3 | L5 | L5 − L3 (pp) |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 29.95 | 39.15 | 40.47 | 40.68 | +0.21 [−0.16, +0.57] |
| Qwen3-VL-4B | 29.39 | 38.39 | 37.60 | 38.56 | +0.95 [+0.14, +1.83] |
| Qwen3.5-4B | 29.62 | 34.99 | 37.23 | 37.23 | 0.00 |
| Gemma-4-E4B | 27.03 | 37.74 | 40.28 | 40.21 | −0.07 [−0.33, +0.19] |

- Almost all of the gain over the bare model (L5 − L0, +7.6 to +13.2 pp) comes
  from giving it the audio evidence as text (L1 − L0, +5.4 to +10.7 pp).
- The student's own Builder-constructed bundle adds nothing that survives Holm
  over L3. The best case is Qwen3-VL-4B, with Holm p = 0.144.

## 2. Transfer: the harness answers IntentQA and Social-IQ questions

Datasets:

- IntentQA **test**: 2134 questions on 567 videos. Annotations come from
  `JoseponLee/IntentQA`; videos come from the `hamedrahimi/IntentQA` Hugging Face mirror.
- Social-IQ 2.0 **val**: 795 questions on 122 videos, from `PediaMedAI/Social-IQ-Video`.
  The test split in that release has no answers.

Each question is rendered into the MME-Emotion prompt grammar, so the frozen
harness runs unchanged. The question goes in the noun slot and the options
become the candidate labels.

```bash
python -m experiments.transfer.manifests
python -m experiments.transfer.row --dataset intentqa_test --perceive-only   # fill the cache
python -m experiments.transfer.run                                           # all students
python -m experiments.transfer.report
```

Result:

- No significant L5 − L0 gain on either dataset.
- On IntentQA, L5 − L0 is significantly negative for Qwen3.5-4B (−11.86 pp)
  and Gemma-4-E4B (−9.61 pp).
- L1 is the best row in 7 of 8 student × dataset cells.

This asks an emotion harness to answer non-emotion questions, which is what
experiment 3 avoids.

## 3. Emotion context: does the harness's context help the bare model?

Each student's own harness runs once per video on its native task: the
MME-Emotion ER_Wild prompt with the labels `neutral, happy, angry, sad, fear,
surprise, disgust`. No question is involved. The context it writes has three
parts:

- the predicted emotion;
- up to three ranked alternatives;
- its step 1–4 analysis, with the evidence-id citations removed.

The bare student then answers a lettered multiple-choice prompt. It sees the
same cached frames (2 fps, up to 32) in every condition, and only one text
block differs:

| condition | text block |
|---|---|
| `none` | nothing |
| `asr` | the L1 transcript and prosody block |
| `emoctx` | the `asr` block, followed by the harness emotion context |

```bash
python -m experiments.emotion_context.manifests
python -m experiments.emotion_context.run --students qwen3vl_4b --smoke-only   # check the setup
setsid nohup python -u -m experiments.emotion_context.run --take-over \
    >> logs/emotion_context/driver.$(hostname).log 2>&1 < /dev/null &
python -m experiments.emotion_context.topup    # re-send calls longer than 16384 tokens
python -m experiments.emotion_context.report
python -m experiments.emotion_context.subset   # emotion questions only
```

Accuracy (%):

| student | Social-IQ none / asr / emoctx | IntentQA none / asr / emoctx |
|---|---|---|
| Qwen3-VL-8B | 61.89 / 70.19 / 66.42 | 70.99 / 72.63 / 73.10 |
| Qwen3-VL-4B | 59.12 / 68.43 / 65.28 | 67.48 / 68.42 / 70.29 |
| Qwen3.5-4B | 65.91 / 71.82 / 72.70 | 79.90 / 77.84 / 76.34 |
| Gemma-4-E4B | 57.74 / 64.91 / 61.13 | 57.31 / 60.59 / 61.62 |

What the harness's own analysis adds, `emoctx − asr` (pp; Holm p over the 12
comparisons per dataset in parentheses):

| student | Social-IQ | IntentQA |
|---|---|---|
| Qwen3-VL-8B | −3.77 (0.056) | +0.47 (0.69) |
| Qwen3-VL-4B | −3.14 (0.089) | +1.87 (0.085) |
| Qwen3.5-4B | +0.88 (0.53) | −1.50 (0.26) |
| Gemma-4-E4B | −3.77 (0.089) | +1.03 (0.69) |

- `emoctx` beats `none` in 7 of 8 cells, but the transcript alone carries
  most of that gain. On Social-IQ, `asr − none` is +5.9 to +9.3 pp, with
  Holm p ≤ 0.008.
- The harness's own analysis survives Holm nowhere. On Social-IQ it is
  negative for 3 of 4 students.
- The emotion-question subset gives the same picture. It is picked by a
  keyword rule on the question text, fixed before computing, and has 262 core
  questions on Social-IQ.
  - On those questions, `emoctx − asr` is:

    | student | pp |
    |---|---|
    | Qwen3-VL-8B | +0.76 |
    | Qwen3-VL-4B | −1.53 |
    | Qwen3.5-4B | −0.38 |
    | Gemma-4-E4B | −4.58 |

    No interval lies above zero.
  - For the two Qwen3-VL students, the drop sits on the non-emotion questions.

### Protocol details that move the numbers

- **Answer parsing.** A reply counts only if it commits to a letter: a
  leading letter, an `<answer>` tag, an explicit "answer is X", or a last line
  that is only the letter. A letter merely mentioned in the analysis never
  counts. Unparsed replies score as wrong and are counted per condition;
  `tests/unit/experiment_scripts/` pins these rules.
- **Qwen3.5-4B output budget.** With thinking off it still reasons before
  answering, so it gets its config's 1024-token budget instead of 64. Its three
  conditions share one prompt, so they stay comparable with each other.
- **Context length.** For the Qwen models, 32 frames of 25 IntentQA videos
  exceed `max_model_len` 16384, and vLLM rejects the call; the same 100
  questions are rejected in every condition. `topup` re-sends only those calls
  at 32768. Errors are never cached, so every successful call is reused
  unchanged.
- **Two hosts, one results tree.** `owner.json` names the only host allowed to
  write; status and logs are per host.

## Layout

```
experiments/
├── common.py            # the four students, their bundles, one owned vLLM server
├── transfer/            # manifests, row, run, report
└── emotion_context/     # manifests, row, run, topup, report, subset
```

Outputs go under `results/transfer/` and `results/emotion_context/`, which are
not tracked by git.
