# EmoHarness

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Can a strong Builder model write an inference-time harness that makes a small,
frozen multimodal model better at emotion understanding, without touching its
weights?**

## Introduction

A **harness** is a directory of prompts, routing rules, schemas and small
verification programs. It turns one question about a clip into four calls to a
frozen student model, over evidence from fixed perception tools. The
**Builder** (GPT 5.6 Luna) improves a harness for one student by proposing
patches from its failures on a construction corpus drawn from outside
MME-Emotion. It keeps only the patches that pass the gates: a local
Recognition gain, Reasoning held above a floor, and the call budget. The
student's weights never change.

Methods form a ladder, so each ingredient is measured separately:

| row | what answers the question |
|---|---|
| **L0** | the student on the native video, one call |
| **L1** | L0 plus the transcript and prosody as text |
| **L2** | L1 with a frozen handcrafted prompt |
| **L3** | the shared four-call scaffold in [`harness/`](harness/) |
| **L4** | the Builder's Stage 2 bundle |
| **L5** | the Builder's Stage 3 bundle, one per student |
| **L6** | the teacher model on the same frames and audio text, as an upper bound |

Four students have their own Stage 3 harness in [`bundles/`](bundles/):
Qwen3-VL-8B, Qwen3-VL-4B, Qwen3.5-4B and Gemma-4-E4B. The research plan and
design decisions are in [`docs/research-plan.md`](docs/research-plan.md) and
[`docs/adr/`](docs/adr/).

## Installation

```bash
conda create -n emoharness python=3.11 -y
conda activate emoharness
curl -sSL https://install.python-poetry.org | python3 -
git clone https://github.com/4R5T/EmoHarness.git
cd EmoHarness
poetry install --with dev,test          # add --with api,audio for the real adapters
```

The students are served with vLLM from a separate environment. Use vLLM 0.11.2
for Qwen3-VL, and 0.19.1 for Qwen3.5 and Gemma 4, which 0.11.2 does not
register. Point `EMOHARNESS_VLLM_VENV` at it.

## Setup

| variable | used for |
|---|---|
| `VLLM_ENDPOINT` | the served student (default `http://127.0.0.1:8000/v1`) |
| `OPENAI_API_KEY` | the Builder and teacher (GPT 5.6 Luna), and the official MME-Emotion judge |
| `GEMINI_API_KEY` | the proxy judge behind the Reasoning gate |
| `MME_EMOTION_ROOT` | the downloaded MME-Emotion tree |

Serve a student:

```bash
EMOHARNESS_STUDENT_MODEL=Qwen/Qwen3-VL-4B-Instruct \
EMOHARNESS_VLLM_VENV=~/.venvs/emoharness-vllm011 scripts/serve_student.sh
```

## Get started

The whole pipeline runs on fakes in a few seconds, with no GPU, data, API keys
or network:

```bash
scripts/run_all_fake.sh                 # writes to /tmp/emoharness-fake
```

Real runs follow the phases below. Each phase is `examples/<step>.py`, with a
wrapper in `scripts/<step>.sh`. Every step accepts `--root` and `--fake`. Every
step also accepts `--force`; without it, a step refuses to overwrite an
existing artifact.

| phase | command | needs |
|---|---|---|
| 1 | `scripts/prepare_benchmark.sh` | `MME_EMOTION_ROOT` |
| 2 | `scripts/prepare_external.sh`, then `scripts/leakage_report.sh` | the corpora in `configs/data/sources.yaml` |
| 3 | `scripts/preprocess.sh` | GPU, ffmpeg |
| 4 | `EMOHARNESS_ROWS=L0,L1,L3 EMOHARNESS_ITEMS=clean scripts/run_ladder.sh` | a served student |
| 5 | `scripts/preregister.sh`, `scripts/calibrate_proxy_judge.sh` | both manifests; `GEMINI_API_KEY`, `OPENAI_API_KEY` |
| 6 | `EMOHARNESS_STAGE=3 scripts/build_harness.sh` | `OPENAI_API_KEY`, `GEMINI_API_KEY`, a served student |
| 7 | `EMOHARNESS_ROW=L5 scripts/judge_official.sh --execute` | `OPENAI_API_KEY`, a checkout of `QwenAudio/MME-Emotion` |
| 8 | `scripts/analyze.sh` | nothing |

Item sets:

| set | items |
|---|---|
| `full` | every benchmark item |
| `clean` | every benchmark item except the 70 whose media also appears in a construction split |
| `subset` | the 1,000 pre-registered items |
| `external:<split>` | one construction split |

[`examples/README.md`](examples/README.md) documents each step, and
[`docs/run-indist.md`](docs/run-indist.md) records a complete run.

## Experiments

| experiment | where | finding |
|---|---|---|
| Per-student harnesses on MME-Emotion (5507 items) | `examples/`, `--items clean` | L1 − L0 is +5.4 to +10.7 pp; L5 − L3 survives Holm for no student |
| Transfer to IntentQA / Social-IQ 2.0 | `experiments/transfer/` | no L5 − L0 gain; −11.9 and −9.6 pp on IntentQA for two students |
| Harness emotion context as input to the bare model | `experiments/emotion_context/` | helps only through the transcript; −3 to −4 pp beyond it on Social-IQ for 3 of 4 students |

Commands, tables and protocol details are in
[`experiments/README.md`](experiments/README.md).

## Package structure

```
EmoHarness/
├── src/emoharness/
│   ├── data/          # tasks, the MME-Emotion prompt grammar, manifests, splits, leakage checks
│   ├── perceive/      # frame sampling, Whisper ASR, openSMILE prosody, the clip cache
│   ├── harness/       # the staged inference pipeline that runs a bundle
│   ├── builder/       # diagnosis packets, candidate patches, gates, provenance
│   ├── target/        # vLLM (OpenAI-compatible) and Gemini clients, the call cache
│   ├── baselines/     # single-call rows L0–L2 and best-of-n
│   ├── evaluation/    # ladder, scoreboard, bootstrap, official and proxy judges
│   ├── runner/        # paths, item sets, rows, construction, analysis, CLI plumbing
│   └── common/        # I/O, hashing, logging, cost ledger, errors
├── harness/           # the shared Stage 1 scaffold
├── bundles/           # the four frozen per-student Stage 3 harnesses
├── configs/           # builder, target, data, ladder and preprocessing configs
├── data/              # manifests, splits, checksums (raw media is not tracked)
├── examples/          # one entry point per pipeline phase
├── scripts/           # wrappers for examples/, serving, end-to-end drivers
├── experiments/       # the out-of-distribution evaluations
├── tests/             # unit and integration tests, all on fakes
└── docs/              # research plan, ADRs, engineering notes, run records
```

Run outputs go to `results/`, which is not tracked by git.

## Contributing

Read [`docs/engineering.md`](docs/engineering.md) before writing code and
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request, and run
the same checks as CI:

```bash
poetry run ruff check src tests examples experiments
poetry run ruff format --check src tests examples experiments
poetry run mypy
poetry run pytest -m 'not integration'
```

## License

[MIT](LICENSE)
