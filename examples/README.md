# Entry points

One runnable step per file, in phase order (`docs/plan.md`). Every one takes
`--root` (where `data/` and `results/` live), `--fake` (run with fakes: no GPU,
no data, no API keys), and `--force` (overwrite existing artifacts). Each has a
matching wrapper in `scripts/`; `scripts/run_all_fake.sh` runs the whole list.

| Phase | Step | What it does |
|---|---|---|
| 1 | `prepare_benchmark.py` | Builds `data/manifests/mme_emotion.jsonl`, its `.sha256` checksums, and its annotation report; `--fake` generates a synthetic MME-Emotion tree under `data/fake/` first. |
| 2 | `prepare_external.py` | Loads whichever construction corpora are present, assigns build/val/ext-test by source id, and reports per-source label histograms. |
| 2 | `leakage_report.py` | Runs the four leakage checks over both manifests and writes `data/checksums/leakage_report.json`; `--drop` removes flagged clips. |
| 3 | `preprocess.py` | Fills the shared clip cache under `data/derived/`, or `--verify` checks the one that exists. |
| 5 | `preregister.py` | Draws the pre-registered subset and renders `docs/preregistration.md` from the artifacts it freezes. |
| 4, 7 | `run_ladder.py` | Runs ladder rows (`--rows L0,L3`) over an item set (`--items full\|clean\|subset\|external:val`; `clean` drops the 70 benchmark items whose media also appears in a construction split) and writes raw predictions; L6 is the teacher Direct upper bound. |
| 6 | `build_harness.py` | Runs one construction stage's Builder loop; `--freeze` records the freeze, evaluates D_ext-test once, and prints the git tag command. |
| 5 | `calibrate_proxy_judge.py` | Measures proxy-versus-reference judge agreement and derives the Reasoning gate margin. |
| 7 | `judge_official.py` | Writes the official judge's per-task inputs and prints its commands; `--execute` runs them. |
| 8 | `analyze.py` | Builds `ladder.json`/`ladder.md`, `per_task.json`, `hypotheses.json`, `multilabel.json`, and `cost.json`. |

## API keys

`--fake` needs none. A real run reads `OPENAI_API_KEY` for the Builder
(`build_harness.py`), for the teacher row L6 (`run_ladder.py`), and for the
official judge (`judge_official.py`); `OPENAI_ENDPOINT` overrides the base URL
of the first two. `GEMINI_API_KEY` is needed only by the proxy judge
(`calibrate_proxy_judge.py` and the Reasoning gate inside `build_harness.py`),
or when a Builder or teacher config selects `provider: gemini`. `VLLM_ENDPOINT`
points at the served Target.

Order matters: `leakage_report.py` reads the preprocessing cache, so run
`preprocess.py` first if you want its visual and transcript checks to see
anything; `preregister.py` needs both manifests and the leakage report;
`analyze.py` needs raw predictions and, for judged scores, `judge_official.py`.
