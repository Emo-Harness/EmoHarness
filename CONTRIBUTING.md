# Contributing

Every change goes through a pull request that passes the checks below.

## Checks

```bash
poetry install --with dev,test
poetry run ruff check src tests examples experiments
poetry run ruff format --check src tests examples experiments
poetry run mypy                                  # strict
poetry run pytest -m 'not integration'
poetry run pytest -m integration                 # the whole pipeline on fakes
```

## Code

- Python 3.10+ with `from __future__ import annotations` and full type hints;
  `mypy --strict` must pass. Single quotes and a line length of 88 (ruff).
- Pydantic models are `frozen=True`. Functions return new objects and never
  mutate their inputs.
- Protocol facts and thresholds live in `emoharness.common.constants` or
  `configs/`, nowhere else.
- Validate at the boundary: everything read from disk, an API or a model
  reply is parsed into a pydantic model or rejected with a `ValidationError`.
- Errors derive from `emoharness.common.errors.EmoHarnessError`. Never
  swallow an exception.
- Every billable call records a ledger entry.
- Heavy runtime dependencies are imported lazily inside the function that
  needs them. This covers vLLM, faster-whisper, openSMILE and the API SDKs.
- The harness never branches on a task name; it only sees the prompt.

## Tests

- `tests/unit/<subpackage>/test_<module>.py` mirrors
  `src/emoharness/<subpackage>/<module>.py`.
- Use fakes, not mocks. Each seam ships its fake next to its real adapter, and
  only `runner/context.py` chooses between them.
- Tests generate their media in `tmp_path` and never touch the network, a GPU
  or an API key.

## Entry points and artifacts

- `examples/<group>/<step>.py` is one pipeline step: argparse, one `main()`, and
  a `--fake` mode. `scripts/<group>/<step>.sh` wraps it with the paper's
  settings as defaults, and `emoharness <command>` runs it
  (`src/emoharness/__main__.py` holds the command table).
- The paper's experiments live in `experiments/`; each pipeline stage is a
  method of `experiments/paper/pipeline.py` and an `emoharness` command.
- Steps write under `data/` and `results/`, neither tracked by git.
  They refuse to overwrite an artifact without `--force`.
- Never edit a frozen bundle in `harness/` or `bundles/`: any change is a new
  bundle with a new content hash.
