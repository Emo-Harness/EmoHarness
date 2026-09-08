# Engineering conventions

These rules apply to every file under `src/`, `tests/`, and `examples/`. Subagents and humans follow them alike.

## Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate poetry-env   # poetry lives here
poetry install --with dev,test                                            # once
poetry run pytest                                                         # tests with coverage
poetry run mypy                                                           # strict, configured in pyproject
poetry run ruff check src tests examples && poetry run ruff format --check src tests examples
```

Heavy runtime dependencies (vLLM, faster-whisper, openSMILE, pyannote, google-genai, openai) are optional. Import them lazily inside the function that needs them and raise a clear `EmoHarnessError` naming the missing package. Tests never import them and never touch the network.

## Code

- Python 3.10+, `from __future__ import annotations`, full type hints, `mypy --strict` clean.
- Single quotes (ruff format), line length 88, imports sorted by ruff.
- Immutability: pydantic models are `frozen=True`; functions return new objects; no in-place mutation of inputs; prefer tuples over lists in models.
- Functions under 50 lines; files under 400 lines; no nesting deeper than four levels.
- No hardcoded protocol facts or thresholds outside `emoharness.common.constants` and `configs/`.
- Validate at boundaries: everything read from disk, an API, or a model response is parsed into a pydantic model or rejected with `ValidationError`.
- Errors derive from `emoharness.common.errors.EmoHarnessError`; never swallow exceptions silently; log context with `emoharness.common.logging.get_logger`.
- Every billable call records a `LedgerEntry` (`emoharness.common.ledger`).
- No module reads `audio_clue/` files except the official-judge wrapper; a test asserts this.
- No module branches on a task name at inference time; the harness only sees the prompt (`emoharness.data.prompt_template.parse_prompt`).

## Contracts (do not change without updating every consumer)

| Module | Owns |
|---|---|
| `emoharness.data.tasks` | `Task`, `Group`, `Kind`, `Split`, task groups |
| `emoharness.data.prompt_template` | official prompt render/parse, `Quantifier`, `Cardinality`, `ParsedPrompt` |
| `emoharness.data.label_set` | `LabelSet`: candidate lookup, answer parsing, cardinality, Recognition, precision/recall/F1 (ADR-0014) |
| `emoharness.data.manifest` | `Example`, `read_manifest`, `write_manifest` |
| `emoharness.harness.schema` | `EvidenceRecord`, `HypothesisSet`, `VerificationResult`, `HarnessOutput`, `StageTrace` |
| `emoharness.target.base` | `TargetRequest` (with `tag`), `TargetResponse`, `TargetModel` protocol |
| `emoharness.builder.schema` | `CandidatePatch`, `BuilderProposal`, `CandidateScores` |
| `emoharness.evaluation.schema` | `PredictionRecord`, `StepScores`, `MethodScores` |
| `emoharness.evaluation.scoreboard` | `Scoreboard`: the only way scores reach the ladder, hypotheses, Builder gate, and calibration (ADR-0014) |
| `emoharness.common.*` | constants, errors, io, hashing, logging, ledger, stats, text, jsonish, `config.load_model` (the only YAML loader) |

## Tests

- `tests/unit/<subpackage>/test_<module>.py` mirrors `src/emoharness/<subpackage>/<module>.py`.
- Fakes over mocks, and every seam owns its fakes next to its real adapters (`target/fake.py`, `harness/fake_responder.py`, `perceive/fakes.py`, `builder/fakes.py`, `evaluation/fakes.py`, `data/fakes.py`). Only the composition root `runner/context.py` imports a fake; `TargetRequest.tag` tells a fake which stage or row is asking, so no fake sniffs another module's prompts.
- Fixtures in `tests/conftest.py`; small synthetic media are generated in `tmp_path`, never committed.
- Property tests with `hypothesis` for parsers and metrics.
- Coverage target 80% per subpackage.

## Entry points

- `examples/<step>.py`: argparse, one `main()` that builds configs, calls library functions, writes artifacts, and returns an exit code. A `--fake` flag runs the step with fakes so that the whole pipeline can be exercised without GPUs, data, or keys.
- `scripts/<step>.sh`: activates the environment and calls the matching example with sensible defaults; every argument can be overridden by environment variables.
- Artifacts go under `data/` or `results/`; never overwrite an existing artifact without `--force`.
