# Provenance: the student-specific full evaluation published 2026-09-19

This branch holds the code for the evaluation behind the published results page
(16 arms x 5,507 items = 88,112 prediction records, four student models).

## What this branch is, exactly

* The **121 input files pinned by the run itself**, listed with sha256 in
  `results/v3/student_specific_full_20260918/execution_manifest.json`
  (`input_sha256`): the harness bundle, the four target configs, the builder
  configs, the data manifests and the whole executed `src/emoharness` tree.
* The scripts that ran, reported and published that evaluation, chosen by what
  the run's own artifacts reference rather than by guesswork:
  `run_student_specific_full.py`, `student_full_row.py`,
  `serve_student_full_20260918.sh`, `budget_admission*.py`, the 09-18
  preprocessing and construction launchers, `report_student_specific_full*.py`,
  `build_student_full_results_page.py`, `student_full_web_examples.py`, and the
  `supervise_student_specific_full*` monitors with their tests.
* The published site payload (`results/web_update_20260919/`), whose data files
  match the deployment record hash for hash.

## What this branch is NOT

**It is not a byte-exact reconstruction of the code that produced those
numbers, and it cannot be.** The last commit before this one is dated
2026-09-12; the evaluation ran on 09-18/09-19 and was never committed. Of the
121 pinned inputs, **105 still match their pinned hash today and 16 do not**.

Fourteen of the sixteen drifted on their own: later optimizer work (v5-v8)
edited them in place after the evaluation had already been generated at
17:34:46Z on 09-19. Their 09-19 content is not recoverable -- it is in no
commit, and none of the v5, v6 or v7 `source_snapshot/` directories preserves
it (checked, 0 of 14 match).

The fourteen that drifted before this branch existed:

* `src/emoharness/builder/config.py`
  pinned `7ad7cada75b17418` -> now `294178dee544cd66`
* `src/emoharness/builder/gates.py`
  pinned `23967af1d7553fc3` -> now `71411a7206343fa9`
* `src/emoharness/builder/loop.py`
  pinned `26483e88ae2ad090` -> now `45c12a302df236a9`
* `src/emoharness/builder/packet.py`
  pinned `f12fcde89e3e3a8f` -> now `12f4a1437b1e36b4`
* `src/emoharness/builder/schema.py`
  pinned `80cadd67e57e145c` -> now `289f07b39f25a1bf`
* `src/emoharness/builder/verifier.py`
  pinned `c95df50f76caeaa2` -> now `c3f15d69304011c2`
* `src/emoharness/harness/_stages.py`
  pinned `3c107468dfd2ec68` -> now `e89e0418f2488d8e`
* `src/emoharness/harness/_verification.py`
  pinned `e8b49e92c86468cf` -> now `357b7f24579e45a3`
* `src/emoharness/harness/bundle_models.py`
  pinned `f8577c52ab83f7ea` -> now `3b8ce1e2512a36fd`
* `src/emoharness/harness/fake_responder.py`
  pinned `09dbb8c82b21e176` -> now `1a4e99de4855be1b`
* `src/emoharness/harness/loader.py`
  pinned `f1eace56fd664d24` -> now `04429c39a1baa507`
* `src/emoharness/harness/pipeline.py`
  pinned `d9e0c996fcbd94b6` -> now `9b707b7bf8d0cbe6`
* `src/emoharness/runner/construction.py`
  pinned `ebe2490e4f87613b` -> now `0931451e677979fb`
* `src/emoharness/runner/context.py`
  pinned `09438298cb8417ce` -> now `96e9b51f74567c4a`

The other two I changed myself, to get the branch through this repository's
own CI (ruff, mypy --strict, pytest, codespell). Both changes are inert:

* `src/emoharness/target/config.py`
  pinned `df04a3dd6029283d` -> now `eec3b0f0381b1036`
  import order only: `Literal, Any` -> `Any, Literal`, plus one blank line
* `src/emoharness/evaluation/official.py`
  pinned `2ac08760de247815` -> now `0bdf27b3f0d4d49e`
  "unparseable" -> "unparsable" inside a log-message string

The codespell hit predates this work -- it is present at `7c3eca3`, the base
of this branch -- so fixing it here is arguably out of scope; I did it anyway
rather than open a pull request with red CI, and record the cost above.

One file had to be added that the pin list does not mention:
`src/emoharness/harness/_mechanism_screen.py`. Today's `pipeline.py` imports
it, and without it a clean checkout of this branch cannot import `emoharness`
at all. It is v5-era code, so its presence is another way of saying that the
`src` tree here is today's code, not 09-19's.

The other 105 pinned inputs verify clean against `input_sha256` as of this
commit. To re-check at any time:

```python
import hashlib, json
from pathlib import Path
pins = json.loads(Path(
    "results/v3/student_specific_full_20260918/execution_manifest.json"
).read_text())["input_sha256"]
bad = [r for r, w in pins.items()
       if hashlib.sha256(Path(r).read_bytes()).hexdigest() != w]
```

## Deliberately excluded

* `results/v3/student_specific_full_20260918/` prediction and log trees (466 MB)
  -- data, not code. Only the run's three manifests are kept, because the
  provenance claims above are checked against them.
* The 12 dataset media files (`.mp4`, `.jpg`) under the site payload's `media/`.
  They are clips and frames from the source benchmarks; republishing them from
  this repository is a licensing decision, not a packaging one. The page's
  example gallery will not render from this branch alone.
* `coordination/` -- inter-session mailbox and internal review notes.

## CI

On a clean checkout of this branch: ruff check and ruff format clean, mypy --strict clean over 99 source files, codespell clean, and 998 passed / 2 skipped with `PYTHONPATH` forced onto this branch's `src`. That last detail matters: the venv installs `emoharness` from the main checkout, so a plain `pytest` run in a worktree silently tests the main tree instead of the branch.
* The v4-v8 optimizer code, which remains uncommitted in the working tree.

## Result this code produced, for the record

| student | L0 | L5 | L5-L0 | own harness? |
|---|---|---|---|---|
| Qwen3-VL-8B | 29.95 | 40.68 | +10.72 | yes, built in an earlier batch and reused here |
| Qwen3-VL-4B | 29.39 | 38.56 | +9.17 | yes, 4 rounds, 2 accepted patches |
| Qwen3.5-4B | 29.62 | 37.23 | +7.61 | **no** -- 4 rounds, 0 accepted; its bundle differs from the shared scaffold only in manifest metadata |
| Gemma-4-E4B | 27.03 | 40.21 | +13.18 | yes, 4 rounds, 2 accepted patches |

No L5-L3 increment survives Holm correction (p = 1.0, 0.144, 1.0, 1.0). Most of
the gain over the bare model arrives at L1, with the fixed audio tools.
