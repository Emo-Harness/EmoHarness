# Frozen per-student harnesses

Each student model has its own harness, built by the Builder (GPT 5.6 Luna)
from the shared scaffold in [`../harness/`](../harness/). The build uses a Stage 3
construction with `configs/builder/v3_full.yaml`, replicate 0, and at most 4
rounds, and the student itself is the Target. These four bundles are the ones
the evaluations in [`../experiments/`](../experiments/) serve. Every
experiment step refuses to run unless the bundle's content hash matches the
value below.

| bundle | student | target config | content hash |
|---|---|---|---|
| `qwen3vl_8b/` | Qwen3-VL-8B-Instruct | `configs/target/qwen3vl_8b_instruct.yaml` | `569c76f9edc0e2e6…` |
| `qwen3vl_4b/` | Qwen3-VL-4B-Instruct | `configs/target/qwen3vl_4b_instruct.yaml` | `4ace41173bb48fda…` |
| `qwen3_5_4b/` | Qwen3.5-4B | `configs/target/qwen3_5_4b.yaml` | `d8b0b2794c4c95fe…` |
| `gemma4_e4b/` | Gemma-4-E4B-it | `configs/target/gemma4_e4b_it.yaml` | `468afcba8c97dd1e…` |

The full hashes are in `experiments/common.py`. The scaffold hashes to
`b38c518b7e4161a3…`. In Qwen3.5-4B's construction no candidate patch passed the
gates, so its bundle behaves exactly like the scaffold; only its manifest
metadata differs.

Check a bundle:

```bash
python -c "from pathlib import Path; from emoharness.harness.loader import load_bundle; print(load_bundle(Path('bundles/qwen3vl_4b')).content_hash())"
```

Rebuild one (with the student served by `scripts/serve_student.sh`,
`OPENAI_API_KEY` set, and the external corpus prepared; this makes paid
Builder calls):

```bash
EMOHARNESS_BUILDER_CONFIG=v3_full.yaml EMOHARNESS_EXTERNAL_MANIFEST=external_v3.jsonl \
EMOHARNESS_RESULTS_DIR=results/construction/qwen3vl_4b \
python examples/build_harness.py --stage 3 --replicate 0 --rounds 4 \
    --bundle harness --target configs/target/qwen3vl_4b_instruct.yaml
```

A bundle is a directory of prompts, YAML routing and sampling settings,
schemas, and verification code. Never edit a frozen bundle in place: any change
makes it a different harness with a different hash.
