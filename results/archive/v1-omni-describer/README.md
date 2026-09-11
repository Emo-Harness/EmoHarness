# v1 run, 2026-09-08/09: with the Qwen2.5-Omni-7B audio describer

This is the first in-distribution effect check, moved here unchanged on
2026-09-11 when the audio-language describer (cue description + opinion
channel) was removed from the harness and the experiment was rerun with
deterministic tools only, a per-round verifier, and per-round cost records.
The numbers in `docs/run-indist.md` under "v1" come from these files:

- `construction/stage2/{0,1,2}/`: three Luna Builder replicates (4 / 2 / 2 rounds).
- `raw_predictions/`: L0, L3, L4, L6 on the 1,000-item subset.
- `metrics/`: `ladder.md`, `per_task.json`, bootstrap deltas, `cost.json`.
- `ledger.jsonl` (machine 1), `ledger.m2.jsonl` (machine 2): every model call.
- `indist/`: the step markers of that run.

Overall local Recognition on the subset: L0 29.2 / L3 41.3 / L4 41.5 / L6 44.4.
Luna spend: $5.42.
