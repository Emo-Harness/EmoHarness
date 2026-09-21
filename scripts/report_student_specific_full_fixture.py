#!/usr/bin/env python3
"""Synthetic arms for exercising scripts/report_student_specific_full.py.

The report gates publication, so its guards need to have been fired at least
once before the real arms land. This writes a throwaway batch root with eight
arms and three planted faults -- an L5 that reused the 8B transfer bundle, an
arm missing 200 items, and a prediction for a media-excluded id -- and rising
accuracy along the ladder so the comparison signs are checkable.

    scripts/report_student_specific_full_fixture.py /tmp/fakebatch
    scripts/report_student_specific_full.py --batch /tmp/fakebatch --resamples 400

Expected: qwen3vl_8b compares cleanly and warns about the stray id; gemma4_e4b
withholds the two L5 differences as FATAL and the two L1 differences for
covering different ids. Never point this at results/v3/.
"""
import json
import random
import shutil
import sys
from pathlib import Path

REAL = Path('/home/max7/EmoHarness/results/v3/student_specific_full_20260918')
FAKE = Path(sys.argv[1])
FAKE.mkdir(parents=True, exist_ok=True)
shutil.copy(REAL/'evaluation_items.jsonl', FAKE/'evaluation_items.jsonl')
shutil.copy(REAL/'evaluation_excluded_ids.json', FAKE/'evaluation_excluded_ids.json')
(FAKE/'.done').mkdir(exist_ok=True)

items = [json.loads(line) for line in (FAKE/'evaluation_items.jsonl')
         .read_text().splitlines() if line.strip()]
excluded = json.loads((FAKE/'evaluation_excluded_ids.json').read_text())['ids']
TRANSFER = '569c76f9edc0e2e6473fdad9f8854fdfa473fcf94964961d48877a338478e9e2'
ACC = {'L0': 0.30, 'L1': 0.36, 'L3_scaffold': 0.44, 'L5_own': 0.47}

def predict(item, correct, rng):
    if correct:
        return [item['labels'][0]]
    wrong = [x for x in item['label_set'] if x not in item['labels']]
    return [rng.choice(wrong)] if wrong else [item['labels'][0]]

for model in ('qwen3vl_8b', 'gemma4_e4b'):
    for slot in ('L0', 'L1', 'L3_scaffold', 'L5_own'):
        rng = random.Random(f'{model}/{slot}')  # noqa: S311 - fixture data
        rows = items[:-200] if (model, slot) == ('gemma4_e4b', 'L1') else items
        out = FAKE/model/slot/'full5507'/'attempt_01'/'raw_predictions'/slot
        out.mkdir(parents=True, exist_ok=True)
        path = out/'full5507.jsonl'
        with path.open('w') as fh:
            for item in rows:
                ok = rng.random() < ACC[slot]
                fh.write(json.dumps({
                    'example_id': item['example_id'], 'row': slot,
                    'task': item['task'], 'labels': predict(item, ok, rng),
                    'response_text': f'<answer>{predict(item, ok, rng)[0]}</answer>',
                    'target_calls': 1 if slot in ('L0','L1') else 4,
                    'parse_failures': 0, 'input_tokens': 5000, 'output_tokens': 400,
                    'latency_s': 3.0, 'cost_usd': 0.0})+'\n')
            if (model, slot) == ('qwen3vl_8b', 'L0'):   # planted: excluded id leaked in
                fh.write(json.dumps({
                    'example_id': excluded[0], 'row': slot, 'task': 'ER_Wild',
                    'labels': ['Neutral'], 'response_text': '<answer>Neutral</answer>',
                    'target_calls': 1, 'parse_failures': 0, 'input_tokens': 1,
                    'output_tokens': 1, 'latency_s': 0.1, 'cost_usd': 0.0})+'\n')
        rd = FAKE/model/slot/'full5507'/'attempt_01'
        (rd/'ledger.jsonl').write_text('')
        bundle_hash = TRANSFER if slot == 'L5_own' else None
        marker = {
            'utc': '2026-09-18T05:00:00Z', 'model': model, 'hf_id': f'x/{model}',
            'row': slot, 'slot': slot, 'items': 'full5507', 'attempt': 1,
            'results_dir': str(rd), 'log': 'logs/x.log', 'returncode': 0,
            'status': 'ok', 'seconds': 1234.5,
            'bundle': str(rd/'final') if slot == 'L5_own' else None,
            'predictions': str(path),
            'item_set_file': str(FAKE/'evaluation_items.jsonl'),
            'n_items_expected': len(items), 'n_items_written': len(rows),
            'provenance': {'git_rev': 'deadbeef', 'vllm_version': '0.19.1',
                           'bundle_hash': bundle_hash},
        }
        (FAKE/'.done'/f'{model}__{slot}__full5507.done').write_text(
            json.dumps(marker, indent=1))
print(FAKE)
