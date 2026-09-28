#!/usr/bin/env python3
"""Re-issue the QA calls that did not fit the published 16384-token context.

    setsid nohup .venv/bin/python -u scripts/emoctx_topup.py >> logs/emoctx/topup.$(hostname).log 2>&1 &

For some IntentQA clips the 32 cached frames alone come to ~22.5k tokens, so
vLLM rejected the QA request (HTTP 400, "decoder prompt is longer than the
maximum model length"). The main run keeps the published serving setting
(``--max-model-len 16384``) for every step, harness included; the harness sends
fewer images per call and is not affected. This pass restarts each student's
server with ``VLLM_MAX_MODEL_LEN=32768`` and re-runs only the QA steps whose
done.json records errors. The content-addressed cache stores responses, never
errors, so every call that already succeeded is served from the cache unchanged
and only the rejected calls reach the new server. Frames, prompts and decoding
are untouched.

The 16384 predictions and done.json are kept beside the new ones as
``*.maxlen16384.*``; the new done.json records what was re-issued.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_emoctx as main_run  # noqa: E402

base, RUN, HOST = main_run.base, main_run.RUN, main_run.HOST
MAX_MODEL_LEN = '32768'
base.STATUS = RUN / f'status.topup.{HOST}.json'
#: Separate logs, so the main run's step and serve logs are never overwritten.
main_run.LOGS = base.LOGS = main_run.LOGS / 'topup'
state, save, log, now = base.state, base.save, base.log, base.now


def pending(student: str) -> list[tuple[str, str]]:
    todo = []
    for cond in main_run.CONDITIONS:
        for ds in main_run.DATASETS:
            done = main_run.step_dir(student, 'qa', cond, ds) / 'done.json'
            if not done.is_file():
                continue
            meta = json.loads(done.read_text())
            if meta.get('n_errors') and not meta.get('topup'):
                todo.append((cond, ds))
    return todo


def topup_step(student: str, cond: str, ds: str) -> None:
    d = main_run.step_dir(student, 'qa', cond, ds)
    before = json.loads((d / 'done.json').read_text())
    for name in ('predictions.jsonl', 'done.json'):
        keep = d / name.replace('.', '.maxlen16384.', 1)
        if not keep.exists():
            shutil.copy2(d / name, keep)
    if not main_run.run_step(student, 'qa', cond, ds):
        log(
            f'{student} qa {cond} {ds}: top-up failed; the 16384 files are untouched copies'
        )
        return
    after = json.loads((d / 'done.json').read_text())
    after['topup'] = {
        'max_model_len': int(MAX_MODEL_LEN),
        'at': now(),
        'errors_before': before['n_errors'],
        'errors_after': after['n_errors'],
        'accuracy_before': before['accuracy'],
        'unparsed_before': before['n_unparsed'],
        'kept': ['predictions.maxlen16384.jsonl', 'done.maxlen16384.json'],
    }
    tmp = d / '.done.json.tmp'
    tmp.write_text(json.dumps(after, indent=2, ensure_ascii=False))
    tmp.replace(d / 'done.json')
    log(
        f'{student} qa {cond} {ds}: errors {before["n_errors"]} -> {after["n_errors"]}, '
        f'acc {before["accuracy"]:.4f} -> {after["accuracy"]:.4f}'
    )


def main() -> int:
    owner = json.loads((RUN / 'owner.json').read_text())
    if owner['host'] != HOST:
        raise SystemExit(f'owner.json names {owner["host"]}; this host is {HOST}')
    os.environ['VLLM_MAX_MODEL_LEN'] = MAX_MODEL_LEN
    main_run.LOGS.mkdir(parents=True, exist_ok=True)
    save()
    for student in base.ORDER:
        todo = pending(student)
        state['pairs'].clear()
        if not todo:
            log(f'{student}: nothing to top up')
            continue
        proc = None
        try:
            main_run.wait_for_gpu()
            state['phase'] = f'serving {student} at max_model_len {MAX_MODEL_LEN}'
            save()
            proc = base.start_server(student)
            for cond, ds in todo:
                state['phase'] = f'{student} top-up qa {cond} on {ds}'
                save()
                topup_step(student, cond, ds)
        except Exception as exc:  # noqa: BLE001 - recorded, next student continues
            state.setdefault('errors', []).append(
                {
                    'at': now(),
                    'student': student,
                    'error': f'{type(exc).__name__}: {exc}',
                }
            )
            log(f'{student}: {type(exc).__name__}: {exc}')
        finally:
            base.stop_server(proc)
    state['phase'] = 'finished'
    state['finished_at'] = now()
    save()
    log('top-up finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
