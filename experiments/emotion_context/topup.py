"""Re-issue the QA calls that did not fit the 16384-token serving context.

    setsid nohup .venv/bin/python -u -m experiments.emotion_context.topup \\
        >> logs/emotion_context/topup.$(hostname).log 2>&1 < /dev/null &

For some IntentQA clips the 32 cached frames alone come to ~22.5k tokens, so
vLLM rejects the QA request (HTTP 400, "decoder prompt is longer than the
maximum model length"). The main run keeps the main evaluation's serving
setting (``--max-model-len 16384``) for every step, harness included; the
harness sends fewer images per call and is not affected. This pass restarts
each student's server with ``VLLM_MAX_MODEL_LEN=32768`` and re-runs only the QA
steps whose done.json records errors. The content-addressed cache stores
responses, never errors, so every call that already succeeded is served from
the cache unchanged and only the rejected calls reach the new server. Frames,
prompts and decoding are untouched.

The 16384 predictions and done.json are kept beside the new ones as
``*.maxlen16384.*``; the new done.json records what was re-issued.
"""

from __future__ import annotations

import json
import shutil
import sys
from typing import Final

from experiments.common import ORDER, Driver, now, write_json
from experiments.emotion_context.run import (
    CONDITIONS,
    DATASETS,
    HOST,
    LOGS,
    NEED_FREE_MIB,
    RUN,
    row_command,
    step_dir,
)

MAX_MODEL_LEN: Final = '32768'


def pending(student: str) -> list[tuple[str, str]]:
    todo = []
    for cond in CONDITIONS:
        for ds in DATASETS:
            done = step_dir(student, 'qa', cond, ds) / 'done.json'
            if not done.is_file():
                continue
            meta = json.loads(done.read_text())
            if meta.get('n_errors') and not meta.get('topup'):
                todo.append((cond, ds))
    return todo


def topup_step(driver: Driver, student: str, cond: str, ds: str) -> None:
    d = step_dir(student, 'qa', cond, ds)
    before = json.loads((d / 'done.json').read_text())
    for name in ('predictions.jsonl', 'done.json'):
        keep = d / name.replace('.', '.maxlen16384.', 1)
        if not keep.exists():
            shutil.copy2(d / name, keep)
    if not driver.run_step(
        str(d.relative_to(RUN)), row_command(student, 'qa', cond, ds), d
    ):
        driver.log(
            f'{student} qa {cond} {ds}: top-up failed; the 16384 copies are kept'
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
    write_json(d / 'done.json', after)
    driver.log(
        f'{student} qa {cond} {ds}: errors {before["n_errors"]} -> {after["n_errors"]}, '
        f'acc {before["accuracy"]:.4f} -> {after["accuracy"]:.4f}'
    )


def main() -> int:
    owner = json.loads((RUN / 'owner.json').read_text())
    if owner['host'] != HOST:
        raise SystemExit(f'owner.json names {owner["host"]}; this host is {HOST}')
    driver = Driver(RUN / f'status.topup.{HOST}.json', LOGS / 'topup')
    driver.save()
    for student in ORDER:
        todo = pending(student)
        driver.state['pairs'].clear()
        if not todo:
            driver.log(f'{student}: nothing to top up')
            continue
        proc = None
        try:
            driver.wait_for_vram(NEED_FREE_MIB)
            driver.set_phase(f'serving {student} at max_model_len {MAX_MODEL_LEN}')
            proc = driver.start_server(student, {'VLLM_MAX_MODEL_LEN': MAX_MODEL_LEN})
            for cond, ds in todo:
                driver.set_phase(f'{student} top-up qa {cond} on {ds}')
                topup_step(driver, student, cond, ds)
        except Exception as exc:  # noqa: BLE001 - recorded, next student continues
            driver.record_error(student, exc)
        finally:
            driver.stop_server(proc)
    driver.state['finished_at'] = now()
    driver.set_phase('finished')
    driver.log('top-up finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
