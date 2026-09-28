#!/usr/bin/env python3
"""Drive the new-dataset transfer evaluation: four students x two datasets.

    setsid nohup .venv/bin/python scripts/run_newdata.py > logs/newdata/driver.log 2>&1 &

One owned vLLM server at a time, started exactly the way the 2026-09-19
evaluation started each student: the same serve script, the same serving venv
per architecture, the hub offline, and the student's pinned HF revision
verified against the local refs/main before launch. Each (student, dataset)
pair runs ``newdata_row.py`` with the student's own published L5 bundle.

Resumable and idempotent: a pair with a valid ``done.json`` is skipped, a pair
that failed is retried once (every finished Target call is in the shared
content-addressed cache, so a retry re-issues only unfinished calls; decoding
is temperature 0, so this is not resampling). Progress and health go to
``status.json`` after every phase change and every 60 s while a row runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'results/newdata/intentqa_socialiq_20260923'
LOGS = ROOT / 'logs/newdata'
STATUS = RUN / 'status.json'
PORT = 8000
ENDPOINT = f'http://127.0.0.1:{PORT}/v1'
STARTUP_S = 2400
MAX_ATTEMPTS = 2
DATASETS = ('social_iq_val', 'intentqa_test')
#: (ladder row, arm directory), in the published run's naming. L5 finished in
#: the first pass and is skipped by its done.json.
ROWS = (('L0', 'L0'), ('L1', 'L1'), ('L3', 'L3_scaffold'), ('L5', 'L5_own'))

STUDENTS = {
    'qwen3vl_8b': (
        'Qwen/Qwen3-VL-8B-Instruct',
        '.venvs/emoharness-vllm011',
        '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    ),
    'qwen3vl_4b': (
        'Qwen/Qwen3-VL-4B-Instruct',
        '.venvs/emoharness-vllm011',
        'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    ),
    'qwen3_5_4b': (
        'Qwen/Qwen3.5-4B',
        '.venvs/emoharness-vllm019',
        '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    ),
    'gemma4_e4b': (
        'google/gemma-4-E4B-it',
        '.venvs/emoharness-vllm019-tf513',
        'ee0ef6023621cff504d758262d4e04895a5af4a2',
    ),
}
ORDER = ('qwen3vl_8b', 'qwen3vl_4b', 'qwen3_5_4b', 'gemma4_e4b')

state: dict = {
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'host': os.uname().nodename,
    'driver_pid': os.getpid(),
    'pairs': {},
    'server': None,
    'phase': 'starting',
}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save() -> None:
    state['updated_at'] = now()
    tmp = STATUS.with_name(f'.status.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATUS)


def log(msg: str) -> None:
    print(f'[{now()}] {msg}', flush=True)


def lib_path() -> str:
    site = ROOT / '.venv/lib/python3.12/site-packages/nvidia'
    extra = f'{site}/cublas/lib:{site}/cudnn/lib'
    return extra + (
        ':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''
    )


def pair_dir(student: str, slot: str, dataset: str) -> Path:
    return RUN / student / slot / dataset


def is_done(student: str, slot: str, dataset: str) -> bool:
    marker = pair_dir(student, slot, dataset) / 'done.json'
    if not marker.is_file():
        return False
    preds = ROOT / json.loads(marker.read_text())['predictions']
    return preds.is_file()


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def port_busy() -> bool:
    import socket

    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex(('127.0.0.1', PORT)) == 0
    finally:
        s.close()


def verify_revision(hf_id: str, pinned: str) -> None:
    repo = (
        Path.home() / '.cache/huggingface/hub' / ('models--' + hf_id.replace('/', '--'))
    )
    ref = (repo / 'refs/main').read_text().strip()
    if ref != pinned or not (repo / 'snapshots' / pinned).is_dir():
        raise RuntimeError(f'{hf_id}: refs/main {ref[:12]} != pinned {pinned[:12]}')


def start_server(student: str):
    hf_id, venv, pinned = STUDENTS[student]
    verify_revision(hf_id, pinned)
    if port_busy():
        raise RuntimeError(f'port {PORT} is already serving; refusing to share it')
    env = {
        **os.environ,
        'EMOHARNESS_STUDENT_MODEL': hf_id,
        'EMOHARNESS_VLLM_VENV': str(Path.home() / venv),
        'VLLM_PORT': str(PORT),
        'VLLM_MAX_NUM_SEQS': '8',
        'VLLM_MM_CACHE_GB': '0',
        'HF_HUB_OFFLINE': '1',
    }
    out = (LOGS / f'serve_{student}.log').open('a')
    proc = subprocess.Popen(
        ['bash', str(ROOT / 'scripts/serve_student_full_20260918.sh')],
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=ROOT,
    )
    state['server'] = {
        'student': student,
        'pid': proc.pid,
        'pgid': os.getpgid(proc.pid),
        'model': hf_id,
        'revision': pinned,
        'venv': venv,
        'started_at': now(),
        'ready_at': None,
    }
    save()
    began = time.time()
    while not healthy():
        if proc.poll() is not None:
            raise RuntimeError(
                f'{student} server exited rc={proc.returncode} before ready'
            )
        if time.time() - began > STARTUP_S:
            stop_server(proc)
            raise RuntimeError(f'{student} server not ready within {STARTUP_S}s')
        time.sleep(10)
    state['server']['ready_at'] = now()
    state['server']['startup_s'] = round(time.time() - began, 1)
    save()
    log(f'{student} ready in {time.time() - began:.0f}s')
    return proc


def stop_server(proc) -> None:
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(2)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    for _ in range(30):
        if not port_busy():
            break
        time.sleep(2)
    state['server'] = None
    save()


def ledger_rows(student: str, slot: str, dataset: str) -> int:
    n = 0
    for p in pair_dir(student, slot, dataset).rglob('ledger*.jsonl'):
        with p.open() as fh:
            n += sum(1 for _ in fh)
    return n


def run_pair(student: str, row: str, slot: str, dataset: str) -> bool:
    key = f'{student}/{slot}/{dataset}'
    entry = state['pairs'].setdefault(key, {'attempts': 0, 'status': 'pending'})
    d = pair_dir(student, slot, dataset)
    d.mkdir(parents=True, exist_ok=True)
    while entry['attempts'] < MAX_ATTEMPTS:
        entry['attempts'] += 1
        entry.update({'status': 'running', 'started_at': now(), 'error': None})
        save()
        env = {
            **os.environ,
            'EMOHARNESS_RESULTS_DIR': str(d.relative_to(ROOT)),
            'VLLM_ENDPOINT': ENDPOINT,
            'EMOHARNESS_WORKERS': '8',
            'EMOHARNESS_VIDEO_WORKERS': '2',
            'LD_LIBRARY_PATH': lib_path(),
            'EMOHARNESS_PREPROCESS_CONFIG': 'preprocess.yaml',
        }
        logf = LOGS / f'row_{student}_{slot}_{dataset}_a{entry["attempts"]}.log'
        with logf.open('w') as out:
            proc = subprocess.Popen(
                [
                    str(ROOT / '.venv/bin/python'),
                    str(ROOT / 'scripts/newdata_row.py'),
                    '--dataset',
                    dataset,
                    '--student',
                    student,
                    '--row',
                    row,
                ],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
            entry['row_pid'] = proc.pid
            while proc.poll() is None:
                entry['target_calls_so_far'] = ledger_rows(student, slot, dataset)
                entry['server_healthy'] = healthy()
                save()
                time.sleep(60)
        entry['returncode'] = proc.returncode
        tail = logf.read_text()[-4000:].strip().splitlines()
        if proc.returncode == 0 and tail:
            prov = json.loads(tail[-1])
            (d / 'done.json').write_text(json.dumps(prov, indent=2))
            entry.update(
                {
                    'status': 'done',
                    'finished_at': now(),
                    'n_written': prov['n_written'],
                    'seconds': prov['seconds'],
                    'target_calls': ledger_rows(student, slot, dataset),
                }
            )
            save()
            log(f'{key} done: {prov["n_written"]} items in {prov["seconds"]}s')
            return True
        entry.update({'status': 'failed', 'error': '\n'.join(tail[-6:])})
        save()
        log(f'{key} attempt {entry["attempts"]} failed rc={proc.returncode}')
        if not healthy():
            return False
    return False


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    save()
    for student in ORDER:
        todo = [
            (row, slot, ds)
            for row, slot in ROWS
            for ds in DATASETS
            if not is_done(student, slot, ds)
        ]
        for row, slot in ROWS:
            for ds in DATASETS:
                if (row, slot, ds) not in todo:
                    state['pairs'].setdefault(f'{student}/{slot}/{ds}', {})[
                        'status'
                    ] = 'done'
        if not todo:
            continue
        state['phase'] = f'serving {student}'
        save()
        proc = None
        try:
            proc = start_server(student)
            for row, slot, ds in todo:
                state['phase'] = f'{student} {slot} on {ds}'
                save()
                if not run_pair(student, row, slot, ds) and not healthy():
                    log(f'{student} server unhealthy; moving to the next student')
                    break
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
            stop_server(proc)
    state['phase'] = 'finished'
    state['finished_at'] = now()
    save()
    log('all students processed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
