#!/usr/bin/env python3
"""Drive the emotion-context experiment: four students x two datasets.

    setsid nohup .venv/bin/python -u scripts/run_emoctx.py \
        >> logs/emoctx/driver.$(hostname).log 2>&1 < /dev/null &

Per student, one owned vLLM server started exactly as ``run_newdata.py``
started it (same serve script, serving venv, pinned revision, hub offline).
On that server, in order: a smoke gate (two Social-IQ videos through all four
steps; the student is skipped if any step fails or more than 20% of a
condition's replies carry no letter), then the student's own harness writes
the per-video emotion contexts for both datasets, then the bare student answers
every question under ``none``, ``asr`` and ``emoctx`` (``emoctx_row.py``
documents what each one sees).

Both boxes see this results tree over NFS, and containers get replaced without
warning. So the driver claims ``owner.json`` for its host (``--take-over``
replaces another host's claim), ``emoctx_row.py`` refuses to write from any
other host, and status and logs are per host: ``status.<host>.json`` and
``logs/emoctx/<host>/``. Before each server start it waits until at least
``NEED_FREE_MIB`` of VRAM is free rather than squeezing in beside another
tenant.

Resumable: a step with a done.json is skipped, a failed step is retried once,
and finished Target calls are reused from the shared content-addressed cache
(temperature 0, so a retry is not a resample). QA calls rejected for exceeding
the 16384-token context are re-sent afterwards by ``emoctx_topup.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_newdata as base  # noqa: E402

ROOT = base.ROOT
RUN = ROOT / 'results/newdata/emotion_context_20260925'
LOGS = ROOT / 'logs/emoctx'
DATASETS = ('social_iq_val', 'intentqa_test')
CONDITIONS = ('none', 'asr', 'emoctx')
NEED_FREE_MIB = 60000
SMOKE_MAX_UNPARSED = 0.2
GPU_POLL_S = 300

HOST = os.uname().nodename
#: Per-host status and logs: both boxes see this tree, and a driver on the other
#: box must never overwrite this one's status or logs (see owner.json).
LOGS = LOGS / HOST
base.RUN, base.LOGS, base.STATUS = RUN, LOGS, RUN / f'status.{HOST}.json'
state = base.state
now, save, log, healthy = base.now, base.save, base.log, base.healthy


def steps() -> list[tuple[str, str | None, str]]:
    return [('context', None, ds) for ds in DATASETS] + [
        ('qa', c, ds) for c in CONDITIONS for ds in DATASETS
    ]


def step_dir(student: str, stage: str, cond: str | None, ds: str) -> Path:
    return RUN / student / ('context' if stage == 'context' else f'qa_{cond}') / ds


def is_done(d: Path) -> bool:
    marker = d / 'done.json'
    return marker.is_file() and json.loads(marker.read_text()).get('n_written', 0) > 0


def free_mib() -> int:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.split()[0])


def wait_for_gpu() -> None:
    began = time.time()
    while (free := free_mib()) < NEED_FREE_MIB:
        state['gpu_wait'] = {
            'since': state.get('gpu_wait', {}).get('since', now()),
            'free_mib': free,
            'need_mib': NEED_FREE_MIB,
            'checked_at': now(),
        }
        state['phase'] = 'waiting for GPU'
        save()
        if time.time() - began < GPU_POLL_S:
            log(f'GPU has {free} MiB free, need {NEED_FREE_MIB}; waiting')
        time.sleep(GPU_POLL_S)
    if 'gpu_wait' in state:
        state.setdefault('gpu_waits', []).append(
            {**state.pop('gpu_wait'), 'ended_at': now()}
        )
        save()


def ledger_rows(d: Path) -> int:
    n = 0
    for p in d.rglob('ledger*.jsonl'):
        with p.open() as fh:
            n += sum(1 for _ in fh)
    return n


def run_step(student: str, stage: str, cond: str | None, ds: str) -> bool:
    d = step_dir(student, stage, cond, ds)
    key = str(d.relative_to(RUN))
    entry = state['pairs'].setdefault(key, {'attempts': 0, 'status': 'pending'})
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / '.venv/bin/python'),
        str(ROOT / 'scripts/emoctx_row.py'),
        '--dataset',
        ds,
        '--student',
        student,
        '--stage',
        stage,
    ]
    if cond:
        cmd += ['--condition', cond]
    while entry['attempts'] < base.MAX_ATTEMPTS:
        entry['attempts'] += 1
        entry.update({'status': 'running', 'started_at': now(), 'error': None})
        save()
        env = {
            **os.environ,
            'EMOHARNESS_RESULTS_DIR': str(d.relative_to(ROOT)),
            'VLLM_ENDPOINT': base.ENDPOINT,
            'EMOHARNESS_WORKERS': '8',
            'EMOHARNESS_VIDEO_WORKERS': '2',
            'LD_LIBRARY_PATH': base.lib_path(),
            'EMOHARNESS_PREPROCESS_CONFIG': 'preprocess.yaml',
        }
        logf = LOGS / f'{key.replace("/", "_")}_a{entry["attempts"]}.log'
        with logf.open('w') as out:
            proc = subprocess.Popen(
                cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT
            )
            entry['row_pid'] = proc.pid
            while proc.poll() is None:
                entry['target_calls_so_far'] = ledger_rows(d)
                entry['server_healthy'] = healthy()
                save()
                time.sleep(60)
        entry['returncode'] = proc.returncode
        tail = logf.read_text()[-20000:].strip().splitlines()
        if proc.returncode == 0 and tail:
            prov = json.loads(tail[-1])
            tmp = d / '.done.json.tmp'
            tmp.write_text(json.dumps(prov, indent=2, ensure_ascii=False))
            tmp.replace(d / 'done.json')
            entry.update(
                {
                    'status': 'done',
                    'finished_at': now(),
                    'n_written': prov['n_written'],
                    'seconds': prov['seconds'],
                    'accuracy': prov.get('accuracy'),
                    'n_unparsed': prov.get('n_unparsed'),
                    'target_calls': ledger_rows(d),
                }
            )
            save()
            log(
                f'{student}/{key} done: {prov["n_written"]} in {prov["seconds"]}s'
                + (
                    f', acc {prov["accuracy"]:.4f}'
                    if prov.get('accuracy') is not None
                    else ''
                )
            )
            return True
        entry.update({'status': 'failed', 'error': '\n'.join(tail[-6:])})
        save()
        log(f'{student}/{key} attempt {entry["attempts"]} failed rc={proc.returncode}')
        if not healthy():
            return False
    return False


def smoke(student: str) -> bool:
    """Two Social-IQ videos through all four steps on the live server before the full run.

    Everything it calls lands in the shared cache, so the full run reuses it.
    Fails the student if any step errors, any Target call fails, or more than
    SMOKE_MAX_UNPARSED of a condition's replies carry no parsable letter.
    """
    ds, report = 'social_iq_val', {}
    for stage, cond in (
        ('context', None),
        ('qa', 'none'),
        ('qa', 'asr'),
        ('qa', 'emoctx'),
    ):
        d = step_dir(student, stage, cond, ds)
        d.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ROOT / '.venv/bin/python'),
            str(ROOT / 'scripts/emoctx_row.py'),
            '--dataset',
            ds,
            '--student',
            student,
            '--stage',
            stage,
            '--smoke',
        ] + (['--condition', cond] if cond else [])
        env = {
            **os.environ,
            'EMOHARNESS_RESULTS_DIR': str(d.relative_to(ROOT)),
            'VLLM_ENDPOINT': base.ENDPOINT,
            'EMOHARNESS_WORKERS': '8',
            'EMOHARNESS_VIDEO_WORKERS': '2',
            'LD_LIBRARY_PATH': base.lib_path(),
            'EMOHARNESS_PREPROCESS_CONFIG': 'preprocess.yaml',
        }
        logf = LOGS / f'smoke_{student}_{stage}_{cond or "ctx"}.log'
        with logf.open('w') as out:
            rc = subprocess.run(
                cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT
            ).returncode
        tail = logf.read_text().strip().splitlines()
        prov = json.loads(tail[-1]) if rc == 0 and tail else {}
        ok = (
            rc == 0
            and bool(prov)
            and not prov.get('n_errors')
            and (
                stage == 'context'
                or prov['n_unparsed'] <= SMOKE_MAX_UNPARSED * prov['n_written']
            )
        )
        report[f'{stage}_{cond or "ctx"}'] = {
            'ok': ok,
            'rc': rc,
            'n': prov.get('n_written'),
            'n_unparsed': prov.get('n_unparsed'),
            'accuracy': prov.get('accuracy'),
            'seconds': prov.get('seconds'),
        }
        state.setdefault('smoke', {})[student] = report
        save()
        if not ok:
            log(f'{student} smoke {stage} {cond or ""} failed; see {logf.name}')
            return False
    log(f'{student} smoke passed: {json.dumps(report)}')
    return True


def claim() -> None:
    """Name this host the only writer (``--take-over`` replaces another host's claim)."""
    owner = RUN / 'owner.json'
    if owner.is_file():
        prior = json.loads(owner.read_text())
        if prior['host'] != HOST and '--take-over' not in sys.argv:
            raise SystemExit(
                f'owner.json names {prior["host"]}; pass --take-over to replace it'
            )
    tmp = RUN / f'.owner.{os.getpid()}.tmp'
    tmp.write_text(
        json.dumps(
            {
                'host': HOST,
                'driver_pid': os.getpid(),
                'pgid': os.getpgid(0),
                'claimed_at': now(),
            },
            indent=2,
        )
    )
    tmp.replace(owner)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    claim()
    save()
    for student in base.ORDER:
        todo = [s for s in steps() if not is_done(step_dir(student, *s))]
        for s in steps():
            if s not in todo:
                key = str(step_dir(student, *s).relative_to(RUN))
                state['pairs'].setdefault(key, {})['status'] = 'done'
        if not todo:
            continue
        proc = None
        try:
            wait_for_gpu()
            state['phase'] = f'serving {student}'
            save()
            proc = base.start_server(student)
            if not smoke(student):
                continue
            for stage, cond, ds in todo:
                if (
                    stage == 'qa'
                    and cond == 'emoctx'
                    and not is_done(step_dir(student, 'context', None, ds))
                ):
                    log(f'{student} {ds}: no contexts, skipping emoctx')
                    continue
                state['phase'] = f'{student} {stage} {cond or ""} on {ds}'.replace(
                    '  ', ' '
                )
                save()
                if not run_step(student, stage, cond, ds) and not healthy():
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
            base.stop_server(proc)
    state['phase'] = 'finished'
    state['finished_at'] = now()
    save()
    log('all students processed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
