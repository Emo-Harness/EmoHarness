"""Drive the emotion-context experiment: four students x two datasets.

    setsid nohup .venv/bin/python -u -m experiments.emotion_context.run \\
        >> logs/emotion_context/driver.$(hostname).log 2>&1 < /dev/null &

    # check a setup end to end: two videos through all four steps, then stop
    python -m experiments.emotion_context.run --students qwen3vl_4b --smoke-only

Per student, one owned vLLM server (``experiments.common.Driver``). On it, in
order: a smoke gate (two Social-IQ videos through all four steps; the student
is skipped if any step fails, any call errors, or more than 20% of a
condition's replies carry no letter), then the student's own harness writes the
per-video emotion contexts for both datasets, then the bare student answers
every question under ``none``, ``asr`` and ``emoctx``
(``experiments.emotion_context.row`` documents what each one sees).

Both boxes see the results tree over NFS and containers get replaced without
warning, so the driver claims ``owner.json`` for its host (``--take-over``
replaces another host's claim), the row refuses to write from any other host,
and status and logs are per host: ``status.<host>.json`` and
``logs/emotion_context/<host>/``. QA calls rejected for exceeding the
16384-token context are re-sent afterwards by
``experiments.emotion_context.topup``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

from experiments.common import (
    ORDER,
    ROOT,
    Driver,
    healthy,
    is_done,
    module_command,
    now,
    step_env,
    write_json,
)

RUN: Final = ROOT / 'results/emotion_context'
HOST: Final = os.uname().nodename
LOGS: Final = ROOT / 'logs/emotion_context' / HOST
DATASETS: Final = ('social_iq_val', 'intentqa_test')
CONDITIONS: Final = ('none', 'asr', 'emoctx')
NEED_FREE_MIB: Final = 60000
SMOKE_MAX_UNPARSED: Final = 0.2
ROW: Final = 'experiments.emotion_context.row'


def steps() -> list[tuple[str, str | None, str]]:
    return [('context', None, ds) for ds in DATASETS] + [
        ('qa', c, ds) for c in CONDITIONS for ds in DATASETS
    ]


def step_dir(student: str, stage: str, cond: str | None, ds: str) -> Path:
    return RUN / student / ('context' if stage == 'context' else f'qa_{cond}') / ds


def row_command(
    student: str, stage: str, cond: str | None, ds: str, *extra: str
) -> list[str]:
    args = ['--dataset', ds, '--student', student, '--stage', stage]
    return module_command(ROW, *args, *(['--condition', cond] if cond else []), *extra)


def claim(take_over: bool) -> None:
    """Name this host the only writer of the results tree."""
    owner = RUN / 'owner.json'
    if owner.is_file():
        prior = json.loads(owner.read_text())
        if prior['host'] != HOST and not take_over:
            raise SystemExit(
                f'owner.json names {prior["host"]}; pass --take-over to replace it'
            )
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(
        owner,
        {
            'host': HOST,
            'driver_pid': os.getpid(),
            'pgid': os.getpgid(0),
            'claimed_at': now(),
        },
    )


def smoke(driver: Driver, student: str) -> bool:
    """Two Social-IQ videos through all four steps on the live server.

    Everything it calls lands in the shared cache, so the full run reuses it.
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
        logf = LOGS / f'smoke_{student}_{stage}_{cond or "ctx"}.log'
        with logf.open('w') as out:
            rc = subprocess.run(
                row_command(student, stage, cond, ds, '--smoke'),
                env=step_env(d),
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                check=False,
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
        driver.state.setdefault('smoke', {})[student] = report
        driver.save()
        if not ok:
            driver.log(f'{student} smoke {stage} {cond or ""} failed; see {logf.name}')
            return False
    driver.log(f'{student} smoke passed: {json.dumps(report)}')
    return True


def run_student(driver: Driver, student: str, todo: list, smoke_only: bool) -> None:
    for stage, cond, ds in [] if smoke_only else todo:
        if (
            stage == 'qa'
            and cond == 'emoctx'
            and not is_done(step_dir(student, 'context', None, ds))
        ):
            driver.log(f'{student} {ds}: no contexts, skipping emoctx')
            continue
        driver.set_phase(f'{student} {stage} {cond or ""} on {ds}'.replace('  ', ' '))
        d = step_dir(student, stage, cond, ds)
        ok = driver.run_step(
            str(d.relative_to(RUN)), row_command(student, stage, cond, ds), d
        )
        if not ok and not healthy():
            driver.log(f'{student} server unhealthy; moving to the next student')
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--students', nargs='+', choices=ORDER, default=list(ORDER))
    parser.add_argument(
        '--smoke-only', action='store_true', help='run the smoke gate only'
    )
    parser.add_argument(
        '--take-over', action='store_true', help="replace another host's claim"
    )
    args = parser.parse_args(argv)
    LOGS.mkdir(parents=True, exist_ok=True)
    claim(args.take_over)
    driver = Driver(RUN / f'status.{HOST}.json', LOGS)
    driver.save()
    for student in [s for s in ORDER if s in args.students]:
        todo = [s for s in steps() if not is_done(step_dir(student, *s))]
        if not todo and not args.smoke_only:
            continue
        proc = None
        try:
            driver.wait_for_vram(NEED_FREE_MIB)
            driver.set_phase(f'serving {student}')
            proc = driver.start_server(student)
            if smoke(driver, student):
                run_student(driver, student, todo, args.smoke_only)
        except Exception as exc:  # noqa: BLE001 - recorded, next student continues
            driver.record_error(student, exc)
        finally:
            driver.stop_server(proc)
    driver.state['finished_at'] = now()
    driver.set_phase('finished')
    driver.log('all students processed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
