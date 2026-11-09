"""Drive the emotion-context experiment: four students x two datasets.

    setsid nohup .venv/bin/python -u -m experiments.emotion_context.run \\
        >> logs/emotion_context/driver.$(hostname).log 2>&1 < /dev/null &

    # check a setup end to end: two videos through all four steps, then stop
    python -m experiments.emotion_context.run --students qwen3vl_4b --smoke-only

    # the paper: IntentQA only, plus the GPT-6.1 Sol harnesses' contexts
    python -m experiments.emotion_context.run --datasets intentqa_test --arms gpt6_1_sol

Per student, one owned vLLM server (``experiments.common.Driver``). On it, in
order: a smoke gate (two Social-IQ videos through all four steps; the student
is skipped if any step fails, any call errors, or more than 20% of a
condition's replies carry no letter), then the student's own harness writes the
per-video emotion contexts for both datasets, then the bare student answers
every question under ``none``, ``asr`` and ``emoctx``
(``experiments.emotion_context.row`` documents what each one sees). With
``--arms``, each listed Builder's rebuilt harness (``experiments.layout``:
``results/mhe/<builder>/<student>/construction/stage3/0/final``) then writes its
own contexts and the student answers once more under ``emoctx`` with them; the
``none`` and ``asr`` rows do not depend on the harness and are shared. A student
whose rebuilt harness does not exist yet is skipped for that arm. The smoke gate
runs on Social-IQ 2.0 val when it is among ``--datasets``, else on the first one.

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
from experiments.emotion_context.row import ARMS
from experiments.layout import LAYOUT

RUN: Final = ROOT / 'results/emotion_context'
HOST: Final = os.uname().nodename
LOGS: Final = ROOT / 'logs/emotion_context' / HOST
DATASETS: Final = ('social_iq_val', 'intentqa_test', 'social_iq_emotion')
CONDITIONS: Final = ('none', 'asr', 'emoctx')
NEED_FREE_MIB: Final = 60000
SMOKE_MAX_UNPARSED: Final = 0.2
ROW: Final = 'experiments.emotion_context.row'


Step = tuple[str, str | None, str, str | None]


def steps(
    datasets: tuple[str, ...] = DATASETS, arms: tuple[str, ...] = ()
) -> list[Step]:
    """(stage, condition, dataset, arm): the bundle's steps, then each arm's."""
    frozen: list[Step] = [('context', None, ds, None) for ds in datasets] + [
        ('qa', c, ds, None) for c in CONDITIONS for ds in datasets
    ]
    rebuilt: list[Step] = [
        step
        for arm in arms
        for step in (
            *(('context', None, ds, arm) for ds in datasets),
            *(('qa', 'emoctx', ds, arm) for ds in datasets),
        )
    ]
    return frozen + rebuilt


def step_dir(
    student: str, stage: str, cond: str | None, ds: str, arm: str | None = None
) -> Path:
    base = RUN / student / f'builder_{arm}' if arm else RUN / student
    return base / ('context' if stage == 'context' else f'qa_{cond}') / ds


def row_command(
    student: str,
    stage: str,
    cond: str | None,
    ds: str,
    *extra: str,
    arm: str | None = None,
) -> list[str]:
    args = ['--dataset', ds, '--student', student, '--stage', stage]
    return module_command(
        ROW,
        *args,
        *(['--condition', cond] if cond else []),
        *(['--arm', arm] if arm else []),
        *extra,
    )


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


def smoke_dataset(datasets: tuple[str, ...]) -> str:
    return 'social_iq_val' if 'social_iq_val' in datasets else datasets[0]


def smoke(driver: Driver, student: str, ds: str = 'social_iq_val') -> bool:
    """Two videos through all four steps on the live server.

    Everything it calls lands in the shared cache, so the full run reuses it.
    """
    report: dict = {}
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


def run_student(
    driver: Driver, student: str, todo: list[Step], smoke_only: bool
) -> None:
    for stage, cond, ds, arm in [] if smoke_only else todo:
        if arm and not LAYOUT.mhe_bundle(arm, student).is_dir():
            driver.log(f'{student} {arm}: no rebuilt harness yet, skipping {stage}')
            continue
        if (
            stage == 'qa'
            and cond == 'emoctx'
            and not is_done(step_dir(student, 'context', None, ds, arm))
        ):
            driver.log(f'{student} {arm or ""} {ds}: no contexts, skipping emoctx')
            continue
        label = f'{student} {arm or ""} {stage} {cond or ""} on {ds}'
        driver.set_phase(' '.join(label.split()))
        d = step_dir(student, stage, cond, ds, arm)
        ok = driver.run_step(
            str(d.relative_to(RUN)), row_command(student, stage, cond, ds, arm=arm), d
        )
        if not ok and not healthy():
            driver.log(f'{student} server unhealthy; moving to the next student')
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--students', nargs='+', choices=ORDER, default=list(ORDER))
    parser.add_argument(
        '--datasets', nargs='+', choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        '--arms',
        nargs='*',
        choices=ARMS,
        default=[],
        help="also run these Builders' rebuilt harnesses (context + emoctx)",
    )
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
    datasets = tuple(ds for ds in DATASETS if ds in args.datasets)
    plan = steps(datasets, tuple(args.arms))
    for student in [s for s in ORDER if s in args.students]:
        todo = [s for s in plan if not is_done(step_dir(student, *s))]
        if not todo and not args.smoke_only:
            continue
        proc = None
        try:
            driver.wait_for_vram(NEED_FREE_MIB)
            driver.set_phase(f'serving {student}')
            proc = driver.start_server(student)
            if smoke(driver, student, smoke_dataset(datasets)):
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
