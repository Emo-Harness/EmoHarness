"""Drive the transfer run: four students x two datasets x ladder rows L0/L1/L3/L5.

    setsid nohup .venv/bin/python -u -m experiments.transfer.run \
        >> logs/transfer/driver.log 2>&1 < /dev/null &

One owned vLLM server per student (``experiments.common.Driver``); on it, every
(row, dataset) pair runs ``experiments.transfer.row``. Rows keep the main
evaluation's names: L0 native video, L1 video plus the audio evidence text, L3
the shared scaffold, L5 the student's own bundle. Progress goes to
``results/transfer/status.json``.
"""

from __future__ import annotations

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
)

RUN: Final = ROOT / 'results/transfer'
LOGS: Final = ROOT / 'logs/transfer'
DATASETS: Final = ('social_iq_val', 'intentqa_test')
#: (ladder row, arm directory), in the main evaluation's naming.
ROWS: Final = (('L0', 'L0'), ('L1', 'L1'), ('L3', 'L3_scaffold'), ('L5', 'L5_own'))


def pair_dir(student: str, slot: str, dataset: str) -> Path:
    return RUN / student / slot / dataset


def main() -> int:
    driver = Driver(RUN / 'status.json', LOGS)
    driver.save()
    for student in ORDER:
        todo = [
            (row, slot, ds)
            for row, slot in ROWS
            for ds in DATASETS
            if not is_done(pair_dir(student, slot, ds))
        ]
        if not todo:
            continue
        driver.set_phase(f'serving {student}')
        proc = None
        try:
            proc = driver.start_server(student)
            for row, slot, ds in todo:
                driver.set_phase(f'{student} {slot} on {ds}')
                command = module_command(
                    'experiments.transfer.row',
                    '--dataset',
                    ds,
                    '--student',
                    student,
                    '--row',
                    row,
                )
                key = f'{student}/{slot}/{ds}'
                ok = driver.run_step(key, command, pair_dir(student, slot, ds))
                if not ok and not healthy():
                    driver.log(
                        f'{student} server unhealthy; moving to the next student'
                    )
                    break
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
