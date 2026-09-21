#!/usr/bin/env python3
"""Narrow-screen real-capture replay for the completion-line fix.

    .venv/bin/python scripts/test_monitor_idlefix3.py

No network, no model call, no GPU, no deployment.

Both fixtures are REAL captures from this host:

* reviews/idlefix_terminal_20260919T0052.ansi -- the 80-column screen the
  running e44ee02 monitor refuses, reason ``empty_tail_no_completion_line``;
* reviews/idlefix_terminal_20260918T2319.ansi -- the wide screen already
  accepted, replayed to prove this change does not disturb it.

THE DEFECT. The renderer puts a cursor-position escape between the word and
the clock: ``· done\x1b[31G11:41 PM``. The cleaner deletes the escape, so
the text becomes ``done11:41 PM`` -- and on the wide capture even
``done9:36PM``. The pattern required ``\\bdone\\b``, but there is no word
boundary between "done" and "11", so a finished turn read as unfinished and
the worker was never woken.

THE FIX, which is narrower overall, not looser. Whitespace between the word
and the clock becomes optional, and the previous gap of "up to 40 arbitrary
characters" between them is removed. Shell job-control output such as
``[1]+ Done  setsid nohup python3 ...`` and prose containing "done" therefore
still fail to match, and are asserted here.

Worth recording because it corrected my own reading: the wide 23:19 capture is
accepted through a DIFFERENT branch -- its footer lines sit after the prompt,
so it takes the per-line footer path, which never consults the completion
line. Only the narrow screen, whose tail is empty, reaches the completion
check. I initially mis-read a window measurement as a regression on the wide
capture; it is not one, and both paths are pinned below.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / 'scripts/supervise_student_specific_full_idlefix2.py'
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix3.py'
REVIEWS = ROOT / 'results/v3/student_specific_full_20260918/reviews'
NARROW = REVIEWS / 'idlefix_terminal_20260919T0052.ansi'
WIDE = REVIEWS / 'idlefix_terminal_20260918T2319.ansi'
ROLE = 'session2-worker'


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def narrow() -> str:
    return NARROW.read_text('utf-8', 'replace')


def wide() -> str:
    return WIDE.read_text('utf-8', 'replace')


def narrow_plus_typed() -> str:
    """The REAL narrow screen with a genuine command typed. MUST stay busy."""
    return narrow() + 'rm -rf /important/data'


def narrow_plus_busy() -> str:
    """The REAL narrow screen with a new turn running. MUST stay busy."""
    return narrow() + '\r\n✻ Cogitating… (esc to interrupt)\r\n'


def narrow_without_completion() -> str:
    """The REAL narrow screen with the completion line removed.

    An empty composer with no finished turn behind it is not evidence of an
    idle worker, so this MUST stay busy even though the tail looks clean.
    """
    return re.sub(r'done\x1b\[\d+G\d{1,2}:\d{2}\s*[AP]M', 'working', narrow())


def shell_job_control() -> str:
    """Shell job-control output, not a finished turn. MUST stay busy."""
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'[1]+  Done   setsid nohup python3 scripts/x.py\r\n'
            f'❯ \xa0\r\n')


CASES = [
    ('REAL 00:52 narrow screen is recognised as idle', narrow, True),
    ('REAL 23:19 wide screen still recognised as idle', wide, True),
    ('narrow screen + typed command stays busy', narrow_plus_typed, False),
    ('narrow screen + new turn cogitating stays busy', narrow_plus_busy, False),
    ('narrow screen with no completion line stays busy',
     narrow_without_completion, False),
    ('shell "[1]+ Done" is not a completion line', shell_job_control, False),
]


def main() -> int:
    deployed = load(DEPLOYED, 'idlefix2_deployed')
    successor = load(SUCCESSOR, 'idlefix3')
    rows: list[tuple[str, bool, str]] = []
    changed: list[str] = []
    for name, make, expected in CASES:
        text = make()
        old = bool(deployed.worker_idle(text, ROLE))
        captured: list[str] = []

        def record(_cleaned, reason, _raw, sink=captured):
            sink.append(reason)

        new = bool(successor.worker_idle(text, ROLE, capture=record))
        note = f'old={old} new={new} want={expected}'
        if not new and captured:
            note += f' | {captured[-1]}'
        if old != new:
            changed.append(f'{name}: {old} -> {new}')
        rows.append((name, new is expected, note))

    # the pattern itself, on cleaned forms taken from the real captures
    for label, sample, want in (
        ('real narrow form "done11:41 PM"',
         'Cogitated for 1m 22s· done11:41 PM', True),
        ('real wide form "done9:36PM"', '✻Brewedfor21s·done9:36PM', True),
        ('classic spaced form', 'Cogitated for 42s · done 7:15 PM', True),
        ('24-hour clock', 'Pondered for 3m 8s · done 19:08', True),
        ('shell job control', '[1]+Donesetsidnohuppython3scripts/x', False),
        ('prose containing done', 'the run is done and I will report later', False),
    ):
        got = bool(successor.COMPLETED_TURN.search(sample))
        rows.append((f'completion pattern: {label}', got is want,
                     f'match={got} want={want}'))

    width = max(len(n) for n, _, _ in rows)
    for name, ok, note in rows:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {note}')
    failed = [n for n, ok, _ in rows if not ok]
    print(f'\n{len(rows) - len(failed)}/{len(rows)} cases pass')
    print('\nWHAT THIS CHANGES vs the running e44ee02:')
    for line in changed:
        print(f'  {line}')
    if not changed:
        print('  nothing')
    if failed:
        print('FAILED: ' + '; '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
