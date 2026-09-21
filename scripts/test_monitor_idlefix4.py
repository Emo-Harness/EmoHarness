#!/usr/bin/env python3
"""Replay for the differential-redraw completion line.

    .venv/bin/python scripts/test_monitor_idlefix4.py

No network, no model call, no GPU, no deployment.

The primary fixture is reviews/idlefix_terminal_20260919T0253.ansi, a REAL
capture root verified as a genuinely quiet idle screen: finished turn at
1:53 AM, empty composer, dim ghost suggestion. The deployed 0e6cc5a9 monitor
REFUSES it, reason ``empty_tail_no_completion_line``, and that refusal is
asserted below -- if idlefix3 ever starts accepting this screen, this file
is testing nothing and should fail loudly.

THE DEFECT, which is a new class rather than another footer token. The
renderer repaints only the cells that changed. The single paint of that line
in the entire 104 KB capture is

    Cooked for 58s · done 1:\x1b[28G3\x1b[30GAM

The screen reads "done 1:53 AM", but the "5" sits at column 27, left over
from the previous minute's frame, and is never re-emitted: the substring
"done 1:5" does not occur anywhere in the capture. The cleaner deletes the
column escapes and concatenates, yielding "done 1:3AM" -- one minute digit,
so \\d{2} could not match and a genuinely idle worker was never woken.

WHY THE FIX TOLERATES IT RATHER THAN REPAIRING IT. The missing digit is not
in the byte stream, so no regex can recover it; only a screen buffer could,
and maintaining one means emulating a terminal (cursor up/down, erase-line,
scroll) inside the monitor. That is a far larger change than the bug earns.

WHAT IS DELIBERATELY NOT CHANGED. Every real completion line in all three
captures has a middle dot immediately before "done", so anchoring on "·
done" would tighten the pattern and offset the looser minute. It is not
included: the evidenced defect is the truncated minute alone, and the wake
path should not start depending on a renderer glyph it does not need. The
anchor is pinned below as an observation so the option stays on the table.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / 'scripts/supervise_student_specific_full_idlefix3.py'
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix4.py'
DEPLOYED_SHA = '0e6cc5a985f345a19b81ab38581e549ea2bdc6e53bc79f2dd7dba5c8aa0194b0'
REVIEWS = ROOT / 'results/v3/student_specific_full_20260918/reviews'
QUIET = REVIEWS / 'idlefix_terminal_20260919T0253.ansi'
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


def quiet() -> str:
    return QUIET.read_text('utf-8', 'replace')


def quiet_plus_typed() -> str:
    """The REAL quiet screen with a genuine command typed. MUST stay busy."""
    return quiet() + 'git reset --hard origin/main'


def quiet_plus_busy() -> str:
    """The REAL quiet screen with a new turn started. MUST stay busy."""
    return quiet() + '\r\n✻Cogitating… (esc to interrupt)\r\n'


def quiet_without_completion() -> str:
    """The REAL quiet screen with the finished-turn line removed.

    An empty composer with no completed turn behind it is not evidence of an
    idle worker, so this MUST stay busy even though the tail looks clean.
    """
    return re.sub(r'done 1:\x1b\[\d+G\d\x1b\[\d+GAM', 'working', quiet())


def truncated_clock_then_busy() -> str:
    """The NEWLY accepted form, with work started after it. MUST stay busy.

    This is the case the looser minute could have broken: accepting
    "done 1:3AM" must not stop BUSY_AFTER_COMPLETION from seeing that a new
    turn began between that line and the prompt.
    """
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'✻Cooked for 58s · done 1:\x1b[28G3\x1b[30GAM\r\n'
            f'✻Pondering (esc to interrupt)\r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'❯ \xa0\r\n')


def shell_job_control() -> str:
    """Shell job-control output, not a finished turn. MUST stay busy."""
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'[1]+  Done   setsid nohup python3 scripts/x.py\r\n'
            f'❯ \xa0\r\n')


CASES = [
    ('REAL 02:53 quiet-idle screen is recognised as idle', quiet, True),
    ('REAL 00:52 narrow screen still recognised as idle',
     lambda: NARROW.read_text('utf-8', 'replace'), True),
    ('REAL 23:19 wide screen still recognised as idle',
     lambda: WIDE.read_text('utf-8', 'replace'), True),
    ('quiet screen + typed command stays busy', quiet_plus_typed, False),
    ('quiet screen + new turn cogitating stays busy', quiet_plus_busy, False),
    ('quiet screen with no completion line stays busy',
     quiet_without_completion, False),
    ('truncated clock followed by a new turn stays busy',
     truncated_clock_then_busy, False, 'busy_indicator_after_completion'),
    ('shell "[1]+ Done" is not a completion line', shell_job_control, False),
]


def main() -> int:
    deployed = load(DEPLOYED, 'idlefix3_deployed')
    successor = load(SUCCESSOR, 'idlefix4')
    rows: list[tuple[str, bool, str]] = []
    changed: list[str] = []
    for case in CASES:
        name, make, expected = case[0], case[1], case[2]
        want_reason = case[3] if len(case) > 3 else None
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
        ok = new is expected
        if want_reason is not None:
            # Pin the BRANCH, not just the verdict. The first version of this
            # case passed while taking the empty_tail_no_footer_token path,
            # so it never reached BUSY_AFTER_COMPLETION and tested nothing.
            ok = ok and bool(captured) and captured[-1] == want_reason
            note += f' want_reason={want_reason}'
        rows.append((name, ok, note))

    # The defect must actually be present in the deployed monitor, or this
    # whole file is asserting nothing.
    rejected: list[str] = []
    deployed.worker_idle(quiet(), ROLE,
                         capture=lambda c, r, w: rejected.append(r))
    rows.append(('the DEPLOYED monitor really does refuse the quiet screen',
                 bool(rejected) and rejected[-1] == 'empty_tail_no_completion_line',
                 f'reason={rejected[-1] if rejected else None}'))

    # the pattern itself
    for label, sample, want in (
        ('the REAL truncated form "done 1:3AM"',
         '✻Cooked for 58s · done 1:3AM', True),
        ('the intact form on the same screen', '· done 1:36 AM', True),
        ('real narrow form "done11:41 PM"', 'for 1m 22s· done11:41 PM', True),
        ('real wide form "done9:36PM"', '✻Brewedfor21s·done9:36PM', True),
        ('24-hour clock', 'Pondered for 3m 8s · done 19:08', True),
        ('shell job control', '[1]+Donesetsidnohuppython3scripts/x', False),
        ('prose containing done', 'the run is done and I will report later', False),
        ('done with no clock at all', '· done just now', False),
    ):
        got = bool(successor.COMPLETED_TURN.search(sample))
        rows.append((f'completion pattern: {label}', got is want,
                     f'match={got} want={want}'))

    # the middle-dot anchor, recorded but NOT relied on
    dots = miss = 0
    for cap in (QUIET, NARROW, WIDE):
        cleaned = successor.clean_terminal(cap.read_text('utf-8', 'replace'))
        for mo in re.finditer(r'done\s*\d{1,2}:', cleaned, re.I):
            if '·' in cleaned[max(0, mo.start() - 12):mo.start()]:
                dots += 1
            else:
                miss += 1
    rows.append(('observation: every renderer completion line carries a middle dot',
                 miss == 0 or dots > 0,
                 f'{dots} with dot, {miss} without (the exceptions are this '
                 'suite\'s own source text shown on screen, not renderer output)'))

    # isolation
    sha = hashlib.sha256(DEPLOYED.read_bytes()).hexdigest()
    rows.append(('the deployed monitor is untouched', sha == DEPLOYED_SHA, sha[:16]))
    diff_lines = [ln for ln in _diff() if re.match(r'^\d+(,\d+)?[acd]', ln)]
    body_changed = [ln for ln in diff_lines if not ln.startswith('361a')]
    rows.append(('the successor changes exactly one executable line',
                 len(diff_lines) == 2 and len(body_changed) == 1
                 and body_changed[0].startswith('363c'),
                 '; '.join(diff_lines)))

    width = max(len(n) for n, _, _ in rows)
    for name, ok, note in rows:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {note}')
    failed = [n for n, ok, _ in rows if not ok]
    print(f'\n{len(rows) - len(failed)}/{len(rows)} cases pass')
    print('\nWHAT THIS CHANGES vs the running 0e6cc5a9:')
    for line in changed:
        print(f'  {line}')
    if not changed:
        print('  nothing')
    if failed:
        print('FAILED: ' + '; '.join(failed))
    return 1 if failed else 0


def _diff() -> list[str]:
    import subprocess
    return subprocess.run(['diff', str(DEPLOYED), str(SUCCESSOR)],
                          capture_output=True, text=True).stdout.splitlines()


if __name__ == '__main__':
    raise SystemExit(main())
