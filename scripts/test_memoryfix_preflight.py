#!/usr/bin/env python3
"""Focused checks for the ONE thing the memoryfix launcher changes.

    .venv/bin/python scripts/test_memoryfix_preflight.py

No GPU, no server, no API key, no network, no launch. Every case drives
``ram_pressure_check()`` in scripts/start_student_construction_20260918_memoryfix.py
with the cgroup files stubbed, so the decision is exercised without needing the
host to actually be in that state.

What has to hold, and why it is the whole point of the successor:

* A cgroup that is 98% full of CLEAN page cache with no stall must ADMIT. That
  is the live state of this host after the 4,102-clip import -- anon under
  1 GiB, file_dirty 0, hundreds of successful reclaims, zero OOM kills -- and
  the frozen 0.60 raw-fraction rule refused it.
* Real memory pressure must still REFUSE, by each of the three rules
  independently.

It also asserts that nothing else in the successor drifted from the frozen
original: the diff is confined to ram_headroom and the admission-wrapper path.

"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / 'scripts/start_student_construction_20260918.py'
SUCCESSOR = ROOT / 'scripts/start_student_construction_20260918_memoryfix.py'
FROZEN_SHA = '7842d6ef7186f863e5d3b45eb9665932597a60ada9525f622fca010a75592702'


def load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location('memoryfix_launcher', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['memoryfix_launcher'] = module
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = '') -> None:
        self.rows.append((name, bool(ok), detail))

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        for name, ok, detail in self.rows:
            print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {detail}')
        failed = [n for n, ok, _ in self.rows if not ok]
        print(f'\n{len(self.rows) - len(failed)}/{len(self.rows)} checks passed')
        if failed:
            print('FAILED: ' + '; '.join(failed))
        return 1 if failed else 0


def main() -> int:
    c = Checks()
    module = load(SUCCESSOR)

    def stub(anon_gib: float, current_gib: float, full60: float) -> None:
        module.cgroup_anon_gib = lambda: anon_gib
        module.cgroup_full_avg60 = lambda: full60

        real_read = Path.read_text

        def fake_read(self: Path, *args: Any, **kwargs: Any) -> str:
            name = str(self)
            if name.endswith('memory.max'):
                return str(int(32.0 * 2**30))
            if name.endswith('memory.current'):
                return str(int(current_gib * 2**30))
            return real_read(self, *args, **kwargs)

        module.Path.read_text = fake_read  # type: ignore[method-assign]

    original_read = Path.read_text
    # anon, memory.current, full avg60, expect_ok, needle
    cases = [
        ('98% of the cgroup in clean cache with no stall is ADMITTED',
         0.8, 31.6, 0.04, True, 'anon 0.8'),
        ('the exact live state of this host is admitted',
         0.8, 25.5, 0.00, True, '80%'),
        ('anon above 26 GiB is refused on its own',
         27.0, 27.5, 0.0, False, 'anon 27.0 GiB > 26.0'),
        ('near-full AND stalling is refused',
         0.8, 31.6, 7.0, False, 'AND full avg60 7.0'),
        ('near-full with only mild stalling is admitted',
         0.8, 31.6, 3.0, True, '99%'),
        ('severe stalling is refused wherever the memory went',
         0.8, 12.0, 25.0, False, 'full avg60 25.0 > 20.0'),
        ('a boundary stall just under the hard rule is admitted',
         0.8, 12.0, 19.9, True, 'full avg60 19.90'),
    ]
    try:
        for name, anon, current, full60, expect_ok, needle in cases:
            stub(anon, current, full60)
            result = module.ram_pressure_check()
            ok = result.ok is expect_ok and needle in result.detail
            c.check(name, ok, f'ok={result.ok} {result.detail[:88]}')
    finally:
        module.Path.read_text = original_read  # type: ignore[method-assign]

    # the successor must differ from the frozen original in exactly three places
    c.check('the frozen original is untouched',
            _sha(FROZEN) == FROZEN_SHA, _sha(FROZEN)[:16])
    # Assert on which ORIGINAL lines changed, not on how many hunks diff chose
    # to emit: diff splits one edited function into several hunks wherever an
    # unchanged line survives inside it, so a hunk count says nothing useful.
    diff = subprocess.run(['diff', str(FROZEN), str(SUCCESSOR)],
                          capture_output=True, text=True)
    import re

    # Two intended changes, so two sets of regions:
    #   ram_headroom  -- docstring line 4, the constant at 128, check_resources
    #                    at 696-708;
    #   wrapper path  -- docstring line 10, the BUDGET_WRAPPER constant inserted
    #                    after 125, and the six call sites that now read it
    #                    (self_hashes 572, preflight 792-793, build command
    #                    1002, evidence 1257, the printed summary 1460).
    allowed = ({4, 10} | {125, 128} | set(range(690, 712))
               | {572, 792, 793, 1002, 1257, 1460})
    touched: set[int] = set()
    for line in diff.stdout.splitlines():
        match = re.match(r'^(\d+)(?:,(\d+))?([acd])', line)
        if match:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            touched |= set(range(start, end + 1))
    stray = sorted(touched - allowed)
    c.check('the successor touches only ram_headroom and the wrapper path',
            not stray,
            f'changed original lines {sorted(touched)}'
            + (f'; STRAY {stray}' if stray else ''))
    body = SUCCESSOR.read_text(encoding='utf-8')
    c.check('the raw-fraction rule is gone',
            'MAX_CGROUP_RAM_FRACTION' not in body, '')
    for knob, value in (('ROUNDS', 'rounds'), ('budget', 'budget_admission.py'),
                        ('serve helper', 'serve_student_full_20260918.sh')):
        c.check(f'the {knob} wiring is unchanged', value in body, value)
    c.check('the watchdog interval is unchanged',
            'WATCHDOG_INTERVAL_S: Final = 30' in body, '')
    c.check('it still requires an approval file',
            'check_approval' in body, '')
    c.check('the wrapper path is single-sourced through BUDGET_WRAPPER',
            body.count("BUDGET_WRAPPER: Final = 'scripts/budget_admission_httpx2.py'") == 1
            and body.count('BUDGET_WRAPPER') == 7
            and "sha256_file(ROOT / 'scripts/budget_admission.py')" not in body,
            f'{body.count("BUDGET_WRAPPER")} references to one constant')
    c.check('the frozen original wrapper is untouched at 7e85952e',
            _sha(ROOT / 'scripts/budget_admission.py')
            == '7e85952e738a69536d622c3f3dd7ce2d7af7b133d92c196e0e8e35a6250c54e7',
            _sha(ROOT / 'scripts/budget_admission.py')[:16])
    return c.report()


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    raise SystemExit(main())
