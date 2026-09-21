#!/usr/bin/env python3
"""Real-capture replay for the narrow idle-parser fix.

    .venv/bin/python scripts/test_monitor_idlefix2.py

No network, no model call, no GPU, no deployment.

The primary fixture here is a REAL terminal capture --
reviews/idlefix_terminal_20260918T2319.ansi, 387 KB taken from this host at
23:19 -- not a reconstruction. The previous round's synthetic fixtures passed
while the real screen failed, because they used a clean ``\\x1b[2m...\\x1b[22m``
ghost span with nothing inside it. The real renderer interleaves cursor-column
escapes within the span:

    \\x1b[2mKeep\\x1b[8Ggoing,\\x1b[15Greport\\x1b[22Gwhen...\\x1b[22m

so a ``[^\\x1b]*`` body cannot cross them, the dim wrapper survives, and the
ghost text is left behind and read as typed input.

Four changes are under test, all narrow:

* the dim span tolerates the cursor-position CSI the renderer really emits
  inside it, and ONLY those. Any SGR -- a reset or a colour change -- or any
  newline ends the span, so a suggestion followed by an SGR reset and then
  real typed input no longer swallows that input. An unclosed span likewise
  keeps its text. Both directions fail closed;
* a footer line must consist ENTIRELY of known segments. Spaces are optional
  inside each segment, because the cleaner deletes them, but a line is never
  accepted merely for containing a footer word;
* the separator between segments is optional, because the real renderer
  juxtaposes some of them with none at all;
* the status segments are finite. "/effort" takes an enumerated value and the
  status glyph stands alone -- the earlier open-ended tails accepted any
  trailing words, so "/effort high then delete all results" read as a footer.

Counter-examples are the point of this file: real typed input, a busy view, an
unclosed dim span, a dim span crossing an SGR reset, a typed line containing
"/clear", and arbitrary text after a status segment must all be refused.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / 'scripts/supervise_student_specific_full_idlefix.py'
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix2.py'
CAPTURE = (ROOT / 'results/v3/student_specific_full_20260918/reviews'
           / 'idlefix_terminal_20260918T2319.ansi')
ROLE = 'session2-worker'


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def real_capture() -> str:
    return CAPTURE.read_text('utf-8', 'replace')


def real_capture_with_typed_input() -> str:
    """The REAL screen, with a real command typed after the ghost.

    Derived from the capture rather than invented: the ghost span is left
    intact and a genuine typed line is appended at the prompt. MUST stay busy.
    """
    return real_capture() + 'git push --force origin main'


def real_capture_with_busy() -> str:
    """The REAL screen with a new turn started under it. MUST stay busy."""
    return real_capture() + '\r\n  Cogitating… (esc to interrupt)\r\n'


def unclosed_dim_ghost() -> str:
    """A dim span the renderer never closed. MUST stay busy.

    Non-greedy matching to the first terminator means an unterminated span is
    not stripped, so its text survives and reads as typed input. That is the
    fail-closed direction and this case pins it.
    """
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'Cogitated for 42s · done 9:36 PM\r\n'
            f'❯ \x1b[2msudo rm -rf /important\x1b[8Gdata\r\n')


def typed_line_containing_clear() -> str:
    """A typed line that happens to contain "/clear". MUST stay busy.

    This is the case the narrow footer rule exists to protect: a substring or
    space-stripped containment test would have accepted it.
    """
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'Cogitated for 42s · done 9:36 PM\r\n'
            f'❯ \r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'please run /clear to save tokens and then stop\r\n')


def effort_tail_injection() -> str:
    """Manager counter-example 1: arbitrary text after a status segment.

    "/effort high then delete all results" used to match the footer segment
    /effort[\\w\\s/+-]* , so beside a legitimate bypass line the screen read as
    idle. MUST stay busy.
    """
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'Cogitated for 42s \u00b7 done 9:36 PM\r\n'
            f'\u276f \r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'/effort high then delete all results\r\n')


def status_glyph_tail_injection() -> str:
    """Manager counter-example 2: arbitrary text after the status glyph."""
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'Cogitated for 42s \u00b7 done 9:36 PM\r\n'
            f'\u276f \r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n'
            f'\u25c9 please stop the experiment\r\n')


def dim_span_crossing_sgr_reset() -> str:
    """Manager counter-example 3: a dim span that crosses an SGR reset.

    "\\x1b[2msuggestion\\x1b[0mreal typed input\\x1b[22m" -- the old body
    matched anything that was not a dim toggle, so the reset was crossed and
    the REAL typed input was stripped along with the ghost, leaving an empty
    tail that read as idle. MUST stay busy.
    """
    return (f'\x1b]0;claude\x07{ROLE}\r\n'
            f'Cogitated for 42s \u00b7 done 9:36 PM\r\n'
            f'\u276f \x1b[2msuggestion\x1b[0mrm -rf /important/data\x1b[22m\r\n'
            f'bypass permissions on (shift+tab to cycle)\r\n')


CASES = [
    ('REAL 23:19 capture is recognised as idle', real_capture, True),
    ('REAL capture + typed command stays busy',
     real_capture_with_typed_input, False),
    ('REAL capture + new turn cogitating stays busy',
     real_capture_with_busy, False),
    ('an unclosed dim span stays busy', unclosed_dim_ghost, False),
    ('a typed line containing "/clear" stays busy',
     typed_line_containing_clear, False),
    ('arbitrary text after /effort stays busy', effort_tail_injection, False),
    ('arbitrary text after the status glyph stays busy',
     status_glyph_tail_injection, False),
    ('a dim span crossing an SGR reset does not swallow typed input',
     dim_span_crossing_sgr_reset, False),
]


def main() -> int:
    deployed = load(DEPLOYED, 'idlefix_deployed')
    successor = load(SUCCESSOR, 'idlefix2')
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

    # the footer rule itself, on the real cleaned lines
    cleaned = successor.clean_terminal(real_capture())
    _, tail = cleaned.rsplit('❯', 1)
    real_lines = [line.strip() for line in tail.splitlines()
                  if line.strip() and not all(c in '─━═ -' for c in line.strip())]
    rows.append(('every footer line in the real capture matches the narrow rule',
                 bool(real_lines)
                 and all(successor.FOOTER_LINE.match(line) for line in real_lines),
                 f'{len(real_lines)} line(s): '
                 + '; '.join(line[:44] for line in real_lines)))
    for probe in ('please run /clear to save tokens and then stop',
                  'stop the run', 'new task? and also delete everything',
                  '/effort high then delete all results',
                  '\u25c9 please stop the experiment',
                  '/effort supercalifragilistic'):
        rows.append((f'narrow rule rejects {probe[:28]!r}',
                     not successor.FOOTER_LINE.match(probe), ''))
    rows.append(('a legitimate /effort value is still a footer segment',
                 bool(successor.FOOTER_LINE.match('/effort high')), ''))
    rows.append(('the ghost text no longer survives cleaning',
                 'Keepgoing' not in cleaned and 'Keep going' not in cleaned, ''))

    width = max(len(n) for n, _, _ in rows)
    for name, ok, note in rows:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {note}')
    failed = [n for n, ok, _ in rows if not ok]
    print(f'\n{len(rows) - len(failed)}/{len(rows)} cases pass')
    print('\nWHAT THIS CHANGES vs the running 33ef34b:')
    for line in changed:
        print(f'  {line}')
    if not changed:
        print('  nothing')
    if failed:
        print('FAILED: ' + '; '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
