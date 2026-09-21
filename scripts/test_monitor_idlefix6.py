#!/usr/bin/env python3
"""Screen-buffer idle detection, with the four draft leaks root reproduced.

    .venv/bin/python scripts/test_monitor_idlefix6.py

No network, no model call, no GPU, no deployment, no terminal written to.

idlefix5 rendered the screen correctly and then threw the result away on the
one row that matters. Root typed four things at the composer and all four came
back idle=True, which is the only direction that can actually hurt: the
monitor would have pasted a manager task over the operator's draft.

  "/effort high"              whitelisted by FOOTER_LINE
  "---"                       erased by the box-drawing flattener
  ESC[5m real typed draft     genuine blink set the dim carrier
  ESC[0;2m real typed draft   combined SGR missed by a literal ESC[2m test

The first two are the same mistake: the composer was being read through
vocabulary meant for STATUS rows. A footer token and a rule character are
perfectly ordinary things to type. The composer is now decided by geometry
alone -- what sits immediately after the prompt, and where the insertion
point is -- and neither FOOTER_LINE nor the flattener can reach it.

The second two are the dim carrier being sloppy. Dim still rides on blink,
because pyte 0.8.2 has no SGR 2 and an attribute stored in the buffer is the
only representation that survives scrolling. But the stream is now rewritten
before it is fed: genuine blink is DISCARDED so only this module can set the
carrier, and parameters are parsed properly so ESC[0;2m is recognised.

Also fixed here: the idlefix5 height check called worker_idle five times
without ever changing the height, so it asserted nothing. It now forces the
render height and compares.
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
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix6.py'
DEPLOYED_SHA = '0e6cc5a985f345a19b81ab38581e549ea2bdc6e53bc79f2dd7dba5c8aa0194b0'
REVIEWS = ROOT / 'results/v3/student_specific_full_20260918/reviews'
CAPTURES = sorted(REVIEWS.glob('idlefix_terminal_*.ansi'))
LIVE = REVIEWS / 'idlefix_terminal_20260919T0354.ansi'
ROLE = 'session2-worker'

#: Verbatim from reviews/monitor_idlefix5_root_review.json.
ROOT_BASE = ('session2-worker\nCogitated for 42s · done 7:15 PM\n'
             '───────────'
             '───── session2-worker ─\n❯ ')
ROOT_SUFFIXES = ('/effort high', '———',
                 '\x1b[5mreal typed draft\x1b[0m',
                 '\x1b[0;2mreal typed draft')


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
        print(f'\n{len(self.rows) - len(failed)}/{len(self.rows)} checks pass')
        if failed:
            print('FAILED: ' + '; '.join(failed))
        return 1 if failed else 0


def main() -> int:  # noqa: C901 - a flat list of independent assertions
    c = Checks()
    new = load(SUCCESSOR, 'idlefix6')
    old = load(ROOT / 'scripts/supervise_student_specific_full_idlefix5.py',
               'idlefix5')

    def verdict(module, text, role=ROLE):
        reasons: list[str] = []
        ok = module.worker_idle(text, role,
                                capture=lambda _c, r, _w: reasons.append(r))
        return bool(ok), (reasons[-1] if reasons else '-')

    # -- 1. the four drafts root reproduced -------------------------------- #
    for suffix in ROOT_SUFFIXES:
        was, _ = verdict(old, ROOT_BASE + suffix)
        got, reason = verdict(new, ROOT_BASE + suffix)
        c.check(f'root draft {suffix[:26]!r} is refused',
                got is False and was is True,
                f'idlefix5={was} -> idlefix6={got} ({reason})')

    # -- 2. the same drafts typed into the REAL screens -------------------- #
    #    injected at the actual composer of each capture, not a mock-up.
    for cap in CAPTURES:
        raw = cap.read_text('utf-8', 'replace')
        lines, _dim, (cur_x, cur_y) = new.render_screen(raw)
        prow = max(i for i, ln in enumerate(lines) if new.PROMPT in ln)
        up = cur_y - prow
        for suffix in ROOT_SUFFIXES:
            typed = raw + f'\x1b[{up}A\x1b[3G' + suffix
            got, reason = verdict(new, typed)
            c.check(f'{cap.name[-17:-5]} + {suffix[:18]!r} stays busy',
                    got is False, f'reason={reason}')

    # -- 3. and the untouched screens are still idle ----------------------- #
    for cap in CAPTURES:
        got, reason = verdict(new, cap.read_text('utf-8', 'replace'))
        c.check(f'REAL {cap.name[-17:-5]} is still recognised as idle', got,
                f'reason={reason}')

    live = LIVE.read_text('utf-8', 'replace')
    lines, dim, (cur_x, cur_y) = new.render_screen(live)
    prompt_row = max(i for i, ln in enumerate(lines) if new.PROMPT in ln)

    # -- 4. the composer is decided by geometry, not vocabulary ------------ #
    body = SUCCESSOR.read_text(encoding='utf-8')
    composer = body[body.index('-- the composer itself'):
                    body.index('-- nothing may be happening below')]
    c.check('the composer block never consults FOOTER_LINE',
            'FOOTER_LINE' not in composer, '')
    c.check('the composer block never flattens rule characters',
            'flatten_rules' not in composer and 'BOX_LEFTOVER' not in composer,
            '')
    c.check('flattening is off by default',
            'def visible(line, dim_row, want_dim=False, flatten_rules=False)'
            in body, '')
    c.check('an ambiguous prompt row fails closed',
            'ambiguous_prompt_row' in body, '')

    # -- 5. the dim carrier cannot be set from the stream ------------------ #
    c.check('genuine blink is discarded before rendering',
            '\x1b[5m' not in new.carry_dim_on_blink('\x1b[5mx\x1b[0m'),
            repr(new.carry_dim_on_blink('\x1b[5mx\x1b[0m')))
    for body_in, opens in (('2', True), ('0;2', True), ('1;2;31', True),
                           ('5', False), ('0', False), ('22', False),
                           ('', False)):
        got = new.dim_left_open(f'\x1b[{body_in}m')
        c.check(f'SGR {body_in or "(empty)"!r} leaves dim open = {opens}',
                got is opens, f'got={got}')
    c.check('dim closed by a combined reset is not "left open"',
            new.dim_left_open('\x1b[0;2mx\x1b[1;22my') is False, '')
    # Dropping a blink request must not become a RESET: that would silently
    # close a dim span that is still open and expose part of a suggestion as
    # non-dim typed input.
    c.check('a blink escape inside a dim span does not end the span',
            new.carry_dim_on_blink('\x1b[2ma\x1b[5mb\x1b[22mc')
            == '\x1b[5mab\x1b[22;25mc',
            repr(new.carry_dim_on_blink('\x1b[2ma\x1b[5mb\x1b[22mc')))

    # -- 6. height independence, actually varying the height --------------- #
    #    the idlefix5 version of this check never passed rows through.
    original_render = new.render_screen
    verdicts: dict[int, bool] = {}
    try:
        for rows in (24, 30, 40, 50, 60):
            new.render_screen = (
                lambda t, rows=rows, cols=None, _o=original_render:
                _o(t, rows=rows, cols=cols))
            verdicts[rows] = bool(new.worker_idle(live, ROLE))
    finally:
        new.render_screen = original_render
    c.check('height does not change the verdict (heights really vary)',
            len(set(verdicts.values())) == 1 and all(verdicts.values()),
            f'{verdicts}')
    heights = {rows: new.render_screen(live, rows=rows)[0][-6:]
               for rows in (24, 40, 60)}
    c.check('the bottom rows render identically at every height',
            len({tuple(v) for v in heights.values()}) == 1,
            f'{len(heights)} heights compared')

    # -- 7. the parser is pinned ------------------------------------------- #
    loaded = new.load_pyte()
    c.check('the pinned parser loads', loaded is not None, '')
    if loaded:
        import pyte  # noqa: PLC0415
        c.check('its fingerprint matches the pin',
                new.parser_fingerprint(Path(pyte.__file__).parent)
                == new.PARSER_PIN, new.PARSER_PIN[:16])
    saved = new.PARSER_PIN
    try:
        new.PARSER_PIN = '0' * 64
        got, reason = verdict(new, live)
        c.check('a parser that does not match the pin is refused',
                got is False and reason == 'terminal_parser_unavailable',
                f'reason={reason}')
    finally:
        new.PARSER_PIN = saved

    # -- 8. everything the earlier rounds pinned --------------------------- #
    ghost = [y for y, ln in enumerate(lines)
             if 'Read the next manager task' in ln]
    if ghost:
        y = ghost[0]
        plain = new.visible(lines[y], dim[y], want_dim=False).strip()
        shaded = new.visible(lines[y], dim[y], want_dim=True).strip()
        c.check('the grey suggestion is still all dim, never typed input',
                not plain and shaded.startswith('Read the next'),
                f'non-dim={plain!r}')
    inline = [cap for cap in CAPTURES
              if 'Keep going' in '\n'.join(new.render_screen(
                  cap.read_text('utf-8', 'replace'))[0])]
    c.check('a suggestion INLINE at the prompt is still idle', bool(inline),
            f'{inline[0].name[-17:-5] if inline else None} carries one')
    for label, suffix, want in (
        ('a new turn running',
         '\x1b[4A\x1b[3G✻Pondering (esc to interrupt)',
         'busy_indicator_after_completion'),
        ('a compaction in progress',
         '\x1b[4A\x1b[3GCompacting conversation',
         'compacting_after_completion'),
        ('a shell prompt under the composer',
         '\x1b[1B\x1b[1Gmax7@box:~/EmoHarness$ ls -la',
         'unrecognised_line_below_prompt'),
    ):
        got, reason = verdict(new, live + suffix)
        c.check(f'{label} keeps the worker busy',
                got is False and reason == want, f'reason={reason}')
    got, reason = verdict(new, live, 'session1-worker')
    c.check('another worker\'s screen is not woken as this one',
            got is False and reason == 'role_absent', f'reason={reason}')
    saved_loader = new.load_pyte
    try:
        new.load_pyte = lambda: None
        got, reason = verdict(new, live)
        c.check('no terminal emulator means NOT idle',
                got is False and reason == 'terminal_parser_unavailable',
                f'reason={reason}')
    finally:
        new.load_pyte = saved_loader

    agree = total = 0
    disagreed: list[str] = []
    for suite in ('test_monitor_idlefix.py', 'test_monitor_idlefix2.py',
                  'test_monitor_idlefix3.py', 'test_monitor_idlefix4.py'):
        mod = load(ROOT / 'scripts' / suite, suite.replace('.', '_'))
        for case in getattr(mod, 'CASES', []):
            total += 1
            if bool(new.worker_idle(case[1](), ROLE)) is bool(case[2]):
                agree += 1
            else:
                disagreed.append(f'{suite}:{case[0]}')
    c.check('every case the four earlier suites pin still holds',
            not disagreed and total >= 38,
            f'{agree}/{total}' + (f' DISAGREE {disagreed}' if disagreed else ''))

    # -- 9. drafts are never touched --------------------------------------- #
    kills = [s for s in ('\\x15', '\\x03', '\\x1b[2K', 'C-u', 'send-keys')
             if s in body]
    c.check('the monitor has no way to clear a draft', not kills, f'{kills}')
    c.check('stdin is written only after idle is confirmed',
            body.count('connection.send') == 2
            and body.index('if not quiet or not worker_idle')
            < body.index('connection.send'), '')
    c.check('the vendoring comment no longer contradicts itself',
            'in-repo copy' not in body and 'NOT vendored' in body, '')
    sha = hashlib.sha256(DEPLOYED.read_bytes()).hexdigest()
    c.check('the deployed monitor is untouched', sha == DEPLOYED_SHA, sha[:16])
    return c.report()


if __name__ == '__main__':
    raise SystemExit(main())
