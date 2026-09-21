#!/usr/bin/env python3
"""Screen-buffer idle detection: composer painting, SGR groups, real geometry.

    .venv/bin/python scripts/test_monitor_idlefix7.py

No network, no model call, no GPU, no deployment, no terminal written to, and
the real rejection sink is redirected to a temporary directory.

THREE THINGS idlefix6 GOT WRONG, all reproduced by root:

1. The composer field was truncated at the first run of three spaces so the
   renderer's right-hand notice fell outside it. Root typed "   do something"
   and pressed Home: the leading spaces ended the field immediately and a real
   draft vanished. There is no truncation any more. What separates a draft
   from the renderer's own status is how the cell is PAINTED -- dim is the
   suggestion, coloured-away-from-home is status, anything else is typed.
   Colour at the home column is ambiguous and fails closed.

2. SGR parameters were mapped one at a time, so the palette index in
   "\x1b[38;5;2m" was read as the dim attribute and the sequence was rewritten
   to "\x1b[38;5m", corrupting the colour too. 38/48/58 groups, indexed and
   RGB, are now copied through whole.

3. "Height independence" was never true; the idlefix6 check also never varied
   the height. The 05:54 capture settles it: at 24 rows the screen is the real
   one, and at 40 and 50 rows the replay keeps a stale spinner and phantom
   prompt rows the real screen has scrolled away. Height is configuration now,
   and this file pins the difference rather than asserting independence.

AND ONE THING NEITHER SUITE COULD SEE, because both only ever drove
worker_idle: the idlefix5 splice dropped _SECRET and the three REJECTION_*
constants. redact() raised NameError inside record_rejection, whose bare
"except Exception: pass" swallowed it, so every rejection was silently
discarded. This file drives the sink end to end.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / 'scripts/supervise_student_specific_full_idlefix3.py'
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix7.py'
DEPLOYED_SHA = '0e6cc5a985f345a19b81ab38581e549ea2bdc6e53bc79f2dd7dba5c8aa0194b0'
REVIEWS = ROOT / 'results/v3/student_specific_full_20260918/reviews'
CAPTURES = sorted(REVIEWS.glob('idlefix_terminal_*.ansi'))
LIVE = REVIEWS / 'idlefix_terminal_20260919T0554.ansi'
ROLE = 'session2-worker'

ROOT_BASE = ('session2-worker\nCogitated for 42s · done 7:15 PM\n'
             '───────────'
             '───── session2-worker ─\n❯ ')
#: Every draft root has reproduced, plus the variants the review asked for.
DRAFTS = (
    ('the idlefix6 counterexample: spaces then Home', '   do something\x1b[3G'),
    ('plain', 'do something'),
    ('space-prefixed', '   do something'),
    ('typed then Home', 'do something\x1b[3G'),
    ('trailing padded', '  padded draft  '),
    ('right-aligned by padding', ' ' * 55 + 'do something'),
    ('reaching the right margin', 'x' * 70),
    ('multiline', 'first line\nsecond line'),
    ('a footer token typed', '/effort high'),
    ('rule characters typed', '———'),
    ('genuine blink', '\x1b[5mreal typed draft\x1b[0m'),
    ('combined dim, unclosed', '\x1b[0;2mreal typed draft'),
    ('indexed colour at home', '\x1b[38;5;2mgreen draft'),
    ('RGB colour at home', '\x1b[38;2;10;200;10mgreen draft'),
)


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
    new = load(SUCCESSOR, 'idlefix7')
    old = load(ROOT / 'scripts/supervise_student_specific_full_idlefix6.py',
               'idlefix6')

    def verdict(module, text, role=ROLE):
        reasons: list[str] = []
        ok = module.worker_idle(text, role,
                                capture=lambda _c, r, _w: reasons.append(r))
        return bool(ok), (reasons[-1] if reasons else '-')

    # -- 1. the full notification path, which no earlier suite drove ------- #
    c.check('redact() no longer raises NameError',
            new.redact('token: sk-abcdef') == 'token: <redacted>',
            new.redact('token: sk-abcdef'))
    for name in ('_SECRET', 'REJECTION_LOG_MAX_BYTES', 'REJECTION_SAMPLE_CHARS',
                 'REJECTION_REASONS'):
        c.check(f'{name} is defined again', hasattr(new, name), '')
    c.check('idlefix6 still has the defect, so this is a real fix',
            _raises_nameerror(old), 'redact() -> NameError')
    saved_root = new.ROOT
    with tempfile.TemporaryDirectory() as tmp:
        try:
            new.ROOT = Path(tmp)
            sink = (Path(tmp) / 'results/v3/student_specific_full_20260918'
                    / 'supervision/idle_rejections.jsonl')
            new.worker_idle(ROOT_BASE + 'token: sk-leakme',
                            ROLE, capture=new.record_rejection)
            wrote = sink.exists() and sink.read_text().strip()
            c.check('a rejection is actually WRITTEN to the sink', bool(wrote),
                    f'{len(wrote.splitlines()) if wrote else 0} record(s)')
            if wrote:
                rec = json.loads(wrote.splitlines()[-1])
                c.check('its reason is a known code, not "unspecified"',
                        rec['reason'] == 'text_typed_at_prompt',
                        f"reason={rec['reason']}")
                c.check('the secret is masked in what was written',
                        'sk-leakme' not in wrote and '<redacted>' in wrote, '')
                c.check('the sink file is 0600',
                        oct(sink.stat().st_mode & 0o777) == '0o600',
                        oct(sink.stat().st_mode & 0o777))
            c.check('every reason the decision can emit is in the vocabulary',
                    not (_emitted_reasons(SUCCESSOR) - set(new.REJECTION_REASONS)),
                    f'unknown: {sorted(_emitted_reasons(SUCCESSOR) - set(new.REJECTION_REASONS))}')
        finally:
            new.ROOT = saved_root

    # -- 2. every real capture, including the new 05:54 -------------------- #
    c.check('five real captures are under test', len(CAPTURES) == 5,
            ', '.join(p.name[-17:-5] for p in CAPTURES))
    for cap in CAPTURES:
        got, reason = verdict(new, cap.read_text('utf-8', 'replace'))
        c.check(f'REAL {cap.name[-17:-5]} is recognised as idle', got,
                f'reason={reason}')

    # -- 3. drafts, on root's base and injected into every real screen ----- #
    for label, suffix in DRAFTS:
        got, reason = verdict(new, ROOT_BASE + suffix)
        c.check(f'draft on root\'s base: {label}', got is False,
                f'reason={reason}')
    for cap in CAPTURES:
        raw = cap.read_text('utf-8', 'replace')
        lines, _d, _f, (_cx, cur_y) = new.render_screen(raw)
        prow = max(i for i, ln in enumerate(lines) if new.PROMPT in ln)
        up = cur_y - prow
        leaked = []
        for label, suffix in DRAFTS:
            typed = raw + (f'\x1b[{up}A' if up > 0 else '') + '\x1b[3G' + suffix
            if verdict(new, typed)[0]:
                leaked.append(label)
        c.check(f'{cap.name[-17:-5]}: all {len(DRAFTS)} drafts stay busy',
                not leaked, f'leaked: {leaked}' if leaked else '')

    # -- 4. SGR groups are copied through, never read as attributes -------- #
    for body, rewritten, opens in (
        ('38;5;114', '\x1b[38;5;114m', False),
        ('38;5;2', '\x1b[38;5;2m', False),
        ('38;2;10;200;10', '\x1b[38;2;10;200;10m', False),
        ('48;5;2', '\x1b[48;5;2m', False),
        ('58;5;2', '\x1b[58;5;2m', False),
        ('2', '\x1b[5m', True),
        ('0;2', '\x1b[0;25;5m', True),
        ('5', '', False),
        ('', '\x1b[0;25m', False),
    ):
        got = new.carry_dim_on_blink(f'\x1b[{body}m')
        c.check(f'SGR {body or "(empty)"!r}: rewrite and dim state',
                got == rewritten and new.dim_left_open(f'\x1b[{body}m') is opens,
                f'-> {got!r} dim_open={new.dim_left_open(chr(27) + "[" + body + "m")}')
    c.check('idlefix6 corrupted the indexed colour, idlefix7 does not',
            old.carry_dim_on_blink('\x1b[38;5;2m') == '\x1b[38;5m'
            and new.carry_dim_on_blink('\x1b[38;5;2m') == '\x1b[38;5;2m',
            f'idlefix6 -> {old.carry_dim_on_blink(chr(27) + "[38;5;2m")!r}')

    # -- 5. geometry is configuration, and the wrong height IS different --- #
    live = LIVE.read_text('utf-8', 'replace')
    shapes = {}
    for rows in (24, 40, 50):
        lines, _d, _f, (_x, _y) = new.render_screen(live, rows=rows)
        shapes[rows] = {
            'prompts': sum(ln.count(new.PROMPT) for ln in lines),
            'stale_spinner': any(re.search(r'esc to interrupt|ing…', ln)
                                 for ln in lines)}
    c.check('at the real height there is ONE prompt and no stale spinner',
            shapes[24]['prompts'] == 1 and not shapes[24]['stale_spinner'],
            f'{shapes[24]}')
    c.check('40 and 50 rows really do differ (height is NOT independent)',
            shapes[40] != shapes[24] and shapes[50] != shapes[24],
            f'40={shapes[40]} 50={shapes[50]}')
    c.check('the height is an explicit setting, not an assumption',
            'TERMINAL_ROWS' in SUCCESSOR.read_text(encoding='utf-8')
            and new.TERMINAL_ROWS == 24, f'TERMINAL_ROWS={new.TERMINAL_ROWS}')

    # -- 6. the composer has no vocabulary and no truncation --------------- #
    body_text = SUCCESSOR.read_text(encoding='utf-8')
    composer = body_text[body_text.index('-- the composer itself'):
                         body_text.index('-- nothing may be happening below')]
    for banned in ('FOOTER_LINE', 'BOX_LEFTOVER', 'flatten_rules', 's{3,}'):
        c.check(f'the composer never uses {banned}', banned not in composer, '')

    # -- 7. everything the earlier rounds pinned --------------------------- #
    # Place these BETWEEN the finished-turn line and the prompt, computed
    # from this capture's own layout. Hardcoding "up 4" was written for the
    # 03:54 screen and on 05:54 it lands below the prompt instead, so the
    # cases passed through a different branch and tested nothing.
    live_lines, _ld, _lf, (_lx, live_cur) = new.render_screen(live)
    live_prompt = max(i for i, ln in enumerate(live_lines) if new.PROMPT in ln)
    between = live_cur - (live_prompt - 2)
    for label, text, want in (
        ('a new turn running', '✻Pondering (esc to interrupt)',
         'busy_indicator_after_completion'),
        ('a compaction in progress', 'Compacting conversation',
         'compacting_after_completion'),
    ):
        suffix = f'\x1b[{between}A\x1b[3G' + text
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
                got is False and reason == 'terminal_parser_unavailable', '')
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
            not disagreed and total >= 38, f'{agree}/{total}')
    kills = [s for s in ('\\x15', '\\x03', '\\x1b[2K', 'C-u', 'send-keys')
             if s in body_text]
    c.check('the monitor has no way to clear a draft', not kills, f'{kills}')
    sha = hashlib.sha256(DEPLOYED.read_bytes()).hexdigest()
    c.check('the deployed monitor is untouched', sha == DEPLOYED_SHA, sha[:16])
    return c.report()


def _raises_nameerror(module) -> bool:
    try:
        module.redact('x')
    except NameError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _emitted_reasons(path: Path) -> set[str]:
    """Every literal passed to no(...) in the decision."""
    return set(re.findall(r'\bno\("([a-z_]+)"\)', path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    raise SystemExit(main())
