#!/usr/bin/env python3
"""Screen-buffer idle detection, replayed against every real capture.

    .venv/bin/python scripts/test_monitor_idlefix5.py

No network, no model call, no GPU, no deployment, no terminal written to.

WHY THIS IS NOT ANOTHER REGEX. Four versions keyed the decision on the
finished-turn timestamp and each died on a new repaint artefact:

  idlefix3  cursor escape inside the clock      "done\x1b[31G11:41 PM"
  idlefix4  minute never re-sent by the emitter "done 1:\x1b[28G3\x1b[30GAM"
  03:54     duration glued on by a lost escape  "Baked for \x1b[16G30\x1b[22Gdone"

The 02:53 screen ends the argument: its missing "5" is not in the byte
stream anywhere, so no amount of pattern work can recover it. The stream is
a sequence of cell updates, not text, and the only correct way to read it is
to replay it into a screen. That is what the successor does, and the clock is
then dropped from the decision entirely -- it was never load-bearing.

WHAT THE RENDER NEEDED. Two things, both asserted below:

  * LNM. All four captures use bare LF and zero CRLF, relying on the terminal
    to turn LF into CR+LF. pyte defaults that mode off, which renders every
    screen as a staircase.
  * dim. pyte 0.8.2 does not implement SGR 2, so the grey autosuggestion
    arrives indistinguishable from a typed command -- and "the ghost is not
    authorization" is the single most important rule here. SGR 2 is carried
    on blink, which the emitter never uses (asserted, 0 occurrences in all
    four captures).
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
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix5.py'
DEPLOYED_SHA = '0e6cc5a985f345a19b81ab38581e549ea2bdc6e53bc79f2dd7dba5c8aa0194b0'
REVIEWS = ROOT / 'results/v3/student_specific_full_20260918/reviews'
CAPTURES = sorted(REVIEWS.glob('idlefix_terminal_*.ansi'))
ROLE = 'session2-worker'
#: The 03:54 screen root rejected idlefix4 on. Offsets from its prompt row are
#: used to inject counter-examples at an exact place on the screen.
LIVE = REVIEWS / 'idlefix_terminal_20260919T0354.ansi'


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
    new = load(SUCCESSOR, 'idlefix5')
    old = load(DEPLOYED, 'idlefix3_deployed')

    def verdict(module, text, role=ROLE):
        reasons: list[str] = []
        ok = module.worker_idle(text, role,
                                capture=lambda _c, r, _w: reasons.append(r))
        return bool(ok), (reasons[-1] if reasons else '-')

    # -- 1. every real capture is a quiet idle screen and must be woken --- #
    c.check('four real captures are under test', len(CAPTURES) == 4,
            ', '.join(p.name[-17:-5] for p in CAPTURES))
    for cap in CAPTURES:
        raw = cap.read_text('utf-8', 'replace')
        got, reason = verdict(new, raw)
        was, was_reason = verdict(old, raw)
        c.check(f'REAL {cap.name[-17:-5]} is recognised as idle', got,
                f'idlefix3={was}({was_reason}) -> idlefix5={got}({reason})')

    live = LIVE.read_text('utf-8', 'replace')
    lines, dim, (cur_x, cur_y) = new.render_screen(live)
    prompt_row = max(i for i, ln in enumerate(lines) if new.PROMPT in ln)

    # -- 2. the grey suggestion is not authorization ---------------------- #
    ghost = [y for y, ln in enumerate(lines)
             if 'Read the next manager task' in ln]
    c.check('the dim suggestion really is on the rendered screen', bool(ghost),
            f'row {ghost[0] if ghost else None}')
    if ghost:
        y = ghost[0]
        plain = new.visible(lines[y], dim[y], want_dim=False).strip()
        shaded = new.visible(lines[y], dim[y], want_dim=True).strip()
        c.check('the suggestion is entirely dim, so it is never typed input',
                not plain and shaded.startswith('Read the next'),
                f'non-dim={plain!r} dim={shaded[:40]!r}')

    # -- 3. counter-examples, injected at an exact row of the REAL screen -- #
    #     cursor sits 3 rows under the prompt, so -4 lands between the
    #     finished-turn line and the prompt, and +1 lands under the prompt.
    for label, suffix, want_reason in (
        ('a typed draft at the composer',
         '\x1b[3A\x1b[3Grm -rf /important/data', 'text_typed_at_prompt'),
        ('a new turn running',
         '\x1b[4A\x1b[3G✻Pondering (esc to interrupt)',
         'busy_indicator_after_completion'),
        ('a compaction in progress',
         '\x1b[4A\x1b[3GCompacting conversation', 'compacting_after_completion'),
        ('a shell prompt under the composer',
         '\x1b[1B\x1b[1Gmax7@box:~/EmoHarness$ ls -la',
         'unrecognised_line_below_prompt'),
    ):
        got, reason = verdict(new, live + suffix)
        c.check(f'{label} keeps the worker busy',
                got is False and reason == want_reason,
                f'idle={got} reason={reason} want={want_reason}')

    # -- 3b. three defects this suite caught in the successor itself ------- #
    #      Found by replaying the older suites through it, not by review.
    unclosed = (f'\x1b]0;claude\x07{ROLE}\r\n'
                f'bypass permissions on (shift+tab to cycle)\r\n'
                f'Cogitated for 42s · done 9:36 PM\r\n'
                f'❯ \x1b[2msudo rm -rf /important\x1b[8Gdata\r\n')
    got, reason = verdict(new, unclosed)
    c.check('an unterminated dim span is refused, not read as a suggestion',
            got is False and reason == 'unterminated_dim_span',
            f'idle={got} reason={reason} -- otherwise a real draft renders '
            'grey and would be pasted over')
    for cap in CAPTURES:
        raw = cap.read_text('utf-8', 'replace')
        opened = raw.rfind(new.DIM_OPEN)
        c.check(f'{cap.name[-17:-5]}: its last dim span IS closed',
                opened >= 0 and bool(
                    new.DIM_CLOSE.search(raw, opened + len(new.DIM_OPEN))),
                'so the rule above costs nothing on a healthy screen')
    redrawn = ('\x1b]0;claude\x07  session2-worker\n'
               '  Cogitated for 42s · done 7:15 PM\n'
               '  bypass permissions on (shift+tab to cycle)\n❯ ')
    c.check('the cursor resting ON the composer row is still idle',
            verdict(new, redrawn)[0] is True, f'{verdict(new, redrawn)}')
    updating = ('\x1b]0;claude\x07  session2-worker\n'
                '  bypass permissions on (shift+tab to cycle)\n'
                '  Cogitated for 42s · done 7:15 PM\n❯ \n'
                'Checking for updates\n')
    c.check('the updater status under the prompt is still idle',
            verdict(new, updating)[0] is True, f'{verdict(new, updating)}')

    # -- 3c. every case the four earlier suites pin, replayed -------------- #
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

    # -- 4. ownership ------------------------------------------------------ #
    got, reason = verdict(new, live, 'session1-worker')
    c.check('another worker\'s screen is not woken as this one',
            got is False and reason == 'role_absent', f'reason={reason}')

    # -- 5. an OLD spinner higher up the screen must not block idle -------- #
    #     grounded in real data: the 03:54 screen carries a finished
    #     "Gusting..." status above its completion line and is still idle.
    stale = [y for y, ln in enumerate(lines) if 'Gusting' in ln]
    done_rows = [y for y in range(prompt_row)
                 if new.TURN_SUMMARY.search(new.BOX_LEFTOVER.sub(' ', lines[y]))]
    c.check('a spent spinner above the finished turn does not hold it busy',
            bool(stale) and bool(done_rows) and stale[0] < done_rows[-1]
            and verdict(new, live)[0],
            f'spinner row {stale[0] if stale else None} < done row '
            f'{done_rows[-1] if done_rows else None}')

    # -- 6. fail closed ----------------------------------------------------- #
    original = new.load_pyte
    try:
        new.load_pyte = lambda: None
        got, reason = verdict(new, live)
        c.check('no terminal emulator means NOT idle, never idle',
                got is False and reason == 'terminal_parser_unavailable',
                f'reason={reason}')
    finally:
        new.load_pyte = original
    got, reason = verdict(new, 'not a terminal at all')
    c.check('junk input is not idle', got is False, f'reason={reason}')

    # -- 7. the two rendering facts the decision rests on -------------------- #
    for cap in CAPTURES:
        raw = cap.read_text('utf-8', 'replace')
        crlf = len(re.findall(r'\r\n', raw))
        blink = len(re.findall(r'\x1b\[5m', raw))
        c.check(f'{cap.name[-17:-5]}: bare LF only, and SGR 5 free',
                crlf == 0 and blink == 0,
                f'CRLF={crlf} SGR5={blink} '
                f'dim={len(re.findall(r"\x1b\[2m", raw))}')
    c.check('pyte alone cannot see dim, which is why it is carried on blink',
            _pyte_lacks_dim(), 'SGR 2 absent from pyte graphics.TEXT')
    c.check('without LNM the same screen does not render',
            _needs_lnm(new, live), 'staircase without newline mode')

    # -- 8. the decision does not depend on a guessed screen height --------- #
    verdicts = {rows: new.worker_idle(live, ROLE)
                for rows in (24, 30, 40, 50, 60)}
    c.check('height does not change the verdict', len(set(map(bool, verdicts.values()))) == 1,
            f'{ {k: bool(v) for k, v in verdicts.items()} }')

    # -- 9. a draft is never touched ---------------------------------------- #
    body = SUCCESSOR.read_text(encoding='utf-8')
    kills = [s for s in ('\\x15', '\\x03', '\\x1b[2K', 'C-u', 'send-keys')
             if s in body]
    c.check('the monitor has no way to clear a draft', not kills,
            f'no kill-line/interrupt sequences; found {kills}')
    c.check('stdin is written only after idle is confirmed',
            body.count('connection.send') == 2
            and body.index('if not quiet or not worker_idle')
            < body.index('connection.send'), '')
    before = live
    new.worker_idle(live, ROLE)
    c.check('worker_idle does not mutate its input', before == live, '')

    # -- 10. isolation ------------------------------------------------------ #
    sha = hashlib.sha256(DEPLOYED.read_bytes()).hexdigest()
    c.check('the deployed monitor is untouched', sha == DEPLOYED_SHA, sha[:16])
    return c.report()


def _pyte_lacks_dim() -> bool:
    sys.path.append('/tmp/emoharness-terminal-parser')
    try:
        from pyte import graphics
    except Exception:  # noqa: BLE001
        return False
    return 2 not in graphics.TEXT


def _needs_lnm(module, text) -> bool:
    """Render the live screen with newline mode OFF and show it breaks."""
    loaded = module.load_pyte()
    if loaded is None:
        return False
    pyte, _modes = loaded
    screen = pyte.Screen(module.detect_cols(text), 50)
    pyte.Stream(screen).feed(text)
    lines = [ln.rstrip() for ln in screen.display]
    # with LNM the prompt sits alone on its row; without it the emitter's
    # bare LFs leave it indented behind whatever the previous line ended with.
    rows = [ln for ln in lines if module.PROMPT in ln]
    return not any(ln.lstrip().startswith(module.PROMPT) and ln.index(
        module.PROMPT) < 4 for ln in rows) or len(rows) == 0


if __name__ == '__main__':
    raise SystemExit(main())
