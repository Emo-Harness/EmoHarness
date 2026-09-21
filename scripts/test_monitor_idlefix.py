#!/usr/bin/env python3
"""Terminal replay tests for the monitor's idle parser.

    .venv/bin/python scripts/test_monitor_idlefix.py

No network, no model call, no GPU, no deployment. Every case is a byte-accurate
ANSI terminal capture replayed through ``worker_idle`` in BOTH the deployed
monitor and the isolated successor, so each row shows what changes and what
does not.

The bug being fixed: a finished Claude turn ends with a line like
"Cogitated for 42s · done 7:15 PM". The deployed matcher required that line to
be the LAST thing before the redrawn prompt, with a middot separator and a
12-hour AM/PM clock. A collapsed or compressed view breaks those incidental
details -- the status footer can land between the completion line and the
prompt, the separator can differ, the clock can be 24-hour -- and the worker is
then judged busy and never woken. session1-worker went unwoken from 19:08 with
manager messages unread.

What must NOT change: real typed input, a busy/thinking view, and a shell
prompt must all still fail closed. Those cases are replayed here too, and a
patch that woke a worker mid-typing would fail this suite.

Honest limit, stated because it bounds what this proves: I do not have the
actual capture from session1-worker's terminal at 19:08, so these fixtures are
reconstructions of the compressed layouts the renderer can produce, not that
exact byte sequence. The successor therefore also records the rejected tail on
every negative decision, so the next occurrence yields a real sample instead of
another reconstruction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / 'scripts/supervise_student_specific_full.py'
SUCCESSOR = ROOT / 'scripts/supervise_student_specific_full_idlefix.py'
ROLE = 'session2-worker'

DIM = '\x1b[2m'
UNDIM = '\x1b[22m'
RESET = '\x1b[0m'
OSC = '\x1b]0;claude\x07'
FOOTER = '  bypass permissions on (shift+tab to cycle)'


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def classic_done() -> str:
    """The layout the deployed matcher was written against. Must stay idle.

    Note the order: the status footer comes BEFORE the completion line, which
    is the last thing before the redrawn prompt. My first draft put the footer
    after the done line, and the deployed matcher rejected it -- meaning the
    "regression" case was really another compressed case and proved nothing.
    Getting this order right is what lets the suite show that the patch changes
    only the compressed layouts and leaves the working one alone.
    """
    return (f'{OSC}  Running {ROLE} diagnostics\n'
            f'{FOOTER}\n'
            f'\n  Cogitated for 42s · done 7:15 PM\n'
            f'❯ \n')


def compressed_footer_after_done() -> str:
    """Collapsed view: the status footer lands BETWEEN done and the prompt."""
    return (f'{OSC}  {ROLE} run\n'
            f'  Cogitated for 42s · done 7:15 PM\n'
            f'{FOOTER}\n'
            f'❯ \n')


def compressed_24h_clock() -> str:
    """A 24-hour clock with no AM/PM."""
    return (f'{OSC}  {ROLE}\n'
            f'  Pondered for 3m 8s · done 19:08\n'
            f'{FOOTER}\n'
            f'❯ \n')


def compressed_alt_separator() -> str:
    """A dash instead of the middot, and no blank line anywhere."""
    return (f'{OSC}  {ROLE}\n'
            f'  Thought for 12s - done 7:15 PM\n'
            f'{FOOTER}\n'
            f'❯\n')


def compressed_no_trailing_newline() -> str:
    """Prompt redrawn immediately, tail is a bare space."""
    return (f'{OSC}  {ROLE}\n'
            f'  Cogitated for 42s · done 7:15 PM\n{FOOTER}\n❯ ')


def dim_ghost_suggestion() -> str:
    """An unaccepted suggestion rendered dim. Not typed input."""
    return (f'{OSC}  {ROLE}\n'
            f'  Cogitated for 9s · done 7:16 PM\n'
            f'{FOOTER}\n'
            f'❯ {DIM}Try "relaunch the queue"{UNDIM}\n')


def dim_ghost_that_looks_typed() -> str:
    """A dim suggestion whose text would read as a command if not stripped."""
    return (f'{OSC}  {ROLE}\n'
            f'  Cogitated for 9s · done 7:16 PM\n'
            f'{FOOTER}\n'
            f'❯ {DIM}git push --force origin main{UNDIM}\n')


def updater_tail() -> str:
    """The updater status after the prompt is still idle."""
    return (f'{OSC}  {ROLE}\n'
            f'{FOOTER}\n'
            f'  Cogitated for 42s · done 7:15 PM\n'
            f'❯ \nChecking for updates\n')


def busy_after_done_before_prompt() -> str:
    """Root's reproduction: a NEW turn started after the old one finished.

    done-line, then a live busy indicator, then the prompt is redrawn. The
    worker is working. MUST stay busy. My first successor returned True here
    because it accepted a completion line anywhere in the window.
    """
    return (f'{OSC}{ROLE}\n'
            f'bypass permissions on (shift+tab to cycle)\n'
            f'Cogitated for 42s \u00b7 done 7:15 PM\n'
            f'Thinking (esc to interrupt)\n'
            f'\u276f ')


def busy_after_done_cogitating() -> str:
    """Same shape, different in-flight word. MUST stay busy."""
    return (f'{OSC}{ROLE}\n{FOOTER}\n'
            f'Cogitated for 42s \u00b7 done 7:15 PM\n'
            f'  Cogitating\u2026\n'
            f'\u276f ')


def token_on_unknown_line() -> str:
    """Root's leak reproduction: a fake token on an unrecognised line.

    This is the input that put the token into the REASON field. The tail was
    masked but the reason interpolated the offending line verbatim, so the
    secret reached disk anyway. MUST stay busy, and must leave no trace of the
    token anywhere in the sink.
    """
    return f'{OSC}{ROLE}\n\u276f \nsk-FAKE-REVIEW-ONLY\n'


def real_typed_input() -> str:
    """A human is mid-command. MUST stay busy."""
    return (f'{OSC}  {ROLE}\n'
            f'  Cogitated for 42s · done 7:15 PM\n'
            f'{FOOTER}\n'
            f'❯ git push --force origin main')


def busy_thinking() -> str:
    """A turn in flight. MUST stay busy."""
    return (f'{OSC}  {ROLE}\n'
            f'❯ \n  Cogitating... (esc to interrupt)\n')


def shell_prompt() -> str:
    """A shell, not Claude. MUST stay busy."""
    return f'{OSC}❯ \nmax7@jupyter-max7:~/EmoHarness$ \n'


def typed_after_footer() -> str:
    """Footer present but a real line typed after it. MUST stay busy."""
    return (f'{OSC}  {ROLE}\n'
            f'  Cogitated for 42s · done 7:15 PM\n'
            f'❯ \n{FOOTER}\nplease stop the run\n')


def no_role() -> str:
    """Another worker's terminal. MUST stay busy for this role."""
    return (f'{OSC}  session1-worker\n'
            f'  Cogitated for 42s · done 7:15 PM\n{FOOTER}\n❯ \n')


#: name, capture, expected-idle, whether the deployed monitor is expected to
#: already agree (used only to report what the patch actually changes)
CASES = [
    ('classic done footer (regression)', classic_done, True),
    ('compressed: footer between done and prompt', compressed_footer_after_done, True),
    ('compressed: 24-hour clock, no AM/PM', compressed_24h_clock, True),
    ('compressed: dash separator, no blank lines', compressed_alt_separator, True),
    ('compressed: prompt redrawn with no trailing newline',
     compressed_no_trailing_newline, True),
    ('dim ghost suggestion at the prompt', dim_ghost_suggestion, True),
    ('dim ghost that reads like a command', dim_ghost_that_looks_typed, True),
    ('updater status after the prompt', updater_tail, True),
    ('busy indicator AFTER done, before prompt, must stay busy',
     busy_after_done_before_prompt, False),
    ('a second turn cogitating after done must stay busy',
     busy_after_done_cogitating, False),
    ('a token on an unknown line must stay busy', token_on_unknown_line, False),
    ('REAL typed input must stay busy', real_typed_input, False),
    ('thinking / esc-to-interrupt must stay busy', busy_thinking, False),
    ('shell prompt must stay busy', shell_prompt, False),
    ('typed line after the footer must stay busy', typed_after_footer, False),
    ('another role\'s terminal must stay busy', no_role, False),
]


class _StubWS:
    """A scripted terminal websocket. No network; no Jupyter; no real token."""

    class WebSocketTimeoutException(Exception):  # noqa: N818 -- must mirror the websocket module's own name
        pass

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent: list[str] = []

    # module-surface used by notify_idle
    def create_connection(self, *_a, **_k):
        return self

    def recv(self):
        if self._frames:
            return self._frames.pop(0)
        raise _StubWS.WebSocketTimeoutException()

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


def drive_notify_idle(module, terminal_text, tmp, monkey_env):
    """Run the REAL notify_idle against a stubbed socket and return its status.

    Root's second finding was that the capture callback was never passed at
    the call site, so the claim that a rejection leaves a sample was untrue.
    Only exercising notify_idle itself can prove the wiring; asserting on
    worker_idle alone cannot.
    """
    import json as _json
    import sys as _sys

    stub = _StubWS([_json.dumps(['stdout', terminal_text])])
    _sys.modules['websocket'] = stub
    old_env = {k: os.environ.get(k) for k in
               ('JUPYTERHUB_API_TOKEN', 'JUPYTERHUB_SERVICE_URL',
                'JUPYTERHUB_SERVICE_PREFIX')}
    os.environ['JUPYTERHUB_API_TOKEN'] = monkey_env
    os.environ.setdefault('JUPYTERHUB_SERVICE_URL', 'http://127.0.0.1:8888')
    old_root = module.ROOT
    module.ROOT = tmp
    try:
        status = module.notify_idle('session2', '2', 'diagnostic please')
    finally:
        module.ROOT = old_root
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _sys.modules.pop('websocket', None)
    return status, stub


def _offvocab_is_scrubbed(module, tmp, secret) -> bool:
    """A future edit that passes free text as a reason must not leak it."""
    old_root = module.ROOT
    module.ROOT = tmp
    try:
        module.record_rejection('tail', f'leaked {secret} in a free-text reason', '')
    finally:
        module.ROOT = old_root
    body = (tmp / 'results/v3/student_specific_full_20260918/supervision'
            / 'idle_rejections.jsonl').read_text(encoding='utf-8')
    last = json.loads(body.splitlines()[-1])
    return last['reason'] == 'unspecified' and secret not in body


def main() -> int:
    deployed = load(DEPLOYED, 'monitor_deployed')
    successor = load(SUCCESSOR, 'monitor_successor')
    rows: list[tuple[str, bool, str]] = []
    changed: list[str] = []
    for name, make, expected in CASES:
        text = make()
        old = bool(deployed.worker_idle(text, ROLE))
        captured: list[tuple[str, str]] = []

        def record(cleaned, reason, _raw, sink=captured):
            sink.append((reason, cleaned))

        new = bool(successor.worker_idle(text, ROLE, capture=record))
        ok = new is expected
        note = f'old={old} new={new} want={expected}'
        if not new and captured:
            note += f' | reason: {captured[-1][0][:52]}'
        if old != new:
            changed.append(f'{name}: {old} -> {new}')
        rows.append((name, ok, note))

    # --- the REAL notify_idle path, both outcomes -------------------------- #
    tmp = Path(tempfile.mkdtemp(prefix='idlefix-'))
    log = tmp / 'results/v3/student_specific_full_20260918/supervision/idle_rejections.jsonl'
    token = 'sk-DO-NOT-LOG-testtoken123456'  # noqa: S105 -- a fake, asserted never to reach disk
    try:
        # busy terminal: must refuse AND must now leave a sample on disk
        status, stub = drive_notify_idle(
            successor, busy_after_done_before_prompt(), tmp, token)
        wrote = log.is_file()
        rows.append(('notify_idle refuses a busy terminal',
                     status == 'busy_or_unverified_queued', f'status={status}'))
        rows.append(('notify_idle sends nothing when busy', not stub.sent,
                     f'{len(stub.sent)} frame(s) sent'))
        rows.append(('the capture callback IS wired: a rejection is recorded',
                     wrote, f'{log} exists={wrote}'))
        if wrote:
            entry = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
            rows.append(('the record names the reason as a fixed code',
                         entry['reason'] == 'busy_indicator_after_completion',
                         entry['reason']))
            rows.append(('the record is bounded',
                         len(entry['tail']) <= 1200, f'{len(entry["tail"])} chars'))
            rows.append(('the record leaks no token',
                         token not in log.read_text(encoding='utf-8'), ''))
            rows.append(('the log file is 0600',
                         oct(log.stat().st_mode & 0o777) == '0o600',
                         oct(log.stat().st_mode & 0o777)))
        # root's leak reproduction, through the real path: the fake token must
        # appear NOWHERE in the sink -- not in the tail, not in the reason.
        leak_token = 'sk-FAKE-REVIEW-ONLY'  # noqa: S105 -- root's reproduction, asserted never to reach disk
        status, stub = drive_notify_idle(
            successor, token_on_unknown_line(), tmp, token)
        body = log.read_text(encoding='utf-8') if log.is_file() else ''
        last = json.loads(body.splitlines()[-1]) if body else {}
        rows.append(('the leak reproduction is still refused',
                     status == 'busy_or_unverified_queued', f'status={status}'))
        rows.append(('the fake token appears NOWHERE in the sink file',
                     leak_token not in body, 'searched the whole file'))
        rows.append(('the reason field carries no terminal text',
                     last.get('reason') in successor.REJECTION_REASONS,
                     f'reason={last.get("reason")!r}'))
        rows.append(('every reason emitted is a fixed code',
                     all(json.loads(line)['reason'] in successor.REJECTION_REASONS
                         for line in body.splitlines() if line.strip()),
                     f'{len(body.splitlines())} record(s)'))
        rows.append(('an off-vocabulary reason is replaced, not written',
                     _offvocab_is_scrubbed(successor, tmp, leak_token),
                     'sink rewrites unknown reasons to "unspecified"'))

        # idle terminal: must send paste then CR, and record nothing new
        before_lines = len(log.read_text(encoding='utf-8').splitlines()) if log.is_file() else 0
        status, stub = drive_notify_idle(
            successor, compressed_footer_after_done(), tmp, token)
        after_lines = len(log.read_text(encoding='utf-8').splitlines()) if log.is_file() else 0
        rows.append(('notify_idle wakes a genuinely idle worker',
                     status == 'sent_awaiting_ack', f'status={status}'))
        # Assert on the DECODED frames. json.dumps escapes ESC as  and
        # CR as \r, so matching raw substrings tests the encoder, not the
        # behaviour -- that is what made my first version of this fail.
        decoded = [json.loads(frame) for frame in stub.sent]
        rows.append(('it sends exactly paste then CR, in that order',
                     len(decoded) == 2
                     and decoded[0][0] == 'stdin'
                     and decoded[0][1].startswith('\x1b[200~')
                     and decoded[0][1].endswith('\x1b[201~')
                     and decoded[1] == ['stdin', '\r'],
                     f'{len(decoded)} frames'))
        rows.append(('no rejection recorded when it succeeds',
                     after_lines == before_lines,
                     f'{before_lines} -> {after_lines} lines'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(n) for n, _, _ in rows)
    for name, ok, note in rows:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {note}')
    failed = [n for n, ok, _ in rows if not ok]
    print(f'\n{len(rows) - len(failed)}/{len(rows)} cases pass')
    print('\nWHAT THE PATCH CHANGES (deployed -> successor):')
    for line in changed:
        print(f'  {line}')
    if not changed:
        print('  nothing -- the patch would be pointless')
    print('\nWHAT IT DOES NOT CHANGE: every must-stay-busy case above is '
          'unchanged and still rejects.')
    if failed:
        print('FAILED: ' + '; '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
