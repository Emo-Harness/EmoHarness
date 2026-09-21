#!/usr/bin/env python3
"""Run one construction step behind a per-call spend admission gate.

    python scripts/budget_admission.py --cap 3.00 --state <dir> \
        --step examples/build_harness.py -- --stage 3 --replicate 0 ...

Why this exists. A four-round Builder run is one subprocess: by the time a
round's cost reaches the ledger the money is spent, so a guard that polls the
ledger can only notice an overrun, never prevent one. This wrapper moves the
decision in front of the call. It patches ``Client.send`` in EVERY HTTP client
package a paid role can use, and before each request to a non-local host it
asks whether a conservative estimate of that call's cost still fits inside what
is left.

There are two such packages in this project's ``.venv``, and an earlier version
of this file patched only the first. That version was blind to the proxy judge
-- roughly 90% of a Stage-3 bill -- which is how a Gemma run recorded $0.4011 in
the ledger while the gate had accounted for $0.1492:

* ``httpx`` 0.27 -- the Builder and the reviewer construct their own clients.
* ``httpx2`` 2.12 -- a SEPARATE distribution. The OpenAI SDK's default client
  has MRO ``openai._DefaultHttpxClient -> httpx2.Client``, so its ``send`` is a
  different function in a different module and the ``httpx`` patch never saw it.

The regression that missed this passed a hand-built ``httpx.Client`` into the
SDK, so it tested the transport the test itself chose rather than the one
production builds. The test now instantiates a DEFAULT ``openai.OpenAI`` client
and mocks underneath it. ``arm`` refuses to run if any installed transport is
left ungated, and the state file records which were covered.

WHAT THIS IS NOT. It is our own bookkeeping, not a provider billing limit, and
it does not give a formal overshoot bound. Three things it cannot promise, all
of them measured and reported rather than asserted:

* The reservation is an ESTIMATE, not an upper bound. Input tokens are estimated
  from the request's own byte count, which over-counts a base64 image payload
  and can under-count dense CJK text; the output allowance is an assumption
  about how much the model may emit, not a limit the request enforces. Every
  call whose actual cost exceeded its reservation is counted in
  ``actual_over_reservation`` with the largest excess, so the assumption is
  auditable instead of trusted.
* A call already in flight when the cap is reached still completes. That is
  visible as ``reserved_in_flight_usd`` while it runs.
* A response whose body cannot be parsed for usage keeps its reservation as its
  recorded cost. That cost may be higher OR lower than what the call actually
  cost -- the reservation is an estimate, so a response we cannot read is
  accounted for by an estimate and nothing more. Those calls are counted in
  ``unparsed_responses``.

Nothing here changes a prompt, a sampling parameter, a gate, or a decision: it
only decides whether a call is allowed to leave the box, and records what left.
Local calls to the served student (127.0.0.1) are never gated -- they cost GPU
time, not dollars. Request URLs are recorded as scheme, host and path only;
query strings, userinfo and headers are never written anywhere.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import runpy
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parent.parent
STATE_FILE: Final = 'budget_admission.json'
LOCAL_HOSTS: Final = frozenset({'127.0.0.1', 'localhost', '::1', '0.0.0.0'})  # noqa: S104 -- matched against a request's host, never bound to
#: Output-token allowance assumed per paid response when sizing a reservation.
#: GPT-5 bills reasoning as output, so this sits far above any answer observed
#: in this project -- but the request does not enforce it, so it is an
#: assumption, not a cap. Override with --reserve-output-tokens.
DEFAULT_RESERVE_OUTPUT: Final = 16_000
#: Bytes per token when estimating a request's input size. Over-counts base64
#: image payloads heavily and can under-count dense multi-byte text.
BYTES_PER_TOKEN: Final = 3.5
#: Once this many calls have settled, the reservation is also floored at the
#: largest actual cost seen so far times a margin, so a systematic
#: under-estimate corrects itself instead of repeating.
LEARN_AFTER_CALLS: Final = 3
LEARNED_MARGIN: Final = 1.5

_CALL_IDS: Final = itertools.count(1)


class BudgetRefused(RuntimeError):  # noqa: N818 -- named for what it does to a call
    """Raised in place of a paid call that would not fit inside the cap."""


class StateReuseRefused(RuntimeError):  # noqa: N818 -- named for what it does to a run
    """Raised when a state directory already holds a run's accounting."""


def utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def safe_url(url: Any) -> str:
    """Scheme, host and path only.

    A request URL can carry credentials in its userinfo or query string, and
    this string is written to disk and to log lines. Everything after the path
    is dropped, and headers are never touched at all.
    """
    scheme = getattr(url, 'scheme', '') or ''
    host = getattr(url, 'host', '') or ''
    port = getattr(url, 'port', None)
    path = getattr(url, 'path', '') or ''
    authority = f'{host}:{port}' if port else host
    return f'{scheme}://{authority}{path}' if scheme or authority else '(unknown)'


@dataclass
class Pricing:
    """USD per million tokens, as the Builder config states them."""

    input: float
    output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


@dataclass
class Gate:
    """The admission decision, and the durable record of what it let through."""

    cap_usd: float
    pricing: Pricing
    state_dir: Path
    reserve_output: int = DEFAULT_RESERVE_OUTPUT
    spent_usd: float = 0.0
    carried_usd: float = 0.0
    calls_allowed: int = 0
    calls_refused: int = 0
    unparsed_responses: int = 0
    actual_over_reservation: int = 0
    max_excess_usd: float = 0.0
    max_actual_usd: float = 0.0
    in_flight: dict[int, dict[str, Any]] = field(default_factory=dict)
    per_call: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state_path(self) -> Path:
        return self.state_dir / STATE_FILE

    @property
    def reserved_usd(self) -> float:
        return sum(call['reserved_usd'] for call in self.in_flight.values())

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self.reserved_usd

    def reservation_for(self, input_tokens: int) -> float:
        """A conservative estimate of what one call may cost. Not a bound."""
        estimate = self.pricing.cost(input_tokens, self.reserve_output)
        if len(self.per_call) >= LEARN_AFTER_CALLS:
            return max(estimate, self.max_actual_usd * LEARNED_MARGIN)
        return estimate

    def admit(self, input_tokens: int, url: str) -> int:
        """Reserve for one call and persist that BEFORE it is sent, or refuse it.

        The write happens while the lock is held and before ``admit`` returns,
        so a process killed mid-call leaves a state file that says a call of
        this size was admitted and never settled. A failed write refuses the
        call: spending money we cannot account for is worse than not spending it.
        """
        with self._lock:
            reservation = self.reservation_for(input_tokens)
            if self.committed_usd + reservation > self.cap_usd:
                self.calls_refused += 1
                self.refusals.append({
                    'utc': utc(), 'url': url,
                    'reservation_usd': round(reservation, 6),
                    'spent_usd': round(self.spent_usd, 6),
                    'reserved_in_flight_usd': round(self.reserved_usd, 6),
                })
                self._write_locked()
                raise BudgetRefused(
                    f'refusing a paid call to {url}: an estimated '
                    f'${reservation:.4f} on top of ${self.spent_usd:.4f} spent and '
                    f'${self.reserved_usd:.4f} in flight would exceed this job\'s '
                    f'${self.cap_usd:.2f} cap'
                )
            call_id = next(_CALL_IDS)
            self.in_flight[call_id] = {
                'call_id': call_id, 'utc': utc(), 'url': url,
                'reserved_usd': round(reservation, 6),
                'estimated_input_tokens': input_tokens,
            }
            self.calls_allowed += 1
            try:
                self._write_locked()
            except OSError as exc:
                self.in_flight.pop(call_id, None)
                self.calls_allowed -= 1
                # Counted as a refusal, not just raised: if the step swallows
                # this the exit code must still be non-zero, exactly as for a
                # refusal at the cap. A call we cannot account for is a call we
                # did not make.
                self.calls_refused += 1
                self.refusals.append({
                    'utc': utc(), 'url': url, 'reason': 'reservation_not_persisted',
                    'error': str(exc),
                    'reservation_usd': round(reservation, 6),
                })
                raise BudgetRefused(
                    f'refusing a paid call to {url}: could not persist its '
                    f'reservation to {self.state_path} ({exc}), so its cost could '
                    f'not be accounted for'
                ) from exc
            return call_id

    def settle(self, call_id: int, usage: dict[str, int] | None, url: str) -> None:
        """Replace a reservation with the call's actual cost, or keep the estimate."""
        with self._lock:
            call = self.in_flight.pop(call_id, None)
            reservation = float(call['reserved_usd']) if call else 0.0
            if usage is None:
                self.unparsed_responses += 1
                actual = reservation
            else:
                actual = self.pricing.cost(usage['input'], usage['output'])
            self.spent_usd += actual
            self.max_actual_usd = max(self.max_actual_usd, actual)
            if usage is not None and actual > reservation:
                self.actual_over_reservation += 1
                self.max_excess_usd = max(self.max_excess_usd, actual - reservation)
            self.per_call.append({
                'call_id': call_id, 'utc': utc(), 'url': url,
                'reserved_usd': round(reservation, 6),
                'actual_usd': round(actual, 6),
                'cost_known': usage is not None,
                'usage': usage,
                'spent_after_usd': round(self.spent_usd, 6),
            })
            # The call is already made, so losing this write must not lose the
            # in-memory total; main()'s final write retries it. Spelled out
            # rather than contextlib.suppress so the reason is at the site.
            try:  # noqa: SIM105
                self._write_locked()
            except OSError:
                pass

    def payload(self) -> dict[str, Any]:
        return {
            'utc': utc(),
            'cap_usd': self.cap_usd,
            'carried_forward_usd': round(self.carried_usd, 6),
            'spent_usd': round(self.spent_usd, 6),
            'reserved_in_flight_usd': round(self.reserved_usd, 6),
            'in_flight_calls': list(self.in_flight.values()),
            'remaining_usd': round(max(0.0, self.cap_usd - self.committed_usd), 6),
            'calls_allowed': self.calls_allowed,
            'calls_refused': self.calls_refused,
            'unparsed_responses': self.unparsed_responses,
            'actual_over_reservation': self.actual_over_reservation,
            'max_excess_over_reservation_usd': round(self.max_excess_usd, 6),
            'max_actual_call_usd': round(self.max_actual_usd, 6),
            'reserve_output_tokens_assumed': self.reserve_output,
            'bytes_per_token_assumed': BYTES_PER_TOKEN,
            'pricing_usd_per_million': {'input': self.pricing.input,
                                        'output': self.pricing.output},
            'gated_transports': armed_transports(),
            'accounting': (
                'Own bookkeeping, not a provider billing limit. The per-call '
                'reservation is a conservative ESTIMATE, not an upper bound: input '
                'tokens are estimated from request bytes and the output allowance '
                'is an assumption the request does not enforce. Calls whose actual '
                'cost exceeded their reservation are counted in '
                'actual_over_reservation. A call already in flight when the cap is '
                'reached still completes. A response whose usage could not be '
                'read is accounted for at its reservation, which may be above '
                'or below its true cost.'
            ),
            'refusals': self.refusals,
            'calls': self.per_call[-200:],
        }

    def _write_locked(self) -> None:
        """Atomic publish of the accounting. Caller holds the lock."""
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(
            f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
        with temp.open('w', encoding='utf-8') as fh:
            json.dump(self.payload(), fh, ensure_ascii=False, indent=2, default=str)
            fh.write('\n')
            fh.flush()
            os.fsync(fh.fileno())
        temp.replace(path)

    def write(self) -> None:
        with self._lock:
            self._write_locked()

    def adopt_prior_state(self, resume: bool) -> dict[str, Any] | None:
        """Refuse a reused state directory, or carry its spend forward.

        A second run in a directory that already has accounting is either a
        retry or a mistake. Starting from zero would silently hand it the whole
        cap again, so by default this refuses. ``--resume-state`` carries the
        prior spend forward instead, counting any call that was admitted and
        never settled at its reservation: its true cost is unknown, and
        assuming it was free is the one assumption that cannot be corrected.
        """
        path = self.state_path
        if not path.is_file():
            return None
        try:
            prior = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise StateReuseRefused(
                f'{path} exists but cannot be read ({exc}); refusing to start '
                f'because the spend already made from this directory is unknown'
            ) from exc
        unsettled = sum(float(call.get('reserved_usd') or 0.0)
                        for call in prior.get('in_flight_calls') or [])
        carried = float(prior.get('spent_usd') or 0.0) + unsettled
        if not resume:
            raise StateReuseRefused(
                f'{path} already accounts for ${carried:.4f} of paid calls '
                f'(including ${unsettled:.4f} admitted but never settled). '
                f'Refusing to start with a fresh cap. Use a new state directory '
                f'for a new attempt, or pass --resume-state to carry the '
                f'${carried:.4f} forward against the same cap.'
            )
        with self._lock:
            self.carried_usd = carried
            self.spent_usd = carried
        retired = path.with_name(f'{STATE_FILE}.superseded.{int(time.time())}')
        path.replace(retired)
        self.write()
        return {'carried_usd': carried, 'unsettled_usd': unsettled,
                'prior_state_kept_at': str(retired)}


def _is_paid(url: Any) -> bool:
    """True when a request leaves this box for a metered provider."""
    host = getattr(url, 'host', '') or ''
    return bool(host) and host not in LOCAL_HOSTS


def _request_tokens(request: Any) -> int:
    """An estimate of one request's input tokens, from its own byte count.

    Deliberately crude and deliberately not called a bound: it over-counts a
    base64 image payload by a large factor and can under-count dense multi-byte
    text. ``Gate.reservation_for`` floors the reservation at the largest actual
    cost seen once a few calls have settled, which is what corrects a
    systematic under-estimate.
    """
    try:
        content = request.content or b''
    except Exception:
        return 0
    return int(len(content) / BYTES_PER_TOKEN) + 1


def _response_usage(response: Any) -> dict[str, int] | None:
    """The usage block of an OpenAI-shaped response, or None if unreadable."""
    try:
        if not hasattr(response, '_content'):
            response.read()
        usage = response.json().get('usage') or {}
    except Exception:
        return None
    try:
        return {'input': int(usage['prompt_tokens']),
                'output': int(usage['completion_tokens'])}
    except (KeyError, TypeError, ValueError):
        return None


#: Every HTTP client package a paid role can reach the network through.
#:
#: There are TWO, and assuming one was the defect that made this gate blind to
#: the proxy judge -- roughly 90% of a Stage-3 bill. The Builder and the
#: reviewer construct their own ``httpx.Client`` (httpx 0.27). The OpenAI SDK
#: the judge uses builds its DEFAULT client, whose MRO is
#: ``openai._DefaultHttpxClient -> httpx2.Client``: a separate distribution,
#: ``httpx2`` 2.12, whose ``send`` lives in its own module and was never
#: patched. Both expose the same ``send(self, request, *, stream, auth,
#: follow_redirects)`` and the same request/response surface the gate reads, so
#: one wrapper serves both.
TRANSPORT_MODULES: Final = ('httpx', 'httpx2')


def arm(gate: Gate) -> dict[str, Any]:
    """Patch ``Client.send`` in every transport a paid role can use.

    Returns the original methods, keyed by module name, so a test can restore
    them. A module that is not installed is skipped rather than failing: the
    point is to cover what exists, and :func:`armed_transports` reports what
    was actually covered so a silent miss cannot recur.
    """
    originals: dict[str, Any] = {}
    for name in TRANSPORT_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        client = getattr(module, 'Client', None)
        if client is None:
            continue
        original = client.send
        originals[name] = original

        def guarded(self: Any, request: Any, _original: Any = original,
                    **kwargs: Any) -> Any:
            if not _is_paid(request.url):
                return _original(self, request, **kwargs)
            url = safe_url(request.url)
            call_id = gate.admit(_request_tokens(request), url)
            usage: dict[str, int] | None = None
            try:
                response = _original(self, request, **kwargs)
                usage = _response_usage(response)
                return response
            finally:
                gate.settle(call_id, usage, url)

        client.send = guarded  # type: ignore[method-assign]
    return originals


def armed_transports() -> dict[str, bool]:
    """Which transports are installed, and whether each one is currently gated."""
    state: dict[str, bool] = {}
    for name in TRANSPORT_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        client = getattr(module, 'Client', None)
        state[name] = bool(client is not None
                           and getattr(client.send, '__name__', '') == 'guarded')
    return state


def disarm(originals: dict[str, Any]) -> None:
    """Restore the unpatched ``Client.send`` methods (tests only)."""
    for name, original in (originals or {}).items():
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        client = getattr(module, 'Client', None)
        if client is not None:
            client.send = original  # type: ignore[method-assign]


def pricing_from_builder_config(name: str) -> Pricing:
    """The price list the Builder config declares.

    ``BuilderPricing`` takes ``input``/``output`` as YAML aliases but stores
    them as ``input_usd_per_million``/``output_usd_per_million``, so reading it
    as a mapping keyed on the aliases silently missed every value and fell
    through to a default. It matched by luck on the frozen config; a config
    with different prices would have been ignored. Read the attributes.
    """
    from emoharness.builder.config import BuilderConfig
    from emoharness.common.config import load_model

    config = load_model(ROOT / 'configs' / 'builder' / name, BuilderConfig)
    return Pricing(input=float(config.pricing.input_usd_per_million),
                   output=float(config.pricing.output_usd_per_million))


def _refusal_in(exc: BaseException) -> bool:
    """Did a BudgetRefused cause this exception, however far it was wrapped?

    The Builder retries its own failures and the harness runs items on a thread
    pool, so a refusal can come back as something else entirely. The exit code
    must still say the job was stopped by the cap.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BudgetRefused):
            return True
        if 'BudgetRefused' in repr(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cap', type=float, required=True,
                        help='USD this job may spend on paid calls')
    parser.add_argument('--state', type=Path, required=True,
                        help=f'directory for {STATE_FILE}')
    parser.add_argument('--step', required=True,
                        help='entry point to run, e.g. examples/build_harness.py')
    parser.add_argument('--builder-config', default='v3_full.yaml',
                        help='configs/builder/<name> the pricing comes from')
    parser.add_argument('--reserve-output-tokens', type=int,
                        default=DEFAULT_RESERVE_OUTPUT,
                        help='output tokens assumed per call when reserving')
    parser.add_argument('--resume-state', action='store_true',
                        help='carry a reused state directory\'s spend forward '
                             'instead of refusing to start')
    parser.add_argument('step_args', nargs=argparse.REMAINDER,
                        help='arguments for the step, after --')
    args = parser.parse_args(argv)

    step_args = [a for a in args.step_args if a != '--']
    gate = Gate(cap_usd=args.cap,
                pricing=pricing_from_builder_config(args.builder_config),
                state_dir=args.state,
                reserve_output=args.reserve_output_tokens)
    try:
        adopted = gate.adopt_prior_state(args.resume_state)
    except StateReuseRefused as exc:
        print(f'{utc()} STATE REUSE REFUSED: {exc}', file=sys.stderr, flush=True)
        return 4
    if adopted:
        print(f'{utc()} carried ${adopted["carried_usd"]:.4f} forward from a prior '
              f'run in this state directory (${adopted["unsettled_usd"]:.4f} of it '
              f'admitted but never settled); prior state kept at '
              f'{adopted["prior_state_kept_at"]}', flush=True)
    arm(gate)
    gate.write()
    covered = armed_transports()
    if not all(covered.values()):
        print(f'{utc()} REFUSING TO RUN: transports {covered} -- an ungated '
              f'client package means paid calls can leave unaccounted',
              file=sys.stderr, flush=True)
        return 5
    print(f'{utc()} budget admission armed over {sorted(covered)}: cap '
          f'${args.cap:.2f}; a 100k-token '
          f'request reserves ${gate.reservation_for(100_000):.4f} as a conservative '
          f'estimate (not an upper bound); state {gate.state_path}', flush=True)
    step = ROOT / args.step
    sys.argv = [str(step), *step_args]
    started = time.time()
    code = 0
    refusal: str | None = None
    try:
        runpy.run_path(str(step), run_name='__main__')
    except SystemExit as exit_code:
        code = int(exit_code.code or 0)
    except BaseException as exc:  # noqa: BLE001 -- classify, then report
        if _refusal_in(exc):
            refusal = repr(exc)
            code = 3
            print(f'{utc()} BUDGET REFUSED: {exc!r}', file=sys.stderr, flush=True)
        else:
            raise
    # The step may have caught the refusal itself and still exited zero -- the
    # Builder retries its own errors. A refused call means the job did not do
    # what it was asked to do, so the exit code says so regardless.
    if gate.calls_refused and code == 0:
        code = 3
        print(f'{utc()} BUDGET REFUSED: the step exited 0 but '
              f'{gate.calls_refused} paid call(s) were refused at the cap',
              file=sys.stderr, flush=True)
    payload = gate.payload()
    payload.update({
        'step': args.step, 'step_args': step_args, 'returncode': code,
        'seconds': round(time.time() - started, 1), 'refusal': refusal,
        'budget_stop': bool(refusal) or gate.calls_refused > 0,
        'carried_forward': adopted,
    })
    path = gate.state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.final.{os.getpid()}.tmp')
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n',
        encoding='utf-8')
    temp.replace(path)
    print(f'{utc()} budget admission closed: ${gate.spent_usd:.4f} of '
          f'${args.cap:.2f} over {gate.calls_allowed} allowed call(s), '
          f'{gate.calls_refused} refused, {gate.actual_over_reservation} call(s) '
          f'cost more than their reservation (largest excess '
          f'${gate.max_excess_usd:.4f}); rc={code}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
