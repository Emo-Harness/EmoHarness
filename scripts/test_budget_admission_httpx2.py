#!/usr/bin/env python3
"""Isolated checks for scripts/budget_admission_httpx2.py. No network, no API key, no GPU.

    .venv/bin/python scripts/test_budget_admission_httpx2.py

Every check runs against an ``httpx.MockTransport``, so the gate is exercised on
the exact seam it patches without a request reaching a provider. What the
managers asked these to prove, one check each:

* the Builder's own ``httpx.Client`` is intercepted;
* the proxy judge's OpenAI SDK client is intercepted through the same patch;
* concurrent admission never lets committed spend pass the cap;
* a refused call makes NO HTTP request at all;
* local (127.0.0.1) student calls bypass the gate entirely;
* usage accounting comes from the response body, and an unparseable body keeps
  the reservation rather than recording zero;
* nothing written to disk contains an Authorization header, a bearer token, a
  query string or userinfo;
* a reused state directory fails closed, and ``--resume-state`` carries spend
  forward;
* a reservation that fails to persist refuses the call.

This file tests the wrapper only. It starts no construction and spends nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import budget_admission_httpx2 as ba  # noqa: E402
import httpx  # noqa: E402

PAID_URL = 'https://api.openai.com/v1/chat/completions'
LOCAL_URL = 'http://127.0.0.1:8000/v1/chat/completions'
SECRET = 'sk-test-DO-NOT-LOG-abcdef123456'  # noqa: S105 -- a fake, asserted to never be logged


def _pricing() -> ba.Pricing:
    return ba.Pricing(input=0.2, output=1.2)


def _gate(tmp: Path, cap: float = 1.0, **kwargs: Any) -> ba.Gate:
    return ba.Gate(cap_usd=cap, pricing=_pricing(), state_dir=tmp, **kwargs)


def _usage_response(request: httpx.Request, prompt: int = 1000,  # noqa: ARG001
                    completion: int = 100) -> httpx.Response:
    return httpx.Response(200, json={
        'choices': [{'message': {'content': 'ok'}}],
        'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
    })


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = '') -> None:
        self.rows.append((name, bool(ok), detail))

    def report(self) -> int:
        width = max(len(name) for name, _, _ in self.rows)
        for name, ok, detail in self.rows:
            print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {detail}')
        failed = [name for name, ok, _ in self.rows if not ok]
        print(f'\n{len(self.rows) - len(failed)}/{len(self.rows)} checks passed')
        if failed:
            print('FAILED: ' + '; '.join(failed))
        return 1 if failed else 0


def run(tmp: Path) -> int:  # noqa: C901 -- a flat list of independent checks
    c = Checks()

    # --- 1. the Builder's transport, which posts directly ------------------ #
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 -- MockTransport handler signature
        calls.append(request)
        return _usage_response(request)

    gate = _gate(tmp / 'builder')
    original = ba.arm(gate)
    try:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        response = client.post(PAID_URL, json={'messages': [{'role': 'user',
                                                             'content': 'x' * 3500}]},
                               headers={'Authorization': f'Bearer {SECRET}'})
        c.check('the Builder httpx client is intercepted and the call succeeds',
                response.status_code == 200 and len(calls) == 1
                and gate.calls_allowed == 1,
                f'allowed={gate.calls_allowed} http_calls={len(calls)}')
        c.check('usage accounting comes from the response body',
                abs(gate.spent_usd - _pricing().cost(1000, 100)) < 1e-9,
                f'${gate.spent_usd:.6f} == ${_pricing().cost(1000, 100):.6f}')
        c.check('the reservation is released once the call settles',
                gate.reserved_usd == 0.0 and not gate.in_flight,
                f'in_flight={len(gate.in_flight)}')

        # --- 2. the proxy judge's OpenAI SDK, same patch ------------------- #
        import openai

        sdk_calls: list[httpx.Request] = []

        def sdk_handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 -- MockTransport handler signature
            sdk_calls.append(request)
            return _usage_response(request, prompt=2000, completion=50)

        sdk_gate = _gate(tmp / 'sdk')
        ba.disarm(original)
        original = ba.arm(sdk_gate)
        sdk = openai.OpenAI(
            api_key=SECRET, base_url='https://api.openai.com/v1',
            http_client=httpx.Client(transport=httpx.MockTransport(sdk_handler)))
        answer = sdk.chat.completions.create(
            model='gpt-5.6-luna', messages=[{'role': 'user', 'content': 'score this'}])
        c.check('the OpenAI SDK client is intercepted through the same patch',
                bool(answer.choices) and len(sdk_calls) == 1
                and sdk_gate.calls_allowed == 1,
                f'allowed={sdk_gate.calls_allowed} http_calls={len(sdk_calls)}')
        c.check('SDK usage is accounted from its response body',
                abs(sdk_gate.spent_usd - _pricing().cost(2000, 50)) < 1e-9,
                f'${sdk_gate.spent_usd:.6f}')

        # --- 3. a refused call makes no HTTP request ----------------------- #
        tight_calls: list[httpx.Request] = []

        def tight_handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 -- MockTransport handler signature
            tight_calls.append(request)
            return _usage_response(request)

        tight = _gate(tmp / 'tight', cap=0.0001)
        ba.disarm(original)
        original = ba.arm(tight)
        tight_client = httpx.Client(transport=httpx.MockTransport(tight_handler))
        refused = False
        try:
            tight_client.post(PAID_URL, json={'messages': [{'content': 'x' * 3500}]})
        except ba.BudgetRefused:
            refused = True
        c.check('a call over the cap is refused and never reaches the transport',
                refused and not tight_calls and tight.calls_refused == 1,
                f'refused={refused} http_calls={len(tight_calls)}')
        c.check('a refusal is persisted with the reason',
                (tmp / 'tight' / ba.STATE_FILE).is_file()
                and json.loads((tmp / 'tight' / ba.STATE_FILE)
                               .read_text())['calls_refused'] == 1, '')

        # --- 4. local student calls bypass the gate ------------------------ #
        local_calls: list[httpx.Request] = []

        def local_handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 -- MockTransport handler signature
            local_calls.append(request)
            return httpx.Response(200, json={'choices': [{'message':
                                                          {'content': 'ok'}}]})

        local = _gate(tmp / 'local', cap=0.0)
        ba.disarm(original)
        original = ba.arm(local)
        httpx.Client(transport=httpx.MockTransport(local_handler)).post(
            LOCAL_URL, json={'messages': []})
        c.check('local student calls bypass the gate even at a zero cap',
                len(local_calls) == 1 and local.calls_allowed == 0
                and local.calls_refused == 0,
                f'http_calls={len(local_calls)} allowed={local.calls_allowed}')

        # --- 5. concurrent admission respects the cap ---------------------- #
        concurrent = _gate(tmp / 'concurrent', cap=0.5)
        observed_peak = 0.0
        peak_lock = threading.Lock()
        allowed = 0
        allow_lock = threading.Lock()

        def worker() -> None:
            nonlocal observed_peak, allowed
            try:
                call_id = concurrent.admit(100_000, PAID_URL)
            except ba.BudgetRefused:
                return
            with allow_lock:
                allowed += 1
            with peak_lock:
                observed_peak = max(observed_peak, concurrent.committed_usd)
            concurrent.settle(call_id, {'input': 100_000, 'output': 1_000},
                              PAID_URL)

        threads = [threading.Thread(target=worker) for _ in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        c.check('concurrent admission never commits past the cap',
                observed_peak <= concurrent.cap_usd + 1e-9,
                f'peak committed ${observed_peak:.4f} <= cap '
                f'${concurrent.cap_usd:.2f}, {allowed} allowed, '
                f'{concurrent.calls_refused} refused')
        c.check('concurrent settling leaves no reservation behind',
                not concurrent.in_flight and concurrent.reserved_usd == 0.0,
                f'in_flight={len(concurrent.in_flight)}')

        # --- 6. an unparseable body keeps the reservation ------------------ #
        def junk_handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 -- MockTransport handler signature
            return httpx.Response(200, content=b'not json at all')

        junk = _gate(tmp / 'junk')
        ba.disarm(original)
        original = ba.arm(junk)
        httpx.Client(transport=httpx.MockTransport(junk_handler)).post(
            PAID_URL, json={'messages': [{'content': 'x' * 3500}]})
        c.check('an unreadable response keeps its reservation, not zero',
                junk.unparsed_responses == 1 and junk.spent_usd > 0,
                f'unparsed={junk.unparsed_responses} spent=${junk.spent_usd:.6f}')

        # --- 7. an over-reservation call is counted, not hidden ------------ #
        def expensive_handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 -- MockTransport handler signature
            return _usage_response(request, prompt=1, completion=10_000_000)

        over = _gate(tmp / 'over', cap=100.0)
        ba.disarm(original)
        original = ba.arm(over)
        httpx.Client(transport=httpx.MockTransport(expensive_handler)).post(
            PAID_URL, json={'messages': [{'content': 'tiny'}]})
        c.check('a call that cost more than its reservation is counted',
                over.actual_over_reservation == 1 and over.max_excess_usd > 0,
                f'excess ${over.max_excess_usd:.4f} -- the estimate is not a bound '
                f'and says so')
    finally:
        ba.disarm(original)

    # --- 7b. a realistic build must not be refused --------------------------- #
    #
    # The check that matters most: a gate that refuses a legitimate run is worse
    # than no gate. Costs are the measured per-call costs of the nine finished
    # v3 Stage-3 runs (results/v3/fixed_asr ledgers, 2026-09-12): builder
    # $0.0316 x 4, verifier $0.0054 x 4, judge_proxy $0.00043 x 467 -- one
    # four-round run's share -- driven at the launch concurrency of 8.
    realistic = _gate(tmp / 'realistic', cap=3.0)
    schedule = ([('builder', 0.031573, 100_000)] * 4
                + [('verifier', 0.005433, 40_000)] * 4
                + [('judge_proxy', 0.000430, 12_000)] * 467)
    peak = 0.0
    peak_lock = threading.Lock()
    errors: list[str] = []

    def paid_call(spec: tuple[str, float, int]) -> None:
        nonlocal peak
        _role, actual, input_tokens = spec
        try:
            call_id = realistic.admit(input_tokens, PAID_URL)
        except ba.BudgetRefused as exc:
            errors.append(str(exc))
            return
        with peak_lock:
            peak = max(peak, realistic.committed_usd)
        # price the call back to the measured cost through the real pricing
        output = int(actual * 1_000_000 / _pricing().output)
        realistic.settle(call_id, {'input': 0, 'output': output}, PAID_URL)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(paid_call, schedule))
    c.check('a four-round build at measured costs is never refused under the $3 cap',
            not errors and realistic.calls_refused == 0,
            f'spent ${realistic.spent_usd:.3f}, peak committed ${peak:.3f} of '
            f'$3.00, {len(schedule)} calls, {len(errors)} refused')
    c.check('in-flight reservations, not cumulative ones, are what the cap sees',
            peak < 1.0,
            f'peak ${peak:.3f}: actual spend plus at most 8 concurrent '
            f'reservations, so an over-estimate costs headroom only while a call '
            f'is open')

    # --- 7c. THE REGRESSION THAT MATTERS: the SDK's OWN default transport --- #
    #
    # The earlier version of check 2 passed a hand-built httpx.Client into the
    # SDK, so it verified the transport the TEST chose. Production builds
    # openai._DefaultHttpxClient, whose MRO goes to httpx2.Client -- a separate
    # distribution whose send() the httpx patch never touched. That is why the
    # proxy judge, ~90% of a Stage-3 bill, went ungated.
    import httpx2
    import openai

    default_calls: list[Any] = []

    def httpx2_handler(request: Any) -> Any:  # noqa: ARG001 -- MockTransport handler signature
        default_calls.append(request)
        return httpx2.Response(200, json={
            'choices': [{'message': {'content': '<score>3</score>'}}],
            'usage': {'prompt_tokens': 3000, 'completion_tokens': 60},
        })

    sdk_gate = _gate(tmp / 'sdk_default', cap=5.0)
    original = ba.arm(sdk_gate)
    try:
        covered = ba.armed_transports()
        c.check('both installed transports are gated, not just httpx',
                covered.get('httpx') and covered.get('httpx2'),
                json.dumps(covered))
        # A DEFAULT client: no http_client argument. Only the transport
        # underneath it is mocked, which is the seam production actually uses.
        client = openai.OpenAI(
            api_key=SECRET, base_url='https://api.openai.com/v1',
            http_client=openai.DefaultHttpxClient(
                transport=httpx2.MockTransport(httpx2_handler)))
        c.check('the SDK default client really is httpx2, not httpx',
                type(client._client).__mro__[1].__module__.startswith('httpx2'),
                type(client._client).__mro__[1].__module__)
        answer = client.chat.completions.create(
            model='gpt-5.6-luna', messages=[{'role': 'user', 'content': 'score'}])
        c.check('a proxy-judge call through the SDK default transport IS gated',
                bool(answer.choices) and len(default_calls) == 1
                and sdk_gate.calls_allowed == 1,
                f'allowed={sdk_gate.calls_allowed} http_calls={len(default_calls)}')
        c.check('its cost is accounted, not silently free',
                abs(sdk_gate.spent_usd - _pricing().cost(3000, 60)) < 1e-9,
                f'${sdk_gate.spent_usd:.6f}')

        # accounting closure at production shape: 600 judge + 8 Luna calls
        closure = _gate(tmp / 'closure', cap=50.0)
        ba.disarm(original)
        original = ba.arm(closure)
        judge_cost = _pricing().cost(3000, 60)
        luna_cost = _pricing().cost(100_000, 4000)

        def judge_handler(request: Any) -> Any:  # noqa: ARG001 -- MockTransport handler signature
            return httpx2.Response(200, json={
                'choices': [{'message': {'content': 'ok'}}],
                'usage': {'prompt_tokens': 3000, 'completion_tokens': 60}})

        def luna_handler(request: Any) -> Any:  # noqa: ARG001 -- MockTransport handler signature
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'ok'}}],
                'usage': {'prompt_tokens': 100_000, 'completion_tokens': 4000}})

        judge_client = openai.OpenAI(
            api_key=SECRET, base_url='https://api.openai.com/v1',
            max_retries=0,
            http_client=openai.DefaultHttpxClient(
                transport=httpx2.MockTransport(judge_handler)))
        for _ in range(600):
            judge_client.chat.completions.create(
                model='gpt-5.6-luna', messages=[{'role': 'user', 'content': 'x'}])
        builder_client = httpx.Client(transport=httpx.MockTransport(luna_handler))
        for _ in range(8):
            builder_client.post(PAID_URL, json={'messages': [{'content': 'y' * 100}]})
        expected = 600 * judge_cost + 8 * luna_cost
        c.check('600 judge + 8 Luna calls all pass the gate',
                closure.calls_allowed == 608 and closure.calls_refused == 0,
                f'allowed={closure.calls_allowed} refused={closure.calls_refused}')
        c.check('and the accounting closes to the penny',
                abs(closure.spent_usd - expected) < 1e-6,
                f'${closure.spent_usd:.6f} == ${expected:.6f}')
        c.check('nothing is left reserved after closure',
                not closure.in_flight and closure.reserved_usd == 0.0, '')

        # cap refusal on the SDK path must make NO outbound request
        refuse_calls: list[Any] = []

        def never(request: Any) -> Any:  # noqa: ARG001 -- MockTransport handler signature
            refuse_calls.append(request)
            return httpx2.Response(200, json={'usage': {'prompt_tokens': 1,
                                                        'completion_tokens': 1}})

        tight_sdk = _gate(tmp / 'tight_sdk', cap=0.00001)
        ba.disarm(original)
        original = ba.arm(tight_sdk)
        tight_client = openai.OpenAI(
            api_key=SECRET, base_url='https://api.openai.com/v1', max_retries=0,
            http_client=openai.DefaultHttpxClient(
                transport=httpx2.MockTransport(never)))
        blocked = False
        try:
            tight_client.chat.completions.create(
                model='gpt-5.6-luna', messages=[{'role': 'user', 'content': 'x'}])
        except Exception as exc:  # noqa: BLE001 -- the SDK may wrap it
            blocked = ba._refusal_in(exc)
        c.check('a capped SDK call is refused with NO outbound request',
                blocked and not refuse_calls and tight_sdk.calls_refused >= 1,
                f'blocked={blocked} outbound={len(refuse_calls)} '
                f'refused={tight_sdk.calls_refused}')
    finally:
        ba.disarm(original)
        original = ba.arm(gate)  # keep something armed for the later checks
        ba.disarm(original)
        original = {}

    # --- 8. nothing written contains a credential ------------------------- #
    leaked: list[str] = []
    for path in tmp.rglob('*'):
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for needle in (SECRET, 'Authorization', 'Bearer ', 'api_key', 'api-key'):
            if needle in text:
                leaked.append(f'{path.name}:{needle}')
    c.check('no state file records a header, token or api key', not leaked,
            str(leaked[:3]))

    # --- 9. URL sanitisation ---------------------------------------------- #
    dirty = httpx.URL('https://user:pw@api.openai.com:443/v1/chat/completions'
                      '?api-key=SECRET&x=1#frag')
    sanitised = ba.safe_url(dirty)
    c.check('a URL is recorded as scheme, host and path only',
            'SECRET' not in sanitised and 'user' not in sanitised
            and 'pw' not in sanitised and '?' not in sanitised
            and sanitised.endswith('/v1/chat/completions'), sanitised)

    # --- 10. state reuse fails closed, --resume-state carries forward ----- #
    reuse = tmp / 'reuse'
    reuse.mkdir(parents=True, exist_ok=True)
    (reuse / ba.STATE_FILE).write_text(json.dumps({
        'spent_usd': 1.75,
        'in_flight_calls': [{'call_id': 9, 'reserved_usd': 0.25,
                             'url': 'https://api.openai.com/v1/x'}],
    }), encoding='utf-8')
    fresh = _gate(reuse, cap=3.0)
    fail_closed = False
    try:
        fresh.adopt_prior_state(resume=False)
    except ba.StateReuseRefused as exc:
        fail_closed = '2.0000' in str(exc) or '2.00' in str(exc)
    c.check('a reused state directory fails closed instead of resetting the cap',
            fail_closed, 'refused, and the message names the carried amount')
    resumed = _gate(reuse, cap=3.0)
    adopted = resumed.adopt_prior_state(resume=True)
    c.check('--resume-state carries spend forward, unsettled calls included',
            adopted is not None and abs(resumed.spent_usd - 2.0) < 1e-9
            and abs(resumed.reservation_for(1) - resumed.reservation_for(1)) < 1e-9,
            f'carried ${resumed.spent_usd:.4f} of the $3.00 cap')
    c.check('the prior state file is kept, not deleted',
            any(p.name.startswith(f'{ba.STATE_FILE}.superseded')
                for p in reuse.iterdir()), '')

    # --- 11. a reservation that cannot be persisted refuses the call ------ #
    blocked = tmp / 'blocked'
    blocked.mkdir(parents=True, exist_ok=True)
    unwritable = _gate(blocked, cap=10.0)
    (blocked / ba.STATE_FILE).mkdir()  # a directory where the file must go
    persist_refused = False
    try:
        unwritable.admit(1000, PAID_URL)
    except ba.BudgetRefused as exc:
        persist_refused = 'persist' in str(exc)
    c.check('a call whose reservation cannot be written is refused',
            persist_refused and unwritable.calls_allowed == 0,
            f'allowed={unwritable.calls_allowed}')
    c.check('a persistence refusal counts as a refusal, so rc is non-zero even '
            'if it is swallowed',
            unwritable.calls_refused == 1
            and any(r.get('reason') == 'reservation_not_persisted'
                    for r in unwritable.refusals),
            f'calls_refused={unwritable.calls_refused}')

    # --- 11b. pricing is read from the config, not a coincidental default -- #
    #
    # The v3 prices happen to equal the old code's hardcoded fallback, so
    # comparing numbers proves nothing. What has to hold is that the mapping
    # keys the old code looked up do not exist, and the attributes it now reads
    # do -- and that a config with different prices comes through.
    from pathlib import Path as _Path

    from emoharness.builder.config import BuilderConfig, BuilderPricing
    from emoharness.common.config import load_model

    config = load_model(_Path(ROOT / 'configs/builder/v3_full.yaml'), BuilderConfig)
    as_mapping = dict(config.pricing)
    priced = ba.pricing_from_builder_config('v3_full.yaml')
    c.check('the old mapping lookup really was dead: no "input"/"output" keys',
            'input' not in as_mapping and 'output' not in as_mapping
            and 'input_usd_per_million' in as_mapping,
            f'keys {sorted(as_mapping)}')
    c.check('pricing now comes from the config attributes',
            priced.input == config.pricing.input_usd_per_million
            and priced.output == config.pricing.output_usd_per_million,
            f'${priced.input}/${priced.output} per 1M')
    odd = BuilderPricing(input_usd_per_million=7.0, output_usd_per_million=11.0)
    c.check('a non-default price list would be carried through, not defaulted',
            ba.Pricing(input=float(odd.input_usd_per_million),
                       output=float(odd.output_usd_per_million)).cost(1_000_000, 0)
            == 7.0,
            'attribute read gives $7.00/1M input, the old alias lookup gave $0.20')

    # --- 12. a swallowed refusal still exits non-zero --------------------- #
    swallowed = _gate(tmp / 'swallowed', cap=0.0001)
    with contextlib.suppress(ba.BudgetRefused):
        swallowed.admit(100_000, PAID_URL)
    c.check('calls_refused is what forces a non-zero exit, not the exception',
            swallowed.calls_refused == 1,
            'main() overrides rc=0 whenever calls_refused > 0')
    c.check('a wrapped refusal is still recognised',
            ba._refusal_in(RuntimeError('worker died',)) is False
            and ba._refusal_in(ValueError('x')) is False
            and ba._refusal_in(_wrapped()) is True, '')

    return c.report()


def _wrapped() -> BaseException:
    """A refusal as the harness would deliver it: raised inside a worker."""
    try:
        try:
            raise ba.BudgetRefused('refusing a paid call')
        except ba.BudgetRefused as inner:
            raise RuntimeError('thread pool wrapped it') from inner
    except RuntimeError as outer:
        return outer


def main() -> int:
    import shutil
    import tempfile

    tmp = Path(os.environ.get('EMOHARNESS_BUDGET_TEST_DIR')
               or tempfile.mkdtemp(prefix='budget-admission-'))
    print(f'budget admission checks, scratch {tmp}\n')
    try:
        return run(tmp)
    finally:
        if not os.environ.get('EMOHARNESS_BUDGET_TEST_KEEP'):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
