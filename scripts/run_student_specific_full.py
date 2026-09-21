#!/usr/bin/env python3
"""Drive the 2026-09-18 student-specific batch on one host.

    .venv/bin/python scripts/run_student_specific_full.py --plan
    .venv/bin/python scripts/run_student_specific_full.py --preflight
    .venv/bin/python scripts/run_student_specific_full.py --selftest
    .venv/bin/python scripts/run_student_specific_full.py \
        --models qwen3vl_4b,qwen3_5_4b --workers 8 --video-workers 2 \
        --max-num-seqs 8 --mm-cache-gb 0

The batch (coordination/full_run_plan_20260918.md, revision 2): three students
each get one independent full-condition four-round Stage 3 from the frozen v3
scaffold, Qwen3-VL-8B reuses its finished full/replicate_0 bundle, and all four
are evaluated on L0, L1, L3 (scaffold) and L5 (the student's own final bundle)
over the frozen eligible item set -- 5,507 rows, not the canonical 5,577,
because 70 canonical rows share an exact media file with a construction item.
16 arms.

This driver runs the full queue. A single first construction is
``scripts/start_student_construction_20260918.py`` (session1-worker's file);
both share ``scripts/budget_admission.py`` and neither edits the other.

Design rules it holds itself to, each one a thing that went wrong somewhere:

* Fail closed. ``run`` refuses to start until the preflight passes; a foreign
  server on the port, an occupied GPU, a drifted input hash or any non-zero
  media overlap stops the batch rather than being logged and worked around.
* One process group per job, and only groups we started are ever signalled.
* A unique attempt directory per attempt, so no step ever needs ``--force`` and
  no finished construction can be overwritten. Failed attempts are archived.
* Finite attempts, counted across invocations in ``state/<session>.attempts.json``,
  and terminal holds in ``state/<session>.holds.json``. A failed build is ALWAYS
  held: restarting Stage 3 from scratch draws fresh Builder proposals, which is a
  second optimisation run wearing the first one's name. Releasing a hold is an
  explicit act (``--release <job_key>``), never a retry loop.
* Spend is decided before the call, not noticed after it: construction runs
  behind ``scripts/budget_admission.py``. That is our bookkeeping, not a provider
  limit, and its residual is reported rather than denied.
* A model answering badly is an outcome, not a driver failure. Truncation, empty
  answers and parse failures are counted into the marker; only structural faults
  -- wrong item set, missing file, missing task, call budget breached -- fail an
  arm. The 24-item interface smoke is what stops a broken contract.
* Resume verifies artifacts and hashes, not just that a marker file exists.
* Each host publishes its own status, index and reports. Nothing is a shared
  single file that two hosts race to rewrite.

Ownership: this file, ``scripts/student_full_row.py``,
``scripts/budget_admission.py`` and ``scripts/test_budget_admission.py``.
Nothing under ``src/``, ``harness/``, ``configs/`` or any existing result is
written here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parent.parent
RUN_NAME: Final = 'student_specific_full_20260918'
RUN_REL: Final = f'results/v3/{RUN_NAME}'
PY_BIN: Final = ROOT / '.venv/bin/python'
ROW_STEP: Final = 'scripts/student_full_row.py'
BUILD_STEP: Final = 'examples/build_harness.py'
#: The successor admission wrapper, which gates BOTH installed HTTP
#: transports. The original scripts/budget_admission.py gated only httpx
#: and so never saw the proxy judge's httpx2 calls; it is left frozen at
#: 7e85952e as the code the finished Gemma run was executed with.
ADMISSION_STEP: Final = ROOT / 'scripts/budget_admission_httpx2.py'
#: The isolated serve helper for this batch. Identical to the frozen
#: scripts/serve_student.sh except that it passes --served-model-name
#: "${MODEL}", so /v1/models advertises the HF id the target configs post
#: rather than the snapshot path vLLM infers under HF_HUB_OFFLINE. Kept as
#: a separate file so the frozen original does not drift.
SERVE_STEP: Final = ROOT / 'scripts/serve_student_full_20260918.sh'
MAILBOX: Final = ROOT / 'scripts/session_mailbox.py'
SCAFFOLD: Final = ROOT / 'harness'
KEY_FILE: Final = Path(os.environ.get('EMOHARNESS_ENV_FILE',
                                      Path.home() / '.emoharness_env'))

#: The frozen eligible item set. ``full`` means the canonical 5,577 in
#: ``src/emoharness/runner/items.py`` and must never be used here.
ITEM_SET: Final = 'full5507'
SMOKE_ITEMS: Final = 'smoke24'
SMOKE_N: Final = 24

#: Ledger roles that cost money. The students are served locally at zero price.
PAID_ROLES: Final = frozenset({'builder', 'verifier', 'judge_proxy', 'teacher_direct'})
BUILD_CAP_USD: Final = float(os.environ.get('EMOHARNESS_BUILD_CAP_USD', '3.0'))
BATCH_CAP_USD: Final = float(os.environ.get('EMOHARNESS_BATCH_CAP_USD', '10.0'))
PREFLIGHT_RESERVE_USD: Final = float(
    os.environ.get('EMOHARNESS_PREFLIGHT_RESERVE_USD', '1.0'))

MEM_LIMIT_GIB: Final = float(os.environ.get('EMOHARNESS_MEM_LIMIT_GIB', '26'))
#: Memory-pressure policy, matching the approved minimal-launcher rule.
#:
#: Raw ``memory.current`` is useless on its own here: this cgroup routinely sits
#: at 98% of its 32 GiB because a 4,102-clip import left ~29 GiB of CLEAN page
#: cache, with zero dirty pages and 329 successful reclaims and no OOM kill. A
#: guard that aborted on that number would abort the batch for cache the kernel
#: is happy to drop. So file cache alone never trips anything: either anon is
#: genuinely large, or the kernel is genuinely stalling on reclaim.
PRESSURE_FULL_AVG60_HARD: Final = float(
    os.environ.get('EMOHARNESS_PRESSURE_HARD', '20'))
PRESSURE_FULL_AVG60_WITH_CURRENT: Final = float(
    os.environ.get('EMOHARNESS_PRESSURE_SOFT', '5'))
CURRENT_FRACTION_WITH_PRESSURE: Final = float(
    os.environ.get('EMOHARNESS_CURRENT_FRACTION', '0.92'))
DISK_FLOOR_GIB: Final = float(os.environ.get('EMOHARNESS_DISK_FLOOR_GIB', '20'))
WATCHDOG_SECONDS: Final = 15
MONITOR_SECONDS: Final = 60
REPORT_SECONDS: Final = 3600
SERVER_READY_SECONDS: Final = 2400
VLLM_PORT: Final = int(os.environ.get('VLLM_PORT', '8000'))
ENDPOINT: Final = f'http://127.0.0.1:{VLLM_PORT}/v1'
HEALTH: Final = f'http://127.0.0.1:{VLLM_PORT}/health'

PREPROCESS_CONFIG: Final = 'preprocess.yaml'
BUILDER_CONFIG: Final = 'v3_full.yaml'
EXTERNAL_MANIFEST: Final = 'external_v3.jsonl'
STAGE: Final = 3
REPLICATE: Final = 0
ROUNDS: Final = 4

_TEMP_COUNTER: Final = itertools.count()


@dataclass(frozen=True)
class Student:
    key: str
    hf_id: str
    target: str
    serving: str
    #: Repo-relative bundle to reuse instead of building. Qwen3-VL-8B only.
    reuse_bundle: str | None = None

    @property
    def builds(self) -> bool:
        return self.reuse_bundle is None


STUDENTS: Final[dict[str, Student]] = {
    'qwen3vl_4b': Student(
        'qwen3vl_4b', 'Qwen/Qwen3-VL-4B-Instruct',
        'configs/target/qwen3vl_4b_instruct.yaml', 'vllm011'),
    'qwen3_5_4b': Student(
        'qwen3_5_4b', 'Qwen/Qwen3.5-4B',
        'configs/target/qwen3_5_4b.yaml', 'vllm019'),
    # vllm019 (torch 2.10+cu128) and never vllm026: 0.26 pins a cu130 torch
    # that cannot initialise CUDA on this box's 570 driver, whatever
    # torch.cuda.is_available() reports once it is imported.
    'gemma4_e4b': Student(
        'gemma4_e4b', 'google/gemma-4-E4B-it',
        'configs/target/gemma4_e4b_it.yaml', 'vllm019_tf513'),
    'qwen3vl_8b': Student(
        'qwen3vl_8b', 'Qwen/Qwen3-VL-8B-Instruct',
        'configs/target/qwen3vl_8b_instruct.yaml', 'vllm011',
        reuse_bundle='results/v3/fixed_asr/full/replicate_0/construction/stage3/0/final'),
}

SERVING_VENVS: Final = {
    'vllm011': '.venvs/emoharness-vllm011',
    'vllm019': '.venvs/emoharness-vllm019',
    # transformers 5.17 raises AmbiguousGlobalPerLayerAttributeError on
    # Gemma 4 E4B's heterogeneous head dimensions, which is what stopped the
    # session1 Gemma launch at rc5 before any build. 5.13.1 loads it; the venv
    # is otherwise the same vLLM 0.19.1 / torch 2.10+cu128 as vllm019.
    'vllm019_tf513': '.venvs/emoharness-vllm019-tf513',
}
#: The 8B transfer bundle. A new student's own L5 must never be this one.
TRANSFER_BUNDLE_HASH: Final = (
    '569c76f9edc0e2e6473fdad9f8854fdfa473fcf94964961d48877a338478e9e2')

#: The HF commit each student's local cache is pinned at, verified by the
#: coordinator against the finished transfer batch. The server is started with
#: the hub offline so a model id cannot silently resolve to a newer revision and
#: answer with different weights than the arms it is being compared to.
HF_REVISIONS: Final[dict[str, str]] = {
    'qwen3vl_8b': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    'qwen3vl_4b': 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    'qwen3_5_4b': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'gemma4_e4b': 'ee0ef6023621cff504d758262d4e04895a5af4a2',
}


@dataclass(frozen=True)
class Arm:
    row: str
    slot: str
    #: None for the single-call rows, 'scaffold' for L3, 'own' for L5.
    bundle: str | None
    needs_perception: bool


#: L0 posts native video to the server and needs no cached clip. L1 reads the
#: audio evidence; L3 and L5 run the harness over both.
ARMS: Final[tuple[Arm, ...]] = (
    Arm('L0', 'L0', None, False),
    Arm('L1', 'L1', None, True),
    Arm('L3', 'L3_scaffold', 'scaffold', True),
    Arm('L5', 'L5_own', 'own', True),
)


@dataclass(frozen=True)
class Job:
    kind: str  # 'build' | 'eval'
    student: str
    arm: Arm | None = None
    items: str = ITEM_SET

    @property
    def key(self) -> str:
        if self.kind == 'build':
            return f'{self.student}__build'
        assert self.arm is not None
        return f'{self.student}__{self.arm.slot}__{self.items}'

    @property
    def label(self) -> str:
        return self.key.replace('__', ' ')

    @property
    def is_smoke(self) -> bool:
        return self.items == SMOKE_ITEMS


def utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _temp_for(path: Path) -> Path:
    """A temp name no other thread or process can collide with."""
    return path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.'
                          f'{next(_TEMP_COUNTER)}.tmp')


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_for(path)
    with temp.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  default=str)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    temp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_for(path)
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_json_or(path: Path, default: Any) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def load_key_file(path: Path = KEY_FILE) -> dict[str, str]:
    """Names and values from the shared key file. Values are never logged."""
    found: dict[str, str] = {}
    if not path.is_file():
        return found
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        name, _, value = line.partition('=')
        found[name.strip()] = value.strip().strip('"').strip("'")
    return found


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            path.read_text(encoding='utf-8').splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# host facts
# --------------------------------------------------------------------------- #

def cgroup_value(name: str) -> int | None:
    try:
        raw = Path(f'/sys/fs/cgroup/{name}').read_text(encoding='utf-8').strip()
    except OSError:
        return None
    return None if raw == 'max' else int(raw.split()[0])


def cgroup_stat(field_name: str) -> int:
    try:
        text = Path('/sys/fs/cgroup/memory.stat').read_text(encoding='utf-8')
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith(f'{field_name} '):
            return int(line.split()[1])
    return 0


def cgroup_pressure() -> dict[str, float]:
    """``some``/``full`` memory stall averages; a rising avg10 precedes an OOM."""
    out: dict[str, float] = {}
    try:
        text = Path('/sys/fs/cgroup/memory.pressure').read_text(encoding='utf-8')
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split()
        for item in parts[1:]:
            key, _, value = item.partition('=')
            if key in ('avg10', 'avg60'):
                try:
                    out[f'{parts[0]}_{key}'] = float(value)
                except ValueError:
                    pass
    return out


def host_mem_available_gib() -> float:
    try:
        text = Path('/proc/meminfo').read_text(encoding='utf-8')
    except OSError:
        return 0.0
    for line in text.splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return 0.0


def anon_gib() -> float:
    return cgroup_stat('anon') / 2**30


def memory_trip_reason(
    anon_limit_gib: float = MEM_LIMIT_GIB,
) -> str | None:
    """Why the memory watchdog should abort, or None.

    Three independent rules, and reclaimable file cache alone satisfies none of
    them:

    * anon above its limit -- real allocation we cannot reclaim;
    * ``memory.current`` near the limit AND the kernel stalling (full avg60
      above the soft threshold) -- near-full plus real stalling;
    * full avg60 above the hard threshold on its own -- stalling badly enough
      that where the memory went does not matter.

    Checked continuously, including while a server is still loading, because
    the watchdog thread starts before the queue does.
    """
    anon = anon_gib()
    if anon > anon_limit_gib:
        return (f'anon {anon:.1f} GiB > {anon_limit_gib} GiB; this is real '
                f'allocation, not reclaimable cache')
    pressure = cgroup_pressure()
    full60 = pressure.get('full_avg60', 0.0)
    if full60 > PRESSURE_FULL_AVG60_HARD:
        return (f'memory stall full avg60 {full60:.1f} > '
                f'{PRESSURE_FULL_AVG60_HARD}; the kernel is thrashing')
    limit = cgroup_value('memory.max')
    current = cgroup_value('memory.current') or 0
    if limit:
        fraction = current / limit
        if (fraction > CURRENT_FRACTION_WITH_PRESSURE
                and full60 > PRESSURE_FULL_AVG60_WITH_CURRENT):
            return (f'memory.current {current / 2**30:.1f} GiB is '
                    f'{fraction:.0%} of the limit AND full avg60 {full60:.1f} > '
                    f'{PRESSURE_FULL_AVG60_WITH_CURRENT}; near-full and stalling')
    return None


def gpu_facts() -> dict[str, Any]:
    query = ('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,'
             'memory.total,driver_version --format=csv,noheader,nounits')
    try:
        out = subprocess.run(query.split(), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {'available': False}
    if out.returncode != 0 or not out.stdout.strip():
        return {'available': False}
    name, util, used, total, driver = (
        x.strip() for x in out.stdout.strip().splitlines()[0].split(','))
    facts: dict[str, Any] = {
        'available': True, 'name': name, 'util_pct': int(util),
        'memory_used_mib': int(used), 'memory_total_mib': int(total),
        'driver': driver, 'compute_apps': [],
    }
    try:
        apps = subprocess.run(
            'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
            .split(), capture_output=True, text=True, timeout=30)
        facts['compute_apps'] = [x.strip() for x in apps.stdout.strip().splitlines()
                                 if x.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return facts


def gpu_pids() -> set[int]:
    found: set[int] = set()
    for row in gpu_facts().get('compute_apps') or []:
        piece = row.split(',')[0].strip()
        if piece.isdigit():
            found.add(int(piece))
    return found


def healthy() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(HEALTH, timeout=5):
            return True
    except Exception:
        return False


def serving_models() -> tuple[str, ...]:
    """Every model id ``/v1/models`` advertises, or an empty tuple."""
    import urllib.request
    try:
        with urllib.request.urlopen(f'{ENDPOINT}/models', timeout=5) as response:
            data = json.load(response).get('data') or []
    except Exception:
        return ()
    return tuple(str(entry.get('id')) for entry in data if entry.get('id'))


def serving_model() -> str | None:
    """The first advertised model id, or None."""
    names = serving_models()
    return names[0] if names else None


def serves_target(hf_id: str) -> tuple[bool, tuple[str, ...]]:
    """Does the server answer to the exact id the Target config will post?

    ``/health`` cannot see this. Under ``HF_HUB_OFFLINE=1`` vLLM 0.11.2 rewrites
    the model id to the snapshot path (``arg_utils.py:592``) and, with no
    ``--served-model-name``, that absolute path becomes the only name
    ``/v1/models`` advertises -- so a healthy server rejects every request the
    Target client makes with its configured ``model_id``. Caught on the real
    Qwen3-VL-4B launch on 2026-09-18 while the server was still loading.
    """
    names = serving_models()
    return hf_id in names, names


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: Any) -> bool:
    if not isinstance(pgid, int):
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def bundle_hash(path: Path) -> str:
    from emoharness.harness.loader import load_bundle

    return load_bundle(path).content_hash()


# --------------------------------------------------------------------------- #
# spend
# --------------------------------------------------------------------------- #

@dataclass
class Spend:
    """What the batch has cost, and the caps it is held to (our bookkeeping)."""

    batch: float
    per_student: dict[str, float]
    build_cap: float = BUILD_CAP_USD
    batch_cap: float = BATCH_CAP_USD
    reserve: float = PREFLIGHT_RESERVE_USD

    @property
    def batch_allowance(self) -> float:
        return max(0.0, self.batch_cap - self.reserve - self.batch)

    def remaining_for(self, student: str) -> float:
        return min(self.build_cap - self.per_student.get(student, 0.0),
                   self.batch_allowance)

    def payload(self) -> dict[str, Any]:
        return {
            'batch_usd': round(self.batch, 4),
            'per_student_usd': {k: round(v, 4)
                                for k, v in sorted(self.per_student.items())},
            'build_cap_usd': self.build_cap,
            'batch_cap_usd': self.batch_cap,
            'preflight_reserve_usd': self.reserve,
            'batch_allowance_usd': round(self.batch_allowance, 4),
            'basis': 'paid ledger roles under this batch root only; no earlier '
                     'experiment enters the cap',
        }


# --------------------------------------------------------------------------- #
# artifact validation
# --------------------------------------------------------------------------- #

def summarize_predictions(path: Path) -> dict[str, Any]:
    rows = jsonl_rows(path)
    calls = [int(row.get('target_calls') or 0) for row in rows] or [0]
    unterminated = sum(1 for row in rows
                       if '</answer>' not in (row.get('response_text') or ''))
    return {
        'n': len(rows),
        'unique_ids': len({row.get('example_id') for row in rows}),
        'tasks': sorted({str(row.get('task')) for row in rows}),
        'calls_min': min(calls), 'calls_max': max(calls),
        'calls_mean': round(sum(calls) / len(calls), 3),
        'parse_failures': sum(int(row.get('parse_failures') or 0) for row in rows),
        'empty_responses': sum(1 for row in rows
                               if not (row.get('response_text') or '').strip()),
        'unterminated': unterminated,
        'unterminated_share': round(unterminated / max(1, len(rows)), 4),
        'items_without_labels': sum(1 for row in rows if not row.get('labels')),
        'input_tokens': sum(int(row.get('input_tokens') or 0) for row in rows),
        'output_tokens': sum(int(row.get('output_tokens') or 0) for row in rows),
    }


def validate_predictions(
    path: Path, arm: Arm, items: str, expected_ids: frozenset[str]
) -> tuple[list[str], dict[str, Any]]:
    """Separate structural faults from model outcomes.

    A structural fault means the artifact is not the thing the report will read:
    a missing file, a different id set, a missing task, a call count outside the
    harness contract. Those fail the arm.

    Model outcomes -- empty answers, truncation, parse failures -- are results.
    A student that answers badly on 300 of 5,507 rows has told us something, and
    re-running the arm would neither fix that nor be honest about it. They are
    counted, reported and kept.

    ``expected_ids`` empty means "do not check identity" (the smoke set, whose
    ids the step chooses deterministically).
    """
    if not path.is_file():
        return [f'no predictions file at {path}'], {}
    rows = jsonl_rows(path)
    summary = summarize_predictions(path)
    structural: list[str] = []
    expected_n = SMOKE_N if items == SMOKE_ITEMS else len(expected_ids)
    if expected_n and summary['n'] != expected_n:
        structural.append(f'{summary["n"]} rows, expected {expected_n}')
    if summary['unique_ids'] != summary['n']:
        structural.append(f'{summary["n"] - summary["unique_ids"]} duplicate id(s)')
    if len(summary['tasks']) != 8:
        structural.append(f'{len(summary["tasks"])} tasks, expected 8')
    if summary['items_without_labels']:
        structural.append(f'{summary["items_without_labels"]} rows without gold labels')
    if expected_ids:
        written = {row.get('example_id') for row in rows}
        stray = written - expected_ids
        missing = expected_ids - written
        if stray:
            structural.append(f'{len(stray)} row(s) outside the frozen eligible set, '
                              f'e.g. {sorted(stray)[:2]}')
        if missing:
            structural.append(f'{len(missing)} frozen row(s) not scored, '
                              f'e.g. {sorted(missing)[:2]}')
    harness = arm.bundle is not None
    # Only the hard cap is a contract. A harness item that resolves in 3
    # calls is a legitimate outcome -- the routing can skip a stage -- so a
    # >=4 floor rejected valid work as a structural fault. The observed
    # minimum is reported as a diagnostic instead.
    if summary['calls_max'] > 6:
        structural.append(f'target_calls reached {summary["calls_max"]}, cap is 6')
    if summary['calls_min'] < 1:
        structural.append('an item recorded zero Target calls')
    diagnostics: dict[str, Any] = {
        'empty_responses': summary['empty_responses'],
        'unterminated': summary['unterminated'],
        'unterminated_share': summary['unterminated_share'],
        'parse_failures': summary['parse_failures'],
        'calls_mean': summary['calls_mean'],
    }
    if harness:
        outputs = path.with_name(f'{items}.outputs.jsonl')
        if not outputs.is_file():
            structural.append(f'harness row without {outputs.name}')
        else:
            records = jsonl_rows(outputs)
            if len(records) != summary['n']:
                structural.append(f'{len(records)} harness outputs for '
                                  f'{summary["n"]} predictions')
            diagnostics['outputs_without_steps'] = sum(
                1 for record in records if not record.get('steps'))
    return structural, {**diagnostics, 'summary': summary}


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

class Driver:
    """One host's queue. Owns its own logs, state, indexes and process groups."""

    def __init__(self, args: argparse.Namespace, root: Path | None = None) -> None:
        self.args = args
        #: The checkout by default; ``--selftest`` points a driver at a scratch
        #: tree so the machinery runs without touching the real batch.
        self.root = root or ROOT
        self.run_root = self.root / RUN_REL
        self.host = socket.gethostname()
        self.session = args.session
        self.queue: list[str] = list(args.models)
        # The frozen item set. Injectable so a test can exercise the verified
        # path against a synthetic set instead of doing fake work over the real
        # 5,507 rows and the real caches.
        batch = ROOT / RUN_REL
        self.items_file = args.item_set_file or (batch / 'evaluation_items.jsonl')
        self.excluded_file = (args.excluded_ids_file
                              or (batch / 'evaluation_excluded_ids.json'))
        self.manifest_file = (args.execution_manifest
                              or (batch / 'execution_manifest.json'))
        self.logs = self.run_root / 'logs'
        self.state_dir = self.run_root / 'state'
        self.done_dir = self.run_root / '.done'
        self.failed_dir = self.run_root / '.failed'
        self.locks_dir = self.run_root / '.locks'
        self.bundles_dir = self.run_root / 'bundles'
        self.reports_dir = self.run_root / 'reports'
        self.supervision_dir = self.run_root / 'supervision' / self.session.split('-')[0]
        self.progress_path = self.run_root / f'progress.{self.session}.jsonl'
        self.log_path = self.logs / f'driver.{self.session}.log'
        self.monitor_path = self.logs / f'monitor.{self.session}.jsonl'
        self.status_path = self.run_root / 'status' / f'{self.session}.json'
        self.index_path = self.run_root / f'index.{self.session}.json'
        self.attempts_path = self.state_dir / f'{self.session}.attempts.json'
        self.holds_path = self.state_dir / f'{self.session}.holds.json'
        self.release_dir = self.state_dir / f'{self.session}.release'
        self._lock = threading.Lock()
        #: Latched by a watchdog before it kills anything. Once set, the queue
        #: starts nothing new -- without it, a successor server started during a
        #: teardown gets killed by that same teardown.
        self._abort = threading.Event()
        #: Serialises every server start and stop with every kill.
        self._lifecycle = threading.RLock()
        self._state: dict[str, Any] = {
            'phase': 'starting', 'job': None, 'pending': [], 'blocked': {},
            'held': {}, 'server_model': None, 'server_pgid': None,
            'job_pgid': None, 'stop': False, 'abort_reason': None,
            'memory_trips': 0, 'budget_trips': 0, 'server_starts': 0, 'notes': [],
        }
        self._locked_students: list[Path] = []
        self._perception: dict[str, Any] = {}
        self._frozen: dict[str, Any] = {}
        for directory in (self.logs, self.state_dir, self.done_dir, self.failed_dir,
                          self.locks_dir, self.bundles_dir, self.reports_dir,
                          self.release_dir, self.status_path.parent):
            directory.mkdir(parents=True, exist_ok=True)

    # -- plumbing ---------------------------------------------------------- #

    def log(self, message: str) -> None:
        line = f'{utc()} [{self.session}] {message}'
        print(line, flush=True)
        with self.log_path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    def record(self, entry: dict[str, Any]) -> None:
        payload = {'utc': utc(), 'host': self.host, 'session': self.session, **entry}
        with self.progress_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')

    def note(self, message: str) -> None:
        with self._lock:
            notes = list(self._state['notes'])
            notes.append(f'{utc()} {message}')
            self._state['notes'] = notes[-40:]

    def set_state(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def get_state(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(name, default)

    def notify(self, kind: str, message: str, reply_to: str | None = None) -> None:
        """Best-effort mailbox message; a broken mailbox never stops the batch."""
        if self.args.no_notify:
            return
        for recipient in self.args.notify:
            if recipient == self.session:
                continue
            cmd = [sys.executable, str(MAILBOX), 'send', '--from', self.session,
                   '--to', recipient, '--kind', kind, '--message', message[:6000]]
            if reply_to:
                cmd += ['--reply-to', reply_to]
            try:
                out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                     timeout=60)
                if out.returncode != 0:
                    self.log(f'mailbox to {recipient} failed rc={out.returncode}: '
                             f'{out.stderr.strip()[:200]}')
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.log(f'mailbox to {recipient} failed: {exc}')

    # -- persistent attempt and hold state --------------------------------- #

    def attempts(self) -> dict[str, Any]:
        return read_json_or(self.attempts_path, {})

    def bump_attempt(self, job: Job, status: str) -> int:
        """Count ONE attempt, in state that survives a restart of the driver.

        Called exactly once per attempt, when it starts. The outcome is recorded
        later by :meth:`set_attempt_status`, which must not count again: the
        earlier version incremented on start AND on finish, so a single failed
        attempt consumed a cap of two and no job was ever retried.
        """
        table = self.attempts()
        entry = table.get(job.key) or {'attempts': 0}
        entry['attempts'] = int(entry.get('attempts', 0)) + 1
        entry['last_status'] = status
        entry['last_utc'] = utc()
        entry['session'] = self.session
        table[job.key] = entry
        write_json_atomic(self.attempts_path, table)
        return entry['attempts']

    def set_attempt_status(self, job: Job, status: str) -> None:
        """Record how the current attempt ended, without counting another one."""
        table = self.attempts()
        entry = table.get(job.key) or {'attempts': 1}
        entry['last_status'] = status
        entry['last_utc'] = utc()
        entry['session'] = self.session
        table[job.key] = entry
        write_json_atomic(self.attempts_path, table)

    def attempts_used(self, job: Job) -> int:
        return int((self.attempts().get(job.key) or {}).get('attempts', 0))

    def holds(self) -> dict[str, Any]:
        return read_json_or(self.holds_path, {})

    def hold(self, job_key: str, reason: str, evidence: Any = None) -> None:
        """Mark a job terminal until it is explicitly released."""
        table = self.holds()
        table[job_key] = {'reason': reason, 'utc': utc(), 'evidence': evidence,
                          'release_required': True, 'session': self.session,
                          'host': self.host}
        write_json_atomic(self.holds_path, table)
        self.set_state(held=table)
        self.log(f'HOLD {job_key}: {reason}')

    def is_held(self, job: Job) -> tuple[bool, str]:
        entry = self.holds().get(job.key)
        if entry is None:
            return False, ''
        release = self.release_dir / f'{job.key}.json'
        if release.is_file():
            table = self.holds()
            table.pop(job.key, None)
            write_json_atomic(self.holds_path, table)
            self.log(f'released hold on {job.key} by {release.name}')
            self.record({'key': job.key, 'status': 'hold_released',
                         'release': str(release.relative_to(self.root))})
            return False, ''
        return True, str(entry.get('reason'))

    def release(self, job_key: str) -> int:
        """Write the release file for one held job, and drop the hold."""
        write_json_atomic(self.release_dir / f'{job_key}.json',
                          {'released_utc': utc(), 'by': self.session,
                           'host': self.host})
        table = self.holds()
        removed = table.pop(job_key, None)
        write_json_atomic(self.holds_path, table)
        print(f'released {job_key}'
              + (f' (was: {removed.get("reason")})' if removed else ' (no hold found)'))
        return 0

    # -- the frozen item set ------------------------------------------------ #

    def frozen(self) -> dict[str, Any]:
        """The eligible item set, verified once per invocation.

        Verification is neither optional nor a warning: a drifted or
        overlapping item set means the number this batch produces is not the
        number the plan asked for.
        """
        if self._frozen:
            return self._frozen
        if str(ROOT / 'scripts') not in sys.path:
            sys.path.insert(0, str(ROOT / 'scripts'))
        from student_full_row import load_frozen_items

        examples, provenance = load_frozen_items(
            self.items_file, self.excluded_file, self.manifest_file,
            expect_count=self.args.expect_count,
            require_manifest=not self.args.allow_unverified_item_set)
        self._frozen = {
            **provenance,
            'ids': frozenset(example.example_id for example in examples),
            'media_hashes': frozenset(example.media_hash for example in examples
                                      if example.media_hash),
        }
        return self._frozen

    def expected_ids(self) -> frozenset[str]:
        return self.frozen()['ids']

    # -- ownership --------------------------------------------------------- #

    def claim(self, student: str) -> bool:
        """Advisory ownership of one student.

        The shared root is NFS mounted ``nolock``, so this is an O_EXCL create
        plus a liveness check rather than a mutex: enough to stop the two hosts
        picking up the same student, and it reports rather than races when it
        cannot tell.
        """
        path = self.locks_dir / f'{student}.json'
        payload = {'session': self.session, 'host': self.host, 'pid': os.getpid(),
                   'claimed_utc': utc()}
        try:
            with path.open('x', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            self._locked_students.append(path)
            return True
        except FileExistsError:
            pass
        held = read_json_or(path, None)
        if held is None:
            self.log(f'{student}: unreadable lock at {path}; treating it as held')
            return False
        if held.get('session') == self.session and held.get('host') == self.host:
            if held.get('pid') == os.getpid() or not pid_alive(held.get('pid')):
                write_json_atomic(path, payload)
                self._locked_students.append(path)
                return True
        self.log(f'{student}: owned by {held.get("session")} on {held.get("host")} '
                 f'(pid {held.get("pid")}); this host will not touch it')
        return False

    def release_claims(self) -> None:
        for path in self._locked_students:
            try:
                path.unlink()
            except OSError:
                pass
        self._locked_students = []

    # -- paths ------------------------------------------------------------- #

    def attempt_dir(self, job: Job, attempt: int) -> str:
        """The repo-relative ``EMOHARNESS_RESULTS_DIR`` of one attempt.

        Evaluation arms use the layout the report script asked for,
        ``<batch>/<model>/<slot>/<items>/attempt_NN``, which is the finished
        transfer batch's layout; builds sit under ``construction/``.
        """
        if job.kind == 'build':
            return f'{RUN_REL}/construction/{job.student}/attempt_{attempt:02d}'
        assert job.arm is not None
        return (f'{RUN_REL}/{job.student}/{job.arm.slot}/{job.items}'
                f'/attempt_{attempt:02d}')

    def next_attempt(self, job: Job) -> int:
        """One past the highest attempt id ever used, live or archived."""
        used = {0}
        parent = (self.root / self.attempt_dir(job, 1)).parent
        for sibling in parent.glob('attempt_*'):
            piece = sibling.name.rsplit('_', 1)[-1]
            if piece.isdigit():
                used.add(int(piece))
        for archived in (self.failed_dir / job.student).glob(f'{job.key}_a*'):
            piece = archived.name.split('_a')[-1].split('_')[0]
            if piece.isdigit():
                used.add(int(piece))
        return max(used) + 1

    def marker(self, job: Job) -> Path:
        return self.done_dir / f'{job.key}.done'

    def bundle_record(self, student: str) -> Path:
        return self.bundles_dir / student / 'bundle.json'

    def bundle_for(self, arm: Arm, student: str) -> Path | None:
        if arm.bundle is None:
            return None
        if arm.bundle == 'scaffold':
            return SCAFFOLD
        record = self.bundle_record(student)
        if not record.is_file():
            return None
        path = self.root / read_json(record)['path']
        return path if path.is_dir() else None

    # -- resume: verify, do not trust -------------------------------------- #

    def marker_is_good(self, job: Job) -> tuple[bool, str]:
        """Does this marker still describe artifacts that exist and match?

        A marker file on shared NFS is a claim, not evidence. A stale marker
        from an earlier item set, or one whose predictions were archived, would
        otherwise make an unrun arm report as finished.
        """
        marker = self.marker(job)
        if not marker.is_file():
            return False, 'no marker'
        entry = read_json_or(marker, None)
        if entry is None:
            return False, 'unreadable marker'
        if entry.get('status') != 'ok':
            return False, f'marker status {entry.get("status")}'
        if job.kind == 'build':
            record = self.bundle_record(job.student)
            if not record.is_file():
                return False, 'no published bundle record'
            published = read_json_or(record, {})
            path = self.root / str(published.get('path', ''))
            if not path.is_dir():
                return False, f'published bundle missing at {path}'
            if entry.get('bundle_hash') and published.get('hash') != entry['bundle_hash']:
                return False, 'published bundle hash differs from the marker'
            # Recompute from the files on disk. Comparing the marker's record
            # against the bundle record's record compares two claims to each
            # other and would pass on a bundle whose contents changed under
            # both of them.
            try:
                actual = bundle_hash(path)
            except Exception as exc:
                return False, f'published bundle does not load: {exc}'
            for claim, where in ((entry.get('bundle_hash'), 'marker'),
                                 (published.get('hash'), 'bundle record')):
                if claim and claim != actual:
                    return False, (f'published bundle hashes to {actual[:12]} on '
                                   f'disk but the {where} says {str(claim)[:12]}')
            return True, 'ok'
        assert job.arm is not None
        predictions = self.root / str(entry.get('predictions') or '')
        if not str(entry.get('predictions') or '') or not predictions.is_file():
            return False, f'predictions missing at {predictions}'
        if not job.is_smoke:
            frozen = self.frozen()
            if entry.get('item_set_sha256') != frozen['item_set_sha256']:
                return False, 'marker scored a different frozen item set'
            if entry.get('n_items_written') != len(frozen['ids']):
                return False, (f'marker wrote {entry.get("n_items_written")} of '
                               f'{len(frozen["ids"])} items')
            # Re-read the FILE, not just the marker's claims about it. A marker
            # is a claim; the predictions are the evidence, and on shared NFS
            # the two can diverge (a truncated write, a half-synced file, an
            # arm rerun into the same path by another process).
            try:
                rows = jsonl_rows(predictions)
            except (OSError, ValueError) as exc:
                return False, f'predictions unreadable: {exc}'
            written = {row.get('example_id') for row in rows}
            if len(rows) != len(frozen['ids']):
                return False, (f'predictions hold {len(rows)} rows, expected '
                               f'{len(frozen["ids"])}')
            if written != frozen['ids']:
                return False, (f'predictions cover a different id set '
                               f'({len(written - frozen["ids"])} stray, '
                               f'{len(frozen["ids"] - written)} missing)')
            if any(row.get('row') not in (None, job.arm.row) for row in rows):
                return False, 'predictions carry another row name'
            # Exact content sha of the file as it is now. Recorded on first
            # validation and compared thereafter, so a file that is rewritten
            # between resumes -- same ids, same count, different answers -- is
            # caught rather than accepted because its shape still matches.
            from emoharness.common.hashing import sha256_file

            actual = sha256_file(predictions)
            claimed = entry.get('predictions_sha256')
            if claimed and claimed != actual:
                return False, (f'predictions content sha {actual[:12]} does not '
                               f'match the marker\'s {str(claimed)[:12]}')
        expected_bundle = self.bundle_for(job.arm, job.student)
        if expected_bundle is not None:
            recorded = (entry.get('provenance') or {}).get('bundle_hash')
            if recorded and recorded != bundle_hash(expected_bundle):
                return False, 'marker used a different bundle than the current one'
        return True, 'ok'

    def is_done(self, job: Job) -> bool:
        good, why = self.marker_is_good(job)
        if not good and self.marker(job).is_file():
            self.log(f'{job.key}: ignoring its marker ({why}); it will be re-run')
        return good

    # -- perception gate --------------------------------------------------- #

    def perception_coverage(self, refresh: bool = False) -> dict[str, Any]:
        """How much of the frozen eligible set the shared clip cache holds.

        Checked on the ``media_hash`` the frozen manifest already carries, which
        is one stat per row rather than one file hash per row.
        ``--verify-media`` re-hashes a sample to confirm the manifest still
        describes what is on disk.
        """
        if self._perception and not refresh:
            return self._perception
        from emoharness.common.config import load_model
        from emoharness.perceive.cache import PreprocessConfig
        from emoharness.runner.paths import ProjectPaths

        paths = ProjectPaths.from_root(self.root, ROOT)
        config = load_model(paths.preprocess_config, PreprocessConfig)
        config_hash = config.config_hash()
        derived = paths.preprocess_cache
        if not self.items_file.is_file():
            return {'config_hash': config_hash, 'cache_root': str(derived),
                    'rows_total': 0, 'media_unique': 0, 'media_cached': 0,
                    'rows_missing': 0, 'missing_by_task': {}, 'complete': False,
                    'error': f'no frozen item set at {self.items_file}'}
        rows = jsonl_rows(self.items_file)
        # A row with no media_hash cannot be looked up in a content-addressed
        # cache at all, so it counts as uncached rather than crashing the
        # monitor. The real frozen manifest carries one for every row; this is
        # the guard, not the expectation.
        hashes = {row['media_hash'] for row in rows if row.get('media_hash')}
        without_hash = sum(1 for row in rows if not row.get('media_hash'))
        cached = {h for h in hashes if (derived / h / f'{config_hash}.json').is_file()}
        missing = [row for row in rows
                   if not row.get('media_hash') or row['media_hash'] not in cached]
        by_task: dict[str, int] = {}
        for row in missing:
            by_task[row['task']] = by_task.get(row['task'], 0) + 1
        result = {
            'config_hash': config_hash,
            'cache_root': str(derived),
            'item_set_file': str(self.items_file),
            'rows_total': len(rows),
            'media_unique': len(hashes),
            'media_cached': len(cached),
            'rows_without_media_hash': without_hash,
            'rows_missing': len(missing),
            'missing_by_task': dict(sorted(by_task.items())),
            'complete': not missing,
            **self._coordinator_marker(config_hash, len(hashes)),
        }
        if self.args.verify_media:
            result['media_sample_check'] = self._sample_media_check(rows)
        self._perception = result
        return result

    def _coordinator_marker(self, config_hash: str,
                            eligible_unique: int) -> dict[str, Any]:
        """Read and CHECK the coordinator's completion marker.

        Existence was never the claim worth trusting. The marker must parse,
        must name the perception config this batch uses, and its canonical
        counts must be a superset of the eligible set -- it verifies against the
        canonical 5,577, so its unique-clip count has to be at least ours. The
        eligible coverage itself is still computed independently from the cache;
        this marker gates on the import having been declared finished, and the
        two together are what open the gate.
        """
        path = self.run_root / 'preprocessing.done.json'
        if not path.is_file():
            return {'coordinator_marker': False,
                    'coordinator_marker_detail': 'absent'}
        payload = read_json_or(path, None)
        if payload is None:
            return {'coordinator_marker': False,
                    'coordinator_marker_detail': f'{path.name} does not parse'}
        problems: list[str] = []
        if payload.get('config_hash') != config_hash:
            problems.append(f'marker config_hash '
                            f'{str(payload.get("config_hash"))[:12]} != this '
                            f'batch\'s {config_hash[:12]}')
        verified_unique = payload.get('verified_unique')
        if not isinstance(verified_unique, int):
            problems.append('marker records no integer verified_unique')
        elif verified_unique < eligible_unique:
            problems.append(f'marker verified {verified_unique} unique clips, '
                            f'fewer than the {eligible_unique} the eligible set '
                            f'needs')
        verified_items = payload.get('verified_items')
        if isinstance(verified_items, int) and verified_items < len(
                self.frozen()['ids']):
            problems.append(f'marker verified {verified_items} items, fewer than '
                            f'the {len(self.frozen()["ids"])} eligible rows')
        return {
            'coordinator_marker': not problems,
            'coordinator_marker_detail': '; '.join(problems) or 'verified',
            'coordinator_marker_payload': {
                k: payload.get(k) for k in
                ('config_hash', 'verified_unique', 'verified_items', 'utc')},
        }

    def _sample_media_check(self, rows: list[dict[str, Any]],
                            sample: int = 12) -> dict[str, Any]:
        from emoharness.common.hashing import sha256_file

        step = max(1, len(rows) // sample)
        checked = mismatched = missing = 0
        examples: list[str] = []
        for row in rows[::step][:sample]:
            path = Path(row['media_path'])
            if not path.is_file():
                missing += 1
                examples.append(f'missing {row["example_id"]}')
                continue
            checked += 1
            if sha256_file(path) != row['media_hash']:
                mismatched += 1
                examples.append(f'hash drift {row["example_id"]}')
        return {'checked': checked, 'mismatched': mismatched,
                'media_missing': missing, 'examples': examples[:5]}

    def ready(self, job: Job) -> tuple[bool, str]:
        held, reason = self.is_held(job)
        if held:
            return False, f'held:{reason}'
        if self.attempts_used(job) >= self.args.max_total_attempts:
            return False, 'attempts_exhausted'
        # A staged construction that may still be running owns this student's
        # GPU, so nothing for that student can start -- not its build, not its
        # rows.
        if self.staged_live_owner(job.student) is not None:
            return False, 'waiting_staged_run'
        if job.kind == 'eval':
            assert job.arm is not None
            if job.arm.bundle is not None and self.bundle_for(job.arm, job.student) is None:
                return False, 'waiting_bundle'
            if job.arm.needs_perception:
                coverage = self.perception_coverage()
                # BOTH, not coverage alone. Coverage is computed from cache
                # files this host can see; the coordinator's marker is what says
                # the import finished and was verified. An NFS import that is
                # 99% landed looks "complete" for any row whose clip happens to
                # be there, and an arm started then would silently invoke
                # Whisper on the box serving the student.
                if not coverage['coordinator_marker']:
                    return False, 'waiting_perception'
                if not coverage['complete']:
                    return False, 'waiting_perception'
        return True, 'ready'

    # -- server lifecycle -------------------------------------------------- #

    def stop_server(self) -> bool:
        """Stop only the server this driver started. Fails closed.

        Ownership is cleared only once the process group is really gone; a
        surviving group keeps its pgid recorded so nothing starts a second
        server onto the same GPU.
        """
        with self._lifecycle:
            pgid = self.get_state('server_pgid')
            if pgid is None:
                return True
            self.log(f'stopping owned server process group {pgid}')
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                for _ in range(90 if sig is signal.SIGTERM else 60):
                    if not group_alive(pgid):
                        break
                    time.sleep(1)
                if not group_alive(pgid):
                    break
            if group_alive(pgid):
                self.log(f'WARNING: server process group {pgid} survived SIGKILL; '
                         'keeping ownership and refusing to start another')
                self.note(f'server pgid {pgid} survived SIGKILL')
                return False
            self.log(f'server process group {pgid} terminated')
            self.set_state(server_pgid=None, server_model=None)
            return True

    def ensure_server(self, student: Student) -> None:
        """Serve exactly this student, on a GPU nothing else is using.

        Fails closed on anything it does not own. A healthy server started by
        someone else has unknown decoding, an unknown venv and an unknown model
        revision, so adopting it silently would make every arm it answers
        unattributable. ``--adopt-server`` is the explicit opt-in and it is
        recorded in the progress log.
        """
        if self.args.fake:
            return
        with self._lifecycle:
            if self._abort.is_set():
                raise SystemExit(f'aborted: {self.get_state("abort_reason")}')
            owned = self.get_state('server_pgid')
            current = serving_model()
            if owned is not None and group_alive(owned):
                # Membership, not first-entry equality: a server may advertise
                # several names, and only "does it answer to ours" matters.
                if healthy() and serves_target(student.hf_id)[0]:
                    self.set_state(server_model=student.hf_id)
                    return
                if not self.stop_server():
                    raise SystemExit('the owned server process group will not die; '
                                     'refusing to start another on the same GPU')
                current = serving_model()
            if current is not None:
                if not self.args.adopt_server:
                    raise SystemExit(
                        f'port {VLLM_PORT} already serves {current} and this driver '
                        f'does not own it. Its serving configuration and venv are '
                        f'unknown, so adopting it would make every arm it answers '
                        f'unattributable. Stop it, or pass --adopt-server to accept '
                        f'it explicitly.')
                ok, names = serves_target(student.hf_id)
                if not ok:
                    raise SystemExit(
                        f'--adopt-server given but port {VLLM_PORT} advertises '
                        f'{list(names)!r}, which does not include '
                        f'{student.hf_id!r}; the Target config posts that id and '
                        f'every call would be rejected')
                self.log(f'ADOPTING the existing {current} server on {VLLM_PORT} '
                         '(--adopt-server); recorded in the arm provenance')
                self.note(f'adopted a foreign server for {current}')
                self.set_state(server_model=current)
                self.record({'model': student.key, 'status': 'server_adopted',
                             'served': current})
                return
            foreign = gpu_pids()
            if foreign and not self.args.adopt_server:
                raise SystemExit(
                    f'the GPU carries {len(foreign)} compute process(es) '
                    f'{sorted(foreign)} this driver did not start. One student '
                    f'server at a time is a plan requirement and this driver never '
                    f'kills a process it does not own; stop them, or pass '
                    f'--adopt-server.')
            venv = SERVING_VENVS[student.serving]
            env = dict(
                os.environ,
                EMOHARNESS_STUDENT_MODEL=student.hf_id,
                EMOHARNESS_VLLM_VENV=str(Path.home() / venv),
                # Conservative scheduling per the launch resource decision:
                # 8 CPU and 32 GiB per host, and the vLLM frontend decodes every
                # video in host RAM. The earlier H200 numbers are historical.
                VLLM_MAX_NUM_SEQS=str(self.args.max_num_seqs),
                VLLM_MM_CACHE_GB=str(self.args.mm_cache_gb),
                # The FlashInfer JIT build for Qwen3.5's kernels starts one
                # compile job per core by default and added 8 GiB of anon in 15
                # seconds on a 10-core box, which restarted the container.
                MAX_JOBS=os.environ.get('MAX_JOBS', '2'),
                NVCC_THREADS=os.environ.get('NVCC_THREADS', '1'),
                CMAKE_BUILD_PARALLEL_LEVEL=os.environ.get(
                    'CMAKE_BUILD_PARALLEL_LEVEL', '2'),
                # Every weight this batch needs is already in the local cache
                # from the finished transfer batch. Offline means a model id
                # cannot resolve to a newer revision mid-batch and answer some
                # arms with different weights than the ones they are compared
                # to. A missing auxiliary asset now fails at startup, where it
                # is a serving problem to diagnose, instead of silently
                # downloading one.
                HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',
            )
            out = (self.logs / f'serve_{student.key}.log').open('a')
            proc = subprocess.Popen(['bash', str(SERVE_STEP)], cwd=ROOT, env=env,
                                    stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            pgid = os.getpgid(proc.pid)
            self.set_state(server_pgid=pgid, server_model=student.hf_id,
                           server_starts=self.get_state('server_starts', 0) + 1)
            self.log(f'started {student.hf_id} from {venv} (pid {proc.pid}, pgid '
                     f'{pgid}, max_num_seqs {self.args.max_num_seqs}, mm_cache '
                     f'{self.args.mm_cache_gb} GB, hub offline, pinned revision '
                     f'{HF_REVISIONS.get(student.key, "unrecorded")[:12]})')
            self.record({'model': student.key, 'status': 'server_started',
                         'hf_id': student.hf_id, 'pgid': pgid, 'venv': venv,
                         'hf_revision_expected': HF_REVISIONS.get(student.key),
                         'hub_offline': True,
                         'max_num_seqs': self.args.max_num_seqs,
                         'mm_cache_gb': self.args.mm_cache_gb})
        deadline = time.time() + SERVER_READY_SECONDS
        while time.time() < deadline:
            if self._abort.is_set():
                raise SystemExit(f'aborted while waiting for the server: '
                                 f'{self.get_state("abort_reason")}')
            if healthy():
                waited = int(SERVER_READY_SECONDS - (deadline - time.time()))
                # Healthy is not enough: the server must answer to the exact id
                # the Target config posts. Fail closed rather than let every
                # request be rejected by a server that looks fine.
                ok, names = serves_target(student.hf_id)
                if not ok:
                    self.record({'model': student.key,
                                 'status': 'served_model_name_mismatch',
                                 'expected': student.hf_id, 'advertised': names})
                    raise SystemExit(
                        f'{student.hf_id} is healthy but /v1/models advertises '
                        f'{list(names)!r}, not {student.hf_id!r}. The Target config '
                        f'posts its own model_id, so every call would be rejected. '
                        f'Under HF_HUB_OFFLINE vLLM rewrites the id to the snapshot '
                        f'path unless the serve script passes '
                        f'--served-model-name; that fix belongs in the serving '
                        f'script, and this driver refuses to run arms against a '
                        f'server whose model name it cannot verify.')
                self.log(f'{student.hf_id} healthy after {waited}s and /v1/models '
                         f'advertises it')
                return
            if proc.poll() is not None:
                raise SystemExit(
                    f'server for {student.hf_id} exited with {proc.returncode}; see '
                    f'{self.logs / f"serve_{student.key}.log"}')
            time.sleep(10)
        raise SystemExit(f'server for {student.hf_id} not healthy within '
                         f'{SERVER_READY_SECONDS}s')

    # -- running one job --------------------------------------------------- #

    def job_env(self, results_dir: str) -> dict[str, str]:
        nvidia = ROOT / '.venv/lib/python3.12/site-packages/nvidia'
        env = dict(
            os.environ,
            EMOHARNESS_RESULTS_DIR=results_dir,
            EMOHARNESS_LEDGER_NAME='ledger.jsonl',
            EMOHARNESS_PREPROCESS_CONFIG=PREPROCESS_CONFIG,
            EMOHARNESS_EXTERNAL_MANIFEST=EXTERNAL_MANIFEST,
            EMOHARNESS_BUILDER_CONFIG=BUILDER_CONFIG,
            VLLM_ENDPOINT=ENDPOINT,
            EMOHARNESS_WORKERS=str(self.args.workers),
            EMOHARNESS_VIDEO_WORKERS=str(self.args.video_workers),
            # ctranslate2 needs the CUDA 12 libraries if any clip misses the
            # perception cache. The gate should make that unreachable; this is
            # the guard behind it, not the plan.
            LD_LIBRARY_PATH=f'{nvidia}/cublas/lib:{nvidia}/cudnn/lib'
            + (f':{os.environ["LD_LIBRARY_PATH"]}'
               if os.environ.get('LD_LIBRARY_PATH') else ''),
        )
        env.update(load_key_file())
        return env

    def job_command(self, job: Job, student: Student, cap_usd: float,
                    results_dir: str) -> list[str]:
        """The step this job runs, with every path absolute.

        ``--root`` is passed explicitly rather than left to default to the
        checkout: ``EMOHARNESS_RESULTS_DIR`` resolves against it, and a driver
        pointed at a scratch tree that silently wrote into the real batch root
        is exactly the mistake ``--selftest`` caught. Target configs and the
        scaffold are committed inputs and always come from the checkout.
        """
        common = ['--root', str(self.root), '--target', str(ROOT / student.target)]
        if job.kind == 'build':
            step_args = ['--stage', str(STAGE), '--replicate', str(REPLICATE),
                         '--rounds', str(ROUNDS), '--bundle', str(SCAFFOLD), *common]
            if self.args.fake:
                step_args.append('--fake')
            # Construction is the only paid work, so it is the only thing behind
            # the admission gate. --state is the attempt directory, so the
            # accounting is archived with the attempt and a retry in a new
            # attempt directory cannot silently inherit a fresh cap.
            return [str(PY_BIN), str(ADMISSION_STEP), '--cap', f'{cap_usd:.4f}',
                    '--state', str(self.root / results_dir),
                    '--builder-config', BUILDER_CONFIG,
                    '--step', BUILD_STEP, '--', *step_args]
        assert job.arm is not None
        cmd = [str(PY_BIN), str(ROOT / ROW_STEP), '--rows', job.arm.row,
               '--items', job.items, *common,
               '--item-set-file', str(self.items_file),
               '--excluded-ids-file', str(self.excluded_file),
               '--execution-manifest', str(self.manifest_file),
               '--expect-count', str(self.args.expect_count)]
        if self.args.allow_unverified_item_set:
            cmd.append('--allow-unverified-item-set')
        bundle = self.bundle_for(job.arm, job.student)
        if bundle is not None:
            cmd += ['--bundle', str(bundle)]
        if self.args.fake:
            cmd.append('--fake')
        return cmd

    def spend(self) -> Spend:
        """Paid-role dollars under this batch root, in total and per student.

        Read straight from the ledger JSONL rather than through the pydantic
        model: that model forbids extra fields, and a guard that cannot read a
        ledger written by a slightly newer runtime is worse than one that
        ignores a field it does not know.
        """
        per_student = {key: 0.0 for key in STUDENTS}
        batch = 0.0
        for ledger in sorted(self.run_root.rglob('ledger*.jsonl')):
            total = 0.0
            try:
                with ledger.open(encoding='utf-8') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue
                        if entry.get('role') in PAID_ROLES:
                            total += float(entry.get('cost_usd') or 0.0)
            except OSError:
                continue
            batch += total
            parts = set(ledger.relative_to(self.run_root).parts)
            for key in STUDENTS:
                if key in parts:
                    per_student[key] += total
        return Spend(batch=batch, per_student=per_student)

    def run_job(self, job: Job, attempt: int) -> dict[str, Any]:
        """One job in its own process group. Never raises on job failure."""
        student = STUDENTS[job.student]
        results_dir = self.attempt_dir(job, attempt)
        (self.root / results_dir).mkdir(parents=True, exist_ok=True)
        before = self.spend()
        cap = before.remaining_for(job.student) if job.kind == 'build' else 0.0
        cmd = self.job_command(job, student, cap, results_dir)
        env = self.job_env(results_dir)
        logfile = self.logs / f'{job.key}_a{attempt:02d}.log'
        started = time.time()
        total_attempts = self.bump_attempt(job, 'running')
        self.log(f'START {job.label} attempt {attempt} (attempt {total_attempts}/'
                 f'{self.args.max_total_attempts} overall) -> {results_dir}'
                 + (f' cap ${cap:.2f}' if job.kind == 'build' else ''))
        self.set_state(job={'key': job.key, 'kind': job.kind, 'model': job.student,
                            'slot': job.arm.slot if job.arm else None,
                            'items': job.items, 'attempt': attempt,
                            'total_attempts': total_attempts,
                            'results_dir': results_dir, 'started_utc': utc(),
                            'cap_usd': round(cap, 4) if job.kind == 'build' else None})
        with logfile.open('a', encoding='utf-8') as out:
            out.write(f'\n===== {utc()} {job.label} attempt {attempt} =====\n')
            out.write(f'command: {" ".join(cmd)}\n')
            out.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            self.set_state(job_pgid=os.getpgid(proc.pid))
            returncode = proc.wait()
        self.set_state(job_pgid=None)
        after = self.spend()
        entry: dict[str, Any] = {
            'utc': utc(), 'host': self.host, 'session': self.session,
            'kind': job.kind, 'key': job.key,
            # both names: the report script keys on 'model'
            'model': job.student, 'student': job.student, 'hf_id': student.hf_id,
            'row': job.arm.row if job.arm else None,
            'slot': job.arm.slot if job.arm else None,
            'items': job.items, 'attempt': attempt,
            'total_attempts': total_attempts, 'results_dir': results_dir,
            'log': str(logfile.relative_to(self.root)),
            'returncode': returncode,
            'status': 'ok' if returncode == 0 else 'failed',
            'seconds': round(time.time() - started, 1),
            'spend_usd': round(after.per_student.get(job.student, 0.0)
                               - before.per_student.get(job.student, 0.0), 4),
            'aborted': self._abort.is_set(),
            'abort_reason': self.get_state('abort_reason'),
        }
        admission = self.root / results_dir / 'budget_admission.json'
        if admission.is_file():
            gate = read_json_or(admission, {})
            entry['budget_admission'] = {
                k: gate.get(k) for k in
                ('cap_usd', 'spent_usd', 'calls_allowed', 'calls_refused',
                 'unparsed_responses', 'actual_over_reservation',
                 'max_excess_over_reservation_usd', 'budget_stop')}
            if gate.get('budget_stop'):
                entry['status'] = 'budget_stop'
        if entry['status'] == 'ok':
            entry.update(self.on_success(job, results_dir))
        if entry['status'] == 'ok':
            write_json_atomic(self.marker(job), entry)
            self.set_attempt_status(job, 'ok')
            self.log(f'OK {job.label} in {entry["seconds"]}s '
                     f'(spend ${entry["spend_usd"]})')
        else:
            self.set_attempt_status(job, entry['status'])
            self.archive_failure(job, attempt, entry)
        self.record(entry)
        self.set_state(job=None)
        self.publish_index()
        return entry

    def on_success(self, job: Job, results_dir: str) -> dict[str, Any]:
        """Validate what the job produced; a zero exit code is not evidence."""
        if job.kind == 'build':
            return self.publish_bundle(job.student, results_dir)
        assert job.arm is not None
        base = self.root / results_dir
        predictions = base / 'raw_predictions' / job.arm.row / f'{job.items}.jsonl'
        provenance = read_json_or(base / 'row_provenance.json', {})
        structural, diagnostics = validate_predictions(
            predictions, job.arm, job.items,
            frozenset() if job.is_smoke else self.expected_ids())
        from emoharness.common.hashing import sha256_file

        extra: dict[str, Any] = {
            'predictions': (str(predictions.relative_to(self.root))
                            if predictions.is_file() else None),
            'predictions_sha256': (sha256_file(predictions)
                                   if predictions.is_file() else None),
            'item_set_file': provenance.get('item_set_file'),
            'item_set_sha256': provenance.get('item_set_sha256'),
            'excluded_ids_sha256': provenance.get('excluded_ids_sha256'),
            'n_items_expected': provenance.get('n_items_expected'),
            'n_items_written': provenance.get('n_items_written'),
            'asr_config_hash': provenance.get('asr_config_hash'),
            'provenance': provenance.get('provenance'),
            'bundle': provenance.get('bundle'),
            'bundle_hash': (provenance.get('provenance') or {}).get('bundle_hash'),
            'diagnostics': diagnostics,
        }
        if structural:
            extra['status'] = 'invalid_artifacts'
            extra['problems'] = structural
            self.log(f'STRUCTURAL FAULT {job.label}: ' + '; '.join(structural[:4]))
            return extra
        # Lineage is the criterion, not the outcome, the same ruling adopt_build
        # already applies. An independent construction can legitimately land on
        # bytes identical to the 8B transfer bundle, so equality here is recorded
        # for review and is NOT a rejection; what establishes that this student
        # built its own bundle is the run_config lineage check in adopt_build
        # (parent bundle, construction manifest, perception config and target).
        if job.arm.slot == 'L5_own' and job.student != 'qwen3vl_8b':
            extra['bundle_matches_transfer'] = (
                extra['bundle_hash'] == TRANSFER_BUNDLE_HASH)
            if extra['bundle_matches_transfer']:
                self.log(f'{job.label}: L5 bundle is byte-identical to the 8B '
                         'transfer bundle; lineage decides, so this is recorded, '
                         'not rejected')
        summary = diagnostics.get('summary') or {}
        self.log(f'{job.label}: {summary.get("n")} rows, '
                 f'{len(summary.get("tasks") or [])} tasks, calls '
                 f'{summary.get("calls_min")}-{summary.get("calls_max")}; model '
                 f'diagnostics empty={diagnostics.get("empty_responses")} '
                 f'unterminated={diagnostics.get("unterminated")} '
                 f'parse_failures={diagnostics.get("parse_failures")}')
        return extra

    def publish_bundle(self, student: str, results_dir: str) -> dict[str, Any]:
        """Copy the student's own final bundle to a stable, attempt-free path."""
        source = (self.root / results_dir / 'construction' / f'stage{STAGE}'
                  / str(REPLICATE) / 'final')
        if not source.is_dir():
            return {'status': 'invalid_artifacts',
                    'problems': [f'no final bundle at {source}']}
        destination = self.bundles_dir / student / 'final'
        staging = self.bundles_dir / student / f'.final.{os.getpid()}.tmp'
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if destination.exists():
            retired = self.bundles_dir / student / f'final.superseded.{utc_stamp()}'
            destination.rename(retired)
            self.log(f'{student}: previous published bundle kept at {retired.name}')
        staging.rename(destination)
        digest = bundle_hash(destination)
        # As above: byte-identity with the transfer bundle is recorded, not
        # refused. Publishing is a copy of what the student actually built, and
        # whether it was built legitimately is decided by the lineage checks,
        # not by how the bytes happened to come out.
        matches_transfer = (student != 'qwen3vl_8b'
                            and digest == TRANSFER_BUNDLE_HASH)
        if matches_transfer:
            self.log(f'{student}: published bundle is byte-identical to the 8B '
                     'transfer bundle; recorded for review, lineage decides')
        record = {
            'student': student, 'model': student,
            'matches_transfer_bundle': matches_transfer,
            'path': str(destination.relative_to(self.root)),
            'source': str(source.relative_to(self.root)),
            'hash': digest, 'source_hash': bundle_hash(source),
            'published_utc': utc(), 'host': self.host, 'session': self.session,
            'stage': STAGE, 'replicate': REPLICATE, 'rounds_cap': ROUNDS,
            'builder_config': f'configs/builder/{BUILDER_CONFIG}',
            'scaffold_hash': bundle_hash(SCAFFOLD),
        }
        write_json_atomic(self.bundle_record(student), record)
        self.log(f'{student}: published its own final bundle {digest[:12]} at '
                 f'{record["path"]}')
        return {'bundle': record['path'], 'bundle_hash': digest}

    def archive_failure(self, job: Job, attempt: int, entry: dict[str, Any]) -> None:
        """Move a failed attempt out of the way, intact, and keep its evidence."""
        source = self.root / entry['results_dir']
        if not source.exists():
            return
        target = (self.failed_dir / job.student
                  / f'{job.key}_a{attempt:02d}_{int(time.time())}')
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(target)
        except OSError as exc:
            self.log(f'could not archive {source}: {exc}')
            return
        entry['archived_to'] = str(target.relative_to(self.root))
        self.log(f'FAILED {job.label} rc={entry["returncode"]} '
                 f'status={entry["status"]}; evidence kept at {entry["archived_to"]}')

    # -- published contract ------------------------------------------------- #

    def publish_index(self) -> None:
        """Name this host's finished arms. One writer: this session.

        Two hosts writing one shared index cannot be made safe on a nolock NFS
        mount -- an interleaved read-modify-write drops the other host's newest
        entry -- so each publishes ``index.<session>.json`` and the report
        script merges them (or reads the ``.done`` markers directly).
        """
        arms: dict[str, Any] = {}
        smoke: dict[str, Any] = {}
        builds: dict[str, Any] = {}
        foreign: dict[str, Any] = {}
        invalid: dict[str, Any] = {}
        for marker in sorted(self.done_dir.glob('*.done')):
            entry = read_json_or(marker, None)
            if entry is None:
                continue
            # Ownership: this index reports what THIS session produced. The
            # .done directory is shared, so the other host's markers are listed
            # separately under "observed_elsewhere" rather than claimed here --
            # otherwise two per-host indexes each assert authorship of all 16
            # arms and the reporter cannot tell who ran what.
            produced_here = (entry.get('session') == self.session
                             and entry.get('host') == self.host)
            if not produced_here:
                foreign[str(entry.get('key'))] = {
                    k: entry.get(k) for k in
                    ('key', 'model', 'slot', 'items', 'kind', 'session', 'host',
                     'predictions', 'bundle', 'bundle_hash', 'utc')}
                continue
            # Validate before counting. An index that counts marker FILES
            # reports an arm as finished whose predictions were archived,
            # truncated or rewritten -- which is exactly the thing the resume
            # check exists to catch, so the two must use the same test.
            job = self._job_for_key(str(entry.get('key')))
            if job is not None:
                good, why = self.marker_is_good(job)
                if not good:
                    invalid[str(entry.get('key'))] = {
                        'reason': why, 'model': entry.get('model'),
                        'slot': entry.get('slot'), 'items': entry.get('items')}
                    continue
            if entry.get('kind') == 'build':
                builds[str(entry.get('model'))] = entry
            elif entry.get('items') == SMOKE_ITEMS:
                smoke[str(entry.get('key'))] = entry
            else:
                arms[str(entry.get('key'))] = entry
        bundles = {record.parent.name: read_json_or(record, {})
                   for record in sorted(self.bundles_dir.glob('*/bundle.json'))}
        frozen = self.frozen()
        write_json_atomic(self.index_path, {
            'run': RUN_NAME, 'updated_utc': utc(),
            'writer': {'session': self.session, 'host': self.host,
                       'pid': os.getpid()},
            'queue': self.queue, 'item_set': ITEM_SET,
            'item_set_file': frozen['item_set_file'],
            'item_set_sha256': frozen['item_set_sha256'],
            'excluded_ids_sha256': frozen['excluded_ids_sha256'],
            'rows_expected': len(frozen['ids']),
            'strata': {'old_subset': frozen.get('n_old_subset_stratum'),
                       'new': frozen.get('n_new_stratum'),
                       'excluded': frozen.get('n_excluded')},
            'arms_expected_batch': len(STUDENTS) * len(ARMS),
            'arms_done_here': len(arms),
            'builds_expected_batch': sum(1 for s in STUDENTS.values() if s.builds),
            'builds_done_here': len(builds),
            'predictions_layout': f'{RUN_REL}/<model>/<slot>/{ITEM_SET}/attempt_NN/'
                                  f'raw_predictions/<row>/{ITEM_SET}.jsonl',
            'note': 'arms/builds/smoke are what THIS session produced; markers '
                    'from another session are under observed_elsewhere. The '
                    'reporter should merge the per-session indexes or read '
                    '.done directly.',
            'builds': builds, 'arms': arms, 'smoke': smoke,
            'observed_elsewhere': foreign, 'invalid_markers': invalid,
            'bundles': bundles,
            'holds': self.holds(), 'attempts': self.attempts(),
        })

    def _job_for_key(self, key: str) -> Job | None:
        """The Job a marker key names, or None when it is not one of ours.

        Used so the index validates a marker with the same test resume uses.
        A key from a student this host does not own resolves fine -- the test is
        about the artifacts, not about ownership.
        """
        parts = key.split('__')
        if len(parts) == 2 and parts[1] == 'build':
            return Job('build', parts[0]) if parts[0] in STUDENTS else None
        if len(parts) != 3 or parts[0] not in STUDENTS:
            return None
        student, slot, items = parts
        for arm in ARMS:
            if arm.slot == slot:
                return Job('eval', student, arm, items)
        return None

    def publish_plan(self) -> None:
        frozen = self.frozen()
        write_json_atomic(self.run_root / f'plan.{self.session}.json', {
            'run': RUN_NAME, 'published_utc': utc(), 'session': self.session,
            'host': self.host, 'queue': self.queue,
            'jobs': [{'kind': job.kind, 'model': job.student, 'key': job.key,
                      'row': job.arm.row if job.arm else None,
                      'slot': job.arm.slot if job.arm else None,
                      'items': job.items,
                      'needs_perception': bool(job.arm and job.arm.needs_perception)}
                     for job in self.plan()],
            'settings': {
                'stage': STAGE, 'replicate': REPLICATE, 'rounds': ROUNDS,
                'item_set': ITEM_SET,
                'item_set_sha256': frozen['item_set_sha256'],
                'rows': len(frozen['ids']),
                'builder_config': f'configs/builder/{BUILDER_CONFIG}',
                'external_manifest': f'data/manifests/{EXTERNAL_MANIFEST}',
                'preprocess_config': f'configs/{PREPROCESS_CONFIG}',
                'workers': self.args.workers,
                'video_workers': self.args.video_workers,
                'max_num_seqs': self.args.max_num_seqs,
                'mm_cache_gb': self.args.mm_cache_gb,
                'mem_limit_gib': MEM_LIMIT_GIB,
                'disk_floor_gib': DISK_FLOOR_GIB,
                'build_cap_usd': BUILD_CAP_USD,
                'batch_cap_usd': BATCH_CAP_USD,
                'max_total_attempts': self.args.max_total_attempts,
                'adopt_server': self.args.adopt_server,
            },
        })

    def declare_terminal(self, state: str, detail: dict[str, Any]) -> None:
        """Tell the manager's monitor this host's queue is really finished.

        Only ``completed``, ``cancelled`` or ``stopped``. Waiting on the shared
        clip cache is not terminal, and declaring it would make the monitor stop
        watching a run that still has work to do.
        """
        if state not in ('completed', 'cancelled', 'stopped'):
            return
        write_json_atomic(self.supervision_dir / 'queue_terminal.json',
                          {'state': state, 'utc': utc(), 'session': self.session,
                           'host': self.host, **detail})

    # -- monitoring -------------------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        limit = cgroup_value('memory.max')
        with self._lock:
            state = {k: v for k, v in self._state.items() if k != 'notes'}
            notes = list(self._state['notes'])[-8:]
        spend = self.spend()
        usage = shutil.disk_usage(self.run_root)
        done = sorted(p.stem for p in self.done_dir.glob('*.done'))
        cpu_max = Path('/sys/fs/cgroup/cpu.max')
        return {
            'utc': utc(), 'host': self.host, 'session': self.session,
            'driver_pid': os.getpid(), 'run': RUN_NAME,
            'phase': state.get('phase'), 'status': state.get('phase'),
            'state': state, 'notes': notes,
            'cgroup': {
                'memory_current_gib': round(
                    (cgroup_value('memory.current') or 0) / 2**30, 2),
                'memory_max_gib': round(limit / 2**30, 2) if limit else None,
                'anon_gib': round(anon_gib(), 2),
                'anon_limit_gib': MEM_LIMIT_GIB,
                'pressure': cgroup_pressure(),
                'cpu_max': (cpu_max.read_text(encoding='utf-8').strip()
                            if cpu_max.exists() else None),
            },
            'host_memory_available_gib': round(host_mem_available_gib(), 2),
            'loadavg': list(os.getloadavg()),
            'gpu': gpu_facts(),
            'disk': {'free_gib': round(usage.free / 2**30, 1),
                     'floor_gib': DISK_FLOOR_GIB},
            'server': {
                'endpoint': ENDPOINT,
                'healthy': None if self.args.fake else healthy(),
                'model': None if self.args.fake else serving_model(),
                'owned_pgid': self.get_state('server_pgid'),
                'owned_alive': group_alive(self.get_state('server_pgid')),
            },
            'spend': spend.payload(),
            'cost_usd': round(spend.batch, 4),
            'spend_usd': round(spend.batch, 4),
            'perception': self.perception_coverage(),
            'completed': len(done), 'done': len(done), 'total': len(self.plan()),
            'failures': sorted(p.name for p in self.failed_dir.glob('*/*')),
            'failure_count': len(list(self.failed_dir.glob('*/*'))),
            'progress': {'done': done,
                         'failed': len(list(self.failed_dir.glob('*/*')))},
            'holds': self.holds(), 'attempts': self.attempts(),
        }

    def publish_status(self) -> dict[str, Any]:
        snap = self.snapshot()
        write_json_atomic(self.status_path, snap)
        with self.monitor_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({
                'utc': snap['utc'],
                'job': (snap['state'].get('job') or {}).get('key'),
                'anon_gib': snap['cgroup']['anon_gib'],
                'memory_current_gib': snap['cgroup']['memory_current_gib'],
                'pressure_some_avg10': snap['cgroup']['pressure'].get('some_avg10'),
                'gpu_util': snap['gpu'].get('util_pct'),
                'gpu_mem_mib': snap['gpu'].get('memory_used_mib'),
                'disk_free_gib': snap['disk']['free_gib'],
                'server_healthy': snap['server']['healthy'],
                'batch_usd': snap['spend']['batch_usd'],
                'done': snap['completed'],
                'perception_missing': snap['perception'].get('rows_missing'),
            }, ensure_ascii=False) + '\n')
        return snap

    def publish_report(self, snap: dict[str, Any], tag: str = '') -> Path:
        base = self.reports_dir / f'{self.session}_{tag or utc_stamp()}'
        text = render_report(snap)
        write_json_atomic(base.with_suffix('.json'), snap)
        write_text_atomic(base.with_suffix('.txt'), text)
        latest = self.reports_dir / f'{self.session}_latest'
        write_json_atomic(latest.with_suffix('.json'), snap)
        write_text_atomic(latest.with_suffix('.txt'), text)
        self.log(f'report published: {base.with_suffix(".txt").relative_to(self.root)}')
        return base.with_suffix('.txt')

    def latch_abort(self, reason: str) -> None:
        """Stop the queue starting anything new, then tear down what is running."""
        if not self._abort.is_set():
            self.set_state(abort_reason=reason)
            self._abort.set()
        self.note(f'abort latched: {reason}')
        self.log(f'ABORT LATCHED: {reason}')

    def kill_job(self, why: str) -> None:
        with self._lifecycle:
            pgid = self.get_state('job_pgid')
            if pgid is None:
                return
            self.log(f'{why}: stopping job process group {pgid}')
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
            for _ in range(30):
                if not group_alive(pgid):
                    return
                time.sleep(1)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def watchdog_loop(self) -> None:
        """Memory, disk and a spend backstop, every 15 seconds, behind a latch.

        The cgroup limit here is enforced by restarting the whole container, so
        the first thing to notice an overrun must be us. The latch is set before
        anything is killed: without it the queue could start the next job's
        server into a teardown that then kills it.
        """
        while not self.get_state('stop'):
            reason = memory_trip_reason()
            if reason is not None:
                self.set_state(memory_trips=self.get_state('memory_trips', 0) + 1)
                self.latch_abort(f'memory watchdog: {reason}')
                with self._lifecycle:
                    self.kill_job('memory watchdog')
                    self.stop_server()
            usage = shutil.disk_usage(self.run_root)
            if usage.free / 2**30 < DISK_FLOOR_GIB:
                self.latch_abort(f'disk floor: {usage.free / 2**30:.1f} GiB free < '
                                 f'{DISK_FLOOR_GIB} GiB')
                with self._lifecycle:
                    self.kill_job('disk floor')
                    self.stop_server()
            # The admission gate refuses paid calls before they leave the box.
            # This is the backstop that notices if a cap was passed anyway.
            job = self.get_state('job')
            if job and job.get('kind') == 'build':
                spend = self.spend()
                spent = spend.per_student.get(job['model'], 0.0)
                if (spent > BUILD_CAP_USD
                        or spend.batch > BATCH_CAP_USD - PREFLIGHT_RESERVE_USD):
                    self.set_state(budget_trips=self.get_state('budget_trips', 0) + 1)
                    self.latch_abort(
                        f'spend backstop: {job["model"]} at ${spent:.2f} of '
                        f'${BUILD_CAP_USD:.2f}, batch ${spend.batch:.2f} of '
                        f'${BATCH_CAP_USD:.2f}; the admission gate should have '
                        f'refused that call first')
                    with self._lifecycle:
                        self.kill_job('spend backstop')
            for _ in range(WATCHDOG_SECONDS):
                if self.get_state('stop'):
                    return
                time.sleep(1)

    def monitor_loop(self) -> None:
        """Status every 60 seconds, a persistent report every hour."""
        last_report = 0.0
        while not self.get_state('stop'):
            try:
                snap = self.publish_status()
                if time.time() - last_report >= REPORT_SECONDS:
                    self.publish_report(snap)
                    last_report = time.time()
            except Exception as exc:  # a monitor must never stop the batch
                self.log(f'monitor tick failed: {type(exc).__name__}: {exc}')
            for _ in range(MONITOR_SECONDS):
                if self.get_state('stop'):
                    return
                time.sleep(1)

    # -- the queue --------------------------------------------------------- #

    def plan(self) -> list[Job]:
        jobs: list[Job] = []
        for key in self.queue:
            if STUDENTS[key].builds:
                jobs.append(Job('build', key))
            jobs.extend(Job('eval', key, arm) for arm in ARMS)
        return jobs

    def seed_reused_bundles(self) -> None:
        """Record the reused 8B bundle so every arm has a bundle provenance entry."""
        for key, student in STUDENTS.items():
            if student.reuse_bundle is None or key not in self.queue:
                continue
            path = self.root / student.reuse_bundle
            if not path.is_dir():
                self.log(f'{key}: reused bundle {student.reuse_bundle} is missing')
                continue
            record = self.bundle_record(key)
            if (record.is_file()
                    and read_json_or(record, {}).get('source') == student.reuse_bundle):
                continue
            digest = bundle_hash(path)
            write_json_atomic(record, {
                'student': key, 'model': key, 'path': student.reuse_bundle,
                'source': student.reuse_bundle, 'hash': digest, 'source_hash': digest,
                'reused_from': student.reuse_bundle, 'published_utc': utc(),
                'host': self.host, 'session': self.session,
                'scaffold_hash': bundle_hash(SCAFFOLD),
                'note': "reused finished full/replicate_0 bundle; 8B is not rebuilt "
                        "and this is the plan's one legitimate reuse",
            })
            self.log(f'{key}: reusing {student.reuse_bundle} ({digest[:12]}) as its '
                     'own L5 bundle')

    def staged_attempts(self, student: str) -> list[dict[str, Any]]:
        """Construction attempts already on disk for this student, newest last.

        ``scripts/start_student_construction_20260918.py`` (session1-worker's
        minimal launcher) writes into the same ``construction/<student>/attempt_NN``
        tree this driver uses, and it does not write this driver's ``.done``
        marker or bundle record. An absent marker therefore does NOT mean no
        construction happened, and starting a fresh Stage 3 on that assumption
        would draw new Builder proposals and quietly replace a completed run.
        """
        base = self.run_root / 'construction' / student
        found: list[dict[str, Any]] = []
        if not base.is_dir():
            return found
        for attempt in sorted(base.glob('attempt_*')):
            # run_config.json and loop_result.json are written by the library to
            # ctx.paths.stage_dir(3, 0) -- attempt/construction/stage3/0/ -- not
            # to the attempt root. Reading them from the root returned {} for
            # every attempt, and a `{}` that is then treated as "nothing to
            # check" turns every provenance test into a no-op. They are read
            # from the real paths here and their absence is a hold, not a pass.
            stage_dir = attempt / 'construction' / f'stage{STAGE}' / str(REPLICATE)
            final = stage_dir / 'final'
            # BOTH launcher records, never one as a fallback for the other.
            # Verified against the real files the frozen launcher wrote on
            # 2026-09-18: launch_evidence.json carries budget, code, elapsed_s,
            # frozen, host, latched, owned_pgids, preflight, returncode,
            # scheduling, session, student, utc and weights -- and NO phase and
            # NO launcher_pid. status.json carries phase, launcher_pid and
            # owned_pgids. Reading either alone means a finished run has no
            # phase (so it waits forever) or no returncode (so it looks live).
            evidence = read_json_or(attempt / 'launch_evidence.json', None)
            status = read_json_or(attempt / 'status.json', None)
            found.append({
                'dir': attempt,
                'relative': str(attempt.relative_to(self.root)),
                'stage_dir': stage_dir,
                'final': final if final.is_dir() else None,
                'launch_evidence': evidence,
                'status': status,
                'run_config': read_json_or(stage_dir / 'run_config.json', None),
                'loop_result': read_json_or(stage_dir / 'loop_result.json', None),
                'budget_admission': read_json_or(attempt / 'budget_admission.json',
                                                 None),
            })
        return found

    #: Launcher phases that mean the run is over, one way or another. Anything
    #: else -- including a missing returncode -- is treated as possibly live.
    TERMINAL_PHASES: Final = frozenset(
        {'done', 'completed', 'complete', 'stopped', 'failed', 'held', 'error',
         'cancelled'})

    def staged_live_owner(self, student: str) -> dict[str, Any] | None:
        """Evidence that a construction for this student may still be running.

        Checked BEFORE claiming the student or touching the server: a staged run
        in flight owns the GPU, and starting a second build or adopting its
        half-written output would corrupt both. Liveness is only directly
        observable for a process on this host; an attempt whose launcher record
        is missing, has no returncode, or reports a non-terminal phase is
        treated as possibly live rather than assumed finished.
        """
        for staged in self.staged_attempts(student):
            evidence = staged['launch_evidence'] or {}
            status = staged['status'] or {}
            if not evidence and not status:
                # An attempt directory with no launcher record at all: if it has
                # no final bundle either, something is mid-flight or was killed.
                if staged['final'] is None:
                    return {'attempt': staged['relative'],
                            'reason': 'attempt directory with no launcher record '
                                      'and no final bundle'}
                continue
            # returncode comes from launch_evidence, phase from status.
            phase = str(status.get('phase') or '').lower()
            returncode = evidence.get('returncode')
            if returncode is None:
                returncode = status.get('returncode')
            if returncode is not None and phase in self.TERMINAL_PHASES:
                continue
            host = evidence.get('host') or status.get('host')
            pgids = [p for p in (list(evidence.get('owned_pgids') or [])
                                 + list(status.get('owned_pgids') or []))
                     if isinstance(p, int)]
            pids = [p for p in [status.get('launcher_pid'), status.get('driver_pid')]
                    if isinstance(p, int)]
            if host == self.host and (pids or pgids):
                if (not any(pid_alive(p) for p in pids)
                        and not any(group_alive(g) for g in pgids)):
                    # Ours, and demonstrably dead: unfinished, not live.
                    continue
            return {'attempt': staged['relative'], 'phase': phase or None,
                    'returncode': returncode, 'host': host,
                    'pids': pids, 'owned_pgids': pgids,
                    'reason': 'launcher record is not terminal'
                              + (' and its process ids belong to another host, so '
                                 'liveness cannot be checked from here'
                                 if host and host != self.host else '')}
        return None

    #: Artifacts whose presence proves Builder work happened in an attempt. An
    #: infrastructure failure -- a server that would not load, a wrong
    #: served-model-name -- never reaches any of them.
    BUILDER_EVIDENCE: Final = (
        'build.log', 'budget_admission.json', 'ledger.jsonl', 'construction',
        'raw_predictions', 'metrics', 'judge')

    def _supersession_record(self, staged: dict[str, Any]) -> dict[str, Any]:
        """Describe a superseded attempt, and decide if it was infra-only.

        Infrastructure-only means: no build log, no budget accounting, no
        ledger, no construction or prediction tree, and no recorded paid spend.
        The launcher's own record must also say it stopped before the build.
        """
        present = sorted(name for name in self.BUILDER_EVIDENCE
                         if (staged['dir'] / name).exists())
        admission = staged['budget_admission'] or {}
        spend = float(admission.get('spent_usd') or 0.0)
        evidence = staged['launch_evidence'] or {}
        status = staged['status'] or {}
        return {
            'attempt': staged['relative'],
            'had_final_bundle': staged['final'] is not None,
            'returncode': evidence.get('returncode', status.get('returncode')),
            'phase': status.get('phase'),
            'latched': evidence.get('latched'),
            'paid_spend_usd': spend,
            'builder_evidence': present,
            'infra_only': not present and spend == 0.0
            and staged['final'] is None,
        }

    def _staged_problems(self, student: str, staged: dict[str, Any]) -> list[str]:
        """Everything that must be positively true before a staged run is adopted.

        The rule throughout: a missing field is never evidence of compatibility.
        An absent run_config, an empty budget file or a launcher record without a
        returncode all block adoption, because the alternative is labelling an
        active or failed build a success and publishing its half-written bundle
        as the student's own.
        """
        problems: list[str] = []
        from emoharness.common.config import load_model
        from emoharness.common.hashing import sha256_file
        from emoharness.perceive.cache import PreprocessConfig
        from emoharness.runner.paths import ProjectPaths

        # 1. the bundle itself must be at the exact expected path and loadable
        expected_final = (staged['stage_dir'] / 'final')
        if staged['final'] is None or staged['final'] != expected_final:
            problems.append(f'no final bundle at the expected path {expected_final}')
            return problems
        try:
            digest = bundle_hash(staged['final'])
        except Exception as exc:
            problems.append(f'its final bundle does not load as a harness bundle: '
                            f'{exc}')
            return problems
        # The coordinator's ruling: lineage is the criterion, not the outcome, so
        # a bundle that happens to be byte-identical to the 8B transfer bundle is
        # not rejected here. The run_config checks below are what establish that
        # this student built it from the scaffold with its own target.

        # 2. BOTH launcher records must positively report a finished, clean run.
        #    launch_evidence.json has returncode/latched/code/weights/preflight
        #    and no phase; status.json has phase and the pids. An absent file is
        #    never read as consent.
        evidence = staged['launch_evidence']
        status = staged['status']
        if evidence is None:
            problems.append('no launch_evidence.json beside the attempt')
        if status is None:
            problems.append('no status.json beside the attempt')
        if evidence is not None:
            returncode = evidence.get('returncode')
            if returncode is None:
                problems.append('its launch_evidence.json records no returncode, so '
                                'the run is unfinished or was killed')
            elif returncode != 0:
                problems.append(f'its launcher reported returncode {returncode}')
            if evidence.get('latched'):
                problems.append(f'its launcher latched an abort: '
                                f'{evidence.get("latched")}')
            # identity: the record must be about this job on this host
            for field, expected in (('student', student), ('host', self.host),
                                    ('session', self.session)):
                if evidence.get(field) != expected:
                    problems.append(f'its launch_evidence {field} is '
                                    f'{evidence.get(field)!r}, not {expected!r}')
            code = evidence.get('code') or {}
            approval_path = (self.run_root / 'approvals'
                             / f'BUILD_READY.{student}.json')
            approval = read_json_or(approval_path, None)
            if approval is None:
                problems.append(f'no manager approval at {approval_path.name}, so '
                                f'the code that produced this bundle is unbound')
                approval = {}
            for field in ('launcher_sha256', 'wrapper_sha256'):
                if not code.get(field):
                    problems.append(f'its launch_evidence records no {field}')
                elif approval.get(field) and code.get(field) != approval[field]:
                    problems.append(f'it ran {field} {str(code.get(field))[:12]}, '
                                    f'but the approval binds '
                                    f'{str(approval[field])[:12]}')
            weights = evidence.get('weights') or {}
            if weights.get('model_id') != STUDENTS[student].hf_id:
                problems.append(f'its weights record names model '
                                f'{weights.get("model_id")!r}')
            expected_rev = HF_REVISIONS.get(student)
            if expected_rev and weights.get('cached_revision') != expected_rev:
                problems.append(f'it served revision '
                                f'{str(weights.get("cached_revision"))[:12]}, not the '
                                f'pinned {expected_rev[:12]}')
            if (weights.get('expected_revision')
                    and weights.get('expected_revision') != weights.get('cached_revision')):
                problems.append('its weights record disagrees with itself on the '
                                'served revision')
            preflight_rows = evidence.get('preflight') or []
            if not preflight_rows:
                problems.append('its launch_evidence records no preflight results')
            else:
                failed = [r.get('name') for r in preflight_rows
                          if r.get('blocking') and not r.get('ok')]
                if failed:
                    problems.append(f'it launched with failing blocking preflight '
                                    f'checks: {failed[:4]}')
            frozen_block = evidence.get('frozen') or {}
            if frozen_block.get('scaffold_hash') != bundle_hash(SCAFFOLD):
                problems.append(f'its frozen scaffold_hash is '
                                f'{str(frozen_block.get("scaffold_hash"))[:12]}, not '
                                f'{bundle_hash(SCAFFOLD)[:12]}')
        if status is not None:
            phase = str(status.get('phase') or '').lower()
            if phase not in self.TERMINAL_PHASES:
                problems.append(f'its status.json phase {phase or "(none)"!r} is not '
                                f'terminal')
            if phase == 'held':
                problems.append('its status.json reports the run held')
            # owned groups must be gone, and that is only checkable on its host
            host = status.get('host') or (evidence or {}).get('host')
            pgids = [p for p in (status.get('owned_pgids') or [])
                     if isinstance(p, int)]
            if host == self.host:
                alive = [g for g in pgids if group_alive(g)]
                if alive:
                    problems.append(f'its owned process group(s) {alive} are still '
                                    f'alive on this host')
            elif pgids:
                problems.append(f'its owned process group(s) {pgids} are on host '
                                f'{host!r}, so this host cannot confirm they are '
                                f'gone')

        # 2b. a latched filesystem hold blocks adoption outright
        student_holds = sorted(
            str(p.relative_to(self.root))
            for p in (self.run_root / 'construction' / student).glob('**/HOLD*'))
        if student_holds:
            problems.append(f'a filesystem hold is latched for this student: '
                            f'{student_holds[:2]}')

        # 3. the loop must have run to a recorded stopping point
        loop_result = staged['loop_result']
        if loop_result is None:
            problems.append(f'no loop_result.json under {staged["stage_dir"]}')
        elif not loop_result.get('stopped_reason'):
            problems.append('its loop_result.json records no stopped_reason')
        elif not loop_result.get('rounds'):
            problems.append('its loop_result.json records no rounds')

        # 4. the budget must be settled: nothing refused, nothing in flight
        admission = staged['budget_admission']
        if admission is None:
            problems.append('no budget_admission.json beside the attempt, so its '
                            'paid spend is unaccounted for')
        else:
            if admission.get('budget_stop'):
                problems.append('its budget accounting reports budget_stop')
            if admission.get('calls_refused'):
                problems.append(f'its budget accounting refused '
                                f'{admission.get("calls_refused")} call(s)')
            # Exactly 0. A missing returncode means the wrapper never
            # recorded an exit, which is the signature of a killed run, not of
            # a clean one.
            if admission.get('returncode') != 0:
                problems.append(f'its budget wrapper returncode is '
                                f'{admission.get("returncode")!r}, not 0')
            if admission.get('in_flight_calls'):
                problems.append(f'{len(admission["in_flight_calls"])} paid call(s) '
                                f'were admitted and never settled, so the run was '
                                f'killed mid-call')
            if float(admission.get('reserved_in_flight_usd') or 0.0) > 0:
                problems.append(f'${admission["reserved_in_flight_usd"]} is still '
                                f'reserved in flight')
            if admission.get('spent_usd') is None:
                problems.append('its budget accounting records no spent_usd')
            elif float(admission['spent_usd']) > BUILD_CAP_USD:
                problems.append(f'it spent ${admission["spent_usd"]}, over the '
                                f'${BUILD_CAP_USD:.2f} per-build cap')

        # 5. lineage: the same scaffold, manifest, target, perception and round cap
        run_config = staged['run_config']
        if run_config is None:
            problems.append(f'no run_config.json under {staged["stage_dir"]}, so its '
                            f'lineage cannot be checked')
            return problems
        scaffold = bundle_hash(SCAFFOLD)
        if run_config.get('parent_bundle_hash') != scaffold:
            problems.append(f'it started from parent bundle '
                            f'{str(run_config.get("parent_bundle_hash"))[:12]}, not '
                            f'the v3 scaffold {scaffold[:12]}')
        manifest = ROOT / 'data/manifests' / EXTERNAL_MANIFEST
        if run_config.get('manifest_hash') != sha256_file(manifest):
            problems.append(f'it used construction manifest hash '
                            f'{str(run_config.get("manifest_hash"))[:12]}, not '
                            f'{EXTERNAL_MANIFEST} ({sha256_file(manifest)[:12]})')
        paths = ProjectPaths.from_root(self.root, ROOT)
        expected_preprocess = load_model(paths.preprocess_config,
                                         PreprocessConfig).config_hash()
        if run_config.get('preprocess_hash') != expected_preprocess:
            problems.append(f'it used preprocess config '
                            f'{str(run_config.get("preprocess_hash"))[:12]}, not '
                            f'{expected_preprocess[:12]}')
        target = run_config.get('target') or {}
        if target.get('model_id') != STUDENTS[student].hf_id:
            problems.append(f'it built against target model '
                            f'{target.get("model_id")!r}, not '
                            f'{STUDENTS[student].hf_id!r}')
        builder = run_config.get('builder') or {}
        if not builder:
            problems.append('its run_config.json records no builder block')
        else:
            if builder.get('max_rounds_per_stage') != ROUNDS:
                problems.append(f'it ran with a round cap of '
                                f'{builder.get("max_rounds_per_stage")}, not {ROUNDS}')
            if builder.get('max_candidates_per_round') != 4:
                problems.append(f'it ran with {builder.get("max_candidates_per_round")} '
                                f'candidates per round, not 4')
            if builder.get('model_id') != 'gpt-5.6-luna':
                problems.append(f'it used Builder model {builder.get("model_id")!r}')
        if run_config.get('build_size') != 399 or run_config.get('val_size') != 394:
            problems.append(f'it ran on D_build {run_config.get("build_size")} / '
                            f'D_val {run_config.get("val_size")}, not 399 / 394')
        return problems

    def adopt_staged_construction(self, student: str) -> str:
        """Take over a construction the minimal launcher already completed.

        Returns one of ``adopted``, ``held`` or ``none``. Adoption is only for a
        run that finished and whose provenance matches this batch; a partial or
        refused attempt is held for diagnosis, because the alternative -- a
        fresh attempt -- is a second independent optimisation run reported as
        the first.
        """
        job = Job('build', student)
        attempts = self.staged_attempts(student)
        if not attempts:
            return 'none'
        complete = [a for a in attempts if a['final'] is not None]
        if not complete:
            self.hold(job.key,
                      'a staged construction attempt exists but produced no final '
                      'bundle; a fresh attempt would resample the Builder, so this '
                      'needs an explicit decision',
                      evidence={'attempts': [a['relative'] for a in attempts]})
            self.notify('QUESTION',
                        f'{self.session}: {student} has {len(attempts)} staged '
                        f'construction attempt(s) under construction/{student}/ with '
                        f'no final bundle. Held rather than started fresh: a new '
                        f'Stage 3 draws new Luna proposals. Evidence '
                        f'{[a["relative"] for a in attempts]}. Reply with import or '
                        f'restart approval.')
            return 'held'
        # Newest complete attempt wins. An infrastructure-only relaunch (the
        # served-model-name fix, a serving-environment repair) produces
        # attempt_02, and it supersedes attempt_01 without being a resampling
        # decision: the Builder work is whatever the adopted attempt actually
        # did, and the superseded attempts are recorded, not deleted. A newest
        # attempt that fails its checks holds rather than falling back to an
        # older one -- picking a different attempt is a human decision.
        staged = complete[-1]
        superseded = [self._supersession_record(other) for other in attempts
                      if other['relative'] != staged['relative']]
        not_infra = [record for record in superseded if not record['infra_only']]
        if not_infra:
            # Being older does not make an attempt infrastructure-only. If a
            # predecessor produced a build log, a ledger, a construction tree or
            # any paid spend, then Builder work happened in it, and adopting a
            # later attempt would silently discard or double that work. Only a
            # human can decide which run counts.
            self.hold(job.key,
                      'an earlier attempt is not provably infrastructure-only, so '
                      'adopting a later one would discard or duplicate Builder work',
                      evidence={'adopting': staged['relative'],
                                'not_infra_only': not_infra})
            self.notify('QUESTION',
                        f'{self.session}: {student} has {len(not_infra)} earlier '
                        f'construction attempt(s) that are NOT infrastructure-only: '
                        f'{[r["attempt"] for r in not_infra]} carry '
                        f'{[r["builder_evidence"] for r in not_infra]}. I will not '
                        f'adopt {staged["relative"]} over them, because that would '
                        f'either discard or double-count Builder work. Held; decide '
                        f'which attempt is the run of record.')
            return 'held'
        if superseded:
            self.log(f'{student}: adopting {staged["relative"]}; '
                     f'{len(superseded)} earlier attempt(s) are infrastructure-only '
                     f'and kept: {[r["attempt"] for r in superseded]}')
        problems = self._staged_problems(student, staged)
        if problems:
            self.hold(job.key, 'staged construction does not match this batch: '
                               + '; '.join(problems),
                      evidence={'attempt': staged['relative'], 'problems': problems})
            self.notify('QUESTION',
                        f'{self.session}: {student} has a staged construction at '
                        f'{staged["relative"]} that I will not adopt: '
                        f'{"; ".join(problems)}. Held; no fresh attempt started.')
            return 'held'
        admission = staged['budget_admission'] or {}
        published = self.publish_bundle(
            student, str(staged['dir'].relative_to(self.root)))
        if published.get('status'):
            self.hold(job.key, f'could not adopt the staged construction: '
                               f'{published.get("problems")}',
                      evidence=published)
            return 'held'
        entry = {
            'utc': utc(), 'host': self.host, 'session': self.session,
            'kind': 'build', 'key': job.key, 'model': student, 'student': student,
            'hf_id': STUDENTS[student].hf_id, 'row': None, 'slot': None,
            'items': 'construction', 'attempt': staged['relative'].split('_')[-1],
            'results_dir': staged['relative'], 'returncode': 0, 'status': 'ok',
            'adopted_from_staged_launcher': True,
            'adopted_attempt': staged['relative'],
            'superseded_attempts': superseded,
            'launch_evidence': staged['launch_evidence'],
            'launcher_status': staged['status'],
            'budget_admission': admission,
            'loop_result': staged['loop_result'],
            # The launcher records elapsed_s; the reporter reads seconds. Carry
            # the real number across rather than a key that is always absent.
            'seconds': (staged['launch_evidence'] or {}).get('elapsed_s')
            or (staged['launch_evidence'] or {}).get('seconds'),
            'elapsed_s': (staged['launch_evidence'] or {}).get('elapsed_s'),
            'spend_usd': admission.get('spent_usd'),
            **published,
        }
        write_json_atomic(self.marker(job), entry)
        self.record(entry)
        self.log(f'{student}: adopted the staged construction at '
                 f'{staged["relative"]} (bundle {str(published["bundle_hash"])[:12]}, '
                 f'spend ${admission.get("spent_usd")}); no new Stage 3 started')
        self.notify('STATUS',
                    f'{self.session}: adopted {student} staged construction from '
                    f'{staged["relative"]} rather than starting a fresh Stage 3. '
                    f'Bundle {str(published["bundle_hash"])[:12]}, accounted spend '
                    f'${admission.get("spent_usd")}. Its L5 arm can now run.')
        self.publish_index()
        return 'adopted'

    def smoke_first(self, job: Job) -> bool:
        """The 24-item interface check for an arm, before its 5,507-row row.

        A zero exit code only says the process ended. Qwen3.5 once exited
        cleanly while emitting no answer tag on 15 of 24 items and burning its
        whole call cap on retries, and the full rows started anyway. This is the
        one place a broken contract stops the arm -- a merely weak model does
        not, because that is a result.
        """
        if job.kind != 'eval' or self.args.skip_smoke:
            return True
        assert job.arm is not None
        smoke = Job('eval', job.student, job.arm, SMOKE_ITEMS)
        if self.is_done(smoke):
            return True
        held, reason = self.is_held(smoke)
        if held:
            self.log(f'{job.key}: its interface smoke is held ({reason})')
            return False
        if self.attempts_used(smoke) >= self.args.max_total_attempts:
            self.hold(smoke.key, 'smoke attempts exhausted')
            self.hold(job.key, 'its interface smoke has no attempts left')
            return False
        entry = self.run_job(smoke, self.next_attempt(smoke))
        if entry['status'] == 'ok':
            return True
        if self.attempts_used(smoke) >= self.args.max_total_attempts:
            self.hold(smoke.key, f'interface smoke failed: {entry["status"]}',
                      evidence=entry.get('archived_to') or entry.get('problems'))
            self.hold(job.key, 'its interface smoke is held; the full row would '
                               'spend hours on a broken contract')
            self.notify('ERROR',
                        f'{self.session}: interface smoke failed for {smoke.key} '
                        f'({entry["status"]}; {str(entry.get("problems"))[:200]}). '
                        f'Evidence {entry.get("archived_to")}. Holding this arm; no '
                        f'{ITEM_SET} row started. Release with --release {job.key} '
                        f'after diagnosis.')
        return False

    def run_one(self, job: Job) -> bool:
        """Run one job once, with its smoke first. True when it finished."""
        if self._abort.is_set():
            return False
        if not self.smoke_first(job):
            return False
        if job.kind == 'build':
            # Never start a Stage 3 just because a marker is missing: the
            # minimal launcher may already have run this construction.
            outcome = self.adopt_staged_construction(job.student)
            if outcome == 'adopted':
                return True
            if outcome == 'held':
                return False
            spend = self.spend()
            remaining = spend.remaining_for(job.student)
            if remaining <= 0:
                self.hold(job.key, 'no build budget left', evidence=spend.payload())
                self.notify('QUESTION',
                            f'{self.session}: {job.student} construction has no '
                            f'budget left (per-student '
                            f'${spend.per_student.get(job.student, 0):.2f} of '
                            f'${BUILD_CAP_USD:.2f}; batch ${spend.batch:.2f} of '
                            f'${BATCH_CAP_USD:.2f}). Held, no Luna calls made. '
                            f'Raise the cap explicitly or drop this job.')
                return False
        entry = self.run_job(job, self.next_attempt(job))
        if entry['status'] == 'ok':
            self.notify('DONE', f'{self.session}: {job.key} ok in '
                                f'{entry["seconds"]}s, spend ${entry["spend_usd"]}, '
                                f'{entry.get("predictions") or entry.get("bundle")}')
            return True
        # A build is never restarted automatically. A fresh Stage 3 draws new
        # Builder proposals, so a silent retry is a second optimisation run
        # reported as the first one.
        if job.kind == 'build':
            self.hold(job.key,
                      f'construction failed ({entry["status"]}); a restart would '
                      f'draw fresh Builder proposals, so it needs an explicit '
                      f'decision',
                      evidence={'archived_to': entry.get('archived_to'),
                                'returncode': entry['returncode'],
                                'budget': entry.get('budget_admission')})
            self.notify('QUESTION',
                        f'{self.session}: {job.key} failed with {entry["status"]} '
                        f'(rc={entry["returncode"]}, spend ${entry["spend_usd"]}). '
                        f'HELD, not retried: restarting Stage 3 from scratch would '
                        f'draw new Luna proposals, which is a second independent '
                        f'run rather than a resume. Evidence '
                        f'{entry.get("archived_to")}. Reply with a decision; '
                        f'release with --release {job.key}.')
            return False
        if entry['status'] == 'plan_violation':
            self.hold(job.key, 'plan violation', evidence=entry.get('problems'))
            self.notify('ERROR', f'{self.session}: {job.key} is a plan violation: '
                                 f'{entry.get("problems")}. Held.')
            return False
        if self.attempts_used(job) >= self.args.max_total_attempts:
            self.hold(job.key,
                      f'{entry["status"]} after {self.attempts_used(job)} attempt(s)',
                      evidence={'archived_to': entry.get('archived_to'),
                                'problems': entry.get('problems')})
            self.notify('ERROR',
                        f'{self.session}: {job.key} failed '
                        f'{self.attempts_used(job)} attempt(s) ({entry["status"]}: '
                        f'{str(entry.get("problems"))[:200]}). Held for diagnosis; '
                        f'evidence {entry.get("archived_to")}.')
        return False

    def execute(self) -> int:
        self.set_state(phase='running')
        self.seed_reused_bundles()
        self.publish_plan()
        self.publish_index()
        all_jobs = self.plan()
        pending = [job for job in all_jobs if not self.is_done(job)]
        frozen = self.frozen()
        self.log(f'queue {self.queue}: {len(pending)} of {len(all_jobs)} job(s) '
                 f'pending; item set {ITEM_SET} ({len(frozen["ids"])} rows, '
                 f'{frozen["item_set_sha256"][:12]})')
        deadline = time.time() + self.args.max_wait_hours * 3600
        while pending and not self.get_state('stop'):
            if self._abort.is_set():
                self.log(f'not starting anything new: '
                         f'{self.get_state("abort_reason")}')
                break
            blocked: dict[str, str] = {}
            progressed = False
            for key in self.queue:
                if self._abort.is_set():
                    break
                mine = [job for job in pending if job.student == key]
                if not mine:
                    continue
                runnable = []
                for job in mine:
                    ok, reason = self.ready(job)
                    if ok:
                        runnable.append(job)
                    else:
                        blocked[job.key] = reason
                if not runnable:
                    continue
                # Before claiming anything or touching the server: a staged
                # construction may still be running for this student, and it
                # owns the GPU. Starting a second build or adopting its
                # half-written output would corrupt both.
                live = self.staged_live_owner(key)
                if live is not None:
                    self.log(f'{key}: a staged construction may still be running '
                             f'({live["attempt"]}: {live["reason"]}); not claiming '
                             f'it, not starting a server, not adopting')
                    for job in mine:
                        blocked[job.key] = 'waiting_staged_run'
                    continue
                if not self.claim(key):
                    for job in mine:
                        blocked[job.key] = 'owned_elsewhere'
                    continue
                try:
                    self.ensure_server(STUDENTS[key])
                except SystemExit as exc:
                    self.log(f'{key}: cannot serve it ({exc})')
                    self.record({'model': key, 'status': 'server_unavailable',
                                 'reason': str(exc)})
                    # Held, not discarded: a job dropped from the queue would
                    # let the run report itself complete with arms missing.
                    for job in mine:
                        blocked[job.key] = 'server_unavailable'
                        self.hold(job.key, f'server unavailable: {exc}'[:400])
                    self.notify('ERROR', f'{self.session}: cannot serve {key}: '
                                         f'{exc}. Its {len(mine)} job(s) are held, '
                                         f'not discarded.')
                    continue
                for job in runnable:
                    if self._abort.is_set():
                        break
                    if self.run_one(job):
                        progressed = True
                    elif not self.args.fake and not healthy():
                        # A dead engine turns every remaining row into a
                        # one-second failure. Leave the rest for the next pass
                        # rather than manufacture failures for this student.
                        self.log(f'{key}: the server is down; leaving the rest of '
                                 'its queue for the next pass')
                        break
                pending = [job for job in all_jobs if not self.is_done(job)]
            pending = [job for job in all_jobs if not self.is_done(job)]
            self.set_state(pending=[job.key for job in pending], blocked=blocked,
                           held=self.holds())
            if not pending or self._abort.is_set():
                break
            # Both are transient external conditions the driver should poll
            # through rather than give up on: the shared clip cache is still
            # importing, or another host's staged build is still running.
            waiting = {k: v for k, v in blocked.items()
                       if v in ('waiting_perception', 'waiting_staged_run')}
            # Terminal state -- held or out of attempts -- is what stops the
            # outer loop. Without this a failing job stays pending forever and
            # the pass count, not the attempt cap, decides how often it re-runs.
            runnable_left = [job for job in pending if self.ready(job)[0]]
            if progressed:
                continue
            if not runnable_left and not waiting:
                self.log(f'nothing can make progress; blocked: '
                         f'{json.dumps(blocked)[:600]}')
                break
            if not waiting:
                self.log('no job advanced this pass and none is waiting on the '
                         'clip cache; stopping rather than looping')
                break
            if not self.args.wait_for_perception:
                self.log(f'{len(waiting)} arm(s) need the shared clip cache; not '
                         'waiting (--no-wait-for-perception)')
                break
            if time.time() > deadline:
                coverage = self.perception_coverage(refresh=True)
                self.log(f'waited {self.args.max_wait_hours}h for the clip cache; '
                         'stopping with those arms unrun')
                self.notify('QUESTION',
                            f'{self.session}: waited {self.args.max_wait_hours}h and '
                            f'{len(waiting)} arm(s) still cannot run: '
                            f'{coverage["rows_missing"]} of {coverage["rows_total"]} '
                            f'frozen rows have no cached clip. Options: (a) keep '
                            f'waiting, (b) let this host preprocess the remainder '
                            f'locally after stopping its server, (c) stop and report '
                            f'the arms that ran. Holding; no local preprocessing '
                            f'started and nothing discarded.')
                break
            if not self.stop_server():
                self.latch_abort('could not stop the server before waiting')
                break
            coverage = self.perception_coverage(refresh=True)
            self.log(f'{len(waiting)} arm(s) waiting on the shared clip cache '
                     f'({coverage["rows_missing"]} of {coverage["rows_total"]} rows '
                     f'uncached, coordinator marker '
                     f'{coverage["coordinator_marker"]}); server stopped, '
                     f're-checking in {self.args.perception_poll_seconds}s')
            slept = 0
            while (slept < self.args.perception_poll_seconds
                   and not self.get_state('stop') and not self._abort.is_set()):
                time.sleep(5)
                slept += 5
            self.perception_coverage(refresh=True)
        pending = [job for job in all_jobs if not self.is_done(job)]
        self.set_state(phase='finishing', pending=[job.key for job in pending],
                       held=self.holds())
        return 0 if not pending else 1

    # -- entry point -------------------------------------------------------- #

    def install_signal_handlers(self) -> None:
        """A plain TERM must not leave a server and a job running unattended."""
        def handler(signum: int, _frame: Any) -> None:
            name = signal.Signals(signum).name
            self.latch_abort(f'received {name}')
            self.set_state(stop=True)
            self.kill_job(f'{name} received')
            self.stop_server()
            self.record({'status': 'interrupted', 'signal': name,
                         'job': self.get_state('job')})

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def run(self) -> int:
        """Preflight, then the queue. Fails closed: no preflight, no batch."""
        self.install_signal_handlers()
        result = preflight(self)
        text = render_preflight(result)
        stamp = utc_stamp()
        for name in (f'preflight_{self.session}_{stamp}',
                     f'preflight_{self.session}_latest'):
            write_json_atomic(self.reports_dir / f'{name}.json', result)
            write_text_atomic(self.reports_dir / f'{name}.txt', text)
        print(text, end='')
        if result['verdict'] != 'ready':
            self.log(f'PREFLIGHT BLOCKED: {result["problems"]}')
            self.set_state(phase='preflight_blocked')
            self.publish_status()
            self.notify('ERROR', f'{self.session}: preflight blocked, nothing '
                                 f'started. Problems: '
                                 f'{json.dumps(result["problems"])[:1200]}. See '
                                 f'reports/preflight_{self.session}_latest.txt')
            self.declare_terminal('stopped', {'reason': 'preflight_blocked',
                                              'problems': result['problems']})
            return 2
        watchdog = threading.Thread(target=self.watchdog_loop, name='watchdog')
        monitor = threading.Thread(target=self.monitor_loop, name='monitor')
        watchdog.start()
        monitor.start()
        self.log(f'memory watchdog at {MEM_LIMIT_GIB} GiB of the '
                 f'{(cgroup_value("memory.max") or 0) / 2**30:.0f} GiB cgroup (anon '
                 f'now {anon_gib():.1f} GiB); disk floor {DISK_FLOOR_GIB} GiB; build '
                 f'cap ${BUILD_CAP_USD} behind the admission gate, batch cap '
                 f'${BATCH_CAP_USD}; workers {self.args.workers}, video '
                 f'{self.args.video_workers}, max_num_seqs {self.args.max_num_seqs}')
        code = 1
        snap: dict[str, Any] = {}
        try:
            code = self.execute()
        except KeyboardInterrupt:
            self.log('interrupted; stopping this host cleanly')
            self.latch_abort('KeyboardInterrupt')
            self.kill_job('interrupt')
            code = 130
        finally:
            self.set_state(stop=True)
            # Join before the final writes: a monitor tick racing the final
            # snapshot is how a status file ends up describing neither state.
            for thread in (watchdog, monitor):
                thread.join(timeout=WATCHDOG_SECONDS + 10)
            self.stop_server()
            self.release_claims()
            self.set_state(phase='stopped')
            self.publish_index()
            snap = self.publish_status()
            self.publish_report(snap, tag=f'final_{utc_stamp()}')
        pending = (snap.get('state') or {}).get('pending') or []
        held = self.holds()
        if not pending:
            self.declare_terminal('completed', {'done': snap.get('completed')})
        elif self._abort.is_set():
            self.declare_terminal('stopped',
                                  {'reason': self.get_state('abort_reason'),
                                   'pending': pending, 'held': sorted(held)})
        self.notify('STATUS' if not pending else 'QUESTION',
                    f'{self.session}: queue {",".join(self.queue)} stopped with '
                    f'{snap.get("completed")} done marker(s), {len(pending)} pending '
                    f'({",".join(pending[:6])}), {len(held)} held '
                    f'({",".join(sorted(held)[:4])}), batch spend '
                    f'${(snap.get("spend") or {}).get("batch_usd")}, abort='
                    f'{self.get_state("abort_reason")}. Report '
                    f'{self.reports_dir.relative_to(self.root)}/'
                    f'{self.session}_latest.txt')
        return code


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def render_report(snap: dict[str, Any]) -> str:
    state = snap['state']
    job = state.get('job') or {}
    cgroup, gpu, spend = snap['cgroup'], snap['gpu'], snap['spend']
    perception = snap['perception']
    lines = [
        f'EmoHarness {snap["run"]} -- {snap["session"]} on {snap["host"]}',
        f'{snap["utc"]}  driver pid {snap["driver_pid"]}  phase {snap["phase"]}',
        f'abort: {state.get("abort_reason") or "(none)"}',
        '',
        'JOB',
        f'  current     {job.get("key") or "(idle)"}'
        + (f' attempt {job.get("attempt")} ({job.get("total_attempts")} overall) '
           f'since {job.get("started_utc")}' if job else ''),
        f'  results_dir {job.get("results_dir") or "-"}',
        f'  done {snap["completed"]}/{snap["total"]}: '
        f'{", ".join(snap["progress"]["done"]) or "(none)"}',
        f'  pending     {", ".join(state.get("pending") or []) or "(none)"}',
        f'  blocked     {json.dumps(state.get("blocked") or {})}',
        '  held        ' + json.dumps({k: v.get('reason') for k, v
                                       in (snap.get('holds') or {}).items()}),
        f'  failed dirs {snap["failure_count"]}',
        '',
        'SPEND (Luna Builder, reviewer and proxy judge; this batch only)',
        f'  batch       ${spend["batch_usd"]:.2f} of ${spend["batch_cap_usd"]:.2f} '
        f'(reserve ${spend["preflight_reserve_usd"]:.2f}, left '
        f'${spend["batch_allowance_usd"]:.2f})',
    ]
    for student, amount in spend['per_student_usd'].items():
        lines.append(f'  {student:<12s} ${amount:.2f} of '
                     f'${spend["build_cap_usd"]:.2f}')
    lines += [
        '  gate        per-call admission (scripts/budget_admission.py); our '
        'bookkeeping, not a provider limit and not a formal bound',
        '',
        'HOST',
        f'  cgroup mem  {cgroup["memory_current_gib"]} / '
        f'{cgroup["memory_max_gib"]} GiB (anon {cgroup["anon_gib"]}, watchdog at '
        f'{cgroup["anon_limit_gib"]})',
        f'  pressure    {json.dumps(cgroup["pressure"])}',
        f'  cpu.max     {cgroup["cpu_max"]}   loadavg {snap["loadavg"]}',
        f'  host avail  {snap["host_memory_available_gib"]} GiB',
        f'  disk        {snap["disk"]["free_gib"]} GiB free (floor '
        f'{snap["disk"]["floor_gib"]})',
        f'  gpu         {gpu.get("name", "n/a")} util {gpu.get("util_pct")}% mem '
        f'{gpu.get("memory_used_mib")}/{gpu.get("memory_total_mib")} MiB apps '
        f'{len(gpu.get("compute_apps") or [])}',
        f'  server      {snap["server"]["model"] or "(none)"} '
        f'healthy={snap["server"]["healthy"]} '
        f'owned_pgid={snap["server"]["owned_pgid"]} '
        f'alive={snap["server"]["owned_alive"]} '
        f'starts={state.get("server_starts")}',
        '',
        'PERCEPTION CACHE (L1, L3, L5 read it; L0 does not)',
        f'  config      {str(perception.get("config_hash"))[:16]}',
        f'  coverage    {perception.get("media_cached")}/'
        f'{perception.get("media_unique")} clips; {perception.get("rows_missing")} '
        f'of {perception.get("rows_total")} frozen rows uncached',
        f'  complete    {perception.get("complete")}   coordinator marker '
        f'{perception.get("coordinator_marker")}',
    ]
    if perception.get('missing_by_task'):
        lines.append(f'  missing     {json.dumps(perception["missing_by_task"])}')
    if perception.get('media_sample_check'):
        lines.append(f'  media check {json.dumps(perception["media_sample_check"])}')
    lines += [
        '',
        'WATCHDOGS',
        f'  memory trips {state.get("memory_trips")}   spend backstop trips '
        f'{state.get("budget_trips")}',
        '',
        'NOTES',
    ]
    lines += [f'  {note}' for note in (snap.get('notes') or ['(none)'])]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def preflight(driver: Driver) -> dict[str, Any]:
    """Every read-only check that must pass before a dollar or a GPU-hour.

    Nothing here calls a model, starts a server, or writes outside the batch
    root. Anything in ``problems`` stops the batch; ``warnings`` are recorded
    and proceed.
    """
    problems: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {'utc': utc(), 'host': driver.host,
                              'session': driver.session, 'queue': driver.queue}

    declared: dict[str, Any] = {}
    if not driver.manifest_file.is_file():
        problems.append(f'no coordinator manifest at {driver.manifest_file}')
    else:
        declared = read_json(driver.manifest_file)

    # 1. the frozen eligible item set, verified against declared hashes
    frozen: dict[str, Any] = {}
    try:
        frozen = driver.frozen()
        result['item_set'] = {
            'label': ITEM_SET, 'file': frozen['item_set_file'],
            'sha256': frozen['item_set_sha256'],
            'excluded_sha256': frozen['excluded_ids_sha256'],
            'rows': len(frozen['ids']), 'excluded': frozen['n_excluded'],
            'old_subset_stratum': frozen.get('n_old_subset_stratum'),
            'new_stratum': frozen.get('n_new_stratum'),
            'tasks': frozen['tasks'],
            'manifest_verified': frozen.get('manifest_verified'),
        }
    except Exception as exc:
        problems.append(f'the frozen item set does not verify: {exc}')
        result['item_set'] = {'error': str(exc)}
    if frozen and not frozen.get('manifest_verified'):
        problems.append('the frozen item set was loaded without verifying it '
                        'against the execution manifest')

    # 2. every input the coordinator hashed is still byte-identical
    from emoharness.common.hashing import sha256_file

    drift: dict[str, dict[str, str]] = {}
    for relative, expected in (declared.get('input_sha256') or {}).items():
        path = ROOT / relative
        if not path.is_file():
            drift[relative] = {'expected': expected, 'actual': 'MISSING'}
        else:
            actual = sha256_file(path)
            if actual != expected:
                drift[relative] = {'expected': expected, 'actual': actual}
    result['input_hash_drift'] = drift
    result['inputs_checked'] = len(declared.get('input_sha256') or {})
    if drift:
        problems.append(f'{len(drift)} declared input(s) changed since the manifest '
                        f'was written: {sorted(drift)[:5]}')

    # 3. zero overlap between the eligible set and construction -- a problem,
    #    never a warning. The whole point of the 5,507 set is that the media
    #    holdout holds.
    if frozen:
        ext = jsonl_rows(ROOT / 'data/manifests' / EXTERNAL_MANIFEST)
        counts: dict[str, int] = {}
        for row in ext:
            counts[row['split']] = counts.get(row['split'], 0) + 1
        ext_ids = {row['example_id'] for row in ext}
        ext_media = {row['media_hash'] for row in ext}
        overlap_ids = frozen['ids'] & ext_ids
        overlap_media = frozen['media_hashes'] & ext_media
        result['isolation'] = {
            'construction_counts': counts,
            'expected': {'build': 399, 'val': 394, 'ext_test': 129},
            'example_id_overlap': len(overlap_ids),
            'media_hash_overlap': len(overlap_media),
        }
        if counts != {'build': 399, 'val': 394, 'ext_test': 129}:
            problems.append(f'construction split sizes are {counts}, expected '
                            '399/394/129')
        if overlap_ids:
            problems.append(f'{len(overlap_ids)} eligible item(s) are also '
                            f'construction items')
        if overlap_media:
            problems.append(
                f'{len(overlap_media)} eligible item(s) still share an exact media '
                f'file with a construction item; the eligible set must have zero '
                f'media overlap, e.g. {sorted(overlap_media)[:2]}')
        subset = set(read_json(ROOT / 'data/splits/subset_1000.json')['ids'])
        old = len(subset & frozen['ids'])
        new = len(frozen['ids']) - old
        result['strata'] = {'old_subset': old, 'new': new,
                            'declared_old': declared.get('n_subset'),
                            'declared_new': declared.get('n_new')}
        if declared.get('n_subset') and old != declared['n_subset']:
            problems.append(f'the old-subset stratum is {old}, the manifest declares '
                            f'{declared["n_subset"]}')
        if declared.get('n_new') and new != declared['n_new']:
            problems.append(f'the new stratum is {new}, the manifest declares '
                            f'{declared["n_new"]}')

    # 4. bundles
    bundles: dict[str, Any] = {}
    reference = declared.get('reference_bundles') or {}
    scaffold_declared = (reference.get('scaffold_v3') or {}).get('hash')
    scaffold = bundle_hash(SCAFFOLD)
    bundles['scaffold_v3'] = {'path': 'harness', 'hash': scaffold,
                              'declared': scaffold_declared,
                              'matches': scaffold == scaffold_declared}
    if scaffold_declared and scaffold != scaffold_declared:
        problems.append('the v3 scaffold does not match its declared hash')
    for key in driver.queue:
        student = STUDENTS[key]
        if student.reuse_bundle is None:
            continue
        path = ROOT / student.reuse_bundle
        if not path.is_dir():
            problems.append(f'{key}: reused bundle {student.reuse_bundle} is missing')
            continue
        actual = bundle_hash(path)
        expected = (reference.get('transfer_full_rep0') or {}).get('hash')
        bundles[key] = {'path': student.reuse_bundle, 'hash': actual,
                        'declared': expected, 'matches': actual == expected}
        if expected and actual != expected:
            problems.append(f'{key}: reused bundle {actual[:12]} != declared '
                            f'{expected[:12]}')
    result['bundles'] = bundles

    # 5. call-cache compatibility.
    #
    # target/cache.py keys an entry on the model id, the prompt, the media
    # content hashes and the decoding parameters, and context.call_cache_root
    # additionally shards the cache by a hash of chat_template_kwargs -- so
    # Qwen3.5 with thinking on and off can never share entries. What is in
    # neither is the ``video`` block, which reaches the server as
    # mm_processor_kwargs and decides how many frames it samples. Two students
    # differing only there would collide, so the identity is asserted, not
    # assumed.
    from emoharness.common.config import load_model
    from emoharness.common.hashing import stable_hash
    from emoharness.target.config import TargetConfig

    video_blocks: dict[str, Any] = {}
    cache_roots: dict[str, str] = {}
    decoding: dict[str, Any] = {}
    for key, student in STUDENTS.items():
        path = ROOT / student.target
        if not path.is_file():
            continue
        config = load_model(path, TargetConfig)
        video_blocks[key] = config.video.model_dump(mode='json')
        decoding[key] = config.decoding.model_dump(mode='json')
        kwargs = dict(config.chat_template_kwargs or {})
        cache_roots[key] = (f'calls/tpl_{stable_hash(kwargs)[:12]}' if kwargs
                            else 'calls')
    distinct_video = {json.dumps(v, sort_keys=True) for v in video_blocks.values()}
    result['call_cache'] = {
        'keyed_on': ['model_id', 'prompt', 'tag', 'system', 'media content hashes',
                     'json_schema', 'max_tokens', 'temperature', 'top_p', 'seed'],
        'sharded_by_chat_template_kwargs': True,
        'video_block_in_key': False,
        'video_blocks': video_blocks,
        'video_blocks_identical': len(distinct_video) == 1,
        'decoding': decoding, 'cache_roots': cache_roots,
    }
    if len(distinct_video) != 1:
        problems.append(
            'the students declare different video blocks, which are not part of the '
            'call-cache key: reusing the shared cache across them would serve frames '
            f'sampled under another setting. Blocks: {video_blocks}')

    # 5b. the local HF cache is pinned at the revisions the transfer batch used.
    #
    # The server runs with the hub offline, so it cannot fetch a newer
    # revision -- but it would happily serve whatever the cache already holds.
    # If that is not the revision the earlier arms were produced with, the
    # comparison is between two different models wearing one id.
    revisions: dict[str, Any] = {}
    hub = Path.home() / '.cache/huggingface/hub'
    for key in driver.queue:
        student = STUDENTS[key]
        expected = HF_REVISIONS.get(key)
        ref = hub / f'models--{student.hf_id.replace("/", "--")}' / 'refs/main'
        actual = ref.read_text(encoding='utf-8').strip() if ref.is_file() else None
        revisions[key] = {'hf_id': student.hf_id, 'expected': expected,
                          'cached': actual, 'matches': actual == expected,
                          'refs_main': str(ref)}
        if expected is None:
            problems.append(f'{key}: no pinned HF revision recorded')
        elif actual is None:
            problems.append(f'{key}: no local HF cache at {ref}; with the hub '
                            f'offline the server cannot start')
        elif actual != expected:
            problems.append(f'{key}: local HF cache is at {actual[:12]}, the batch '
                            f'is pinned to {expected[:12]}; serving it would '
                            f'compare two different models under one id')
    result['hf_revisions'] = revisions
    result['hub_offline'] = True

    # 6. perception
    coverage = driver.perception_coverage(refresh=True)
    result['perception'] = coverage
    if not coverage['complete']:
        warnings.append(f'{coverage["rows_missing"]} of {coverage["rows_total"]} '
                        f'frozen rows have no cached clip; L1, L3 and L5 wait for '
                        f'the shared cache. L0 can run now.')
    sample = coverage.get('media_sample_check')
    if sample and (sample['mismatched'] or sample['media_missing']):
        problems.append(f'media sample check found {sample["mismatched"]} hash '
                        f'drift(s) and {sample["media_missing"]} missing file(s)')

    # 7. credentials: presence only, never a value
    keys = load_key_file()
    result['credentials'] = {
        'key_file': str(KEY_FILE),
        'mode': oct(KEY_FILE.stat().st_mode & 0o777) if KEY_FILE.is_file() else None,
        'names_present': sorted(keys),
        'openai_api_key_present': bool(keys.get('OPENAI_API_KEY')
                                       or os.environ.get('OPENAI_API_KEY')),
    }
    needs_key = any(STUDENTS[key].builds for key in driver.queue)
    if needs_key and not result['credentials']['openai_api_key_present']:
        problems.append(f'no OPENAI_API_KEY in {KEY_FILE} or the environment; the '
                        'Builder cannot run')

    # 8. serving environments and the steps
    serving: dict[str, Any] = {}
    for key in driver.queue:
        student = STUDENTS[key]
        venv = Path.home() / SERVING_VENVS[student.serving]
        serving[key] = {
            'hf_id': student.hf_id, 'serving': student.serving, 'venv': str(venv),
            'venv_present': (venv / 'bin/vllm').is_file(),
            'target_config': student.target,
            'target_present': (ROOT / student.target).is_file(),
            'builds': student.builds,
        }
        if not serving[key]['venv_present']:
            problems.append(f'{key}: no vllm in {venv}')
        if not serving[key]['target_present']:
            problems.append(f'{key}: missing target config {student.target}')
    result['serving'] = serving
    steps = {'python': PY_BIN, 'row_step': ROOT / ROW_STEP,
             'build_step': ROOT / BUILD_STEP, 'admission': ADMISSION_STEP,
             'serve': SERVE_STEP}
    result['steps'] = {name: {'path': str(path), 'present': path.exists()}
                       for name, path in steps.items()}
    for name, path in steps.items():
        if not path.exists():
            problems.append(f'missing {name} at {path}')

    # 9. host, GPU, budget, disk
    limit = cgroup_value('memory.max')
    usage = shutil.disk_usage(driver.run_root)
    cpu_max = Path('/sys/fs/cgroup/cpu.max')
    result['host_facts'] = {
        'cpu_max': (cpu_max.read_text(encoding='utf-8').strip()
                    if cpu_max.exists() else None),
        'memory_max_gib': round(limit / 2**30, 2) if limit else None,
        'anon_gib': round(anon_gib(), 2), 'mem_limit_gib': MEM_LIMIT_GIB,
        'disk_free_gib': round(usage.free / 2**30, 1), 'gpu': gpu_facts(),
        'concurrency': {'workers': driver.args.workers,
                        'video_workers': driver.args.video_workers,
                        'max_num_seqs': driver.args.max_num_seqs,
                        'mm_cache_gb': driver.args.mm_cache_gb,
                        'max_jobs_jit': os.environ.get('MAX_JOBS', '2')},
    }
    if not result['host_facts']['gpu'].get('available'):
        problems.append('nvidia-smi reports no GPU')
    if limit and limit / 2**30 < MEM_LIMIT_GIB + 2:
        problems.append(f'the memory watchdog at {MEM_LIMIT_GIB} GiB leaves no room '
                        f'in a {limit / 2**30:.0f} GiB cgroup')
    if usage.free / 2**30 < DISK_FLOOR_GIB:
        problems.append(f'only {usage.free / 2**30:.1f} GiB free, below the '
                        f'{DISK_FLOOR_GIB} GiB floor')
    spend = driver.spend()
    result['spend'] = spend.payload()
    if spend.batch_allowance <= 0 and needs_key:
        problems.append(f'the batch allowance is exhausted (${spend.batch:.2f} of '
                        f'${BATCH_CAP_USD:.2f})')

    # 10. the port and the GPU must be free, or explicitly adopted
    current = serving_model()
    foreign = gpu_pids()
    result['port'] = {'endpoint': ENDPOINT, 'serving': current, 'healthy': healthy(),
                      'gpu_pids': sorted(foreign),
                      'adopt_server': driver.args.adopt_server}
    if current is not None and not driver.args.adopt_server:
        problems.append(f'port {VLLM_PORT} already serves {current} and this driver '
                        f'does not own it; stop it or pass --adopt-server')
    if foreign and not driver.args.adopt_server:
        problems.append(f'the GPU carries compute process(es) {sorted(foreign)} this '
                        f'driver did not start; one student server at a time')

    # 11. what this host would do
    result['jobs'] = []
    for job in driver.plan():
        good, why = driver.marker_is_good(job)
        ready, reason = driver.ready(job)
        result['jobs'].append({'key': job.key, 'kind': job.kind, 'done': good,
                               'marker': why, 'ready': ready, 'reason': reason,
                               'attempts_used': driver.attempts_used(job)})
    result['holds'] = driver.holds()
    result['problems'] = problems
    result['warnings'] = warnings
    result['verdict'] = 'blocked' if problems else 'ready'
    return result


def render_preflight(result: dict[str, Any]) -> str:
    lines = [f'PREFLIGHT {result["session"]} on {result["host"]} -- {result["utc"]}',
             f'queue: {", ".join(result["queue"])}',
             f'verdict: {result["verdict"].upper()}', '']
    items = result.get('item_set', {})
    isolation = result.get('isolation', {})
    lines += [
        'ITEM SET (frozen, run-local; the canonical 5,577 is NOT used)',
        f'  label          {items.get("label")}',
        f'  file           {items.get("file")}',
        f'  sha256         {items.get("sha256")}',
        f'  rows           {items.get("rows")}  (excluded {items.get("excluded")}, '
        f'old-subset stratum {items.get("old_subset_stratum")}, new '
        f'{items.get("new_stratum")})',
        f'  tasks          {len(items.get("tasks") or [])}   manifest verified '
        f'{items.get("manifest_verified")}',
        '',
        'ISOLATION',
        f'  construction   {json.dumps(isolation.get("construction_counts"))}',
        f'  id overlap     {isolation.get("example_id_overlap")} (must be 0)',
        f'  media overlap  {isolation.get("media_hash_overlap")} (must be 0)',
        f'  strata         {json.dumps(result.get("strata"))}',
        f'  inputs checked {result.get("inputs_checked")} '
        f'({len(result.get("input_hash_drift") or {})} drifted)',
        '',
        'BUNDLES',
    ]
    for name, record in (result.get('bundles') or {}).items():
        lines.append(f'  {name:<14s} {record["hash"][:16]} declared_match='
                     f'{record.get("matches")}  {record["path"]}')
    cache = result.get('call_cache', {})
    perception = result.get('perception', {})
    host = result.get('host_facts', {})
    gpu = host.get('gpu', {})
    lines += [
        '',
        'CALL CACHE COMPATIBILITY',
        f'  sharded by chat_template_kwargs  '
        f'{cache.get("sharded_by_chat_template_kwargs")}',
        f'  video block part of the key      {cache.get("video_block_in_key")}',
        f'  video blocks identical           {cache.get("video_blocks_identical")}',
        f'  cache roots                      {json.dumps(cache.get("cache_roots"))}',
        '',
        f'MODEL REVISIONS (hub offline: {result.get("hub_offline")})',
    ]
    for key, record in (result.get('hf_revisions') or {}).items():
        lines.append(f'  {key:<12s} cached {str(record.get("cached"))[:12]} '
                     f'expected {str(record.get("expected"))[:12]} '
                     f'matches={record.get("matches")}')
    lines += [
        '',
        'PERCEPTION',
        f'  cached clips   {perception.get("media_cached")}/'
        f'{perception.get("media_unique")}',
        f'  uncached rows  {perception.get("rows_missing")}/'
        f'{perception.get("rows_total")}',
        f'  complete       {perception.get("complete")}  coordinator marker '
        f'{perception.get("coordinator_marker")}',
        '',
        'HOST',
        f'  cpu.max {host.get("cpu_max")}  memory.max {host.get("memory_max_gib")} '
        f'GiB  anon {host.get("anon_gib")} GiB  disk {host.get("disk_free_gib")} GiB',
        f'  gpu {gpu.get("name")} {gpu.get("memory_total_mib")} MiB driver '
        f'{gpu.get("driver")} apps {len(gpu.get("compute_apps") or [])}',
        f'  concurrency {json.dumps(host.get("concurrency"))}',
        f'  port {result.get("port", {}).get("endpoint")} serving '
        f'{result.get("port", {}).get("serving")} gpu_pids '
        f'{result.get("port", {}).get("gpu_pids")}',
        f'  key file {result.get("credentials", {}).get("key_file")} mode '
        f'{result.get("credentials", {}).get("mode")} OPENAI_API_KEY present='
        f'{result.get("credentials", {}).get("openai_api_key_present")}',
        '',
        'JOBS',
    ]
    for job in result.get('jobs', []):
        lines.append(f'  {job["key"]:<34s} done={str(job["done"]):<5s} '
                     f'ready={str(job["ready"]):<5s} attempts={job["attempts_used"]} '
                     f'{job["reason"]}')
    lines += ['', 'SPEND', f'  {json.dumps(result.get("spend"))}']
    if result.get('holds'):
        lines += ['', 'HOLDS'] + [f'  - {k}: {v.get("reason")}'
                                  for k, v in result['holds'].items()]
    if result.get('warnings'):
        lines += ['', 'WARNINGS'] + [f'  - {w}' for w in result['warnings']]
    if result.get('problems'):
        lines += ['', 'PROBLEMS (each of these stops the batch)'] + [
            f'  - {p}' for p in result['problems']]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='qwen3vl_4b,qwen3_5_4b',
                        help='comma-separated student keys, in queue order')
    parser.add_argument('--session',
                        default=os.environ.get('EMOHARNESS_SESSION',
                                               'session2-worker'),
                        help='coordination role name this host runs as')
    parser.add_argument('--notify', default='session2-manager',
                        help='comma-separated mailbox recipients')
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--plan', action='store_true',
                        help='print the matrix this host would run, and exit')
    parser.add_argument('--preflight', action='store_true',
                        help='run every read-only check; non-zero if blocked')
    parser.add_argument('--selftest', action='store_true',
                        help='drive the machinery with fakes in a scratch root')
    parser.add_argument('--report', action='store_true',
                        help='publish one status report from the shared disk, and exit')
    parser.add_argument('--release', default=None, metavar='JOB_KEY',
                        help='release one terminal hold, then exit')
    parser.add_argument('--fake', action='store_true',
                        help='pass --fake to every step: no GPU, no server, no API')
    parser.add_argument('--skip-smoke', action='store_true')
    parser.add_argument('--adopt-server', action='store_true',
                        help='accept a server this driver did not start (recorded)')
    parser.add_argument('--max-total-attempts', type=int, default=2,
                        help='attempts per job, counted across driver invocations')
    parser.add_argument('--workers', type=int, default=8,
                        help='client request concurrency (EMOHARNESS_WORKERS)')
    parser.add_argument('--video-workers', type=int, default=2,
                        help='concurrent video requests (EMOHARNESS_VIDEO_WORKERS)')
    parser.add_argument('--max-num-seqs', type=int, default=8,
                        help='vLLM --max-num-seqs for the student server')
    parser.add_argument('--mm-cache-gb', type=int, default=0,
                        help='vLLM multimodal processor cache; 0 saves host RAM')
    parser.add_argument('--verify-media', action='store_true',
                        help='re-hash a sample of media files during the preflight')
    parser.add_argument('--item-set-file', type=Path, default=None,
                        help='testing only: use another frozen eligible set')
    parser.add_argument('--excluded-ids-file', type=Path, default=None,
                        help='testing only: use another exclusion list')
    parser.add_argument('--execution-manifest', type=Path, default=None,
                        help='testing only: use another execution manifest')
    parser.add_argument('--expect-count', type=int, default=5507,
                        help='rows the frozen eligible set must carry')
    parser.add_argument('--allow-unverified-item-set', action='store_true',
                        help='testing only: skip execution-manifest verification')
    parser.add_argument('--wait-for-perception', dest='wait_for_perception',
                        action='store_true', default=True)
    parser.add_argument('--no-wait-for-perception', dest='wait_for_perception',
                        action='store_false',
                        help='report the arms that need the shared cache and stop')
    parser.add_argument('--perception-poll-seconds', type=int, default=300)
    parser.add_argument('--max-wait-hours', type=float, default=24.0)
    args = parser.parse_args(argv)
    args.models = [key.strip() for key in args.models.split(',') if key.strip()]
    args.notify = [name.strip() for name in args.notify.split(',') if name.strip()]
    unknown = [key for key in args.models if key not in STUDENTS]
    if unknown:
        parser.error(f'unknown student key(s): {", ".join(unknown)}; expected from '
                     f'{", ".join(STUDENTS)}')
    return args


def print_plan(driver: Driver) -> int:
    frozen = driver.frozen()
    print(f'run root: {driver.run_root.relative_to(driver.root)}')
    print(f'session:  {driver.session} on {driver.host}')
    print(f'queue:    {", ".join(driver.queue)}')
    print(f'item set: {ITEM_SET}  {len(frozen["ids"])} rows  sha '
          f'{frozen["item_set_sha256"][:16]}  (strata '
          f'{frozen.get("n_old_subset_stratum")} old / '
          f'{frozen.get("n_new_stratum")} new, {frozen["n_excluded"]} excluded)\n')
    for key in driver.queue:
        student = STUDENTS[key]
        origin = ('builds its own bundle' if student.builds
                  else f'reuses {student.reuse_bundle}')
        print(f'{key:12s} {student.hf_id:30s} serving={student.serving}  {origin}')
        if student.builds:
            job = Job('build', key)
            print(f'    {"build":22s} stage {STAGE} rep {REPLICATE} rounds {ROUNDS} '
                  f'from harness/  cap ${BUILD_CAP_USD:.0f} behind the admission '
                  f'gate  done={driver.marker_is_good(job)[0]}')
        for arm in ARMS:
            job = Job('eval', key, arm)
            print(f'    {arm.slot:22s} row={arm.row:3s} items={ITEM_SET} '
                  f'bundle={arm.bundle or "-":9s} perception={arm.needs_perception}  '
                  f'done={driver.marker_is_good(job)[0]}')
    jobs = driver.plan()
    print(f'\njobs on this host: {len(jobs)} '
          f'({sum(1 for j in jobs if j.kind == "build")} build, '
          f'{sum(1 for j in jobs if j.kind == "eval")} eval)')
    print(f'whole batch: {sum(1 for s in STUDENTS.values() if s.builds)} builds, '
          f'{len(STUDENTS) * len(ARMS)} evaluation arms')
    return 0


def _synthetic_item_set(scratch: Path) -> tuple[Path, Path, Path]:
    """A small frozen set, its exclusions and a manifest, all inside the scratch.

    The selftest must not do fake work against the real 5,507-row set or the
    real caches, and it must still exercise the VERIFIED load path rather than
    the test-only escape hatch. So it builds its own declared triple from the
    fake benchmark and hashes it exactly as the coordinator does.
    """
    from emoharness.common.hashing import sha256_file

    rows = jsonl_rows(scratch / 'data/manifests/mme_emotion.jsonl')
    items = scratch / 'frozen/evaluation_items.jsonl'
    items.parent.mkdir(parents=True, exist_ok=True)
    with items.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    excluded = scratch / 'frozen/evaluation_excluded_ids.json'
    known = {row['example_id'] for row in rows}
    fake_excluded = [f'synthetic_excluded_{index}' for index in range(70)]
    assert not (set(fake_excluded) & known)
    excluded.write_text(json.dumps({
        'reason': 'synthetic selftest exclusion list', 'ids': fake_excluded},
        indent=2), encoding='utf-8')
    manifest = scratch / 'frozen/execution_manifest.json'
    manifest.write_text(json.dumps({
        'evaluation_file': items.name,
        'evaluation_sha256': sha256_file(items),
        'exclusions_sha256': sha256_file(excluded),
        'n_full': len(rows), 'n_excluded': 70,
        'note': 'synthetic manifest written by --selftest',
    }, indent=2), encoding='utf-8')
    return items, excluded, manifest


def selftest(args: argparse.Namespace) -> int:
    """Drive the real queue logic end to end with fakes, in a scratch root.

    Exercised here: subprocess plumbing and process groups, attempt directories,
    artifact validation with structural faults separated from model outcomes,
    resume verification, terminal holds and the attempt cap across invocations,
    failed-attempt archiving, bundle publication, the per-host index, the ledger
    reader, the admission gate, the status and report writers, and that none of
    it touches the real batch root, the real frozen set or the real caches.
    No GPU, no server, no API key, no network.
    """
    scratch = Path(os.environ.get(
        'EMOHARNESS_SELFTEST_ROOT',
        Path(os.environ.get('TMPDIR', '/tmp')) / f'emoharness-selftest-{os.getpid()}'))
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    print(f'selftest root: {scratch}')
    for step in ('prepare_benchmark.py', 'prepare_external.py'):
        out = subprocess.run(
            [str(PY_BIN), str(ROOT / 'examples' / step), '--root', str(scratch),
             '--fake'],
            cwd=ROOT, env=dict(os.environ, EMOHARNESS_RESULTS_DIR='results'),
            capture_output=True, text=True)
        if out.returncode != 0:
            print(f'{step} failed:\n{out.stdout}\n{out.stderr}')
            return 1
    # The build step reads data/manifests/<EXTERNAL_MANIFEST> under --root, and
    # the fake prepare step writes external.jsonl. Give the scratch root the
    # v3 name so construction exercises the same env the real run uses.
    # v3_full.yaml sets require_isolated_splits, and the fake generator reuses
    # media paths and source ids across splits, so give each synthetic row its
    # own identity. Fake mode never opens the media, and this is the same
    # property prepare_v3_splits.py establishes for the real manifest.
    fake_external = scratch / 'data/manifests/external.jsonl'
    if fake_external.is_file():
        target = fake_external.with_name(EXTERNAL_MANIFEST)
        seen: set[str] = set()
        with target.open('w', encoding='utf-8') as fh:
            for row in jsonl_rows(fake_external):
                # media_path must stay a real file -- construction stats it even
                # in fake mode -- so isolation comes from dropping the rows that
                # reuse a path and giving the rest their own hash and source id.
                if row['media_path'] in seen:
                    continue
                seen.add(row['media_path'])
                row['media_hash'] = f'selftest_{row["example_id"]}'
                row['source_id'] = row['example_id']
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    items, excluded, manifest = _synthetic_item_set(scratch)
    synthetic_rows = len(jsonl_rows(items))
    checks: list[tuple[str, bool, str]] = []
    fake_args = parse_args([
        '--models', 'qwen3vl_4b', '--session', 'selftest', '--no-notify', '--fake',
        '--skip-smoke', '--max-total-attempts', '1', '--no-wait-for-perception',
        '--item-set-file', str(items), '--excluded-ids-file', str(excluded),
        '--execution-manifest', str(manifest),
        '--expect-count', str(synthetic_rows)])
    driver = Driver(fake_args, root=scratch)
    real_before = {p for p in (ROOT / RUN_REL).rglob('*') if p.is_file()}

    # 1. the real frozen set is verified read-only, and never worked against
    real_driver = Driver(parse_args(['--models', 'qwen3vl_4b', '--no-notify']),
                         root=scratch)
    real = real_driver.frozen()
    external_media = {row['media_hash'] for row in
                      jsonl_rows(ROOT / 'data/manifests' / EXTERNAL_MANIFEST)}
    checks.append(('the REAL frozen set verifies to 5,507 rows and 70 exclusions',
                   len(real['ids']) == 5507 and real['n_excluded'] == 70
                   and real['manifest_verified'] is True,
                   f'{len(real["ids"])} rows, {real["n_excluded"]} excluded, '
                   f'verified={real["manifest_verified"]}'))
    checks.append(('the REAL frozen set has zero construction media overlap',
                   not (real['media_hashes'] & external_media), 'overlap 0'))
    checks.append(('the strata are 986 old / 4,521 new, never 1,000 / 4,577',
                   real.get('n_old_subset_stratum') == 986
                   and real.get('n_new_stratum') == 4521,
                   f'{real.get("n_old_subset_stratum")} / '
                   f'{real.get("n_new_stratum")}'))
    checks.append(('the item-set label is not "full"', ITEM_SET == 'full5507',
                   ITEM_SET))

    # 2. the selftest itself runs against a synthetic declared triple
    frozen = driver.frozen()
    checks.append(('the selftest uses a synthetic frozen set, still verified',
                   len(frozen['ids']) == synthetic_rows
                   and frozen['manifest_verified'] is True
                   and str(scratch) in frozen['item_set_file'],
                   f'{len(frozen["ids"])} synthetic rows inside the scratch root'))

    # 3. the queue and its published plan
    jobs = driver.plan()
    checks.append(('plan is one build then four arms',
                   [j.kind for j in jobs] == ['build'] + ['eval'] * 4,
                   str([j.key for j in jobs])))
    driver.publish_plan()
    checks.append(('per-session plan published',
                   (driver.run_root / 'plan.selftest.json').is_file(), ''))

    # 4. ownership is advisory but exclusive
    checks.append(('claims its student', driver.claim('qwen3vl_4b'), ''))
    other = Driver(parse_args(['--models', 'qwen3vl_4b', '--session', 'other',
                               '--no-notify', '--fake']), root=scratch)
    checks.append(('a second session cannot claim it',
                   not other.claim('qwen3vl_4b'), ''))

    # 5. attempt directories and the agreed layout
    build = Job('build', 'qwen3vl_4b')
    first = driver.next_attempt(build)
    (scratch / driver.attempt_dir(build, first)).mkdir(parents=True)
    checks.append(('attempt ids advance past what exists',
                   driver.next_attempt(build) == first + 1,
                   f'{first} -> {driver.next_attempt(build)}'))
    shutil.rmtree(scratch / driver.attempt_dir(build, first))
    checks.append(('eval layout is <model>/<slot>/<items>/attempt_NN',
                   driver.attempt_dir(Job('eval', 'qwen3vl_4b', ARMS[0]), 1)
                   == f'{RUN_REL}/qwen3vl_4b/L0/{ITEM_SET}/attempt_01',
                   driver.attempt_dir(Job('eval', 'qwen3vl_4b', ARMS[0]), 1)))

    # 6. the build step behind the admission gate
    entry = driver.run_job(build, driver.next_attempt(build))
    checks.append(('fake build succeeds through the admission wrapper',
                   entry['status'] == 'ok',
                   json.dumps({k: entry.get(k) for k in
                               ('status', 'returncode', 'seconds', 'bundle')})))
    checks.append(('the admission gate recorded its accounting',
                   bool(entry.get('budget_admission')),
                   json.dumps(entry.get('budget_admission'))))
    checks.append(('the build published a bundle record',
                   driver.bundle_record('qwen3vl_4b').is_file(), ''))
    checks.append(('the build wrote a VERIFIED done marker',
                   driver.marker_is_good(build)[0], driver.marker_is_good(build)[1]))

    # 7. an evaluation arm and the marker schema the report consumes
    arm_job = Job('eval', 'qwen3vl_4b', ARMS[0])
    entry = driver.run_job(arm_job, driver.next_attempt(arm_job))
    needed = ('predictions', 'item_set_file', 'item_set_sha256',
              'excluded_ids_sha256', 'n_items_expected', 'n_items_written',
              'provenance', 'asr_config_hash')
    checks.append(('the arm wrote predictions where its marker says',
                   bool(entry.get('predictions'))
                   and (scratch / entry['predictions']).is_file(),
                   str(entry.get('predictions'))))
    checks.append(('the marker carries every field the report script asked for',
                   all(entry.get(k) is not None for k in needed),
                   'missing: ' + str([k for k in needed if entry.get(k) is None])))
    checks.append(('the marker names the model, not only the student',
                   entry.get('model') == 'qwen3vl_4b', str(entry.get('model'))))
    checks.append(('the arm scored the whole frozen set',
                   entry.get('n_items_written') == synthetic_rows,
                   str(entry.get('n_items_written'))))
    checks.append(('model outcomes are recorded, not treated as failures',
                   entry['status'] == 'ok' and 'diagnostics' in entry,
                   json.dumps({k: v for k, v in (entry.get('diagnostics') or {}).items()
                               if k != 'summary'})))

    # 8. structural faults fail; model outcomes do not
    def synth(path: Path, n: int, tasks: int = 8, bad: int = 0) -> Path:
        ids = sorted(frozen['ids'])[:n]
        with path.open('w', encoding='utf-8') as fh:
            for index, example_id in enumerate(ids):
                text = '' if index < bad else '<answer>happy</answer>'
                fh.write(json.dumps({
                    'example_id': example_id, 'task': f'T{index % tasks}',
                    'labels': ['happy'], 'response_text': text, 'target_calls': 1,
                    'parse_failures': 1 if index < bad else 0,
                    'input_tokens': 10, 'output_tokens': 5}) + '\n')
        return path

    good = synth(scratch / 'good.jsonl', synthetic_rows)
    structural, _ = validate_predictions(good, ARMS[0], ITEM_SET, frozen['ids'])
    checks.append(('a complete row over the exact id set passes', structural == [],
                   str(structural)))
    rough = synth(scratch / 'rough.jsonl', synthetic_rows,
                  bad=max(1, synthetic_rows // 4))
    structural, diagnostics = validate_predictions(rough, ARMS[0], ITEM_SET,
                                                   frozen['ids'])
    checks.append(('a quarter of the answers empty is an outcome, not a fault',
                   structural == [] and diagnostics['empty_responses'] > 0,
                   f'structural={structural} empty={diagnostics["empty_responses"]}'))
    short = synth(scratch / 'short.jsonl', synthetic_rows - 5)
    structural, _ = validate_predictions(short, ARMS[0], ITEM_SET, frozen['ids'])
    checks.append(('a short row is a structural fault',
                   any('not scored' in p or 'rows, expected' in p
                       for p in structural), str(structural[:2])))
    few = synth(scratch / 'few.jsonl', synthetic_rows, tasks=7)
    structural, _ = validate_predictions(few, ARMS[0], ITEM_SET, frozen['ids'])
    checks.append(('a missing task is a structural fault',
                   any('tasks' in p for p in structural), str(structural[:2])))
    swapped = scratch / 'swapped.jsonl'
    with swapped.open('w', encoding='utf-8') as fh:
        for index in range(synthetic_rows):
            fh.write(json.dumps({
                'example_id': f'not_in_the_frozen_set_{index}',
                'task': f'T{index % 8}', 'labels': ['happy'],
                'response_text': '<answer>happy</answer>', 'target_calls': 1,
                'parse_failures': 0, 'input_tokens': 1, 'output_tokens': 1}) + '\n')
    structural, _ = validate_predictions(swapped, ARMS[0], ITEM_SET, frozen['ids'])
    checks.append(('a same-length but different id set is caught',
                   any('outside the frozen eligible set' in p for p in structural),
                   str(structural[:1])))

    # 9. resume verification rejects stale markers
    marker = driver.marker(arm_job)
    intact_marker = read_json(marker)
    stale = dict(intact_marker)
    stale['item_set_sha256'] = 'deadbeef'
    write_json_atomic(marker, stale)
    ok, why = driver.marker_is_good(arm_job)
    checks.append(('a marker from a different item set is rejected',
                   not ok and 'item set' in why, why))
    stale['item_set_sha256'] = frozen['item_set_sha256']
    stale['predictions'] = 'nowhere/at/all.jsonl'
    write_json_atomic(marker, stale)
    ok, why = driver.marker_is_good(arm_job)
    checks.append(('a marker whose predictions are gone is rejected',
                   not ok and 'missing' in why, why))
    # Put the real marker back: later checks read the file it points at.
    write_json_atomic(marker, intact_marker)

    # 10. terminal holds and the attempt cap survive a restart
    driver.hold('qwen3vl_4b__build', 'selftest hold')
    reborn = Driver(fake_args, root=scratch)
    held, reason = reborn.is_held(build)
    checks.append(('a hold survives a new driver process', held, reason))
    checks.append(('a held job is not ready', not reborn.ready(build)[0],
                   reborn.ready(build)[1]))
    reborn.release('qwen3vl_4b__build')
    checks.append(('--release clears the hold', not reborn.is_held(build)[0], ''))
    exhausted = Job('eval', 'qwen3vl_4b', ARMS[1])
    reborn.bump_attempt(exhausted, 'failed')
    checks.append(('the attempt cap is enforced across invocations',
                   Driver(fake_args, root=scratch).ready(exhausted)[1]
                   == 'attempts_exhausted',
                   Driver(fake_args, root=scratch).ready(exhausted)[1]))

    # 10b. staged-construction adoption, and the gates that must refuse it.
    #
    # The first version of this read run_config.json and loop_result.json from
    # the attempt root, where the library never writes them, so every lineage
    # test silently passed on an empty dict. These checks pin the real nested
    # paths and prove each gate refuses rather than tolerates.
    def stage_staged(student: str, attempt: str = 'attempt_07', *,
                     final: bool = True, run_config: dict[str, Any] | None = None,
                     loop_result: dict[str, Any] | None = None,
                     admission: dict[str, Any] | None = None,
                     evidence: dict[str, Any] | None = None,
                     status: dict[str, Any] | None = None) -> Path:
        """Stage a construction using the REAL frozen launcher record schema.

        Taken verbatim from the files scripts/start_student_construction_20260918.py
        wrote during the 2026-09-18 Qwen3-VL-4B launch: launch_evidence.json
        carries returncode/latched/code/weights/frozen/preflight/scheduling and
        has no phase and no launcher_pid; status.json carries phase and the pids.
        An invented schema would have let the either/or read pass here and hang
        on the real thing.
        """
        from emoharness.common.hashing import sha256_file

        base = driver.run_root / 'construction' / student / attempt
        stage = base / 'construction' / f'stage{STAGE}' / str(REPLICATE)
        stage.mkdir(parents=True, exist_ok=True)
        if final:
            shutil.copytree(SCAFFOLD, stage / 'final', dirs_exist_ok=True)
        good_config = {
            'parent_bundle_hash': bundle_hash(SCAFFOLD),
            'manifest_hash': sha256_file(ROOT / 'data/manifests' / EXTERNAL_MANIFEST),
            'preprocess_hash': driver.perception_coverage()['config_hash'],
            'target': {'model_id': STUDENTS[student].hf_id},
            'builder': {'max_rounds_per_stage': ROUNDS,
                        'max_candidates_per_round': 4, 'model_id': 'gpt-5.6-luna'},
            'build_size': 399, 'val_size': 394,
        }
        write_json_atomic(stage / 'run_config.json',
                          good_config if run_config is None else run_config)
        write_json_atomic(stage / 'loop_result.json',
                          {'stopped_reason': 'builder_stopped', 'rounds': [{}, {}]}
                          if loop_result is None else loop_result)
        write_json_atomic(base / 'budget_admission.json',
                          {'returncode': 0, 'budget_stop': False, 'calls_refused': 0,
                           'in_flight_calls': [], 'reserved_in_flight_usd': 0.0,
                           'spent_usd': 0.34}
                          if admission is None else admission)
        approval = read_json_or(
            driver.run_root / 'approvals' / f'BUILD_READY.{student}.json', {})
        good_evidence = {
            'utc': utc(), 'host': driver.host, 'session': driver.session,
            'student': student, 'returncode': 0, 'latched': None,
            'elapsed_s': 1234.5, 'owned_pgids': [999999],
            'code': {'launcher_sha256': approval.get('launcher_sha256'),
                     'wrapper_sha256': approval.get('wrapper_sha256')},
            'weights': {'model_id': STUDENTS[student].hf_id,
                        'expected_revision': HF_REVISIONS[student],
                        'cached_revision': HF_REVISIONS[student]},
            'frozen': {'scaffold_hash': bundle_hash(SCAFFOLD),
                       'builder_config': BUILDER_CONFIG,
                       'external_manifest': EXTERNAL_MANIFEST},
            'preflight': [{'name': 'attempt_unique', 'ok': True, 'blocking': True}],
            'scheduling': {'workers': 8, 'video_workers': 2, 'max_num_seqs': 8},
            'budget': {'cap_usd': BUILD_CAP_USD},
        }
        write_json_atomic(base / 'launch_evidence.json',
                          good_evidence if evidence is None
                          else {**good_evidence, **evidence})
        write_json_atomic(base / 'status.json',
                          {'phase': 'done', 'host': driver.host,
                           'student': student, 'session': driver.session,
                           'launcher_pid': 999999, 'driver_pid': 999999,
                           'owned_pgids': [999999], 'elapsed_s': 1234.5,
                           'latched': None}
                          if status is None else status)
        return base

    staged_student = 'qwen3_5_4b'
    # The manager approval that binds the code hashes; adoption requires it.
    write_json_atomic(
        driver.run_root / 'approvals' / f'BUILD_READY.{staged_student}.json',
        {'student': staged_student, 'session': driver.session,
         'launcher_sha256': 'selftest-launcher-sha',
         'wrapper_sha256': 'selftest-wrapper-sha',
         'approved_by': 'selftest'})
    base = stage_staged(staged_student)
    attempts = driver.staged_attempts(staged_student)
    checks.append(('staged metadata is read from attempt/construction/stage3/0/',
                   len(attempts) == 1 and attempts[0]['run_config'] is not None
                   and attempts[0]['loop_result'] is not None
                   and attempts[0]['final'] is not None,
                   'run_config, loop_result and final all found at the nested path'))
    checks.append(('a fully-evidenced staged run has no problems',
                   driver._staged_problems(staged_student, attempts[0]) == [],
                   str(driver._staged_problems(staged_student, attempts[0]))))

    good_run_config = dict(attempts[0]['run_config'] or {})

    def problems_with(**kwargs: Any) -> list[str]:
        """Re-stage with one thing wrong, everything else still valid.

        Overrides are merged into the good run_config rather than replacing it,
        so each case isolates the one gate it is meant to exercise instead of
        tripping several at once.
        """
        if 'run_config' in kwargs and kwargs['run_config']:
            kwargs['run_config'] = {**good_run_config, **kwargs['run_config']}
        shutil.rmtree(driver.run_root / 'construction' / staged_student)
        stage_staged(staged_student, **kwargs)
        return driver._staged_problems(staged_student,
                                       driver.staged_attempts(staged_student)[0])

    cases = [
        ('a missing run_config blocks adoption instead of passing',
         {'run_config': {}}, 'parent bundle'),
        ('a wrong parent bundle blocks adoption',
         {'run_config': {'parent_bundle_hash': 'deadbeef'}}, 'parent bundle'),
        ('a wrong round cap blocks adoption',
         {'run_config': {'builder': {'max_rounds_per_stage': 8,
                                     'max_candidates_per_round': 4,
                                     'model_id': 'gpt-5.6-luna'}}}, 'round cap'),
        ('a build against another student\'s model blocks adoption',
         {'run_config': {'target': {'model_id': 'Qwen/Qwen3-VL-8B-Instruct'}}},
         'built against target model'),
        ('a wrong D_build/D_val size blocks adoption',
         {'run_config': {'build_size': 300, 'val_size': 394}}, 'D_build'),
        ('a launch_evidence with no returncode blocks adoption',
         {'evidence': {'returncode': None}}, 'no returncode'),
        ('a non-zero launcher returncode blocks adoption',
         {'evidence': {'returncode': 1}}, 'returncode 1'),
        ('a latched abort blocks adoption even at returncode 0',
         {'evidence': {'latched': 'launcher received SIGTERM'}}, 'latched an abort'),
        ('a non-terminal status phase blocks adoption',
         {'status': {'phase': 'building', 'host': socket.gethostname(),
                     'owned_pgids': []}},
         'is not terminal'),
        ('a held status phase blocks adoption',
         {'status': {'phase': 'held', 'host': socket.gethostname(),
                     'owned_pgids': []}},
         'run held'),
        ('code hashes that do not match the approval block adoption',
         {'evidence': {'code': {'launcher_sha256': 'wrong',
                                'wrapper_sha256': 'wrong'}}}, 'approval binds'),
        ('a served revision other than the pin blocks adoption',
         {'evidence': {'weights': {'model_id': STUDENTS[staged_student].hf_id,
                                   'expected_revision': 'deadbeef',
                                   'cached_revision': 'deadbeef'}}},
         'not the pinned'),
        ('a failing blocking preflight check blocks adoption',
         {'evidence': {'preflight': [{'name': 'gpu_free', 'ok': False,
                                      'blocking': True}]}},
         'failing blocking preflight'),
        ('live owned process groups on this host block adoption',
         {'status': {'phase': 'done', 'host': socket.gethostname(),
                     'owned_pgids': [os.getpgid(0)]}},
         'still alive on this host'),
        ('an unsettled in-flight paid call blocks adoption',
         {'admission': {'returncode': 0, 'in_flight_calls': [{'call_id': 1}],
                        'spent_usd': 0.2}}, 'never settled'),
        ('a refused paid call blocks adoption',
         {'admission': {'returncode': 0, 'calls_refused': 2, 'spent_usd': 0.2,
                        'in_flight_calls': []}}, 'refused'),
        ('a missing budget file blocks adoption',
         {'admission': {}}, 'no spent_usd'),
        ('a loop_result with no stopped_reason blocks adoption',
         {'loop_result': {}}, 'stopped_reason'),
        ('a missing final bundle blocks adoption',
         {'final': False}, 'no final bundle'),
    ]
    # The two stale outcome-hash guards were removed on 2026-09-18: an
    # independent construction can legitimately produce bytes identical to the
    # 8B transfer bundle, so identity is recorded and lineage decides. These
    # cases pin both halves of that: identical bytes alone must NOT block, and
    # removing the guard must not have weakened the lineage checks.
    shutil.rmtree(driver.run_root / 'construction' / staged_student)
    identical_base = stage_staged(staged_student)
    identical = driver.staged_attempts(staged_student)[0]
    real_transfer = TRANSFER_BUNDLE_HASH
    forged = bundle_hash(identical['final'])
    globals()['TRANSFER_BUNDLE_HASH'] = forged
    try:
        checks.append(('a legitimate build with bytes identical to the 8B '
                       'transfer bundle still adopts',
                       driver._staged_problems(staged_student, identical) == [],
                       str(driver._staged_problems(staged_student, identical))))
        published = driver.publish_bundle(
            staged_student, str(identical_base.relative_to(driver.root)))
        # publish_bundle returns a summary and writes the full record to disk,
        # so the recorded flag is asserted where it actually lives.
        recorded = read_json_or(driver.bundle_record(staged_student), {})
        checks.append(('publish_bundle records identical bytes instead of '
                       'refusing them',
                       published.get('status') != 'plan_violation'
                       and published.get('bundle_hash') == forged
                       and recorded.get('matches_transfer_bundle') is True,
                       str({'returned': published,
                            'recorded_flag':
                                recorded.get('matches_transfer_bundle')})))
        shutil.rmtree(driver.run_root / 'construction' / staged_student)
        stage_staged(staged_student,
                     run_config={**good_run_config,
                                 'parent_bundle_hash': 'deadbeef'})
        wrong_lineage = driver._staged_problems(
            staged_student, driver.staged_attempts(staged_student)[0])
        checks.append(('identical bytes do NOT excuse a wrong parent bundle',
                       any('parent bundle' in problem
                           for problem in wrong_lineage),
                       str(wrong_lineage)))
    finally:
        globals()['TRANSFER_BUNDLE_HASH'] = real_transfer
    shutil.rmtree(driver.run_root / 'construction' / staged_student)
    stage_staged(staged_student)

    for name, kwargs, needle in cases:
        found = problems_with(**kwargs)
        checks.append((name, any(needle in p for p in found),
                       str(found[:1]) if found else 'NO PROBLEMS REPORTED'))

    # a non-terminal launcher record must also make the student un-runnable
    shutil.rmtree(driver.run_root / 'construction' / staged_student)
    stage_staged(staged_student, evidence={'phase': 'building', 'returncode': None,
                                           'host': 'some-other-host',
                                           'launcher_pid': 4242})
    live = driver.staged_live_owner(staged_student)
    checks.append(('a staged run on another host is treated as possibly live',
                   live is not None and 'another host' in live['reason'],
                   str(live and live['reason'])[:90]))
    checks.append(('a possibly-live staged run makes every job for that student wait',
                   driver.ready(Job('build', staged_student))[1] == 'waiting_staged_run'
                   and driver.ready(Job('eval', staged_student, ARMS[0]))[1]
                   == 'waiting_staged_run',
                   'build and L0 both report waiting_staged_run'))
    shutil.rmtree(driver.run_root / 'construction' / staged_student)

    # 10c. the five coordinator 053827 launch gates
    #
    # (a) the perception gate needs the coordinator's verified marker, not
    #     coverage alone: a 99%-landed NFS import looks complete for whichever
    #     clips happen to be there.
    driver._perception = {}
    marker_path = driver.run_root / 'preprocessing.done.json'
    fully_cached = {'config_hash': 'x', 'cache_root': 'y', 'rows_total': 10,
                    'media_unique': 10, 'media_cached': 10, 'rows_missing': 0,
                    'missing_by_task': {}, 'complete': True,
                    'coordinator_marker': False}
    driver._perception = dict(fully_cached)
    gated = driver.ready(Job('eval', 'qwen3vl_4b', ARMS[2]))
    checks.append(('full coverage without the coordinator marker still waits',
                   gated == (False, 'waiting_perception'), str(gated)))
    write_json_atomic(marker_path, {'verified_unique': 10, 'utc': utc()})
    driver._perception = {**fully_cached, 'coordinator_marker': True}
    gated = driver.ready(Job('eval', 'qwen3vl_4b', ARMS[2]))
    checks.append(('coverage plus the marker together open the gate',
                   gated[0] is True, str(gated)))
    driver._perception = {**fully_cached, 'coordinator_marker': True,
                          'complete': False, 'rows_missing': 1}
    gated = driver.ready(Job('eval', 'qwen3vl_4b', ARMS[2]))
    checks.append(('the marker alone does not open the gate either',
                   gated == (False, 'waiting_perception'), str(gated)))
    driver._perception = {}
    marker_path.unlink()

    # (b) a harness item that answers in 3 calls is legitimate, not a fault
    three = scratch / 'three_calls.jsonl'
    ids = sorted(frozen['ids'])
    with three.open('w', encoding='utf-8') as fh:
        for index, example_id in enumerate(ids):
            fh.write(json.dumps({
                'example_id': example_id, 'task': f'T{index % 8}',
                'labels': ['happy'], 'response_text': '<answer>happy</answer>',
                'target_calls': 3 if index % 2 else 4, 'parse_failures': 0,
                'input_tokens': 1, 'output_tokens': 1}) + '\n')
    (three.with_name('three_calls.outputs.jsonl')
     .write_text('\n'.join(json.dumps({'steps': ['a']}) for _ in ids) + '\n',
                 encoding='utf-8'))
    structural, diagnostics = validate_predictions(three, ARMS[2], 'three_calls',
                                                   frozen['ids'])
    checks.append(('a harness arm answering some items in 3 calls is accepted',
                   structural == [] and diagnostics['calls_mean'] < 4,
                   f'structural={structural} calls_mean={diagnostics["calls_mean"]}'))
    over = scratch / 'seven_calls.jsonl'
    with over.open('w', encoding='utf-8') as fh:
        for index, example_id in enumerate(ids):
            fh.write(json.dumps({
                'example_id': example_id, 'task': f'T{index % 8}',
                'labels': ['happy'], 'response_text': '<answer>happy</answer>',
                'target_calls': 7, 'parse_failures': 0,
                'input_tokens': 1, 'output_tokens': 1}) + '\n')
    (over.with_name('seven_calls.outputs.jsonl')
     .write_text('\n'.join(json.dumps({'steps': ['a']}) for _ in ids) + '\n',
                 encoding='utf-8'))
    structural, _ = validate_predictions(over, ARMS[2], 'seven_calls', frozen['ids'])
    checks.append(('but breaching the hard cap of 6 calls is still a fault',
                   any('cap is 6' in p for p in structural), str(structural[:1])))

    # (c) resume re-reads the predictions file, not just the marker's claims
    resume_job = Job('eval', 'qwen3vl_4b', ARMS[0])
    resume_marker = driver.marker(resume_job)
    claim = read_json_or(resume_marker, None)
    if claim is None:
        checks.append(('resume content validation exercised', False,
                       'no marker from the earlier arm run'))
    else:
        target = driver.root / claim['predictions']
        original = target.read_text(encoding='utf-8')
        # same row count, different ids: every field the marker records still
        # agrees, so only reading the file can catch this.
        swapped_rows = [json.loads(line) for line in original.splitlines() if line.strip()]
        for index, row in enumerate(swapped_rows):
            row['example_id'] = f'substituted_{index}'
        target.write_text('\n'.join(json.dumps(r) for r in swapped_rows) + '\n',
                          encoding='utf-8')
        ok, why = driver.marker_is_good(resume_job)
        checks.append(('resume rejects a file whose ids were substituted',
                       not ok and 'different id set' in why, why))
        truncated = swapped_rows[:-3]
        target.write_text('\n'.join(json.dumps(r) for r in truncated) + '\n',
                          encoding='utf-8')
        ok, why = driver.marker_is_good(resume_job)
        checks.append(('resume rejects a truncated predictions file',
                       not ok and 'rows, expected' in why, why))
        target.write_text(original, encoding='utf-8')
        ok, why = driver.marker_is_good(resume_job)
        checks.append(('resume accepts the intact file again', ok, why))

    # (d) one attempt counts once
    counted = Job('eval', 'qwen3_5_4b', ARMS[0])
    before_count = driver.attempts_used(counted)
    driver.bump_attempt(counted, 'running')
    driver.set_attempt_status(counted, 'failed')
    checks.append(('a started-then-finished attempt counts exactly once',
                   driver.attempts_used(counted) == before_count + 1,
                   f'{before_count} -> {driver.attempts_used(counted)}'))

    # (e) index ownership: another session's marker is not claimed as ours
    write_json_atomic(driver.done_dir / 'gemma4_e4b__L0__full5507.done',
                      {'key': 'gemma4_e4b__L0__full5507', 'kind': 'eval',
                       'model': 'gemma4_e4b', 'slot': 'L0', 'items': ITEM_SET,
                       'status': 'ok', 'session': 'session1-worker',
                       'host': 'jupyter-max7', 'predictions': 'somewhere.jsonl'})
    driver.publish_index()
    index_now = read_json(driver.index_path)
    checks.append(('another session\'s marker is not claimed in our index',
                   'gemma4_e4b__L0__full5507' not in index_now['arms']
                   and 'gemma4_e4b__L0__full5507' in index_now['observed_elsewhere'],
                   f'arms={sorted(index_now["arms"])} '
                   f'elsewhere={sorted(index_now["observed_elsewhere"])}'))
    (driver.done_dir / 'gemma4_e4b__L0__full5507.done').unlink()

    # 10d. session1-manager 055821: supersession must PROVE infra-only,
    #      marker validation must hash real content, index must validate.
    shutil.rmtree(driver.run_root / 'construction' / staged_student,
                  ignore_errors=True)
    stage_staged(staged_student, 'attempt_02')
    # attempt_01: infrastructure failure -- server never loaded, no build
    infra = driver.run_root / 'construction' / staged_student / 'attempt_01'
    infra.mkdir(parents=True, exist_ok=True)
    write_json_atomic(infra / 'launch_evidence.json',
                      {'returncode': 5, 'latched': 'server exited with 1 while '
                                                   'loading', 'host': driver.host,
                       'student': staged_student, 'session': driver.session})
    write_json_atomic(infra / 'status.json',
                      {'phase': 'held', 'host': driver.host, 'owned_pgids': []})
    (infra / 'server.log').write_text('server died\n', encoding='utf-8')
    records = [driver._supersession_record(a)
               for a in driver.staged_attempts(staged_student)
               if a['relative'].endswith('attempt_01')]
    checks.append(('a server-only failure is classified infrastructure-only',
                   records and records[0]['infra_only'] is True,
                   f'builder_evidence={records[0]["builder_evidence"]} '
                   f'spend=${records[0]["paid_spend_usd"]}'))
    checks.append(('adopting over an infra-only predecessor is allowed',
                   driver.adopt_staged_construction(staged_student) == 'adopted',
                   'attempt_02 adopted, attempt_01 recorded as superseded'))
    adopted_marker = read_json_or(driver.marker(Job('build', staged_student)), {})
    checks.append(('the adoption marker records which attempt was superseded',
                   adopted_marker.get('adopted_attempt', '').endswith('attempt_02')
                   and len(adopted_marker.get('superseded_attempts') or []) == 1,
                   f'adopted={adopted_marker.get("adopted_attempt")} '
                   f'superseded={[r["attempt"] for r in adopted_marker.get("superseded_attempts") or []]}'))
    # now make attempt_01 carry real Builder work: adoption must HOLD
    driver.marker(Job('build', staged_student)).unlink(missing_ok=True)
    table = driver.holds()
    table.pop(f'{staged_student}__build', None)
    write_json_atomic(driver.holds_path, table)
    write_json_atomic(infra / 'budget_admission.json',
                      {'returncode': 0, 'spent_usd': 0.87, 'calls_allowed': 40,
                       'calls_refused': 0, 'in_flight_calls': []})
    (infra / 'build.log').write_text('round 0 ...\n', encoding='utf-8')
    records = [driver._supersession_record(a)
               for a in driver.staged_attempts(staged_student)
               if a['relative'].endswith('attempt_01')]
    checks.append(('an earlier attempt that spent money is NOT infra-only',
                   records and records[0]['infra_only'] is False
                   and records[0]['paid_spend_usd'] == 0.87,
                   f'builder_evidence={records[0]["builder_evidence"]}'))
    outcome = driver.adopt_staged_construction(staged_student)
    checks.append(('adoption HOLDS when a predecessor did Builder work',
                   outcome == 'held'
                   and 'infrastructure-only' in
                   (driver.holds().get(f'{staged_student}__build') or {}).get('reason', ''),
                   f'outcome={outcome}'))
    shutil.rmtree(driver.run_root / 'construction' / staged_student,
                  ignore_errors=True)
    driver.marker(Job('build', staged_student)).unlink(missing_ok=True)
    table = driver.holds()
    table.pop(f'{staged_student}__build', None)
    write_json_atomic(driver.holds_path, table)

    # the build branch must hash the bundle on disk, not compare two records
    build_marker = driver.marker(build)
    if build_marker.is_file():
        claim = read_json(build_marker)
        published_path = driver.root / read_json(driver.bundle_record('qwen3vl_4b'))['path']
        # Tamper so the bundle STILL LOADS but hashes differently -- appending
        # to a prompt is the realistic case. Replacing a config with junk would
        # trip the "does not load" guard instead and prove the wrong thing.
        prompt = published_path / 'prompts/synthesize.md'
        prompt.write_text(prompt.read_text(encoding='utf-8') + '\n<!-- t -->\n',
                          encoding='utf-8')
        ok, why = driver.marker_is_good(build)
        checks.append(('a tampered published bundle is caught by rehashing it',
                       not ok and 'hashes to' in why, why[:100]))
        del claim

    # the index validates markers instead of counting files
    bad_key = 'qwen3vl_4b__L1__full5507'
    write_json_atomic(driver.done_dir / f'{bad_key}.done',
                      {'key': bad_key, 'kind': 'eval', 'model': 'qwen3vl_4b',
                       'slot': 'L1', 'items': ITEM_SET, 'status': 'ok',
                       'session': driver.session, 'host': driver.host,
                       'predictions': 'gone/missing.jsonl',
                       'item_set_sha256': frozen['item_set_sha256'],
                       'n_items_written': len(frozen['ids'])})
    driver.publish_index()
    index_checked = read_json(driver.index_path)
    checks.append(('the index refuses to count a marker whose artifacts are gone',
                   bad_key not in index_checked['arms']
                   and bad_key in index_checked['invalid_markers'],
                   f'invalid={sorted(index_checked["invalid_markers"])}'))
    (driver.done_dir / f'{bad_key}.done').unlink()

    # the isolated serve helper and the Gemma transformers pin are consumed
    checks.append(('the driver uses the isolated serve helper with '
                   '--served-model-name',
                   SERVE_STEP.name == 'serve_student_full_20260918.sh'
                   and '--served-model-name' in SERVE_STEP.read_text(encoding='utf-8'),
                   SERVE_STEP.name))
    checks.append(('Gemma is pinned to the transformers-5.13 venv',
                   STUDENTS['gemma4_e4b'].serving == 'vllm019_tf513'
                   and (Path.home() / SERVING_VENVS['vllm019_tf513']
                        / 'bin/vllm').is_file(),
                   SERVING_VENVS[STUDENTS['gemma4_e4b'].serving]))

    # 10e. session1-manager 060654: pressure-based watchdog, verified marker
    #
    # Reclaimable file cache alone must never abort the batch -- this cgroup
    # sits at 98% of 32 GiB from a clean 29 GiB page cache with 329 successful
    # reclaims and zero OOM kills, and a raw-current rule would have aborted
    # every arm.
    module = sys.modules[__name__]
    real = (module.anon_gib, module.cgroup_pressure, module.cgroup_value)

    def fake_memory(anon: float, current: float, full60: float) -> None:
        module.anon_gib = lambda: anon
        module.cgroup_pressure = lambda: {'full_avg60': full60,
                                          'some_avg60': full60}
        module.cgroup_value = lambda name: (
            int(32.0 * 2**30) if name == 'memory.max'
            else int(current * 2**30) if name == 'memory.current'
            else real[2](name))

    # anon GiB, memory.current GiB, full avg60, expected substring (None = no trip)
    pressure_cases = [
        ('clean cache at 98% of the cgroup alone does not trip',
         0.8, 31.6, 0.04, None),
        ('large anon trips on its own', 27.0, 27.5, 0.0, 'real allocation'),
        ('near-full AND stalling trips', 0.8, 31.6, 7.0, 'near-full and stalling'),
        ('near-full with only mild stalling does not trip', 0.8, 31.6, 3.0, None),
        ('severe stalling trips on its own, wherever the memory went',
         0.8, 12.0, 25.0, 'thrashing'),
    ]
    try:
        for name, anon, current, full60, expected in pressure_cases:
            fake_memory(anon, current, full60)
            got = module.memory_trip_reason()
            if expected is None:
                checks.append((name, got is None, str(got)))
            else:
                checks.append((name, got is not None and expected in got,
                               str(got)[:80]))
    finally:
        module.anon_gib, module.cgroup_pressure, module.cgroup_value = real

    # the coordinator marker is verified, not merely present
    driver._perception = {}
    marker = driver.run_root / 'preprocessing.done.json'
    config_hash = driver.perception_coverage(refresh=True)['config_hash']
    marker_cases = [
        ('an absent completion marker is not verified', None, False, 'absent'),
        ('an unparseable marker is not verified', 'not json at all', False,
         'does not parse'),
        ('a marker for another perception config is rejected',
         {'config_hash': 'deadbeef', 'verified_unique': 99999,
          'verified_items': 99999}, False, 'config_hash'),
        # The synthetic fixture's rows carry no media_hash, so the unique-clip
        # superset test is vacuous there; verified_items is the one that bites
        # on this fixture. Both rules are exercised on the real manifest.
        ('a marker verifying fewer items than the eligible set is rejected',
         {'config_hash': config_hash, 'verified_unique': 99999,
          'verified_items': 1}, False, 'fewer than'),
        ('a marker that is a proper superset is verified',
         {'config_hash': config_hash, 'verified_unique': 99999,
          'verified_items': 99999}, True, 'verified'),
    ]
    for name, payload, expected_ok, needle in marker_cases:
        if payload is None:
            marker.unlink(missing_ok=True)
        elif isinstance(payload, str):
            marker.write_text(payload, encoding='utf-8')
        else:
            write_json_atomic(marker, payload)
        driver._perception = {}
        coverage = driver.perception_coverage(refresh=True)
        checks.append((name,
                       coverage['coordinator_marker'] is expected_ok
                       and needle in coverage['coordinator_marker_detail'],
                       coverage['coordinator_marker_detail'][:70]))
    marker.unlink(missing_ok=True)
    driver._perception = {}

    # 11. the abort latch stops the queue starting anything
    latched = Driver(fake_args, root=scratch)
    latched.latch_abort('selftest')
    checks.append(('a latched abort makes run_one refuse to start a job',
                   not latched.run_one(Job('eval', 'qwen3vl_4b', ARMS[0])),
                   'run_one returned False without starting a subprocess'))

    # 12. the ledger reader behind the spend backstop
    ledger = scratch / driver.attempt_dir(build, 1) / 'ledger.jsonl'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a', encoding='utf-8') as fh:
        for role, cost in (('builder', 1.25), ('judge_proxy', 0.75),
                           ('target', 99.0), ('verifier', 0.5)):
            fh.write(json.dumps({'role': role, 'cost_usd': cost,
                                 'unknown_future_field': 1}) + '\n')
    spend = driver.spend()
    checks.append(('the guard sums paid roles, ignores the local student and '
                   'tolerates an unknown field',
                   abs(spend.per_student['qwen3vl_4b'] - 2.5) < 1e-6,
                   f'{spend.per_student["qwen3vl_4b"]:.4f}'))
    checks.append(('the remaining build allowance shrinks',
                   abs(spend.remaining_for('qwen3vl_4b') - 0.5) < 1e-6,
                   f'{spend.remaining_for("qwen3vl_4b"):.4f}'))

    # 13. status, report, per-host index
    snap = driver.publish_status()
    checks.append(('status published at status/<session>.json',
                   driver.status_path.is_file()
                   and driver.status_path.name == 'selftest.json', ''))
    checks.append(('status carries the keys the manager monitor samples',
                   all(k in snap for k in ('phase', 'status', 'spend', 'cost_usd',
                                           'completed', 'total', 'failure_count')),
                   ''))
    report = driver.publish_report(snap, tag='selftest')
    checks.append(('report text published and substantive',
                   report.is_file()
                   and len(report.read_text(encoding='utf-8')) > 800,
                   f'{len(report.read_text(encoding="utf-8"))} chars'))
    driver.publish_index()
    checks.append(('the index is per session, not shared',
                   driver.index_path.name == 'index.selftest.json', ''))
    index = read_json(driver.index_path)
    checks.append(('the index separates arms from smoke rows and expects 16',
                   'smoke' in index and index['arms_expected_batch'] == 16,
                   f'arms={index["arms_done_here"]} smoke={len(index["smoke"])}'))

    # 14. a terminal declaration is only for a terminal queue
    driver.declare_terminal('waiting_perception', {})
    checks.append(('waiting on perception is never declared terminal',
                   not (driver.supervision_dir / 'queue_terminal.json').is_file(), ''))
    driver.declare_terminal('completed', {'done': 1})
    checks.append(('a completed queue is declared',
                   (driver.supervision_dir / 'queue_terminal.json').is_file(), ''))

    # 15. and the real batch root is untouched BY THIS SELFTEST.
    #
    # Attributed by ownership, not by timing: the other host's worker writes
    # into the same shared root, so a before/after diff alone reports its files
    # as our leak. Anything this selftest could produce carries the selftest
    # session name or lands under a path this driver owns for it.
    real_after = {p for p in (ROOT / RUN_REL).rglob('*') if p.is_file()}
    appeared = sorted(str(p.relative_to(ROOT)) for p in real_after - real_before)
    mine = [p for p in appeared
            if 'selftest' in p or f'{RUN_REL}/qwen3vl_4b/' in p]
    others = [p for p in appeared if p not in mine]
    checks.append(('the selftest wrote nothing of its own into the real batch root',
                   not mine,
                   (f'{len(others)} file(s) appeared from another session while this '
                    f'ran, e.g. {others[:1]}' if others else 'nothing appeared')))
    driver.release_claims()

    failures = [name for name, ok, _ in checks if not ok]
    width = max(len(name) for name, _, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}s}  {detail}')
    print(f'\n{len(checks) - len(failures)}/{len(checks)} checks passed')
    print(f'artifacts under {driver.run_root}')
    if failures:
        print('FAILED: ' + '; '.join(failures))
        return 1
    if not os.environ.get('EMOHARNESS_SELFTEST_KEEP'):
        shutil.rmtree(scratch, ignore_errors=True)
        print('scratch root removed (EMOHARNESS_SELFTEST_KEEP=1 to keep it)')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import emoharness  # noqa: F401
    except ImportError:
        print('run this with the project interpreter: '
              f'{PY_BIN} scripts/run_student_specific_full.py ...', file=sys.stderr)
        return 2
    if args.selftest:
        return selftest(args)
    driver = Driver(args)
    if args.release:
        return driver.release(args.release)
    if args.plan:
        return print_plan(driver)
    if args.preflight:
        result = preflight(driver)
        text = render_preflight(result)
        print(text, end='')
        stamp = utc_stamp()
        for name in (f'preflight_{driver.session}_{stamp}',
                     f'preflight_{driver.session}_latest'):
            write_json_atomic(driver.reports_dir / f'{name}.json', result)
            write_text_atomic(driver.reports_dir / f'{name}.txt', text)
        return 0 if result['verdict'] == 'ready' else 1
    if args.report:
        snap = driver.publish_status()
        print(render_report(snap), end='')
        driver.publish_report(snap)
        return 0
    return driver.run()


if __name__ == '__main__':
    raise SystemExit(main())
