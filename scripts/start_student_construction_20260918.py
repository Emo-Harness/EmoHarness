#!/usr/bin/env python3
"""Start ONE student's Stage-3 construction for the 2026-09-18 batch.

    scripts/start_student_construction_20260918.py --student gemma4_e4b --session session1-worker
    scripts/start_student_construction_20260918.py --student gemma4_e4b --session session1-worker --go
    scripts/start_student_construction_20260918.py --student gemma4_e4b --session session1-worker --mock

One build, one attempt, no retry. This is a launcher, not a framework: it runs
preflight, brings up one student server it owns, runs the existing
``examples/build_harness.py`` behind session2-worker's ``budget_admission.py``,
watches for trouble, and cleans up its own process groups. Every scientific
knob -- rounds, candidates, reasoning effort, gates, decoding, the scaffold it
starts from -- is *asserted* against the frozen configs rather than passed in,
so this script cannot quietly change what is being measured.

Default behaviour is preflight only: it prints what it would do and exits. It
refuses to launch without ``--go`` *and* a manager BUILD_READY approval file,
because the plan puts the launch decision after manager review.

Three things it will not do:

* Retry. A refused, crashed, or watchdog-stopped build leaves its evidence and
  stops. Re-running a failed build is a fresh-resampling decision that belongs
  to a human, not to a launcher, and the plan forbids it outright.
* Race a successor. When the watchdog trips it latches a hold file, and the
  presence of that file blocks any later launch for the same student until
  someone clears it deliberately.
* Signal anything it did not start. Cleanup targets only the process group ids
  this process created. On a shared NFS checkout a pid from another container
  means nothing here, which this project has already been bitten by.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
BATCH: Final = ROOT / 'results/v3/student_specific_full_20260918'

#: Per-build spend reservation. Three jobs at $3 inside the $10 batch
#: authorization, leaving $1 for tracked preflight.
BUILD_CAP_USD: Final = 3.00
#: Conservative scheduling fixed by the coordinator for both A100 hosts. These
#: are scheduling only; none of them changes a prompt or a sampling parameter.
WORKERS: Final = 8
VIDEO_WORKERS: Final = 2
MAX_NUM_SEQS: Final = 8
MM_CACHE_GB: Final = 0
MAX_JOBS: Final = 2
MAX_MODEL_LEN: Final = 16384
PORT: Final = 8000

#: The frozen Stage-1/v3 scaffold every construction starts from.
SCAFFOLD_HASH: Final = (
    'b38c518b7e4161a3ba4f417f31ff4b359b3ea32474d4675cc2f8ec6f0df8ab17'
)
BUILDER_CONFIG: Final = 'v3_full.yaml'
EXTERNAL_MANIFEST: Final = 'external_v3.jsonl'
PREPROCESS_CONFIG: Final = 'preprocess.yaml'
STAGE: Final = 3
REPLICATE: Final = 0
ROUNDS: Final = 4
CANDIDATES: Final = 4

#: Students this launcher may build, and how each is served, taken from the
#: environments the finished transfer batch actually served them from -- the
#: serve logs of 2026-09-14 are the authority, not the venv names.
#:
#: Gemma 4 needs the tf513 variant specifically. On transformers 5.17 vLLM's
#: config convertor reads a global ``head_dim`` and 5.17 answers that with
#: AmbiguousGlobalPerLayerAttributeError, so the server dies during load; 5.13.1
#: returns the value and serves. That is also the reproducibility argument: the
#: transfer numbers for Gemma came from 5.13.1, so another version would not be
#: comparable even if it loaded. The 0.26 venv is cu130 against a 570 driver and
#: is never used: torch.cuda.is_available() lies there.
STUDENTS: Final[dict[str, dict[str, str]]] = {
    'gemma4_e4b': {
        'target': 'configs/target/gemma4_e4b_it.yaml',
        'model': 'google/gemma-4-E4B-it',
        'venv': 'emoharness-vllm019-tf513',
        'transformers': '5.13.1',
        'revision': 'ee0ef6023621cff504d758262d4e04895a5af4a2',
    },
    'qwen3vl_4b': {
        'target': 'configs/target/qwen3vl_4b_instruct.yaml',
        'model': 'Qwen/Qwen3-VL-4B-Instruct',
        'venv': 'emoharness-vllm011',
        'transformers': '4.57.6',
        'revision': 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    },
    'qwen3_5_4b': {
        'target': 'configs/target/qwen3_5_4b.yaml',
        'model': 'Qwen/Qwen3.5-4B',
        'venv': 'emoharness-vllm019',
        'transformers': '5.17.0',
        'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    },
}
#: 8B is deliberately absent: the plan reuses its existing bundle and forbids
#: building it again.
SESSIONS: Final = ('session1-worker', 'session2-worker')
#: The GPU each session is expected to own. An unknown card is refused rather
#: than used, because a swapped host is a real event here, not a hypothetical.
EXPECTED_GPU_NAME: Final = 'NVIDIA A100-SXM4-80GB'
EXPECTED_GPU_UUID: Final[dict[str, str]] = {
    'session1-worker': 'GPU-7407b89f-2011-b0ff-ca87-8f45bdd431b5',
}
#: The 8B bundle this batch reuses rather than rebuilds; recorded, not served.
REUSED_8B_REVISION: Final = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
HF_HUB: Final = Path.home() / '.cache/huggingface/hub'
MIN_FREE_GPU_MIB: Final = 60_000
MIN_FREE_DISK_GIB: Final = 20
MAX_CGROUP_RAM_FRACTION: Final = 0.60
SERVER_READY_TIMEOUT_S: Final = 1800
WATCHDOG_INTERVAL_S: Final = 30


def utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.tmp')
    temp.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    temp.replace(path)


@dataclass
class Check:
    """One preflight question and its answer."""

    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass
class Run:
    """Everything this launch owns, so cleanup and evidence stay honest."""

    student: str
    session: str
    attempt_dir: Path
    attempt: int = 1
    #: A mock run must not write the HOLD or the shared status that the real
    #: batch and the supervisor read. On 2026-09-18 test code latching into the
    #: real path overwrote a genuine failure HOLD, so isolation is structural
    #: here rather than a rule someone has to remember.
    sandbox: bool = False
    #: (label, child, pgid) in start order. Kept together because cleanup has
    #: to both signal the group and reap the child: an unreaped zombie keeps
    #: its process group alive, so a cleanup that only signals would wait out
    #: every timeout on a shutdown where nothing is actually still running.
    owned: list[tuple[str, subprocess.Popen[bytes], int]] = field(
        default_factory=list
    )
    latched: str | None = None
    started: float = field(default_factory=time.monotonic)

    def adopt(self, label: str, child: subprocess.Popen[bytes]) -> None:
        self.owned.append((label, child, os.getpgid(child.pid)))

    @property
    def pgids(self) -> list[int]:
        return [pgid for _, _, pgid in self.owned]

    def child(self, label: str) -> subprocess.Popen[bytes] | None:
        for name, process, _ in self.owned:
            if name == label:
                return process
        return None

    @property
    def server(self) -> subprocess.Popen[bytes] | None:
        return self.child('server')

    @property
    def build(self) -> subprocess.Popen[bytes] | None:
        return self.child('build')

    @property
    def status_path(self) -> Path:
        return self.attempt_dir / 'status.json'

    @property
    def hold_path(self) -> Path:
        return self.attempt_dir.parent / 'HOLD'

    @property
    def predecessor_dir(self) -> Path | None:
        """The attempt immediately before this one, if there is one."""
        if self.attempt <= 1:
            return None
        return self.attempt_dir.parent / f'attempt_{self.attempt - 1:02d}'

    @property
    def shared_status_path(self) -> Path:
        """Where the supervisor looks: <batch>/status/<session>.json.

        The per-attempt status is for whoever reads the attempt directory; the
        monitor samples the batch-root status directory, so the same record is
        published to both. A launch the supervisor cannot see is a launch
        nobody is watching.
        """
        if self.sandbox:
            return self.attempt_dir / 'shared_status.json'
        return BATCH / 'status' / f'{self.session}.json'

    def publish(self, phase: str, **extra: object) -> None:
        record = {
            'utc': utc(),
            'host': socket.gethostname(),
            'session': self.session,
            'student': self.student,
            'phase': phase,
            'attempt_dir': str(self.attempt_dir),
            'launcher_pid': os.getpid(),
            # The supervisor looks for driver_pid; same number, its name.
            'driver_pid': os.getpid(),
            'launcher_pgid': os.getpgid(0),
            'model': STUDENTS[self.student]['model'],
            'owned_pgids': list(self.pgids),
            'elapsed_s': round(time.monotonic() - self.started, 1),
            'latched': self.latched,
            **extra,
        }
        write_json_atomic(self.status_path, record)
        write_json_atomic(self.shared_status_path, record)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def _nvidia(query: str) -> list[str]:
    out = subprocess.run(
        ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def check_gpu(session: str) -> list[Check]:
    """Exactly one expected, idle-enough GPU. Refuse anything unfamiliar."""
    rows = _nvidia('name,uuid,memory.free')
    if len(rows) != 1:
        return [
            Check('gpu', False, f'expected exactly one GPU, nvidia-smi gave {rows}')
        ]
    name, uuid, free = (part.strip() for part in rows[0].split(','))
    checks = [
        Check(
            'gpu_name',
            name == EXPECTED_GPU_NAME,
            f'{name} (expected {EXPECTED_GPU_NAME})',
        )
    ]
    expected_uuid = EXPECTED_GPU_UUID.get(session)
    if expected_uuid is None:
        checks.append(
            Check(
                'gpu_uuid',
                True,
                f'{uuid} accepted: no UUID pinned for {session} yet. Pin it in '
                'EXPECTED_GPU_UUID once that host declares one',
                blocking=False,
            )
        )
    else:
        checks.append(
            Check('gpu_uuid', uuid == expected_uuid, f'{uuid} (expected {expected_uuid})')
        )
    checks.append(
        Check(
            'gpu_free',
            int(free) >= MIN_FREE_GPU_MIB,
            f'{free} MiB free, need {MIN_FREE_GPU_MIB}',
        )
    )
    return checks


def check_gpu_compute_apps() -> Check:
    """No process may already be computing on the GPU.

    Preflight runs before this launcher owns anything, so *every* resident
    compute PID is by definition someone else's. A leftover context from a
    crashed container also shows up here, and sharing the card with one would
    both corrupt throughput measurement and risk an out-of-memory mid-build.
    """
    out = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=20, check=False,
    )
    rows = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if out.returncode != 0:
        return Check('gpu_compute_apps', False,
                     f'could not query compute apps: {out.stderr.strip()[:200]}')
    return Check(
        'gpu_compute_apps',
        not rows,
        f'{len(rows)} unknown compute process(es) on the GPU: {rows}'
        if rows
        else 'no compute process resident on the GPU',
    )


def check_listener() -> Check:
    """Nothing may already hold the serving port.

    A live listener is either another student server or something unrelated;
    either way this launch does not own it and must not assume it can use or
    replace it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        busy = probe.connect_ex(('127.0.0.1', PORT)) == 0
    return Check(
        'port_free',
        not busy,
        f'port {PORT} is already serving; this launch owns no server there'
        if busy
        else f'port {PORT} free',
    )


def check_no_rival_server() -> Check:
    out = subprocess.run(
        ['pgrep', '-af', 'vllm'], capture_output=True, text=True, check=False
    )
    lines = [line for line in out.stdout.splitlines() if 'vllm serve' in line]
    return Check(
        'no_rival_server',
        not lines,
        '; '.join(lines) if lines else 'no vllm serve process on this host',
    )


def check_hashes() -> list[Check]:
    """The scaffold and the construction manifest must be the frozen ones."""
    from emoharness.common.hashing import directory_hash, sha256_file

    scaffold = directory_hash(ROOT / 'harness')
    isolation = json.loads(
        (ROOT / 'data/splits/external_v3_isolation.json').read_text()
    )
    manifest = sha256_file(ROOT / 'data/manifests' / EXTERNAL_MANIFEST)
    return [
        Check(
            'scaffold_hash',
            scaffold == SCAFFOLD_HASH,
            f'{scaffold[:16]}... (frozen {SCAFFOLD_HASH[:16]}...)',
        ),
        Check(
            'external_manifest_hash',
            manifest == isolation['manifest_hash'],
            f"{manifest[:16]}... (isolation record {isolation['manifest_hash'][:16]}...)",
        ),
        Check(
            'isolation_counts',
            isolation['counts'] == {'build': 399, 'val': 394, 'ext_test': 129},
            f"{isolation['counts']}",
        ),
    ]


def check_manifest_inputs() -> list[Check]:
    """Every file the batch recorded a digest for must still hash to it.

    ``execution_manifest.json`` pins 121 inputs -- configs, manifests, splits,
    and the evaluation source. Checking a sample would leave the interesting
    case uncovered, so all of them are hashed: a build that starts from
    different inputs than the batch declared is not the experiment that was
    reviewed, whichever file happened to drift.
    """
    from emoharness.common.hashing import sha256_file

    manifest_path = BATCH / 'execution_manifest.json'
    if not manifest_path.is_file():
        return [Check('manifest_inputs', False, f'{manifest_path} is missing')]
    declared = json.loads(manifest_path.read_text()).get('input_sha256')
    if not isinstance(declared, dict) or not declared:
        return [
            Check('manifest_inputs', False,
                  'execution_manifest.json declares no input_sha256 map')
        ]
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in sorted(declared.items()):
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            mismatched.append(relative)
    ok = not missing and not mismatched
    detail = f'{len(declared)} declared inputs all match'
    if not ok:
        detail = (
            f'{len(declared)} declared: {len(missing)} missing '
            f'{missing[:3]}, {len(mismatched)} changed {mismatched[:3]}'
        )
    return [Check('manifest_inputs', ok, detail)]


def check_approval(
    run: Run, launcher_sha: str, wrapper_sha: str, helper_sha: str
) -> Check:
    """The approval must name this exact build: attempt, predecessor, and code.

    An approval that is just a file anyone could touch is a formality. Binding
    it to the student, session, hostname and all three code digests stops a
    review being reused for another student, replayed on the other host, or
    satisfied by code that changed after it was read.

    It also binds the attempt number and the predecessor's evidence digest,
    which is the part that actually matters for recovery. Without it an
    approval written for attempt_02 would silently authorise attempt_03 after
    attempt_02 burned a Builder run -- a generic retry, which the plan forbids.
    An approval names one attempt and one predecessor state, so a failed paid
    attempt needs a new human decision rather than a re-run of this command.
    """
    path = approval_path(run.student)
    if not path.is_file():
        return Check('approval', False, f'{path} absent')
    try:
        record = json.loads(path.read_text())
    except ValueError as exc:
        return Check('approval', False, f'{path.name} is not readable JSON: {exc}')
    expected = {
        'student': run.student,
        'session': run.session,
        'host': socket.gethostname(),
        'launcher_sha256': launcher_sha,
        'wrapper_sha256': wrapper_sha,
        'serve_helper_sha256': helper_sha,
        'attempt': run.attempt,
        'attempt_dir': _repo_relative(run.attempt_dir),
        'predecessor_attempt': (
            None if run.predecessor_dir is None else run.predecessor_dir.name
        ),
        'predecessor_evidence_sha256': predecessor_evidence_sha(run),
    }
    wrong = {
        key: f'approval says {record.get(key)!r}, expected {value!r}'
        for key, value in expected.items()
        if record.get(key) != value
    }
    return Check(
        'approval',
        not wrong,
        f'{path.name} binds this build' if not wrong else f'{path.name}: {wrong}',
    )


ZERO_BUILD_FORBIDDEN: Final[tuple[str, ...]] = (
    'build.log',
    'budget_admission.json',
    'ledger.jsonl',
    'construction',
)


def predecessor_evidence_sha(run: Run) -> str | None:
    """Digest of the previous attempt's launch evidence, if it has any."""
    previous = run.predecessor_dir
    if previous is None:
        return None
    evidence = previous / 'launch_evidence.json'
    if not evidence.is_file():
        return None
    from emoharness.common.hashing import sha256_file

    return sha256_file(evidence)


def check_predecessor(run: Run) -> list[Check]:
    """A later attempt is only legitimate if the earlier one never built anything.

    This is the rule that stops a generic ``--attempt`` from becoming a retry
    loop. attempt_02 may exist because attempt_01 died before spending money;
    it may not exist because attempt_01 ran a Builder and produced a result
    somebody dislikes. So the predecessor must show no build log, no budget
    state, no ledger and no construction output, and its process groups must
    be gone -- otherwise this launch would be a fresh resample, or would race
    something still alive.
    """
    previous = run.predecessor_dir
    if previous is None:
        return [Check('predecessor', True, 'attempt 1 has no predecessor')]
    if not previous.is_dir():
        return [
            Check('predecessor', False, f'{previous} does not exist; attempts must '
                  'be consecutive so the zero-build chain can be checked')
        ]
    checks: list[Check] = []
    produced = [name for name in ZERO_BUILD_FORBIDDEN if (previous / name).exists()]
    checks.append(
        Check(
            'predecessor_zero_build',
            not produced,
            f'{previous.name} produced {produced}, so it spent or built '
            'something; a further attempt would be a resample'
            if produced
            else f'{previous.name} has no build.log, budget state, ledger or '
            'construction output',
        )
    )
    evidence_path = previous / 'launch_evidence.json'
    if not evidence_path.is_file():
        checks.append(
            Check('predecessor_evidence', False, f'{evidence_path} is missing')
        )
        return checks
    evidence = json.loads(evidence_path.read_text())
    checks.append(
        Check(
            'predecessor_host',
            evidence.get('host') == socket.gethostname(),
            f'ran on {evidence.get("host")}, this host is {socket.gethostname()}',
        )
    )
    alive = [pgid for pgid in evidence.get('owned_pgids') or [] if _group_alive(pgid)]
    checks.append(
        Check(
            'predecessor_groups_dead',
            not alive,
            f'process groups still alive: {alive}'
            if alive
            else f'all owned groups from {previous.name} are gone '
            f'({evidence.get("owned_pgids")})',
        )
    )
    return checks


def self_hashes() -> tuple[str, str, str]:
    """Launcher, budget wrapper and serving helper digests.

    All three are named by the approval: the launcher decides what runs, the
    wrapper decides what it may spend, and the helper decides how the student
    is served. A review that pinned only the first would not actually pin the
    behaviour it reviewed.
    """
    from emoharness.common.hashing import sha256_file

    return (
        sha256_file(Path(__file__).resolve()),
        sha256_file(ROOT / 'scripts/budget_admission.py'),
        sha256_file(ROOT / SERVE_HELPER),
    )


def check_perception_cache() -> list[Check]:
    """Every build and val clip must already be preprocessed.

    The full 5,507-item evaluation cache is still being built remotely and is
    deliberately NOT a prerequisite here: construction only ever reads build
    and val.
    """
    import yaml

    from emoharness.perceive.pipeline import PreprocessConfig

    config = PreprocessConfig.model_validate(
        yaml.safe_load((ROOT / 'configs' / PREPROCESS_CONFIG).read_text())
    )
    config_hash = config.config_hash()
    derived = ROOT / 'data/derived'
    rows = [
        json.loads(line)
        for line in (ROOT / 'data/manifests' / EXTERNAL_MANIFEST)
        .read_text()
        .splitlines()
        if line.strip()
    ]
    checks = []
    for split, expected in (('build', 399), ('val', 394)):
        wanted = [row for row in rows if row['split'] == split]
        have = sum(
            1
            for row in wanted
            if (derived / row['media_hash'] / f'{config_hash}.json').is_file()
        )
        checks.append(
            Check(
                f'{split}_cache',
                have == len(wanted) == expected,
                f'{have}/{len(wanted)} cached at config {config_hash[:16]}...',
            )
        )
    return checks


def check_builder_config() -> list[Check]:
    """The scientific knobs are asserted, never set from here."""
    import yaml

    config = yaml.safe_load(
        (ROOT / 'configs/builder' / BUILDER_CONFIG).read_text()
    )
    verifier = config.get('verifier') or {}
    return [
        Check('builder_model', config.get('model_id') == 'gpt-5.6-luna',
              str(config.get('model_id'))),
        Check('builder_effort', config.get('reasoning_effort') == 'high',
              str(config.get('reasoning_effort'))),
        Check('builder_rounds', config.get('max_rounds_per_stage') == ROUNDS,
              str(config.get('max_rounds_per_stage'))),
        Check('builder_candidates',
              config.get('max_candidates_per_round') == CANDIDATES,
              str(config.get('max_candidates_per_round'))),
        Check('reviewer_effort',
              verifier.get('enabled') is True
              and verifier.get('reasoning_effort') == 'high',
              f"enabled={verifier.get('enabled')} "
              f"effort={verifier.get('reasoning_effort')}"),
        Check('isolated_splits_required',
              config.get('require_isolated_splits') is True,
              str(config.get('require_isolated_splits'))),
    ]


def cached_revision(model_id: str) -> str | None:
    """The commit the local HF cache has checked out as refs/main."""
    ref = HF_HUB / f'models--{model_id.replace("/", "--")}' / 'refs/main'
    try:
        return ref.read_text().strip()
    except OSError:
        return None


def check_revision(student: str) -> Check:
    """The served weights must be the same commit the transfer batch used.

    Serving by model id alone lets a fetch pull a newer revision, which would
    compare this student's construction against different weights than every
    prior result in the batch. The launcher also exports HF_HUB_OFFLINE and
    TRANSFORMERS_OFFLINE so that cannot happen silently; this check makes the
    intended commit an assertion rather than a hope.
    """
    expected = STUDENTS[student]['revision']
    found = cached_revision(STUDENTS[student]['model'])
    if found is None:
        return Check(
            'weights_revision',
            False,
            f'no local HF cache ref for {STUDENTS[student]["model"]}; offline '
            'serving would fail and a fetch is not permitted',
        )
    return Check(
        'weights_revision',
        found == expected,
        f'{found[:12]}... (transfer batch used {expected[:12]}...)',
    )


def check_target_config(student: str) -> Check:
    """The student's decoding contract must be the shared one."""
    import yaml

    path = ROOT / STUDENTS[student]['target']
    config = yaml.safe_load(path.read_text())
    decoding = config.get('decoding') or {}
    ok = (
        config.get('model_id') == STUDENTS[student]['model']
        and decoding.get('temperature') == 0.0
        and decoding.get('max_tokens') == 1024
    )
    return Check('target_config', ok, f'{path.name}: {config.get("model_id")} {decoding}')


def check_resources() -> list[Check]:
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    limit = int((Path('/sys/fs/cgroup/memory.max').read_text() or '0').strip())
    current = int((Path('/sys/fs/cgroup/memory.current').read_text() or '0').strip())
    fraction = current / limit if limit else 1.0
    return [
        Check('disk', free_gib >= MIN_FREE_DISK_GIB, f'{free_gib:.0f} GiB free'),
        Check(
            'ram_headroom',
            fraction <= MAX_CGROUP_RAM_FRACTION,
            f'{current / 2**30:.1f} of {limit / 2**30:.0f} GiB used '
            f'({fraction:.0%})',
        ),
    ]


def check_credentials() -> Check:
    """Availability by key name only. The value is never read or logged."""
    env_file = Path.home() / '.emoharness_env'
    present = False
    if env_file.is_file():
        present = any(
            line.split('=', 1)[0].strip() == 'OPENAI_API_KEY'
            for line in env_file.read_text().splitlines()
        )
    present = present or bool(os.environ.get('OPENAI_API_KEY'))
    return Check('credentials', present, 'OPENAI_API_KEY available by name')


def check_attempt_unique(run: Run) -> list[Check]:
    """One unique attempt per student, and no launch under a latched hold."""
    return [
        Check(
            'attempt_unique',
            not run.attempt_dir.exists(),
            f'{run.attempt_dir} already exists; this launcher never retries a '
            'build, and re-running one is a fresh-resampling decision for a human'
            if run.attempt_dir.exists()
            else f'{run.attempt_dir.relative_to(ROOT)} is new',
        ),
        Check(
            'no_latched_hold',
            not run.hold_path.exists(),
            f'{run.hold_path} is latched: {run.hold_path.read_text()[:200]}'
            if run.hold_path.exists()
            else 'no hold latched',
        ),
    ]


def venv_versions(venv: Path) -> tuple[str | None, str | None]:
    """The vllm and transformers versions a serving environment would use."""
    out = subprocess.run(
        [str(venv / 'bin/python'), '-c',
         'import json,transformers,vllm;'
         'print(json.dumps([vllm.__version__, transformers.__version__]))'],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if out.returncode != 0:
        return None, None
    try:
        vllm_version, transformers_version = json.loads(out.stdout.strip())
    except (ValueError, TypeError):
        return None, None
    return vllm_version, transformers_version


def check_serving_stack(student: str) -> list[Check]:
    """The serving environment must be the one this student was served from.

    Checking that a venv merely exists is not enough: the first real launch of
    this batch died 90 seconds in because the environment was present, had a
    working vllm, and still could not load the model. The transformers version
    is the part that actually differs, so it is asserted.
    """
    venv = Path.home() / '.venvs' / STUDENTS[student]['venv']
    if not (venv / 'bin/vllm').is_file():
        return [Check('serving_venv', False, f'{venv} has no bin/vllm')]
    _, transformers_version = venv_versions(venv)
    expected = STUDENTS[student]['transformers']
    return [
        Check('serving_venv', True, str(venv)),
        Check(
            'serving_transformers',
            transformers_version == expected,
            f'{transformers_version} (transfer batch served this student on '
            f'{expected})',
        ),
    ]


def check_dependencies() -> list[Check]:
    """The two files this launcher delegates the real work to."""
    return [
        Check(
            'budget_wrapper',
            (ROOT / 'scripts/budget_admission.py').is_file(),
            'scripts/budget_admission.py (owned by session2-worker, not edited here)',
        ),
        Check('serve_helper', (ROOT / SERVE_HELPER).is_file(), SERVE_HELPER),
        Check(
            'build_step',
            (ROOT / 'examples/build_harness.py').is_file(),
            'examples/build_harness.py',
        ),
    ]


def preflight(run: Run) -> list[Check]:
    checks: list[Check] = []
    checks += check_attempt_unique(run)
    checks += check_predecessor(run)
    checks += check_dependencies()
    checks += check_serving_stack(run.student)
    checks += check_hashes()
    checks += check_manifest_inputs()
    checks += check_perception_cache()
    checks += check_builder_config()
    checks.append(check_target_config(run.student))
    checks.append(check_revision(run.student))
    checks += check_gpu(run.session)
    checks.append(check_gpu_compute_apps())
    checks.append(check_listener())
    checks.append(check_no_rival_server())
    checks += check_resources()
    checks.append(check_credentials())
    return checks


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------


ENV_FILE: Final = Path.home() / '.emoharness_env'
#: The student is always the locally served one. Pinning this closes the
#: accident where an inherited VLLM_ENDPOINT points a "local" run at another
#: host's server, which would silently evaluate a different model.
LOCAL_ENDPOINT: Final = f'http://127.0.0.1:{PORT}/v1'


def private_env() -> dict[str, str]:
    """Secrets from ``~/.emoharness_env``, loaded into memory and never printed.

    Returned separately from the rest of the environment so that nothing in
    this module can log it by accident: callers merge it into a child's env and
    never into anything that gets written to disk or a status file.
    """
    values: dict[str, str] = {}
    if not ENV_FILE.is_file():
        return values
    for line in ENV_FILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, _, value = stripped.partition('=')
        values[key.strip()] = value.strip().strip('\'"')
    return values


def _repo_relative(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    The runner takes this as its results directory. In a real launch the
    attempt always sits under the checkout, but refusing to render a path that
    does not is pointless brittleness: it made the launcher untestable outside
    the repo and turned a harmless case into a crash.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_environment(run: Run) -> dict[str, str]:
    """Scheduling and routing for the build. No scientific setting is here."""
    env = dict(os.environ)
    env.update(private_env())
    env['VLLM_ENDPOINT'] = LOCAL_ENDPOINT
    env.update(
        EMOHARNESS_SKIP_CONDA='1',
        EMOHARNESS_PYTHON=str(ROOT / '.venv/bin/python'),
        EMOHARNESS_WORKERS=str(WORKERS),
        EMOHARNESS_VIDEO_WORKERS=str(VIDEO_WORKERS),
        EMOHARNESS_BUILDER_CONFIG=BUILDER_CONFIG,
        EMOHARNESS_EXTERNAL_MANIFEST=EXTERNAL_MANIFEST,
        EMOHARNESS_PREPROCESS_CONFIG=PREPROCESS_CONFIG,
        EMOHARNESS_LEDGER_NAME='ledger.jsonl',
        EMOHARNESS_RESULTS_DIR=_repo_relative(run.attempt_dir),
        EMOHARNESS_SESSION=run.session,
        MAX_JOBS=str(MAX_JOBS),
        OMP_NUM_THREADS='1',
        MALLOC_ARENA_MAX='2',
    )
    return env


SERVE_HELPER: Final = 'scripts/serve_student_full_20260918.sh'


def server_command() -> list[str]:
    """The helper picks the model and venv up from the environment.

    This batch uses an isolated copy of serve_student.sh whose only difference
    is ``--served-model-name``, so the id on /v1/models is the HF id the target
    configs ask for. Keeping it separate means the frozen original cannot drift
    underneath a run that already hashed it.
    """
    return ['bash', str(ROOT / SERVE_HELPER)]


def server_environment(student: str) -> dict[str, str]:
    env = dict(os.environ)
    env['VLLM_ENDPOINT'] = LOCAL_ENDPOINT
    env.update(
        EMOHARNESS_VLLM_VENV=str(Path.home() / '.venvs' / STUDENTS[student]['venv']),
        EMOHARNESS_STUDENT_MODEL=STUDENTS[student]['model'],
        VLLM_PORT=str(PORT),
        VLLM_MAX_NUM_SEQS=str(MAX_NUM_SEQS),
        VLLM_MAX_MODEL_LEN=str(MAX_MODEL_LEN),
        VLLM_MM_CACHE_GB=str(MM_CACHE_GB),
        MAX_JOBS=str(MAX_JOBS),
        # Serve the weights the finished transfer batch already downloaded.
        # Offline is the enforcement; the preflight revision check is the
        # assertion. Together they stop a model-id launch from mixing a newly
        # fetched revision into a batch built on the old one.
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
    )
    return env


def spawn(command: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    """Start a child in its own process group so cleanup can be precise."""
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open('ab')
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_for_server(run: Run, timeout: int = SERVER_READY_TIMEOUT_S) -> str | None:
    """Wait for /health, watching resources throughout.

    Returns None when the server is ready, or the reason it never got there.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{PORT}/health"
    while time.monotonic() < deadline:
        if run.server is not None and run.server.poll() is not None:
            return f"server exited with {run.server.returncode} while loading"
        reason = trip(run, "server load")
        if reason is not None:
            return reason
        if _healthy(url):
            served = _served_models()
            wanted = STUDENTS[run.student]['model']
            if wanted in served:
                return None
            # Health can come up before the model is registered, and a server
            # advertising a different id would silently answer requests for a
            # model nobody asked for. Keep waiting rather than accept either.
            run.publish('server_registering', served_models=sorted(served))
        else:
            run.publish('server_starting')
        time.sleep(10)
    return (
        f'server did not advertise {STUDENTS[run.student]["model"]} on /v1/models '
        f'within {timeout}s'
    )


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
            return bool(response.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _served_models() -> frozenset[str]:
    """Model ids the server currently advertises on /v1/models."""
    try:
        with urllib.request.urlopen(  # noqa: S310
            f'http://127.0.0.1:{PORT}/v1/models', timeout=5
        ) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return frozenset()
    return frozenset(
        str(entry.get('id'))
        for entry in payload.get('data', [])
        if isinstance(entry, dict) and entry.get('id')
    )


def build_command(run: Run) -> list[str]:
    """The build, wrapped in the per-call spend gate, with frozen arguments."""
    return [
        str(ROOT / '.venv/bin/python'),
        str(ROOT / 'scripts/budget_admission.py'),
        '--cap',
        f'{BUILD_CAP_USD:.2f}',
        '--state',
        str(run.attempt_dir),
        '--builder-config',
        BUILDER_CONFIG,
        '--step',
        'examples/build_harness.py',
        '--',
        '--root',
        str(ROOT),
        '--stage',
        str(STAGE),
        '--replicate',
        str(REPLICATE),
        '--rounds',
        str(ROUNDS),
        '--bundle',
        str(ROOT / 'harness'),
        '--target',
        str(ROOT / STUDENTS[run.student]['target']),
    ]


# --------------------------------------------------------------------------
# Watchdog and cleanup
# --------------------------------------------------------------------------


#: Anonymous memory ceiling. This is the number that actually kills this
#: container: on 2026-09-12 a FlashInfer JIT build plus repeated video decodes
#: pushed anon past the supervisor tolerance and the whole 32 GiB container
#: restarted with oom_kill still at 0, so the cgroup OOM killer never saw it.
ANON_CEILING_GIB: Final = 26.0
#: Sustained full stall. Brief pressure during model load is normal; a minute
#: of it means the box is thrashing rather than working.
PRESSURE_FULL_AVG60: Final = 20.0
#: The lower bar that makes a high memory.current meaningful rather than just
#: page cache doing its job.
PRESSURE_CONFIRM_AVG60: Final = 5.0


def _cgroup_ints(path: str) -> dict[str, int]:
    try:
        text = Path(path).read_text()
    except OSError:
        return {}
    fields: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            fields[parts[0].rstrip(":")] = int(parts[1])
    return fields


def _pressure_full_avg60() -> float:
    try:
        text = Path("/sys/fs/cgroup/memory.pressure").read_text()
    except OSError:
        return 0.0
    for line in text.splitlines():
        if line.startswith("full"):
            for token in line.split():
                if token.startswith("avg60="):
                    return float(token.removeprefix("avg60="))
    return 0.0


def trip(run: Run, phase: str) -> str | None:
    """What, if anything, means this launch must stop now.

    Called during server load as well as during the build. Loading is where
    memory actually spikes -- weight load, kernel JIT and the first video
    decodes all land there -- so leaving it unwatched would mean the one phase
    most likely to take the container down is the one phase nothing guards.
    """
    limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
    current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    anon_gib = _cgroup_ints("/sys/fs/cgroup/memory.stat").get("anon", 0) / 2**30
    if anon_gib > ANON_CEILING_GIB:
        return (
            f"anonymous memory {anon_gib:.1f} GiB over the "
            f"{ANON_CEILING_GIB} GiB ceiling during {phase}"
        )
    pressure = _pressure_full_avg60()
    if pressure > PRESSURE_FULL_AVG60:
        return f"memory pressure full avg60 {pressure:.1f} during {phase}"
    # memory.current counts reclaimable page cache, and loading tens of GiB of
    # weights fills it by design. On its own that is healthy, so a high total
    # only counts as trouble when the kernel is also actually stalling.
    if limit and current / limit > 0.92 and pressure > PRESSURE_CONFIRM_AVG60:
        return (
            f"cgroup memory at {current / limit:.0%} of {limit / 2**30:.0f} GiB "
            f"with pressure full avg60 {pressure:.1f} during {phase}"
        )
    if shutil.disk_usage(ROOT).free / 2**30 < 5:
        return "less than 5 GiB of disk left"
    if run.server is not None and run.server.poll() is not None:
        return f"student server exited with {run.server.returncode}"
    return None


def latch(run: Run, reason: str) -> None:
    """Record a hold that blocks any successor launch for this student.

    Latching rather than restarting is the point: a build that died from
    resource pressure will die again, and a launcher that retries would spend
    real money discovering that.
    """
    run.latched = reason
    try:
        run.hold_path.parent.mkdir(parents=True, exist_ok=True)
        run.hold_path.write_text(
            f'{utc()} {socket.gethostname()} {run.session} {run.student}\n'
            f'{reason}\n'
            'Cleared only by a human after reading the evidence in this directory.\n',
            encoding='utf-8',
        )
    except OSError as exc:
        # The hold is a courtesy to the next launch; the evidence and the
        # cleanup that follow matter more, so a failure here must not abort
        # the shutdown path that writes them.
        print(f'{utc()} could not latch hold at {run.hold_path}: {exc}',
              file=sys.stderr, flush=True)


def cleanup(run: Run) -> None:
    """Terminate and reap only the process groups this launcher created.

    The group is signalled whether or not its leader is still alive. A dead
    leader says nothing about its descendants: serve_student.sh execs vLLM,
    which forks workers, and a video decode forks ffmpeg, so skipping the
    group because the process we happened to spawn has exited is exactly how
    an orphaned vLLM keeps holding the GPU after this launcher is gone.
    """
    for label, child, pgid in reversed(run.owned):
        for sig, grace in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 5.0)):
            if not _group_alive(pgid):
                break
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                break
            _reap(child, grace)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and _group_alive(pgid):
                time.sleep(0.5)
            if not _group_alive(pgid):
                break
            print(f'{utc()} {label} pgid {pgid} survived '
                  f'{signal.Signals(sig).name}', file=sys.stderr, flush=True)
        _reap(child, 5.0)
        if _group_alive(pgid):
            print(f'{utc()} WARNING {label} pgid {pgid} still has live members '
                  'after SIGKILL', file=sys.stderr, flush=True)


def _group_alive(pgid: int) -> bool:
    """Whether any member of a process group is still around."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap(child: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    """Collect a finished child so it stops holding its process group open."""
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=timeout)


def supervise(run: Run) -> int:
    """Watch the build to its end, or stop it and latch a hold."""
    if run.build is None:
        raise RuntimeError('supervise called before the build started')
    while True:
        code = run.build.poll()
        if code is not None:
            run.publish('build_finished', returncode=code)
            return code
        reason = trip(run, "build")
        if reason is not None:
            latch(run, reason)
            run.publish('watchdog_stop', reason=reason)
            cleanup(run)
            return 4
        run.publish('building', returncode=None)
        time.sleep(WATCHDOG_INTERVAL_S)


def verify_success(run: Run) -> str | None:
    """A zero exit code is a claim; the artifacts are the evidence.

    ``build_harness.py`` can exit 0 having produced nothing usable -- an empty
    loop, a bundle that was never written -- and a launcher that reported that
    as success would hand the next stage a phantom. Both artifacts must exist
    and the loop result must name a final bundle that is actually on disk.
    """
    results = list(run.attempt_dir.rglob('loop_result.json'))
    if not results:
        return f'no loop_result.json under {run.attempt_dir}'
    loop = results[0]
    try:
        record = json.loads(loop.read_text())
    except (OSError, ValueError) as exc:
        return f'{loop} is unreadable: {exc}'
    declared = record.get('final_bundle')
    if not declared:
        return f'{loop} names no final_bundle'
    bundle = Path(declared)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    if not bundle.is_dir():
        return f'final bundle {bundle} does not exist'
    if not any(bundle.iterdir()):
        return f'final bundle {bundle} is empty'
    return None


def final_evidence(run: Run, code: int, checks: list[Check]) -> None:
    write_json_atomic(
        run.attempt_dir / 'launch_evidence.json',
        {
            'utc': utc(),
            'host': socket.gethostname(),
            'session': run.session,
            'student': run.student,
            'returncode': code,
            'latched': run.latched,
            'owned_pgids': run.pgids,
            'elapsed_s': round(time.monotonic() - run.started, 1),
            'frozen': {
                'scaffold_hash': SCAFFOLD_HASH,
                'builder_config': BUILDER_CONFIG,
                'external_manifest': EXTERNAL_MANIFEST,
                'preprocess_config': PREPROCESS_CONFIG,
                'stage': STAGE,
                'replicate': REPLICATE,
                'rounds': ROUNDS,
                'candidates': CANDIDATES,
            },
            'scheduling': {
                'workers': WORKERS,
                'video_workers': VIDEO_WORKERS,
                'max_num_seqs': MAX_NUM_SEQS,
                'mm_cache_gb': MM_CACHE_GB,
                'max_jobs': MAX_JOBS,
                'max_model_len': MAX_MODEL_LEN,
            },
            'budget': {
                'cap_usd': BUILD_CAP_USD,
                'wrapper': 'scripts/budget_admission.py',
            },
            'weights': {
                'model_id': STUDENTS[run.student]['model'],
                'expected_revision': STUDENTS[run.student]['revision'],
                'cached_revision': cached_revision(STUDENTS[run.student]['model']),
                'offline': ['HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1'],
                'reused_8b_revision': REUSED_8B_REVISION,
            },
            'code': dict(
                zip(
                    ('launcher_sha256', 'wrapper_sha256', 'serve_helper_sha256'),
                    self_hashes(),
                    strict=True,
                )
            ),
            'serve_helper': SERVE_HELPER,
            'preflight': [
                {'name': c.name, 'ok': c.ok, 'detail': c.detail, 'blocking': c.blocking}
                for c in checks
            ],
        },
    )


# --------------------------------------------------------------------------


def run_build(run: Run, checks: list[Check], mock: int) -> int:
    """Bring up the server, run the build once, supervise, clean up."""
    interrupted: list[str] = []

    def on_signal(signum: int, _frame: object) -> None:
        """Shut down on a signal without ever losing the evidence.

        This is the path that runs when something has already gone wrong, so
        every step is attempted independently: one failure here must not cost
        us the child cleanup or the record of what happened.
        """
        name = signal.Signals(signum).name
        interrupted.append(name)
        for step, action in (
            ('latch', lambda: latch(run, f'launcher received {name}')),
            ('status', lambda: run.publish('terminating', signal=name)),
            ('cleanup', lambda: cleanup(run)),
            ('evidence', lambda: final_evidence(run, 130, checks)),
        ):
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - report and keep shutting down
                print(f'{utc()} shutdown step {step} failed: {exc!r}',
                      file=sys.stderr, flush=True)
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    run.attempt_dir.mkdir(parents=True, exist_ok=False)
    run.publish('starting_server')
    code = 1
    try:
        code = _lifecycle(run, mock)
    except SystemExit as exc:
        # The signal handler raises this after it has already latched and
        # recorded 130. Preserve that code instead of relabelling a clean
        # shutdown as a crash.
        code = exc.code if isinstance(exc.code, int) else 130
        raise
    except KeyboardInterrupt:
        latch(run, 'launcher interrupted')
        code = 130
        raise
    except Exception as exc:
        latch(run, f'launcher raised {type(exc).__name__}: {exc}')
        run.publish('crashed', error=repr(exc)[:400])
        code = 70
        raise
    finally:
        # Whatever happened -- clean exit, watchdog stop, unexpected exception,
        # or a signal that got past the handler -- the children this launcher
        # started are its responsibility and the evidence must exist.
        cleanup(run)
        final_evidence(run, code, checks)
        # The supervisor reads this phase, so it has to say what happened:
        # publishing 'done' after a crash would make a failed launch look
        # finished to the one process watching it.
        if code == 0 and run.latched is None:
            phase = 'done'
        elif run.latched is not None:
            phase = 'held'
        else:
            phase = 'failed'
        run.publish(phase, returncode=code)
    return code


def _lifecycle(run: Run, mock: int) -> int:
    """Server up, one build, supervise. Cleanup and evidence are the caller's."""
    if mock:
        run.adopt('server', spawn(['sleep', str(mock * 4)], dict(os.environ),
                                  run.attempt_dir / 'server.log'))
        run.publish('server_ready', mock=True)
    else:
        run.adopt('server', spawn(server_command(),
                                  server_environment(run.student),
                                  run.attempt_dir / 'server.log'))
        failure = wait_for_server(run)
        if failure is not None:
            latch(run, failure)
            run.publish("server_failed", reason=failure)
            return 5
        run.publish('server_ready')

    command = ['sleep', str(mock)] if mock else build_command(run)
    run.adopt('build', spawn(command, build_environment(run),
                             run.attempt_dir / 'build.log'))
    run.publish('building', command=command if mock else '[frozen build command]')
    code = supervise(run)
    if code == 0 and not mock:
        unusable = verify_success(run)
        if unusable is not None:
            latch(run, f'build exited 0 but produced nothing usable: {unusable}')
            run.publish('unusable_result', reason=unusable)
            return 6
    return code


def approval_path(student: str) -> Path:
    return BATCH / 'approvals' / f'BUILD_READY.{student}.json'


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--student', required=True, choices=sorted(STUDENTS))
    parser.add_argument('--session', required=True, choices=SESSIONS)
    parser.add_argument(
        '--attempt',
        type=int,
        default=1,
        metavar='N',
        help='attempt number, chosen deliberately by whoever approves the run. '
        'It is never auto-incremented: attempt_unique still refuses an existing '
        'directory, so naming a new attempt stays a human decision',
    )
    parser.add_argument(
        '--go',
        action='store_true',
        help='actually launch; requires the manager BUILD_READY approval file',
    )
    parser.add_argument(
        '--mock',
        nargs='?',
        type=int,
        const=5,
        default=0,
        metavar='SECONDS',
        help='exercise process groups, watchdog, status and cleanup with no GPU, '
        'no server and no paid call; the optional value is how long the stand-in '
        'build runs, so a longer one can be signalled mid-flight',
    )
    args = parser.parse_args()

    # A mock lives entirely under the safety-check area so it cannot touch the
    # real attempt tree, the real HOLD, or the status file the supervisor reads.
    attempt_dir = (
        BATCH
        / 'construction/_safety_checks_20260918/mock'
        / args.student
        / f'mock_attempt_{args.attempt:02d}'
        if args.mock
        else BATCH / 'construction' / args.student / f'attempt_{args.attempt:02d}'
    )
    run = Run(
        student=args.student,
        session=args.session,
        attempt_dir=attempt_dir,
        attempt=args.attempt,
        sandbox=bool(args.mock),
    )
    checks = preflight(run)
    if args.mock:
        checks = [
            c
            for c in checks
            if c.name not in {'gpu_name', 'gpu_uuid', 'gpu_free', 'gpu_compute_apps'}
        ]
    for check in checks:
        mark = 'ok  ' if check.ok else ('FAIL' if check.blocking else 'warn')
        print(f'{mark} {check.name}: {check.detail}')
    blocking = [c for c in checks if not c.ok and c.blocking]

    launcher_sha, wrapper_sha, helper_sha = self_hashes()
    approval = approval_path(args.student)
    print(f'\nattempt      : {run.attempt}'
          f'  (predecessor: {run.predecessor_dir.name if run.predecessor_dir else "none"})')
    print(f'attempt dir  : {run.attempt_dir}')
    print(f'launcher sha : {launcher_sha}')
    print(f'wrapper sha  : {wrapper_sha}')
    print(f'helper sha   : {helper_sha}  ({SERVE_HELPER})')
    print(f'approval     : {approval} '
          f'({"present" if approval.is_file() else "ABSENT"})')
    print(f'budget cap   : ${BUILD_CAP_USD:.2f} via scripts/budget_admission.py')
    print(f'endpoint     : {LOCAL_ENDPOINT} (forced)')
    print('approval must bind: ' + json.dumps({
        'student': args.student,
        'session': args.session,
        'host': socket.gethostname(),
        'launcher_sha256': launcher_sha,
        'wrapper_sha256': wrapper_sha,
        'serve_helper_sha256': helper_sha,
        'attempt': run.attempt,
        'attempt_dir': _repo_relative(run.attempt_dir),
        'predecessor_attempt': (
            None if run.predecessor_dir is None else run.predecessor_dir.name
        ),
        'predecessor_evidence_sha256': predecessor_evidence_sha(run),
    }))

    if blocking:
        print(f'\nREFUSED: {len(blocking)} blocking preflight failures.')
        return 2
    if not args.go and not args.mock:
        print('\nPreflight only. Re-run with --go once BUILD_READY review lands.')
        return 0
    if args.go:
        bound = check_approval(run, launcher_sha, wrapper_sha, helper_sha)
        print(f'{"ok  " if bound.ok else "FAIL"} approval: {bound.detail}')
        if not bound.ok:
            print('\nREFUSED: the BUILD_READY approval does not bind this build. '
                  'A manager must record one naming this student, session, host '
                  'and both code hashes.')
            return 2
        checks = [*checks, bound]
    return run_build(run, checks, args.mock)


if __name__ == '__main__':
    raise SystemExit(main())
