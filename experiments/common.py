"""What the out-of-distribution experiments share: the four students and one owned server.

Every experiment evaluates the same four students with the frozen harness each
of them was given, served exactly the way the main evaluation served them: the
same serve script, the serving venv for the architecture, the Hugging Face hub
offline, and the student's pinned revision checked against the local cache
before launch.

Drivers run one vLLM server at a time and never share a port. A step is one
subprocess (``python -m experiments.<name>.row ...``) whose last stdout line
is its provenance record; a step with a ``done.json`` is skipped on resume, and
a failed step is retried once. Every finished Target call is in the shared
content-addressed cache and decoding is temperature 0, so a retry re-issues
only unfinished calls and is not a resample.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
SERVE_SCRIPT: Final = ROOT / 'scripts/serve_student.sh'

#: The shared scaffold (the L3 row) and its content hash.
SCAFFOLD: Final = 'harness'
SCAFFOLD_HASH: Final = (
    'b38c518b7e4161a3ba4f417f31ff4b359b3ea32474d4675cc2f8ec6f0df8ab17'
)
#: Hash of the perception config (Whisper ASR + openSMILE prosody + frames) the
#: main evaluation used; every step refuses to run under any other.
PERCEPTION_CONFIG_HASH: Final = (
    '6ce8cc9178cefb39a0f109c5d10586a2e2041550a7ac515e8851c63c11cb2c00'
)


@dataclass(frozen=True)
class Student:
    """One student: how it is served, and the frozen harness it was given."""

    name: str
    hf_id: str
    revision: str
    venv: str
    target: str
    bundle: str
    bundle_hash: str


STUDENTS: Final[dict[str, Student]] = {
    'qwen3vl_8b': Student(
        name='Qwen3-VL-8B',
        hf_id='Qwen/Qwen3-VL-8B-Instruct',
        revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        venv='.venvs/emoharness-vllm011',
        target='configs/target/qwen3vl_8b_instruct.yaml',
        bundle='bundles/qwen3vl_8b',
        bundle_hash='569c76f9edc0e2e6473fdad9f8854fdfa473fcf94964961d48877a338478e9e2',
    ),
    'qwen3vl_4b': Student(
        name='Qwen3-VL-4B',
        hf_id='Qwen/Qwen3-VL-4B-Instruct',
        revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',
        venv='.venvs/emoharness-vllm011',
        target='configs/target/qwen3vl_4b_instruct.yaml',
        bundle='bundles/qwen3vl_4b',
        bundle_hash='4ace41173bb48fda24a4a0fd9c8134bc659f40d2f6fa2d428d1324c53cafe513',
    ),
    'qwen3_5_4b': Student(
        name='Qwen3.5-4B',
        hf_id='Qwen/Qwen3.5-4B',
        revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        venv='.venvs/emoharness-vllm019',
        target='configs/target/qwen3_5_4b.yaml',
        bundle='bundles/qwen3_5_4b',
        bundle_hash='d8b0b2794c4c95fee8df06ffcc979df9e83242c877848d80d50dd3a77ad4175e',
    ),
    'gemma4_e4b': Student(
        name='Gemma-4-E4B',
        hf_id='google/gemma-4-E4B-it',
        revision='ee0ef6023621cff504d758262d4e04895a5af4a2',
        venv='.venvs/emoharness-vllm019-tf513',
        target='configs/target/gemma4_e4b_it.yaml',
        bundle='bundles/gemma4_e4b',
        bundle_hash='468afcba8c97dd1e98a8822fe892b5832d31f999d5a70fc8a5ff0f269a5a9f17',
    ),
}
ORDER: Final = ('qwen3vl_8b', 'qwen3vl_4b', 'qwen3_5_4b', 'gemma4_e4b')
NAMES: Final = {key: student.name for key, student in STUDENTS.items()}

PORT: Final = 8000
ENDPOINT: Final = f'http://127.0.0.1:{PORT}/v1'
STARTUP_S: Final = 2400
MAX_ATTEMPTS: Final = 2


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_json(path: Path, document: Any) -> None:
    """Atomic within one directory: a reader never sees half a file."""
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def lib_path() -> str:
    """CUDA 12 cuBLAS/cuDNN from the project venv, which faster-whisper needs."""
    site = ROOT / '.venv/lib/python3.12/site-packages/nvidia'
    extra = f'{site}/cublas/lib:{site}/cudnn/lib'
    current = os.environ.get('LD_LIBRARY_PATH')
    return f'{extra}:{current}' if current else extra


def verify_revision(hf_id: str, pinned: str) -> None:
    repo = (
        Path.home() / '.cache/huggingface/hub' / ('models--' + hf_id.replace('/', '--'))
    )
    ref = (repo / 'refs/main').read_text().strip()
    if ref != pinned or not (repo / 'snapshots' / pinned).is_dir():
        raise RuntimeError(f'{hf_id}: refs/main {ref[:12]} != pinned {pinned[:12]}')


def ledger_rows(results_dir: Path) -> int:
    n = 0
    for path in results_dir.rglob('ledger*.jsonl'):
        with path.open() as fh:
            n += sum(1 for _ in fh)
    return n


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as r:
            return bool(r.status == 200)
    except Exception:  # noqa: BLE001 - any failure means not healthy
        return False


def port_busy() -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(('127.0.0.1', PORT)) == 0


def free_vram_mib() -> int:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.split()[0])


def step_env(results_dir: Path) -> dict[str, str]:
    """The environment every step runs in; client concurrency stays at its defaults."""
    return {
        **os.environ,
        'EMOHARNESS_RESULTS_DIR': str(results_dir.relative_to(ROOT)),
        'VLLM_ENDPOINT': ENDPOINT,
        'EMOHARNESS_WORKERS': '8',
        'EMOHARNESS_VIDEO_WORKERS': '2',
        'LD_LIBRARY_PATH': lib_path(),
        'EMOHARNESS_PREPROCESS_CONFIG': 'preprocess.yaml',
    }


def module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, '-m', module, *args]


class Driver:
    """Status file, logs, and at most one owned vLLM server."""

    def __init__(self, status: Path, logs: Path) -> None:
        self.status = status
        self.logs = logs
        self.state: dict[str, Any] = {
            'started_at': now(),
            'host': os.uname().nodename,
            'driver_pid': os.getpid(),
            'pairs': {},
            'server': None,
            'phase': 'starting',
        }

    def save(self) -> None:
        self.state['updated_at'] = now()
        write_json(self.status, self.state)

    def log(self, message: str) -> None:
        print(f'[{now()}] {message}', flush=True)

    def set_phase(self, phase: str) -> None:
        self.state['phase'] = phase
        self.save()

    def record_error(self, student: str, exc: BaseException) -> None:
        error = f'{type(exc).__name__}: {exc}'
        self.state.setdefault('errors', []).append(
            {'at': now(), 'student': student, 'error': error}
        )
        self.save()
        self.log(f'{student}: {error}')

    def wait_for_vram(self, need_mib: int, poll_s: int = 300) -> None:
        """Wait for free VRAM instead of squeezing in beside another tenant."""
        since = now()
        while (free := free_vram_mib()) < need_mib:
            self.state['gpu_wait'] = {
                'since': since,
                'free_mib': free,
                'need_mib': need_mib,
                'checked_at': now(),
            }
            self.set_phase('waiting for GPU')
            time.sleep(poll_s)
        if 'gpu_wait' in self.state:
            wait = self.state.pop('gpu_wait')
            self.state.setdefault('gpu_waits', []).append({**wait, 'ended_at': now()})
            self.save()

    def start_server(
        self, key: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.Popen[bytes]:
        student = STUDENTS[key]
        verify_revision(student.hf_id, student.revision)
        if port_busy():
            raise RuntimeError(f'port {PORT} is already serving; refusing to share it')
        env = {
            **os.environ,
            'EMOHARNESS_STUDENT_MODEL': student.hf_id,
            'EMOHARNESS_VLLM_VENV': str(Path.home() / student.venv),
            'VLLM_PORT': str(PORT),
            'VLLM_MAX_NUM_SEQS': '8',
            'VLLM_MM_CACHE_GB': '0',
            'HF_HUB_OFFLINE': '1',
            **(extra_env or {}),
        }
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / f'serve_{key}.log').open('a') as out:
            proc = subprocess.Popen(
                ['bash', str(SERVE_SCRIPT)],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=ROOT,
            )
        self.state['server'] = {
            'student': key,
            'pid': proc.pid,
            'pgid': os.getpgid(proc.pid),
            'model': student.hf_id,
            'revision': student.revision,
            'venv': student.venv,
            'started_at': now(),
            'ready_at': None,
        }
        self.save()
        began = time.time()
        while not healthy():
            if proc.poll() is not None:
                raise RuntimeError(
                    f'{key} server exited rc={proc.returncode} before ready'
                )
            if time.time() - began > STARTUP_S:
                self.stop_server(proc)
                raise RuntimeError(f'{key} server not ready within {STARTUP_S}s')
            time.sleep(10)
        self.state['server'].update(
            ready_at=now(), startup_s=round(time.time() - began, 1)
        )
        self.save()
        self.log(f'{key} ready in {time.time() - began:.0f}s')
        return proc

    def stop_server(self, proc: subprocess.Popen[bytes] | None) -> None:
        """Stop only the process group this driver started."""
        if proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for _ in range(60):
            if proc.poll() is not None:
                break
            time.sleep(2)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        for _ in range(30):
            if not port_busy():
                break
            time.sleep(2)
        self.state['server'] = None
        self.save()

    def run_step(self, key: str, command: list[str], results_dir: Path) -> bool:
        """Run one step to completion, retrying once; write ``done.json`` on success."""
        entry = self.state['pairs'].setdefault(
            key, {'attempts': 0, 'status': 'pending'}
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        while entry['attempts'] < MAX_ATTEMPTS:
            entry['attempts'] += 1
            entry.update({'status': 'running', 'started_at': now(), 'error': None})
            self.save()
            logf = self.logs / f'{key.replace("/", "_")}_a{entry["attempts"]}.log'
            with logf.open('w') as out:
                proc = subprocess.Popen(
                    command,
                    env=step_env(results_dir),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                )
                entry['row_pid'] = proc.pid
                while proc.poll() is None:
                    entry['target_calls_so_far'] = ledger_rows(results_dir)
                    entry['server_healthy'] = healthy()
                    self.save()
                    time.sleep(60)
            entry['returncode'] = proc.returncode
            tail = logf.read_text()[-20000:].strip().splitlines()
            if proc.returncode == 0 and tail:
                provenance = json.loads(tail[-1])
                write_json(results_dir / 'done.json', provenance)
                entry.update(
                    {
                        'status': 'done',
                        'finished_at': now(),
                        'n_written': provenance['n_written'],
                        'seconds': provenance['seconds'],
                        'accuracy': provenance.get('accuracy'),
                        'n_unparsed': provenance.get('n_unparsed'),
                        'target_calls': ledger_rows(results_dir),
                    }
                )
                self.save()
                self.log(
                    f'{key} done: {provenance["n_written"]} in {provenance["seconds"]}s'
                )
                return True
            entry.update({'status': 'failed', 'error': '\n'.join(tail[-6:])})
            self.save()
            self.log(f'{key} attempt {entry["attempts"]} failed rc={proc.returncode}')
            if not healthy():
                return False
        return False


def is_done(results_dir: Path) -> bool:
    """A step is done when its ``done.json`` names an output that exists."""
    marker = results_dir / 'done.json'
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    output = record.get('predictions') or record.get('contexts')
    return bool(output) and (ROOT / output).is_file()
