"""Serve one student with vLLM on port 8000, the way every stage serves it.

    emoharness serve --student qwen3vl_4b
    emoharness serve --student gemma4_e4b --max-model-len 32768

Runs ``scripts/serve_student.sh`` in the foreground with the student's Hugging
Face id, its pinned revision checked against the local cache, and its vLLM
environment (``~/<venv>`` from ``experiments/common.py``; ``--venv`` overrides
it). The stages reuse a server that already serves the student they need.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from experiments.common import PORT, SERVE_SCRIPT, STUDENTS, verify_revision


def command(
    student: str, venv: Path | None, max_model_len: int
) -> tuple[list[str], dict[str, str]]:
    spec = STUDENTS[student]
    env = {
        **os.environ,
        'EMOHARNESS_STUDENT_MODEL': spec.hf_id,
        'EMOHARNESS_VLLM_VENV': str(venv or Path.home() / spec.venv),
        'VLLM_PORT': str(PORT),
        'VLLM_MAX_MODEL_LEN': str(max_model_len),
    }
    return ['bash', str(SERVE_SCRIPT)], env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--student', required=True, choices=sorted(STUDENTS))
    parser.add_argument('--venv', type=Path, default=None, help='the vLLM environment')
    parser.add_argument('--max-model-len', type=int, default=16384)
    parser.add_argument(
        '--no-verify', action='store_true', help='skip the pinned-revision check'
    )
    parser.add_argument('--dry-run', action='store_true', help='print the command only')
    args = parser.parse_args(argv)
    cmd, env = command(args.student, args.venv, args.max_model_len)
    shown = ' '.join(
        f'{k}={env[k]}'
        for k in (
            'EMOHARNESS_STUDENT_MODEL',
            'EMOHARNESS_VLLM_VENV',
            'VLLM_PORT',
            'VLLM_MAX_MODEL_LEN',
        )
    )
    print(f'{shown} {" ".join(cmd)}', flush=True)
    if args.dry_run:
        return 0
    if not args.no_verify:
        spec = STUDENTS[args.student]
        verify_revision(spec.hf_id, spec.revision)
    os.execvpe(cmd[0], cmd, env)  # noqa: S606 - the repository's own serve script
    return 0  # pragma: no cover - execvpe does not return


if __name__ == '__main__':
    sys.exit(main())
