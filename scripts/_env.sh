#!/usr/bin/env bash
# Shared environment for every step wrapper: strict mode, repo root detection,
# the poetry-env conda environment when it exists, and one runner function.
# Override anything through the environment:
#   EMOHARNESS_PYTHON      interpreter command (default: "poetry run python")
#   EMOHARNESS_ROOT        project root holding data/ and results/
#   EMOHARNESS_CONDA_SETUP path to conda.sh
#   EMOHARNESS_CONDA_ENV   environment name (default: poetry-env)
#   EMOHARNESS_SKIP_CONDA  set to any value to skip activation
set -euo pipefail

EMOHARNESS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMOHARNESS_REPO_ROOT="$(cd "${EMOHARNESS_SCRIPT_DIR}/.." && pwd)"
EMOHARNESS_CONDA_SETUP="${EMOHARNESS_CONDA_SETUP:-${HOME}/anaconda3/etc/profile.d/conda.sh}"

if [[ -z "${EMOHARNESS_SKIP_CONDA:-}" && -f "${EMOHARNESS_CONDA_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${EMOHARNESS_CONDA_SETUP}"
  conda activate "${EMOHARNESS_CONDA_ENV:-poetry-env}"
fi

PYTHON="${EMOHARNESS_PYTHON:-poetry run python}"
EMOHARNESS_ROOT="${EMOHARNESS_ROOT:-${EMOHARNESS_REPO_ROOT}}"

# run_example <file.py> [args...] - run one entry point from the repository root.
run_example() {
  local name="$1"
  shift
  # shellcheck disable=SC2086
  (cd "${EMOHARNESS_REPO_ROOT}" && ${PYTHON} "examples/${name}" --root "${EMOHARNESS_ROOT}" "$@")
}
