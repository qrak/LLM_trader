#!/usr/bin/env bash
set -euo pipefail

#
# scripts/run_all_tests.sh
# Purpose: Run the full test suite in the project's .venv
# Usage: bash ./scripts/run_all_tests.sh [-v] [-x] [--skip-install]
#
# Options:
#   -v              Verbose output (no -q flag)
#   -x              Stop on first failure
#   --skip-install  Skip pip install step
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_PATH}/bin/python"
PIP_BIN="${VENV_PATH}/bin/pip"

VERBOSE="false"
STOP_FIRST="false"
SKIP_INSTALL="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -x|--stop-first)
            STOP_FIRST="true"
            shift
            ;;
        --skip-install)
            SKIP_INSTALL="true"
            shift
            ;;
        -h|--help)
            echo "Usage: bash ./scripts/run_all_tests.sh [-v] [-x] [--skip-install]"
            echo ""
            echo "Options:"
            echo "  -v              Verbose output"
            echo "  -x              Stop on first failure"
            echo "  --skip-install  Skip pip install step"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash ./scripts/run_all_tests.sh [-v] [-x] [--skip-install]" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "Creating virtual environment at '${VENV_PATH}'..."
    python3 -m venv "${VENV_PATH}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python binary not found at ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ "${SKIP_INSTALL}" != "true" ]]; then
    if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
        echo "Installing/updating dependencies from requirements.txt..."
        "${PIP_BIN}" install --upgrade pip
        "${PIP_BIN}" install -r "${REPO_ROOT}/requirements.txt"
    else
        echo "No requirements.txt found; skipping pip install."
    fi
else
    echo "Skipping dependency installation (--skip-install provided)."
fi

cd "${REPO_ROOT}"

PYTEST_ARGS=()
if [[ "${VERBOSE}" != "true" ]]; then
    PYTEST_ARGS+=("-q")
fi
if [[ "${STOP_FIRST}" == "true" ]]; then
    PYTEST_ARGS+=("-x")
fi
PYTEST_ARGS+=("tests/")

echo ""
echo "Running: ${PYTHON_BIN} -m pytest ${PYTEST_ARGS[*]}"
echo ""

"${PYTHON_BIN}" -m pytest "${PYTEST_ARGS[@]}"
