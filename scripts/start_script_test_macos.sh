#!/usr/bin/env bash
set -euo pipefail

#
# start_script_test_macos.sh
# Purpose: Prepare .venv, ensure on 'develop' git branch (if repo), install requirements, and run start.py
# Usage: ./scripts/start_script_test_macos.sh [symbol] [-t timeframe] [--skip-install]
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_PATH}/bin/python"
PIP_BIN="${VENV_PATH}/bin/pip"

SYMBOL=""
TIMEFRAME=""
SKIP_INSTALL="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--timeframe)
            TIMEFRAME="${2:-}"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL="true"
            shift
            ;;
        -h|--help)
            echo "Usage: ./scripts/start_script_test_macos.sh [symbol] [-t timeframe] [--skip-install]"
            exit 0
            ;;
        *)
            if [[ -z "${SYMBOL}" ]]; then
                SYMBOL="$1"
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

echo "== scripts/start_script_test_macos.sh (develop) =="
echo "Repository root: ${REPO_ROOT}"

if [[ -d "${REPO_ROOT}/.git" ]]; then
    if command -v git >/dev/null 2>&1; then
        echo "Switching to 'develop' branch..."
        git -C "${REPO_ROOT}" fetch --all --prune
        git -C "${REPO_ROOT}" checkout develop
    else
        echo "Git command not found; skipping branch checkout."
    fi
else
    echo "No .git folder found; skipping branch checkout."
fi

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "Creating virtual environment at '${VENV_PATH}'..."
    python3 -m venv "${VENV_PATH}"
else
    echo "Virtual environment '${VENV_PATH}' already exists."
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

START_ARGS=()
if [[ -n "${SYMBOL}" ]]; then
    START_ARGS+=("${SYMBOL}")
fi
if [[ -n "${TIMEFRAME}" ]]; then
    START_ARGS+=("-t" "${TIMEFRAME}")
fi

if [[ ${#START_ARGS[@]} -gt 0 ]]; then
    echo "Running start.py with arguments: ${START_ARGS[*]}..."
else
    echo "Running start.py with default settings..."
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/start.py" "${START_ARGS[@]}"
