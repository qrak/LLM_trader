#!/usr/bin/env bash
set -euo pipefail

#
# start_script_main_macos.sh
# Purpose: Prepare .venv, install requirements, and run start.py (main branch)
# Usage: ./scripts/start_script_main_macos.sh [symbol] [-t timeframe] [--skip-install]
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
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            TIMEFRAME="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL="true"
            shift
            ;;
        -h|--help)
            echo "Usage: ./scripts/start_script_main_macos.sh [symbol] [-t timeframe] [--skip-install]"
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

echo "== scripts/start_script_main_macos.sh (main) =="
echo "Graceful stop: use Ctrl+C (app shows confirmation popup)."
echo "In-place reload: press SHIFT+R in the app console (auto-restarts, no manual restart)."
echo "Repository root: ${REPO_ROOT}"

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

install_requirements() {
    "${PIP_BIN}" install --upgrade pip
    "${PIP_BIN}" install -r "${REPO_ROOT}/requirements.txt"
}

if [[ "${SKIP_INSTALL}" != "true" ]]; then
    if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
        echo "Checking installed packages against requirements.txt (version-aware)..."
        CHECK_RC=0
        MISSING="$("${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_requirements.py" "${REPO_ROOT}/requirements.txt")" || CHECK_RC=$?
        if [[ ${CHECK_RC} -ne 0 ]]; then
            echo "Requirement check failed (exit ${CHECK_RC}); running pip install to be safe."
            install_requirements
        elif [[ -z "${MISSING}" ]]; then
            echo "All requirements satisfied; skipping pip install."
        else
            echo "Missing or mismatched requirements detected:"
            printf '%s\n' "${MISSING}"
            echo "Installing/updating dependencies from requirements.txt..."
            install_requirements
        fi
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

# In-place reload: the bot exits with code 42 on SHIFT+R; restart it here so no
# manual stop/start is needed. LLM_TRADER_RELOAD_SUPPORTED tells the bot this
# launcher can restart it (otherwise SHIFT+R is politely refused).
# Note: ${START_ARGS[@]+"${START_ARGS[@]}"} keeps the empty-array expansion safe
# on stock macOS bash 3.2, where "${arr[@]}" with set -u errors out.
RELOAD_EXIT_CODE=42
export LLM_TRADER_RELOAD_SUPPORTED=1
while true; do
    EXIT_CODE=0
    "${PYTHON_BIN}" "${REPO_ROOT}/start.py" ${START_ARGS[@]+"${START_ARGS[@]}"} || EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq ${RELOAD_EXIT_CODE} ]]; then
        echo ""
        echo "=== Reload requested - restarting start.py in place... ==="
        echo ""
        sleep 1
        continue
    fi
    break
done

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo ""
    echo "=== Process exited with error code: ${EXIT_CODE} ==="
    echo ""
fi

exit "${EXIT_CODE}"
