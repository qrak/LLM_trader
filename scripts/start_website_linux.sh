#!/usr/bin/env bash
set -euo pipefail

#
# start_website_linux.sh
# Purpose: Install dependencies (if needed) and start the Astro website in dev or preview mode
# Usage: ./scripts/start_website_linux.sh [mode]
#   mode: dev (default) - Start Astro dev server with hot reload
#         build         - Build static site to dist/
#         preview       - Preview the built site locally
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEBSITE_DIR="${REPO_ROOT}/website"
NODE_MODULES="${WEBSITE_DIR}/node_modules"

MODE="${1:-dev}"

# Handle help flags
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    echo "Usage: $0 [dev|build|preview]" >&2
    echo "  dev     - Start dev server (default)" >&2
    echo "  build   - Build static site to dist/" >&2
    echo "  preview - Preview the built site" >&2
    exit 0
fi

echo "== scripts/start_website_linux.sh =="
echo "Website directory: ${WEBSITE_DIR}"

if [[ ! -d "${WEBSITE_DIR}" ]]; then
    echo "ERROR: Website directory not found at ${WEBSITE_DIR}" >&2
    exit 1
fi

cd "${WEBSITE_DIR}"

# Install dependencies if node_modules missing
if [[ ! -d "${NODE_MODULES}" ]]; then
    echo "Installing npm dependencies..."
    npm install
else
    echo "node_modules exists; skipping npm install."
fi

case "${MODE}" in
    dev)
        echo "Starting Astro dev server (hot reload)..."
        npm run dev
        ;;
    build)
        echo "Building static site..."
        npm run build
        echo "Build complete. Output in: ${WEBSITE_DIR}/dist/"
        ;;
    preview)
        echo "Starting Astro preview server..."
        npm run preview
        ;;
    *)
        echo "Usage: $0 [dev|build|preview]" >&2
        echo "  dev     - Start dev server (default)" >&2
        echo "  build   - Build static site to dist/" >&2
        echo "  preview - Preview the built site" >&2
        exit 1
        ;;
esac
