#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This bootstrap script requires macOS on ARM64." >&2
  exit 1
fi

for tool in brew git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done

brew install python@3.13 cmake ninja bison
git -C "$ROOT_DIR" submodule update --init booksim2 TOPSIS-Python

PYTHON_BIN="/opt/homebrew/bin/python3.13"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.13 was not installed at $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/python" -m pip install -r "$ROOT_DIR/requirements-m4.txt"

echo "Environment ready. Run: source $ROOT_DIR/.venv/bin/activate"
