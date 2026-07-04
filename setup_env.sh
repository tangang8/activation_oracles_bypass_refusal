#!/usr/bin/env bash
# Set up the shared virtual environment for this repo + the upstream activation_oracles.
#
# Layers this repo's lightweight extras (openai, scikit-learn, matplotlib) on top of the
# upstream activation_oracles locked dependency set, in a single shared .venv at the parent
# folder. Safe to re-run: creates the venv if missing, otherwise just re-syncs.
#
# Usage (from anywhere):
#   ./activation_oracles_bypass_refusal/setup_env.sh
#   source <parent-folder>/.venv/bin/activate
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
SIBLING_DIR="$PARENT_DIR/activation_oracles"
VENV_DIR="$PARENT_DIR/.venv"
EXTRAS_REQ="$REPO_DIR/judge_calibration/requirements.txt"

if [[ ! -d "$SIBLING_DIR" ]]; then
  echo "error: sibling repo not found at $SIBLING_DIR" >&2
  echo "       clone adamkarvonen/activation_oracles next to this repo (exact name)." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup_env] creating shared venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 1) Upstream locked GPU stack (torch, transformers, peft, ...) — authoritative pins.
echo "[setup_env] uv sync against $SIBLING_DIR/uv.lock"
uv sync --project "$SIBLING_DIR" --active

# 2) This repo's lightweight, pure-Python extras (judge-calibration pipeline).
#    Layered on top because `uv sync` prunes anything not in the upstream lock.
echo "[setup_env] installing extras from $EXTRAS_REQ"
uv pip install -r "$EXTRAS_REQ"

echo "[setup_env] done. Activate it in your shell with:"
echo "    source $VENV_DIR/bin/activate"
