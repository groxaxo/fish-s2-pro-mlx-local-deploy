#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/Users/op/fish-speech-int4-patch/.venv-mlx/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${ROOT}/.venv-mlx/bin/python"
fi

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
"${PYTHON}" - <<'PY'
from local_mlx.patches.fish_speech_fastpath import apply_fish_speech_patch

apply_fish_speech_patch()
print("fish_speech_fastpath patch applied (runtime monkey-patch).")
PY
