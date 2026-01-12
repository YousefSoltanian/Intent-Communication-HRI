#!/usr/bin/env bash
set -euo pipefail

# Must have conda on PATH
command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is not on PATH. Install Miniconda/Miniforge or add conda to PATH."
  exit 1
}

CONDA_BASE="$(conda info --base)"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

[ -f "$CONDA_SH" ] || {
  echo "ERROR: conda.sh not found at: $CONDA_SH"
  echo "CONDA_BASE was: $CONDA_BASE"
  exit 1
}

# Enable conda activate
source "$CONDA_SH"
conda activate intent_comm_nav

# Strip pyenv shims + python2.7 bin for this shell (optional but helpful)
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.pyenv" | grep -v "$HOME/python2.7/bin" | paste -sd ':' -)"
hash -r

# Convenience vars for imports + headless plotting
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
export MPLBACKEND=Agg

echo "python -> $(command -v python)"
python --version

