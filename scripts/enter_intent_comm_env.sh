#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-intent_comm_nav}"

# Load conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found in \$HOME/miniconda3 or \$HOME/anaconda3" >&2
  exit 1
fi

conda activate "$ENV_NAME"

# Remove pyenv shims + python2.7 from this shell session
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.pyenv" | grep -v "$HOME/python2.7/bin" | paste -sd ':' -)"
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

