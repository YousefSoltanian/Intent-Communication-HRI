#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate the conda env + sanitize PATH for this process
source "$SCRIPT_DIR/enter_intent_comm_env.sh" intent_comm_nav

# Ensure imports work when running from repo root
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Headless-safe plotting
export MPLBACKEND=Agg

echo "Using python: $(which python)"
python -c "import sys; print(sys.version)"

python "$REPO_ROOT/src/Navigation/navigation_npace_influence_vs_human_demo.py"

echo "Done. Outputs should be saved next to the demo script in src/Navigation/"

