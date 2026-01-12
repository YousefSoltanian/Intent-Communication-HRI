#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MPLBACKEND=Agg
export PYTHONPATH="$ROOT/src"

conda run -n intent_comm_nav python "$ROOT/src/Navigation/navigation_npace_influence_vs_human_demo.py"

