#!/usr/bin/env bash
# Run a command within this project's share of the machine: 40% CPU (8 of 20 cores), 48 GB RAM, low priority.
# Usage: scripts/run_capped.sh python -m tcic.something --args
set -euo pipefail
exec systemd-run --user --scope --quiet \
    -p CPUQuota=800% -p MemoryMax=48G \
    nice -n 10 ionice -c 3 "$@"
