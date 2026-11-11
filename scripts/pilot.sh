#!/usr/bin/env bash
# Detached, capped, resumable M2 pilot extraction.
#   scripts/pilot.sh start [ifs|gefs|both] [--hours 20-8]   start (or resume) in the background (all hours unless --hours)
#   scripts/pilot.sh status                                  progress summary
#   scripts/pilot.sh log                                     follow the log
#   scripts/pilot.sh stop                                    graceful stop (finishes the current run unit)
# Others first: at most 4 cores / 16 GB, CPUWeight=1 and IOWeight=1 (lowest share under contention), nice 19, idle I/O class;
# downloads ≤85% of the link minus other users' current traffic; pauses while the machine is busy (load / memory).
# Safe to kill at any time: finished units are kept and the next start resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
# after an ssh reconnect the shell may lack these; systemctl --user / systemd-run --user need them
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
UNIT=tcic-pilot
case "${1:-status}" in
  start)
    shift; MODEL="${1:-both}"; shift || true
    if systemctl --user is-active --quiet "$UNIT"; then echo "already running"; exit 0; fi
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --collect \
      -p CPUQuota=400% -p MemoryMax=16G -p CPUWeight=1 -p IOWeight=1 -p Nice=19 -p IOSchedulingClass=idle -p KillSignal=SIGTERM -p TimeoutStopSec=330 \
      --working-directory="$PWD" .venv/bin/python -m tcic.pilot.extract --model "$MODEL" "$@"
    echo "started $UNIT (model=$MODEL). Logs: scripts/pilot.sh log";;
  stop) systemctl --user stop "$UNIT" && echo "stopped (current unit finished or abandoned cleanly)";;
  status)
    systemctl --user is-active "$UNIT" >/dev/null 2>&1 && echo "service: running" || echo "service: not running"
    [ -f data/pilot/status.json ] && cat data/pilot/status.json; echo
    echo "units done: ifs $(ls data/pilot/ifs/*/done.json 2>/dev/null | wc -l), gefs $(ls data/pilot/gefs/*/done.json 2>/dev/null | wc -l)"
    du -sh data/pilot 2>/dev/null || true
    tail -3 data/pilot/run.log 2>/dev/null || true;;
  log) tail -f data/pilot/run.log;;
  *) echo "usage: $0 start [ifs|gefs|both] [--hours A-B] | stop | status | log"; exit 1;;
esac
