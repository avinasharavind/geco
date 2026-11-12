#!/usr/bin/env bash
# Rebuild everything downstream of the inputs, in order, as a detached capped service that survives ssh disconnects.
#   scripts/rerun_chain.sh start     start (or resume: finished steps have a marker in data/rerun_chain/ and are skipped)
#   scripts/rerun_chain.sh status    which steps are done; tail of the log
#   scripts/rerun_chain.sh reset     forget the markers (next start reruns everything)
# Same limits as the pilot: at most 4 cores / 16 GB, lowest CPU and I/O share, nice 19.
set -euo pipefail
cd "$(dirname "$0")/.."
# after an ssh reconnect the shell may lack these; systemctl --user / systemd-run --user need them
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
UNIT=tcic-chain
MARK=data/rerun_chain
STEPS=(
  "features   .venv/bin/python -m tcic.consensus.features"
  "m15        .venv/bin/python scripts/m15_run.py"
  "members    .venv/bin/python -m tcic.ensemble.members"
  "gridded    .venv/bin/python -m tcic.gridded.features"
  "m2         .venv/bin/python scripts/m2_run.py"
  "ens_run    .venv/bin/python scripts/ens_run.py"
  "ens_smooth .venv/bin/python scripts/ens_smooth.py"
  "ens_spread .venv/bin/python scripts/ens_spread.py"
  "ens_calib  .venv/bin/python scripts/ens_calib.py"
  "ccsr       .venv/bin/python scripts/ens_smoothspread.py"
  "gridcentre .venv/bin/python scripts/ens_gridcentre.py"
  "geco       .venv/bin/python scripts/ens_geco.py"
  "multiseason .venv/bin/python scripts/ens_multiseason.py"
  "ecmwf_tf   .venv/bin/python -m tcic.data.ecmwf_tf"
  "ieco       .venv/bin/python scripts/ens_ieco.py"
  "k_curve    .venv/bin/python scripts/k_curve.py"
  "fnv3       .venv/bin/python scripts/fnv3_compare.py"
  "fig_m15    .venv/bin/python scripts/report_figures.py"
  "fig_ens    .venv/bin/python scripts/report_figures_ens.py"
  "cases      .venv/bin/python scripts/case_plots.py"
)
case "${1:-status}" in
  start)
    if systemctl --user is-active --quiet "$UNIT"; then echo "already running"; exit 0; fi
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --collect \
      -p CPUQuota=400% -p MemoryMax=16G -p CPUWeight=1 -p IOWeight=1 -p Nice=19 -p IOSchedulingClass=idle \
      --working-directory="$PWD" "$PWD/scripts/rerun_chain.sh" _run
    echo "started $UNIT. Progress: scripts/rerun_chain.sh status";;
  _run)
    mkdir -p "$MARK"
    for s in "${STEPS[@]}"; do
      name=${s%% *}; cmd=${s#* }
      [ -f "$MARK/$name.done" ] && { echo "$(date '+%F %T') skip $name (done)" >> "$MARK/chain.log"; continue; }
      echo "$(date '+%F %T') start $name" >> "$MARK/chain.log"
      if bash -c "$cmd" > "$MARK/$name.log" 2>&1; then
        touch "$MARK/$name.done"; echo "$(date '+%F %T') done $name" >> "$MARK/chain.log"
      else
        echo "$(date '+%F %T') FAILED $name (see $MARK/$name.log)" >> "$MARK/chain.log"; exit 1
      fi
    done
    echo "$(date '+%F %T') chain finished" >> "$MARK/chain.log";;
  status)
    systemctl --user is-active "$UNIT" >/dev/null 2>&1 && echo "service: running" || echo "service: not running"
    (ls "$MARK"/*.done 2>/dev/null || true) | xargs -r -n1 basename | sed "s/.done//" | tr "\n" " "; echo
    tail -4 "$MARK/chain.log" 2>/dev/null || true;;
  reset) rm -f "$MARK"/*.done && echo "markers cleared";;
  *) echo "usage: $0 start | status | reset"; exit 1;;
esac
