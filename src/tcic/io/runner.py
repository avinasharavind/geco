"""Resumable, interruption-aware batch runner for long jobs on the shared Spark.

Design (the machine may reboot or our processes may be killed at any time):
- Work is split into small units, each with its own output path. A unit is done iff its output exists. Outputs are
  written to a temp file and atomically renamed, so a kill never leaves a half-written "done" file.
- A lock file (flock) prevents two copies running. The lock file also holds a heartbeat; if a new run finds a stale
  heartbeat from a process that is gone, it records that the previous run ended uncleanly.
- SIGTERM/SIGINT/SIGHUP set a stop flag: the current unit is allowed to finish (up to `grace_s`), then the run exits
  cleanly and says so in the log. A second signal aborts immediately (still safe: outputs are atomic).
- Before each unit the machine is checked; if load or memory pressure is high (other people's jobs), the runner
  waits instead of adding to it.
- Failures are logged per unit with the traceback and retried on later passes, up to `max_attempts`.
- `status.json` is rewritten after every unit (done/failed/remaining, throughput, ETA) for `scripts/pilot_status.sh`.
"""
import fcntl
import json
import os
import signal
import socket
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from tcic.io.throttle import NetCounter, bucket


@dataclass
class Unit:
    key: str  # unique, filesystem-safe id, e.g. "ifs/2025102600"
    output: Path  # the unit is done iff this exists
    payload: dict = field(default_factory=dict)


@dataclass
class Guard:
    max_load_frac: float = 0.70  # pause if 5-min load average > this fraction of cores (whole machine)
    min_mem_avail_gb: float = 24.0  # pause if MemAvailable drops below this
    min_disk_free_gb: float = 100.0  # stop (not pause) if disk gets this full
    check_every_s: float = 60.0
    active_hours: tuple[int, int] | None = None  # e.g. (20, 8): only work 20:00-08:00 local time

    @staticmethod
    def _meminfo_gb(key: str) -> float:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) / 1024**2
        return float("nan")

    def busy_reason(self) -> str | None:
        if self.active_hours is not None:
            a, b = self.active_hours
            h = time.localtime().tm_hour
            inside = (a <= h < b) if a < b else (h >= a or h < b)
            if not inside:
                return f"outside active hours {a:02d}-{b:02d}"
        load5 = os.getloadavg()[1]
        ncpu = os.cpu_count() or 1
        if load5 > self.max_load_frac * ncpu:
            return f"load5={load5:.1f} > {self.max_load_frac:.0%} of {ncpu} cores"
        mem = self._meminfo_gb("MemAvailable")
        if mem < self.min_mem_avail_gb:
            return f"MemAvailable={mem:.1f} GB < {self.min_mem_avail_gb} GB"
        return None


def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _release_memory():
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class Runner:
    def __init__(self, name: str, work_dir: Path, guard: Guard | None = None, max_attempts: int = 3, grace_s: float = 300):
        self.name = name
        self.dir = Path(work_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.guard = guard or Guard()
        self.max_attempts = max_attempts
        self.grace_s = grace_s
        self.log_path = self.dir / "run.log"
        self.fail_path = self.dir / "failures.jsonl"
        self.status_path = self.dir / "status.json"
        self.lock_path = self.dir / "lock"
        self._stop = False
        self._signals = 0

    # ---------- logging / state ----------
    def log(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _attempts(self) -> dict:
        out = {}
        if self.fail_path.exists():
            for line in self.fail_path.read_text().splitlines():
                try:
                    k = json.loads(line)["key"]
                    out[k] = out.get(k, 0) + 1
                except (ValueError, KeyError):
                    continue  # a line torn by a kill mid-write
        return out

    def _record_failure(self, unit: Unit, err: BaseException):
        rec = {"key": unit.key, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": repr(err),
               "traceback": traceback.format_exc(limit=8)}
        with open(self.fail_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _write_status(self, **kw):
        kw.update(updated=time.strftime("%Y-%m-%dT%H:%M:%S"), pid=os.getpid(), host=socket.gethostname())
        atomic_write_bytes(self.status_path, json.dumps(kw, indent=1).encode())

    # ---------- lock + heartbeat ----------
    def _acquire_lock(self):
        self._lock_fh = open(self.lock_path, "a+")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{self.name}: another run holds {self.lock_path}; not starting a second copy")
        self._lock_fh.seek(0)
        prev = self._lock_fh.read().strip()
        if prev:
            try:
                info = json.loads(prev)
                if info.get("state") == "running":
                    self.log(f"previous run (pid {info.get('pid')}) ended UNCLEANLY; last heartbeat {info.get('heartbeat')}, "
                             f"last unit {info.get('unit')!r}. Its finished units are kept; resuming.")
            except ValueError:
                pass
        self._heartbeat(state="running", unit=None)

    def _heartbeat(self, state: str, unit: str | None):
        self._lock_fh.seek(0)
        self._lock_fh.truncate()
        self._lock_fh.write(json.dumps({"pid": os.getpid(), "state": state, "unit": unit,
                                        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        self._lock_fh.flush()
        os.fsync(self._lock_fh.fileno())

    # ---------- signals ----------
    def _on_signal(self, signum, _frame):
        self._signals += 1
        name = signal.Signals(signum).name
        if self._signals == 1:
            self._stop = True
            self.log(f"received {name}: finishing the current unit (≤{self.grace_s:.0f} s), then stopping cleanly")
        else:
            self.log(f"received {name} again: aborting now (finished units are safe)")
            self._heartbeat(state="aborted", unit=None)
            os._exit(130)

    # ---------- main loop ----------
    def run(self, units: Iterable[Unit], work: Callable[[Unit], dict | None]):
        units = list(units)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, self._on_signal)
        self._acquire_lock()
        attempts = self._attempts()
        todo = [u for u in units if not u.output.exists() and attempts.get(u.key, 0) < self.max_attempts]
        given_up = [u.key for u in units if not u.output.exists() and attempts.get(u.key, 0) >= self.max_attempts]
        self.log(f"start: {len(units)} units, {len(units) - len(todo) - len(given_up)} already done, {len(todo)} to do, "
                 f"{len(given_up)} given up after {self.max_attempts} failures")
        t_start, n_done, n_fail, bytes_in = time.time(), 0, 0, 0
        try:
            for i, unit in enumerate(todo):
                if self._stop:
                    break
                # back off while the machine is busy with other people's work
                while (reason := self.guard.busy_reason()) and not self._stop:
                    self._write_status(state="paused", reason=reason, done=n_done, failed=n_fail, remaining=len(todo) - i,
                                       net_rate_mbps=round(bucket().rate / 1e6, 1))
                    self._heartbeat(state="running", unit=f"paused: {reason}")
                    self.log(f"pausing: {reason}")
                    time.sleep(self.guard.check_every_s)
                if self._stop:
                    break
                free_gb = os.statvfs(self.dir).f_bavail * os.statvfs(self.dir).f_frsize / 1024**3
                if free_gb < self.guard.min_disk_free_gb:
                    self.log(f"stopping: only {free_gb:.0f} GB free on disk")
                    break
                self._heartbeat(state="running", unit=unit.key)
                t0 = time.time()
                try:
                    with NetCounter() as net:
                        info = work(unit) or {}
                    if not unit.output.exists():
                        raise RuntimeError("work() returned without writing its output")
                    n_done += 1
                    bytes_in += net.nbytes
                    dt = time.time() - t0
                    self.log(f"done {unit.key} in {dt:.0f} s, {net.nbytes / 1e6:.0f} MB {json.dumps(info) if info else ''}")
                except Exception as e:  # noqa: BLE001 - one bad unit must not stop the night
                    n_fail += 1
                    self._record_failure(unit, e)
                    self.log(f"FAILED {unit.key}: {e!r} (attempt {attempts.get(unit.key, 0) + 1}/{self.max_attempts})")
                _release_memory()  # don't sit on freed-but-retained heap (we may idle for hours while paused)
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = len(todo) - i - 1
                self._write_status(state="running", done_this_run=n_done, failed_this_run=n_fail, remaining=remaining,
                                   net_rate_mbps=round(bucket().rate / 1e6, 1),
                                   total=len(units), mb_per_s=round(bytes_in / 1e6 / max(elapsed, 1), 2),
                                   eta_h=round(remaining / rate / 3600, 2) if rate else None, last_unit=unit.key)
        finally:
            state = "stopped" if self._stop else "finished"
            self._heartbeat(state=state, unit=None)
            self._write_status(state=state, done_this_run=n_done, failed_this_run=n_fail,
                               remaining=sum(not u.output.exists() for u in units), total=len(units))
            self.log(f"{state}: {n_done} units done, {n_fail} failed this run, "
                     f"{sum(not u.output.exists() for u in units)} still to do")
