import json
import os
import signal
import subprocess
import sys
import textwrap
import time

from tcic.io.runner import Guard, Runner, Unit, atomic_write_bytes

NO_GUARD = Guard(max_load_frac=1e9, min_mem_avail_gb=0, min_disk_free_gb=0)


def test_resume_skips_done_and_retries_failures(tmp_path):
    out = tmp_path / "out"
    calls = []

    def work(u):
        calls.append(u.key)
        if u.key == "bad":
            raise ValueError("boom")
        atomic_write_bytes(u.output, b"ok")

    units = [Unit(k, out / f"{k}.bin") for k in ("a", "b", "bad")]
    Runner("t", tmp_path, guard=NO_GUARD, max_attempts=2).run(units, work)
    assert calls == ["a", "b", "bad"]
    calls.clear()
    Runner("t", tmp_path, guard=NO_GUARD, max_attempts=2).run(units, work)
    assert calls == ["bad"]  # done units skipped, failed unit retried
    calls.clear()
    Runner("t", tmp_path, guard=NO_GUARD, max_attempts=2).run(units, work)
    assert calls == []  # gave up after max_attempts
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "finished" and status["remaining"] == 1


def test_sigterm_stops_cleanly_and_unclean_kill_is_detected(tmp_path):
    script = textwrap.dedent(f"""
        import time, sys
        from pathlib import Path
        from tcic.io.runner import Guard, Runner, Unit, atomic_write_bytes
        units = [Unit(str(i), Path({str(tmp_path)!r}) / "out" / f"{{i}}.bin") for i in range(50)]
        def work(u):
            time.sleep(0.2)
            atomic_write_bytes(u.output, b"x")
        Runner("t", Path({str(tmp_path)!r}), guard=Guard(max_load_frac=1e9, min_mem_avail_gb=0, min_disk_free_gb=0)).run(units, work)
    """)
    # 1) SIGTERM: finishes the current unit and exits cleanly
    p = subprocess.Popen([sys.executable, "-c", script])
    time.sleep(1.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 0
    done1 = len(list((tmp_path / "out").glob("*.bin")))
    assert 0 < done1 < 50
    assert not list((tmp_path / "out").glob(".*tmp*"))
    assert json.loads((tmp_path / "lock").read_text())["state"] == "stopped"
    # 2) SIGKILL (like a crash/reboot): next run notices the unclean end and resumes
    p = subprocess.Popen([sys.executable, "-c", script])
    time.sleep(1.5)
    p.kill()
    p.wait()
    p = subprocess.Popen([sys.executable, "-c", script])
    assert p.wait(timeout=30) == 0
    assert len(list((tmp_path / "out").glob("*.bin"))) == 50
    assert "ended UNCLEANLY" in (tmp_path / "run.log").read_text()


def test_second_copy_refuses_to_start(tmp_path):
    import fcntl

    fh = open(tmp_path / "lock", "a+")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        Runner("t", tmp_path, guard=NO_GUARD).run([], lambda u: None)
        raise AssertionError("should have refused")
    except SystemExit as e:
        assert "another run" in str(e)
