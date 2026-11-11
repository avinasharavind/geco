"""Process-wide download throttling.

Two entry points:
- `throttled_get` for HTTP reads we make ourselves (ATCF decks, GRIB byte ranges): bytes are debited as they stream.
- `TokenBucket.consume` for reads done inside libraries we can't hook (icechunk/zarr): debit the measured bytes
  afterwards (e.g. from `NetCounter`); the bucket goes into debt and the next call sleeps it off, so the
  average rate holds even though individual bursts don't.
"""
import threading
import time

import requests

from tcic.config import NET_IFACE, NET_RATE_MBPS, NET_RATE_WIRED_MBPS


class TokenBucket:
    def __init__(self, rate_bytes_per_s: float, burst_bytes: float | None = None, rate_fn=None, refresh_s: float = 60.0):
        self.rate_fn, self.refresh_s, self._checked = rate_fn, refresh_s, 0.0
        self.rate = rate_bytes_per_s
        self.capacity = burst_bytes if burst_bytes is not None else rate_bytes_per_s  # one second of burst
        self.tokens = self.capacity
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.consumed = 0  # total bytes debited by this process (used to tell our traffic from other users')

    def _refill(self):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def consume(self, nbytes: int):
        """Debit nbytes, sleeping until the bucket is out of debt."""
        with self.lock:
            if self.rate_fn is not None and time.monotonic() - self._checked > self.refresh_s:
                self.rate, self._checked = self.rate_fn(), time.monotonic()
            self._refill()
            self.consumed += nbytes
            self.tokens -= nbytes
            wait = -self.tokens / self.rate if self.tokens < 0 else 0.0
        if wait > 0:
            time.sleep(wait)


def link_share() -> float:
    """85% of the measured ceiling of whichever link is in use: wired ~30 MB/s, Wi-Fi ~20 MB/s."""
    return (NET_RATE_WIRED_MBPS if not default_iface().startswith("wl") else NET_RATE_MBPS) * 1e6


_last = {"t": None, "rx": None, "ours": None}


def current_rate(floor_mbps: float = 3.0) -> float:
    """Our share of the link minus what other users are downloading right now (they get priority), never below `floor`.

    Other users' traffic = interface receive rate - our own debited rate, over the time since the last call (~60 s).
    """
    cap = link_share()
    now, ours = time.monotonic(), _BUCKET.consumed
    try:
        rx = NetCounter()._rx()
    except RuntimeError:
        return cap
    others = 0.0
    if _last["t"] is not None and now > _last["t"]:
        dt = now - _last["t"]
        others = max(0.0, (rx - _last["rx"]) - (ours - _last["ours"])) / dt
    _last.update(t=now, rx=rx, ours=ours)
    return max(floor_mbps * 1e6, min(cap, cap - others))


_BUCKET = TokenBucket(NET_RATE_MBPS * 1e6, rate_fn=lambda: current_rate())
_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=32))


def bucket() -> TokenBucket:
    return _BUCKET


def throttled_get(url: str, byte_range: tuple[int, int] | None = None, timeout: float = 60,
                  chunk_size: int = 1 << 20) -> bytes:
    """GET url (optionally an inclusive byte range), streaming under the global rate limit."""
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    with _SESSION.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        parts = []
        for part in r.iter_content(chunk_size):
            _BUCKET.consume(len(part))
            parts.append(part)
    return b"".join(parts)


def default_iface() -> str:
    """Interface holding the default route (lowest metric). It has switched between Wi-Fi and wired on this machine."""
    best = None
    try:
        with open("/proc/net/route") as f:
            next(f)
            for line in f:
                p = line.split()
                if p[1] == "00000000":  # default route
                    metric = int(p[6])
                    if best is None or metric < best[1]:
                        best = (p[0], metric)
    except OSError:
        pass
    return best[0] if best else NET_IFACE


class NetCounter:
    """Bytes received on the internet interface between enter and exit (machine-wide, so approximate)."""

    def __init__(self, iface: str | None = None):
        self.iface = iface or default_iface()
        self.nbytes = 0

    def _rx(self) -> int:
        with open("/proc/net/dev") as f:
            for line in f:
                if line.strip().startswith(self.iface + ":"):
                    return int(line.split(":")[1].split()[0])
        raise RuntimeError(f"interface {self.iface} not found")

    def __enter__(self):
        self.start = self._rx()
        return self

    def __exit__(self, *exc):
        self.nbytes = self._rx() - self.start
        return False
