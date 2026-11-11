import time

from tcic.io.throttle import TokenBucket


def test_bucket_holds_average_rate():
    b = TokenBucket(rate_bytes_per_s=1_000_000, burst_bytes=100_000)
    t0 = time.monotonic()
    for _ in range(10):
        b.consume(100_000)
    elapsed = time.monotonic() - t0
    # 1 MB at 1 MB/s with a 0.1 MB burst allowance -> ~0.9 s
    assert 0.8 <= elapsed <= 1.2


def test_bucket_debt_is_slept_off():
    b = TokenBucket(rate_bytes_per_s=1_000_000, burst_bytes=0)
    t0 = time.monotonic()
    b.consume(500_000)  # post-hoc debit of a large read
    assert 0.45 <= time.monotonic() - t0 <= 0.7
