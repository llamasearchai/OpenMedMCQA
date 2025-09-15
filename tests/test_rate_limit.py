from __future__ import annotations

from medagent.rate_limit import RateLimiter


def test_token_bucket_basic():
    t = [0.0]

    def now():
        return t[0]

    rl = RateLimiter(rate_per_sec=2.0, burst=2, now=now)
    # Burst tokens allow 2 immediate
    assert rl.check("k")
    assert rl.check("k")
    # Third should fail without time elapsing
    assert not rl.check("k")
    # Advance 0.5s -> +1 token
    t[0] += 0.5
    assert rl.check("k")
    # Again immediate should fail
    assert not rl.check("k")
    # After another 0.5s -> +1 token
    t[0] += 0.5
    assert rl.check("k")

