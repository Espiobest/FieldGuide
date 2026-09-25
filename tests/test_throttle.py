import pytest

from fieldguide.throttle import Pacer, is_rate_limited


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def pacer(clock, **options):
    return Pacer(sleep=clock.sleep, clock=clock, **options)


def test_calls_are_spaced_by_minimum_interval():
    clock = Clock()
    limiter = pacer(clock, min_interval=4.0)
    stamps = []
    for _ in range(3):
        limiter.call(lambda: stamps.append(clock()))
    assert stamps == [0.0, 4.0, 8.0]
    assert limiter.idle_seconds == 8.0


def test_rate_limit_backs_off_exponentially_then_succeeds():
    clock = Clock()
    limiter = pacer(clock, base_backoff=2.0)
    attempts = []

    def flaky():
        attempts.append(clock())
        if len(attempts) < 4:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert limiter.call(flaky) == "ok"
    assert attempts == [0.0, 2.0, 6.0, 14.0]
    assert limiter.rate_limit_retries == 3


def test_other_errors_are_not_retried():
    limiter = pacer(Clock())
    with pytest.raises(ValueError):
        limiter.call(lambda: (_ for _ in ()).throw(ValueError("bad schema")))
    assert limiter.calls == 1


def test_retries_are_bounded():
    limiter = pacer(Clock(), max_retries=2)
    with pytest.raises(RuntimeError):
        limiter.call(lambda: (_ for _ in ()).throw(RuntimeError("429")))
    assert limiter.calls == 3


def test_rate_limit_detection_by_class_name():
    class GoogleRateLimitError(Exception):
        pass

    assert is_rate_limited(GoogleRateLimitError("quota"))
    assert not is_rate_limited(ValueError("bad"))
