"""Call pacing with exponential backoff for rate-limited model APIs."""

import time


def is_rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in ("429", "RateLimit", "RESOURCE_EXHAUSTED"))


class Pacer:
    def __init__(
        self,
        min_interval: float = 0.0,
        *,
        max_retries: int = 6,
        base_backoff: float = 5.0,
        max_backoff: float = 60.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.min_interval, self.max_retries = min_interval, max_retries
        self.base_backoff, self.max_backoff = base_backoff, max_backoff
        self._sleep, self._clock = sleep, clock
        self._last_call = None
        self.idle_seconds = 0.0
        self.calls = 0
        self.rate_limit_retries = 0

    def _idle(self, seconds: float):
        if seconds > 0:
            self.idle_seconds += seconds
            self._sleep(seconds)

    def call(self, function, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            if self._last_call is not None:
                self._idle(self.min_interval - (self._clock() - self._last_call))
            self._last_call = self._clock()
            self.calls += 1
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                if not is_rate_limited(exc) or attempt == self.max_retries:
                    raise
                self.rate_limit_retries += 1
                self._idle(min(self.base_backoff * 2**attempt, self.max_backoff))

    def wrap(self, chain):
        return PacedChain(chain, self)


class PacedChain:
    def __init__(self, chain, pacer: Pacer):
        self.chain, self.pacer = chain, pacer

    def invoke(self, payload):
        return self.pacer.call(self.chain.invoke, payload)
