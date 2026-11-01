import functools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("findex.timing")


def timed[F: Callable[..., Any]](fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info("%s took %.2f ms", fn.__name__, elapsed_ms)

    return wrapper  # type: ignore[return-value]
