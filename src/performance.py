import logging
import time
from contextlib import contextmanager
from typing import Iterator


logger = logging.getLogger("rag.performance")


@contextmanager
def timed_stage(stage: str, **details: object) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        suffix = " ".join(f"{key}={value}" for key, value in details.items())
        message = f"[PERF] {stage:<28} {elapsed_ms:8.1f} ms"
        if suffix:
            message += f" {suffix}"
        logger.info(message)
        print(message)


def log_total(stage: str, started: float, **details: object) -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    message = f"[PERF] {stage:<28} {elapsed_ms:8.1f} ms"
    if suffix:
        message += f" {suffix}"
    logger.info(message)
    print(message)
