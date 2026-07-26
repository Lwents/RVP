"""Shared ``AsyncOpenAI`` clients for every AI service in this package.

Building a client per request also builds a fresh httpx connection pool that is
never closed, so a single review/dubbing job used to leak dozens of pools.  The
clients hold no per-request state, so one instance per configuration is reused
for the whole process.

``max_retries`` defaults to 0 on purpose: every caller here already retries at
the pipeline level, and letting the SDK retry too multiplies the wall-clock cost
of a failing request by the pipeline retry count.
"""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI


DEFAULT_TIMEOUT_SECONDS = 90.0


@lru_cache(maxsize=16)
def _cached_client(
    api_key: str,
    base_url: str,
    timeout: float,
    max_retries: int,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_async_openai(
    api_key: str | None,
    base_url: str | None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 0,
) -> AsyncOpenAI:
    """Return a process-wide client shared by every caller with this config."""

    return _cached_client(api_key or "", base_url or "", float(timeout), int(max_retries))
