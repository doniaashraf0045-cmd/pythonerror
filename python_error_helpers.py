"""Error-handling helpers for Playwright-based agents.

Features:
- run_with_retry: retry wrapper with exponential backoff + optional jitter
- safe_call: wrapper which catches/records exceptions, returns SafeCallResult
- log_exception: writes stacktrace to errors.log and journal.jsonl

Design goals:
- Small, dependency-free, well-typed (PEP 484)
- Works with sync and async callables
- Uses Python logging by default; optional file/jsonl append
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, overload

T = TypeVar("T")

DEFAULT_ERROR_LOG = Path("errors.log")
DEFAULT_JOURNAL = Path("agent_journal.jsonl")

@dataclass(frozen=True, slots=True)
class SafeCallResult:
    ok: bool
    result: Optional[Any] = None
    error: Optional[BaseException] = None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log_exception(
    context: str,
    exc: BaseException,
    *,
    logger: Optional[logging.Logger] = None,
    error_log_path: Optional[Path] = DEFAULT_ERROR_LOG,
    journal_path: Optional[Path] = DEFAULT_JOURNAL,
) -> None:
    """Log an exception.

    - Always logs to `logger` (defaults to module logger).
    - Optionally appends JSON to `error_log_path` and a compact entry to `journal_path`.

    The file format is JSON Lines (one JSON object per line).
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.exception("%s", context, exc_info=exc)

    entry = {
        "time": _utc_iso(),
        "context": context,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": tb,
    }

    if error_log_path is not None:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        error_log_path.open("a", encoding="utf-8").write(json.dumps(entry, ensure_ascii=False) + "\n")

    if journal_path is not None:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.open("a", encoding="utf-8").write(
            json.dumps({"t": entry["time"], "ctx": context, "err": entry["error"], "type": entry["error_type"]}, ensure_ascii=False)
            + "\n"
        )


@overload
def run_with_retry(
    func: Callable[..., T],
    *args: Any,
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: Optional[float] = None,
    jitter: float = 0.0,
    context: str = "",
    on_exception: Optional[Callable[[int, BaseException], None]] = None,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> tuple[bool, Optional[T]]: ...


@overload
def run_with_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: Optional[float] = None,
    jitter: float = 0.0,
    context: str = "",
    on_exception: Optional[Callable[[int, BaseException], None]] = None,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> Awaitable[tuple[bool, Optional[T]]]: ...


def run_with_retry(
    func: Union[Callable[..., T], Callable[..., Awaitable[T]]],
    *args: Any,
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: Optional[float] = None,
    jitter: float = 0.0,
    context: str = "",
    on_exception: Optional[Callable[[int, BaseException], None]] = None,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> Union[tuple[bool, Optional[T]], Awaitable[tuple[bool, Optional[T]]]]:
    """Run a callable with retry + exponential backoff.

    Supports both sync and async callables.

    Args:
        retries: total attempts (>=1). If 1, runs once with no retry.
        initial_delay: delay before the 2nd attempt.
        backoff: delay multiplier for subsequent attempts.
        max_delay: optional cap for delay.
        jitter: adds +/- jitter seconds to each delay (0 disables jitter).
        exceptions: exception types that trigger a retry.
        on_exception: callback (attempt_number, exception).
    """

    if retries < 1:
        raise ValueError("retries must be >= 1")
    if initial_delay < 0:
        raise ValueError("initial_delay must be >= 0")
    if backoff < 1:
        raise ValueError("backoff must be >= 1")
    if jitter < 0:
        raise ValueError("jitter must be >= 0")

    async def _async_impl() -> tuple[bool, Optional[T]]:
        delay = initial_delay
        attempt = 0
        while True:
            try:
                result = func(*args, **kwargs)  # type: ignore[misc]
                if asyncio.iscoroutine(result):
                    result = await result  # type: ignore[assignment]
                return True, result  # type: ignore[return-value]
            except exceptions as exc:  # type: ignore[misc]
                attempt += 1
                if on_exception is not None:
                    on_exception(attempt, exc)
                if context:
                    log_exception(f"{context} attempt={{attempt}}", exc)

                if attempt >= retries:
                    return False, None

                sleep_for = delay
                if jitter:
                    sleep_for = max(0.0, sleep_for + random.uniform(-jitter, jitter))
                if max_delay is not None:
                    sleep_for = min(sleep_for, max_delay)

                await asyncio.sleep(sleep_for)
                delay *= backoff

    # If func returns an awaitable, caller can await us.
    result = func(*args, **kwargs)  # type: ignore[misc]
    if asyncio.iscoroutine(result):
        # We already executed once; the async impl should handle first attempt consistently.
        # Re-run via async impl to avoid double execution.
        return _async_impl()

    # Sync path (no awaitables involved)
    delay = initial_delay
    attempt = 0
    while True:
        try:
            return True, result  # type: ignore[return-value]
        except exceptions as exc:
            attempt += 1
            if on_exception is not None:
                on_exception(attempt, exc)
            if context:
                log_exception(f"{context} attempt={{attempt}}", exc)

            if attempt >= retries:
                return False, None

            sleep_for = delay
            if jitter:
                sleep_for = max(0.0, sleep_for + random.uniform(-jitter, jitter))
            if max_delay is not None:
                sleep_for = min(sleep_for, max_delay)

            time.sleep(sleep_for)
            delay *= backoff


@overload
def safe_call(
    func: Callable[..., T],
    *args: Any,
    context: str = "",
    log: bool = True,
    **kwargs: Any,
) -> SafeCallResult: ...


@overload
def safe_call(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    context: str = "",
    log: bool = True,
    **kwargs: Any,
) -> Awaitable[SafeCallResult]: ...


def safe_call(
    func: Union[Callable[..., T], Callable[..., Awaitable[T]]],
    *args: Any,
    context: str = "",
    log: bool = True,
    **kwargs: Any,
) -> Union[SafeCallResult, Awaitable[SafeCallResult]]:
    """Safely call a function; return a structured result instead of raising."""

    async def _async_impl() -> SafeCallResult:
        try:
            result = func(*args, **kwargs)  # type: ignore[misc]
            if asyncio.iscoroutine(result):
                result = await result  # type: ignore[assignment]
            return SafeCallResult(ok=True, result=result)
        except Exception as exc:
            if log and context:
                log_exception(context, exc)
            return SafeCallResult(ok=False, result=None, error=exc)

    result = func(*args, **kwargs)  # type: ignore[misc]
    if asyncio.iscoroutine(result):
        return _async_impl()

    try:
        return SafeCallResult(ok=True, result=result)
    except Exception as exc:  # pragma: no cover
        if log and context:
            log_exception(context, exc)
        return SafeCallResult(ok=False, result=None, error=exc)


def safe_playwright_goto(page: Any, url: str, *, timeout_ms: int = 15_000) -> Union[bool, Awaitable[bool]]:
    """Retry wrapper for Playwright's page.goto().

    Works for both sync and async Playwright.
    """

    return_result = run_with_retry(
        lambda: page.goto(url, wait_until="networkidle", timeout=timeout_ms),
        retries=3,
        initial_delay=1.0,
        backoff=2.0,
        jitter=0.25,
        context=f"goto {{url}}",
    )

    if asyncio.iscoroutine(return_result):
        async def _awaited() -> bool:
            ok, _ = await return_result  # type: ignore[misc]
            return ok
        return _awaited()

    ok, _ = return_result
    return ok


__all__ = [
    "SafeCallResult",
    "log_exception",
    "run_with_retry",
    "safe_call",
    "safe_playwright_goto",
]