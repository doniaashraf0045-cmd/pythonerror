"""
Error-handling helpers for Playwright-based agents.
- run_with_retry: retry wrapper with exponential backoff
- safe_call: wrapper which catches/records exceptions, returns (ok, result, error)
- log_exception: writes stacktrace to errors.log and agent_journal.jsonl
"""

import time
import traceback
import json
from typing import Callable, Any, Tuple, Optional

ERROR_LOG = "errors.log"
JOURNAL = "agent_journal.jsonl"


def log_exception(context: str, exc: Exception):
    tb = traceback.format_exc()
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "context": context,
             "error": str(exc),
             "traceback": tb}
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": entry["time"], "ctx": context, "err": str(exc)}) + "\n")


def run_with_retry(func: Callable, *args, retries: int = 3, backoff: float = 2.0, context: str = "", **kwargs) -> Tuple[bool, Optional[Any]]:
    attempt = 0
    delay = 1.0
    while attempt < retries:
        try:
            result = func(*args, **kwargs)
            return True, result
        except Exception as e:
            attempt += 1
            log_exception(f"{context} attempt={attempt}", e)
            if attempt >= retries:
                return False, None
            time.sleep(delay)
            delay *= backoff
    return False, None


def safe_call(func: Callable, *args, context: str = "", **kwargs) -> Tuple[bool, Optional[Any], Optional[Exception]]:
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        log_exception(context, e)
        return False, None, e


def safe_playwright_goto(page, url: str, timeout: int = 15000) -> bool:
    ok, _ = run_with_retry(lambda: page.goto(url, wait_until="networkidle", timeout=timeout), retries=3, backoff=2.0, context=f"goto {url}")
    return ok


# Example usage in existing agent:
# from python_error_helpers import run_with_retry, log_exception
# ok, _ = run_with_retry(page.click, 'button:has-text("Add")', retries=2, context="click add button")
# if not ok:
#     log_exception("click add", Exception("click failed after retries"))
