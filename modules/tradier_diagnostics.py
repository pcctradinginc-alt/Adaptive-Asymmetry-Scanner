"""Typed, cost-free diagnostics for Tradier option-data requests.

No retries, additional API calls, price backfills, or trading decisions.
Callers fail closed (no contract returned) but log a stable diagnostic code.
Never log provider response bodies, credentials, or Authorization headers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import requests

_FAILURES: Counter[str] = Counter()
_QUOTA_MARKERS = ("quota", "rate limit", "rate-limit", "too many request",
                  "requests per minute", "request limit", "usage limit",
                  "limit exceeded", "exceeded limit")


@dataclass(frozen=True)
class ProviderDataError(RuntimeError):
    category: str
    endpoint: str

    def __str__(self) -> str:
        return f"{self.endpoint}: {self.category}"


def _text(response: Any) -> str:
    """Only inspect a bounded response body; never return/log raw content."""
    try:
        body = getattr(response, "text", "")
        if not isinstance(body, str):
            return ""
        return body[:2048].lower()
    except (AttributeError, TypeError, ValueError):
        return ""


def status_category(response: Any) -> str | None:
    """HTTP 400 can be a provider quota error, not a missing option chain."""
    try:
        status = int(response.status_code)
    except (AttributeError, TypeError, ValueError):
        return None  # Let requests.raise_for_status handle incomplete stubs.
    if 200 <= status < 300:
        return None
    body = _text(response)
    if status == 429 or (status in (400, 403) and
                         any(marker in body for marker in _QUOTA_MARKERS)):
        return "QUOTA_EXHAUSTED"
    if status >= 500:
        return "PROVIDER_DOWN"
    if status in (401, 403):
        return "AUTH_OR_PERMISSION"
    return "HTTP_ERROR"


def check_response(response: Any, endpoint: str) -> None:
    category = status_category(response)
    if category:
        raise ProviderDataError(category, endpoint)
    response.raise_for_status()


def _embedded_error(data: dict) -> str | None:
    """Some APIs return HTTP 200 with structured API-error JSON."""
    error = data.get("errors") or data.get("error") or data.get("fault")
    if not error:
        return None
    message = str(error).lower()[:1024]
    return ("QUOTA_EXHAUSTED" if any(s in message for s in _QUOTA_MARKERS)
            else "PROVIDER_DOWN")


def expirations(data: Any) -> list[str]:
    if not isinstance(data, dict):
        raise ProviderDataError("SCHEMA_CHANGED", "expirations")
    error = _embedded_error(data)
    if error:
        raise ProviderDataError(error, "expirations")
    if "expirations" not in data:
        raise ProviderDataError("SCHEMA_CHANGED", "expirations")
    value = data["expirations"]
    if value is None:  # documented empty results
        return []
    if not isinstance(value, dict) or "date" not in value:
        raise ProviderDataError("SCHEMA_CHANGED", "expirations")
    dates = value["date"]
    if dates is None:
        return []
    if isinstance(dates, str):
        dates = [dates]
    if not isinstance(dates, list) or any(not isinstance(d, str) or not d for d in dates):
        raise ProviderDataError("SCHEMA_CHANGED", "expirations")
    return sorted(dates)


def chain(data: Any) -> list[dict]:
    if not isinstance(data, dict):
        raise ProviderDataError("SCHEMA_CHANGED", "chains")
    error = _embedded_error(data)
    if error:
        raise ProviderDataError(error, "chains")
    if "options" not in data:
        raise ProviderDataError("SCHEMA_CHANGED", "chains")
    value = data["options"]
    if value is None:
        return []
    if not isinstance(value, dict) or "option" not in value:
        raise ProviderDataError("SCHEMA_CHANGED", "chains")
    options = value["option"]
    if options is None:
        return []
    if isinstance(options, dict):
        options = [options] if options else []
    if not isinstance(options, list) or any(not isinstance(o, dict) for o in options):
        raise ProviderDataError("SCHEMA_CHANGED", "chains")
    return options


def error_category(exc: Exception) -> str:
    if isinstance(exc, ProviderDataError):
        return exc.category
    if isinstance(exc, requests.exceptions.Timeout):
        return "TIMEOUT"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "PROVIDER_DOWN"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "SCHEMA_CHANGED"
    if isinstance(exc, requests.exceptions.HTTPError):
        return status_category(exc.response) or "HTTP_ERROR"
    return "PROVIDER_ERROR"


def record(endpoint: str, category: str) -> None:
    """Process-local count for incident diagnostics; no disk/API side effects."""
    _FAILURES[f"{endpoint}:{category}"] += 1


def counters() -> dict[str, int]:
    return dict(_FAILURES)
