"""Shared graph-memory client, request limits, and retry policy.

The knowledge graph used to live in Zep Cloud. It is now self-hosted
(Graphiti + Neo4j, see ``graph_memory.py``); the ``zep`` names are kept so the
services that import these helpers did not need to change.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

import httpx

from ..config import Config
from .graph_memory import GraphMemoryClient, GraphMemoryError, NotFoundError
from .logger import get_logger

logger = get_logger("mirofish.zep")

T = TypeVar("T")

ZepApiError = GraphMemoryError

ZEP_HTTP_REQUEST_TIMEOUT_SECONDS = 60.0
# Self-hosted extraction runs one LLM pipeline per chunk, so a large document
# can take a long time. Configurable through GRAPH_INGESTION_TIMEOUT_SECONDS.
ZEP_INGESTION_WAIT_TIMEOUT_SECONDS = Config.GRAPH_INGESTION_TIMEOUT_SECONDS
MAX_ZEP_SEARCH_QUERY_CHARS = 400
MAX_ZEP_SEARCH_RESULTS = 50


def normalize_zep_search_query(query: Any) -> str:
    """Return a non-empty, bounded search query."""

    if not isinstance(query, str):
        raise ValueError("Graph search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Graph search query must not be empty")
    return normalized[:MAX_ZEP_SEARCH_QUERY_CHARS]


def normalize_zep_search_limit(limit: Any) -> int:
    """Clamp a search result limit."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Graph search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Graph search limit must be at least 1")
    return min(normalized, MAX_ZEP_SEARCH_RESULTS)


@lru_cache(maxsize=4)
def _cached_zep_client(timeout: float) -> GraphMemoryClient:
    return GraphMemoryClient(timeout=timeout)


def get_zep_client(api_key: str | None = None, timeout: float | None = None) -> GraphMemoryClient:
    """Return the process-shared graph-memory client.

    ``api_key`` is accepted for backwards compatibility and ignored; Neo4j
    credentials come from ``NEO4J_*`` settings.
    """

    request_timeout = float(
        timeout if timeout is not None else ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
    )
    if request_timeout <= 0:
        raise ValueError("Graph request timeout must be greater than 0")
    return _cached_zep_client(request_timeout)


def clear_zep_client_cache() -> None:
    """Clear cached clients. Intended for tests and controlled reconfiguration."""

    _cached_zep_client.cache_clear()


def is_retryable_zep_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry."""

    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    try:
        from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
    except ImportError:  # pragma: no cover
        pass
    else:
        if isinstance(error, (ServiceUnavailable, SessionExpired, TransientError)):
            return True
    if isinstance(error, ZepApiError):
        status_code = error.status_code
        return status_code in {408, 429} or (
            status_code is not None and 500 <= status_code <= 599
        )
    return False


def _retry_after_seconds(error: BaseException) -> float | None:
    if not isinstance(error, ZepApiError) or not error.headers:
        return None
    value = next(
        (
            header_value
            for header_name, header_value in error.headers.items()
            if header_name.lower() == "retry-after"
        ),
        None,
    )
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def call_zep_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe graph read only for transport, 408, 429, or 5xx errors."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_zep_error(error):
                raise

            retry_after = _retry_after_seconds(error)
            delay = min(
                retry_after if retry_after is not None else initial_delay * (2 ** (attempt - 1)),
                max_delay,
            )
            logger.warning(
                "Zep %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")
