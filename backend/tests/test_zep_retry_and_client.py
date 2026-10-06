from pathlib import Path
from types import SimpleNamespace

import pytest

from app.utils import zep
from app.utils.zep import ZepApiError


def test_permanent_zep_errors_fail_without_retry():
    calls = []

    def operation():
        calls.append(True)
        raise ZepApiError(status_code=400, body={"message": "bad query"})

    with pytest.raises(ZepApiError):
        zep.call_zep_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_rate_limit_retry_respects_retry_after():
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise ZepApiError(
                status_code=429,
                headers={"Retry-After": "7"},
                body={"message": "slow down"},
            )
        return "ok"

    result = zep.call_zep_read_with_retry(
        operation,
        operation_name="rate limited read",
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_graph_client_is_shared_per_timeout(monkeypatch):
    created = []

    def fake_client(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(zep, "GraphMemoryClient", fake_client)
    zep.clear_zep_client_cache()

    first = zep.get_zep_client(timeout=12)
    second = zep.get_zep_client("ignored-legacy-key", timeout=12)

    assert first is second
    assert created == [{"timeout": 12.0}]
    zep.clear_zep_client_cache()


def test_graph_client_rejects_non_positive_timeout():
    with pytest.raises(ValueError):
        zep.get_zep_client(timeout=0)


def test_neo4j_transient_errors_are_retryable():
    from neo4j.exceptions import ServiceUnavailable

    assert zep.is_retryable_zep_error(ServiceUnavailable("down"))
    assert not zep.is_retryable_zep_error(ZepApiError("bad", status_code=400))
