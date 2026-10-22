"""Regression for Docker's transient connection reset before API startup."""

import http.client
import importlib.util
from pathlib import Path
import urllib.error
import pytest

spec = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).parents[1] / "scripts/smoke_test_api.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_startup_transients_then_ready():
    outcomes = iter(
        [
            ConnectionResetError(),
            http.client.RemoteDisconnected(),
            urllib.error.HTTPError("url", 503, "starting", {}, None),
            {"status": "ready"},
        ]
    )

    def call(_):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    assert smoke.wait_for_ready(call, attempts=4, delay=0) == {"status": "ready"}


def test_permanent_startup_failure_is_not_hidden():
    def call(_):
        raise ConnectionResetError("not ready")

    with pytest.raises(ConnectionResetError):
        smoke.wait_for_ready(call, attempts=2, delay=0)


def test_wrong_readiness_route_fails_immediately():
    calls = []

    def call(_):
        calls.append(1)
        raise urllib.error.HTTPError("url", 404, "wrong endpoint", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        smoke.wait_for_ready(call, attempts=5, delay=0)
    assert len(calls) == 1
