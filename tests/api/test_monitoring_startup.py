"""Tests du démarrage de l'API côté monitoring : publication du token.

Le lifespan lit `MONITORING_API_TOKEN` via `MonitoringConfig` et le publie
dans `runtime.monitoring_api_token`, utilisé par les endpoints
`/monitoring/*`. Sans token, l'API démarre normalement et un `warning`
signale que ces endpoints seront indisponibles.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from agritech.api.core import runtime
from agritech.api.main import app


_WARNING_TEXT = "MONITORING_API_TOKEN is not set"


def test_monitoring_token_is_published_in_runtime(monkeypatch: pytest.MonkeyPatch):
    """Avec `MONITORING_API_TOKEN`, le lifespan publie le token puis le retire à l'arrêt."""
    monkeypatch.setenv("MONITORING_API_TOKEN", "test-monitoring-token")

    with TestClient(app):
        assert runtime.monitoring_api_token == "test-monitoring-token"

    assert runtime.monitoring_api_token is None


def test_missing_monitoring_token_logs_warning_and_api_still_starts(
    caplog: pytest.LogCaptureFixture,
):
    """Sans token, l'API démarre, `/health` répond et un warning est émis."""
    with caplog.at_level(logging.WARNING, logger="agritech.api.main"):
        with TestClient(app) as client:
            assert runtime.monitoring_api_token is None
            response = client.get("/health")

    assert response.status_code == 200
    assert any(_WARNING_TEXT in record.getMessage() for record in caplog.records)


def test_configured_monitoring_token_logs_no_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Avec un token, aucun warning n'est émis et le token n'apparaît pas dans les logs."""
    monkeypatch.setenv("MONITORING_API_TOKEN", "test-monitoring-token")

    with caplog.at_level(logging.DEBUG):
        with TestClient(app):
            pass

    messages = [record.getMessage() for record in caplog.records]
    assert not any(_WARNING_TEXT in message for message in messages)
    assert not any("test-monitoring-token" in message for message in messages)
