"""Tests unitaires de `configure_logfire`.

Aucun test ne fait d'appel réseau réel : `logfire.configure` est mocké via
`monkeypatch.setattr`, ce qui permet de vérifier les arguments passés sans
dépendre d'un token réel.
"""

from __future__ import annotations

import logging

import logfire
import pytest

from agritech.monitoring.config import MonitoringConfig
from agritech.observability.logfire_setup import configure_logfire


def _config(**overrides) -> MonitoringConfig:
    """Construit un `MonitoringConfig` de test avec token vide par défaut."""
    base = {
        "database_url": "sqlite:///:memory:",
        "environment": "test",
        "logfire_token": None,
        "logfire_environment": "test",
        "logfire_service_name": "agritech-answers",
    }
    base.update(overrides)
    return MonitoringConfig(**base)


def test_configure_logfire_without_token_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Sans token, `logfire.configure` n'est pas appelée et aucun log émis."""
    configure_calls = []
    monkeypatch.setattr(
        logfire, "configure", lambda **kwargs: configure_calls.append(kwargs)
    )

    with caplog.at_level(logging.WARNING):
        configure_logfire(_config(logfire_token=None))

    assert configure_calls == []
    assert caplog.records == []


def test_configure_logfire_empty_token_is_silent(monkeypatch: pytest.MonkeyPatch):
    """Un token vide (chaîne '') se comporte comme absence — pas de configuration."""
    called = []
    monkeypatch.setattr(logfire, "configure", lambda **kwargs: called.append(kwargs))

    configure_logfire(_config(logfire_token=""))

    assert called == []


def test_configure_logfire_with_token_calls_configure_with_expected_args(
    monkeypatch: pytest.MonkeyPatch,
):
    """Avec un token, `configure` reçoit le service, l'environnement et le token."""
    configure_kwargs = {}

    def _fake_configure(**kwargs):
        configure_kwargs.update(kwargs)

    monkeypatch.setattr(logfire, "configure", _fake_configure)

    configure_logfire(
        _config(
            logfire_token="fake-test-token",
            logfire_environment="prod",
            logfire_service_name="agritech-answers",
        )
    )

    assert configure_kwargs["service_name"] == "agritech-answers"
    assert configure_kwargs["environment"] == "prod"
    assert configure_kwargs["send_to_logfire"] == "if-token-present"
    # Console silencieuse : pas de bruit sur stdout, ni en dev ni en tests.
    assert configure_kwargs["console"] is False
    # Le token est passé à logfire.configure ; il n'est jamais recopié ailleurs.
    assert configure_kwargs["token"] == "fake-test-token"


def test_configure_logfire_swallows_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Une erreur de `logfire.configure` est loggée en warning, sans lever."""

    def _broken_configure(**kwargs):
        raise RuntimeError("simulated logfire failure")

    monkeypatch.setattr(logfire, "configure", _broken_configure)

    with caplog.at_level(logging.WARNING):
        configure_logfire(_config(logfire_token="fake-token"))  # ne doit pas lever

    assert any(
        "logfire configuration failed" in record.message for record in caplog.records
    )
