"""Tests unitaires de `agritech.monitoring.config`.

Chaque test isole complètement l'environnement via `monkeypatch` pour ne
dépendre ni d'un éventuel `.env` local, ni de l'ordre d'exécution.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agritech.monitoring.config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_ENVIRONMENT,
    DEFAULT_LOGFIRE_SERVICE_NAME,
    load_config,
)


MONITORED_ENV_VARS = (
    "DATABASE_URL",
    "ENVIRONMENT",
    "LOGFIRE_TOKEN",
    "LOGFIRE_ENVIRONMENT",
    "LOGFIRE_SERVICE_NAME",
    "MONITORING_API_TOKEN",
    "MONITORING_DEMO_HISTORY",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vide toutes les variables d'environnement lues par `load_config`.

    Sans cette fixture, un `.env` chargé par un autre outil ou une variable
    exportée dans le shell pollueraient les tests.
    """
    for name in MONITORED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_load_config_uses_documented_defaults_when_env_is_empty():
    """Sans aucune variable d'environnement, les défauts documentés s'appliquent."""
    config = load_config()

    assert config.database_url == DEFAULT_DATABASE_URL
    assert config.environment == DEFAULT_ENVIRONMENT
    assert config.logfire_token is None
    assert config.logfire_environment == DEFAULT_ENVIRONMENT
    assert config.logfire_service_name == DEFAULT_LOGFIRE_SERVICE_NAME
    assert config.api_token is None
    assert config.demo_history is False


def test_load_config_reads_all_environment_variables(monkeypatch: pytest.MonkeyPatch):
    """Chaque variable d'environnement renseignée surcharge le défaut."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:////tmp/custom.sqlite")
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("LOGFIRE_TOKEN", "secret-token")
    monkeypatch.setenv("LOGFIRE_ENVIRONMENT", "production")
    monkeypatch.setenv("LOGFIRE_SERVICE_NAME", "custom-service")
    monkeypatch.setenv("MONITORING_API_TOKEN", "monitoring-token")
    monkeypatch.setenv("MONITORING_DEMO_HISTORY", "true")

    config = load_config()

    assert config.database_url == "sqlite:////tmp/custom.sqlite"
    assert config.environment == "prod"
    assert config.logfire_token == "secret-token"
    assert config.logfire_environment == "production"
    assert config.logfire_service_name == "custom-service"
    assert config.api_token == "monitoring-token"
    assert config.demo_history is True


def test_load_config_logfire_environment_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    """`LOGFIRE_ENVIRONMENT` non renseigné reprend la valeur d'`ENVIRONMENT`."""
    monkeypatch.setenv("ENVIRONMENT", "staging")

    config = load_config()

    assert config.environment == "staging"
    assert config.logfire_environment == "staging"


def test_load_config_empty_logfire_token_is_none(monkeypatch: pytest.MonkeyPatch):
    """`LOGFIRE_TOKEN=""` (chaîne vide) est traité comme absence."""
    monkeypatch.setenv("LOGFIRE_TOKEN", "")

    config = load_config()

    assert config.logfire_token is None


def test_load_config_empty_monitoring_api_token_is_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """`MONITORING_API_TOKEN=""` (chaîne vide) est traité comme absence."""
    monkeypatch.setenv("MONITORING_API_TOKEN", "")

    config = load_config()

    assert config.api_token is None


def test_load_config_empty_string_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    """Une chaîne vide sur les autres variables retombe aussi sur les défauts."""
    for name in MONITORED_ENV_VARS:
        monkeypatch.setenv(name, "")

    config = load_config()

    assert config.database_url == DEFAULT_DATABASE_URL
    assert config.environment == DEFAULT_ENVIRONMENT
    assert config.logfire_token is None
    assert config.logfire_environment == DEFAULT_ENVIRONMENT
    assert config.logfire_service_name == DEFAULT_LOGFIRE_SERVICE_NAME
    assert config.api_token is None
    assert config.demo_history is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), (" 1 ", True), ("yes", True), ("on", True),
     ("false", False), ("0", False), ("no", False), ("oui", False)],
)
def test_load_config_demo_history_flag(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool):
    """`MONITORING_DEMO_HISTORY` s'active comme `NOTIFICATIONS_ENABLED` : 1, true, yes ou on."""
    monkeypatch.setenv("MONITORING_DEMO_HISTORY", value)

    assert load_config().demo_history is expected


def test_monitoring_config_is_frozen():
    """`MonitoringConfig` est immuable : toute modification lève `FrozenInstanceError`."""
    config = load_config()

    with pytest.raises(FrozenInstanceError):
        config.environment = "other"  # type: ignore[misc]
