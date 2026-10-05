"""Tests de ``agritech.ui.settings``."""

from __future__ import annotations

import pytest

from agritech.ui import settings


def test_default_api_url_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRITECH_API_URL", raising=False)
    assert settings.api_base_url() == "http://localhost:8000"


def test_default_api_url_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRITECH_API_URL", "")
    assert settings.api_base_url() == "http://localhost:8000"


def test_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRITECH_API_URL", "http://api.example.test")
    assert settings.api_base_url() == "http://api.example.test"


def test_env_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRITECH_API_URL", "  http://api.example.test  ")
    assert settings.api_base_url() == "http://api.example.test"


def test_timeout_default() -> None:
    assert settings.request_timeout_seconds() == 5.0


# --- token de monitoring ----------------------------------------------------

def test_monitoring_token_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONITORING_API_TOKEN", "test-token")
    assert settings.monitoring_api_token() == "test-token"


def test_monitoring_token_missing_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONITORING_API_TOKEN", raising=False)
    assert settings.monitoring_api_token() is None


def test_monitoring_token_empty_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONITORING_API_TOKEN", "")
    assert settings.monitoring_api_token() is None


def test_monitoring_token_only_spaces_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONITORING_API_TOKEN", "   ")
    assert settings.monitoring_api_token() is None


def test_monitoring_token_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONITORING_API_TOKEN", "  test-token  ")
    assert settings.monitoring_api_token() == "test-token"
