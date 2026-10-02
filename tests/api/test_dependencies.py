"""Tests des dépendances `/monitoring/*` : token Bearer et session monitoring.

Aucun endpoint `/monitoring/*` n'existe encore : une petite application
FastAPI locale à ce fichier déclare des routes de test qui utilisent les
vraies dépendances. Elle reprend les handlers d'erreur enregistrés sur
l'application réelle, ce qui vérifie aussi leur enregistrement dans `main.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agritech.api.core import runtime
from agritech.api.dependencies import get_monitoring_session, require_monitoring_token
from agritech.api.main import app as main_app
from agritech.monitoring.config import MonitoringConfig
from agritech.monitoring.models import ApiRequest, Base
from agritech.monitoring.session import create_monitoring_engine, create_session_factory


SECRET = "test-monitoring-secret"


# --- Application de test ---------------------------------------------------

dependency_app = FastAPI()
for exception_class, handler in main_app.exception_handlers.items():
    dependency_app.add_exception_handler(exception_class, handler)


@dependency_app.get("/_test/auth", dependencies=[Depends(require_monitoring_token)])
def protected_route() -> dict:
    return {"access": "granted"}


@dependency_app.get("/_test/session")
def session_route(session: Session = Depends(get_monitoring_session)) -> dict:
    count = session.scalar(select(func.count()).select_from(ApiRequest))
    return {"session": type(session).__name__, "api_requests": count}


@dependency_app.get("/_test/protected-session", dependencies=[Depends(require_monitoring_token)])
def protected_session_route(session: Session = Depends(get_monitoring_session)) -> dict:
    count = session.scalar(select(func.count()).select_from(ApiRequest))
    return {"api_requests": count}


@dependency_app.get("/_test/session-error")
def failing_session_route(session: Session = Depends(get_monitoring_session)) -> dict:
    raise RuntimeError("endpoint failure")


# --- Session factory réelle, espionnée ---------------------------------------


class _SessionFactorySpy:
    """Enveloppe une vraie session factory et compte ouvertures et fermetures."""

    def __init__(self, factory):
        self._factory = factory
        self.opened = 0
        self.closed = 0

    def __call__(self) -> Session:
        session = self._factory()
        self.opened += 1
        original_close = session.close

        def close() -> None:
            self.closed += 1
            original_close()

        session.close = close
        return session


@pytest.fixture
def session_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Base SQLite jetable, publiée dans `runtime` via une factory espionnée."""
    config = MonitoringConfig(
        database_url=f"sqlite:///{tmp_path / 'dependencies.sqlite'}",
        environment="test",
        logfire_token=None,
        logfire_environment="test",
        logfire_service_name="agritech-answers",
        api_token=None,
    )
    engine = create_monitoring_engine(config)
    Base.metadata.create_all(engine)
    spy = _SessionFactorySpy(create_session_factory(engine))
    monkeypatch.setattr(runtime, "monitoring_session_factory", spy)
    yield spy
    engine.dispose()


@pytest.fixture
def configured_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Publie le token de test côté serveur."""
    monkeypatch.setattr(runtime, "monitoring_api_token", SECRET)
    return SECRET


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _assert_unauthorized(response) -> None:
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json() == {
        "error": "unauthorized",
        "message": "Authentication required.",
        "details": None,
    }


# --- Authentification -------------------------------------------------------


def test_missing_server_token_returns_503(monkeypatch: pytest.MonkeyPatch):
    """Sans token serveur, l'accès est refusé en 503 générique, même avec un en-tête."""
    monkeypatch.setattr(runtime, "monitoring_api_token", None)

    response = TestClient(dependency_app).get("/_test/auth", headers=_bearer(SECRET))

    assert response.status_code == 503
    assert response.json() == {
        "error": "monitoring_unavailable",
        "message": "Monitoring is unavailable.",
        "details": None,
    }
    assert "token" not in response.text.lower()


def test_missing_authorization_header_returns_401(configured_token: str):
    """En-tête `Authorization` absent → 401 + `WWW-Authenticate: Bearer`."""
    _assert_unauthorized(TestClient(dependency_app).get("/_test/auth"))


def test_wrong_scheme_returns_401(configured_token: str):
    """Schéma autre que `Bearer` (ici `Basic`) → 401, même avec le bon secret."""
    response = TestClient(dependency_app).get(
        "/_test/auth", headers={"Authorization": f"Basic {configured_token}"}
    )

    _assert_unauthorized(response)


def test_empty_bearer_token_returns_401(configured_token: str):
    """`Authorization: Bearer ` sans token → 401."""
    response = TestClient(dependency_app).get("/_test/auth", headers={"Authorization": "Bearer "})

    _assert_unauthorized(response)


def test_wrong_token_returns_401(configured_token: str):
    """Token incorrect → 401."""
    _assert_unauthorized(TestClient(dependency_app).get("/_test/auth", headers=_bearer("wrong-token")))


def test_correct_token_grants_access(configured_token: str):
    """Bon token → la route protégée répond 200."""
    response = TestClient(dependency_app).get("/_test/auth", headers=_bearer(configured_token))

    assert response.status_code == 200
    assert response.json() == {"access": "granted"}


def test_tokens_never_appear_in_responses_or_logs(
    configured_token: str, caplog: pytest.LogCaptureFixture
):
    """Ni le secret serveur, ni le token reçu n'apparaissent dans les réponses ou les logs."""
    client = TestClient(dependency_app)
    with caplog.at_level(logging.DEBUG):
        responses = [
            client.get("/_test/auth", headers=_bearer("received-wrong-token")),
            client.get("/_test/auth", headers=_bearer(configured_token)),
            client.get("/_test/auth"),
        ]

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    for response in responses:
        exposed = response.text + str(dict(response.headers))
        assert SECRET not in exposed
        assert "received-wrong-token" not in exposed
    assert SECRET not in log_text
    assert "received-wrong-token" not in log_text


# --- Session monitoring -----------------------------------------------------


def test_missing_session_factory_returns_503(monkeypatch: pytest.MonkeyPatch):
    """Sans session factory, la dépendance renvoie un 503 générique."""
    monkeypatch.setattr(runtime, "monitoring_session_factory", None)

    response = TestClient(dependency_app).get("/_test/session")

    assert response.status_code == 503
    assert response.json()["error"] == "monitoring_unavailable"
    assert "sqlite" not in response.text.lower()


def test_session_is_provided_then_closed(session_factory: _SessionFactorySpy):
    """La route reçoit une vraie session, fermée une fois la requête terminée."""
    response = TestClient(dependency_app).get("/_test/session")

    assert response.status_code == 200
    assert response.json() == {"session": "Session", "api_requests": 0}
    assert session_factory.opened == 1
    assert session_factory.closed == 1


def test_session_is_closed_when_endpoint_fails(session_factory: _SessionFactorySpy):
    """Même si l'endpoint lève une exception, la session est fermée."""
    response = TestClient(dependency_app, raise_server_exceptions=False).get("/_test/session-error")

    assert response.status_code == 500
    assert session_factory.opened == 1
    assert session_factory.closed == 1


# --- Isolation --------------------------------------------------------------


def test_auth_failure_does_not_open_a_session(
    configured_token: str, session_factory: _SessionFactorySpy
):
    """Auth puis session : un appel refusé en 401 n'ouvre aucune session."""
    response = TestClient(dependency_app).get("/_test/protected-session", headers=_bearer("wrong"))

    assert response.status_code == 401
    assert session_factory.opened == 0


def test_missing_server_token_does_not_open_a_session(
    monkeypatch: pytest.MonkeyPatch, session_factory: _SessionFactorySpy
):
    """Token serveur absent : 503 sans ouvrir de session."""
    monkeypatch.setattr(runtime, "monitoring_api_token", None)

    response = TestClient(dependency_app).get("/_test/protected-session", headers=_bearer(SECRET))

    assert response.status_code == 503
    assert session_factory.opened == 0


def test_dependencies_do_not_write_to_api_requests(
    configured_token: str, session_factory: _SessionFactorySpy
):
    """Après plusieurs appels autorisés ou refusés, la table reste vide."""
    client = TestClient(dependency_app)
    client.get("/_test/protected-session", headers=_bearer(configured_token))
    client.get("/_test/protected-session", headers=_bearer("wrong"))
    client.get("/_test/session")

    response = client.get("/_test/protected-session", headers=_bearer(configured_token))

    assert response.json() == {"api_requests": 0}
    assert session_factory.opened == session_factory.closed == 3
