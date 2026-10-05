"""Tests du démarrage de l'API côté monitoring : token et base SQLite.

Le lifespan lit `MONITORING_API_TOKEN` via `MonitoringConfig` et le publie
dans `runtime.monitoring_api_token`, utilisé par les endpoints
`/monitoring/*`. Sans token, l'API démarre normalement et un `warning`
signale que ces endpoints seront indisponibles.

La base SQLite de monitoring n'est jamais bloquante : si elle ne peut pas
être initialisée, l'API démarre quand même, `/predict`, `/recommend` et
`/health` fonctionnent, et `/monitoring/*` répond 503. Les ressources métier
(bundles, contexte `/recommend`) restent, elles, obligatoires.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select

from agritech.api import main as api_main
from agritech.api.core import runtime
from agritech.api.main import app
from agritech.monitoring.models import ApiRequest


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


# ===========================================================================
# Base SQLite de monitoring : initialisation non bloquante
# ===========================================================================

TOKEN = "test-monitoring-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PREDICT_PAYLOAD = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}
RECOMMEND_PAYLOAD = {"iso3": "FRA"}
MONITORING_PATHS = ("/monitoring/summary", "/monitoring/requests")
_UNAVAILABLE_TEXT = "monitoring database unavailable"


@pytest.fixture
def disposed_engines(monkeypatch: pytest.MonkeyPatch) -> list:
    """Garde la vraie création d'engine, mais note chaque appel à `dispose()`."""
    disposals = []
    real_create_engine = api_main.create_monitoring_engine

    def spying_create_engine(config):
        engine = real_create_engine(config)
        real_dispose = engine.dispose

        def dispose(*args, **kwargs):
            disposals.append(engine)
            real_dispose(*args, **kwargs)

        engine.dispose = dispose
        return engine

    monkeypatch.setattr(api_main, "create_monitoring_engine", spying_create_engine)
    return disposals


def _assert_business_endpoints_work(client: TestClient) -> None:
    """`/predict`, `/recommend` et `/health` répondent normalement."""
    assert client.post("/predict", json=PREDICT_PAYLOAD).status_code == 200
    assert client.post("/recommend", json=RECOMMEND_PAYLOAD).status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def _assert_monitoring_endpoints_unavailable(client: TestClient) -> None:
    """Auth valide → 503 ; mauvais token → toujours 401."""
    for path in MONITORING_PATHS:
        response = client.get(path, headers=AUTH)
        assert response.status_code == 503
        assert response.json()["error"] == "monitoring_unavailable"
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


def _unavailable_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if _UNAVAILABLE_TEXT in r.getMessage()]


def test_normal_startup_initialises_monitoring_database(
    monkeypatch: pytest.MonkeyPatch, disposed_engines: list
):
    """Succès : table créée, factory publiée, appels archivés, engine disposée une fois à l'arrêt."""
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)

    with TestClient(app) as client:
        factory = runtime.monitoring_session_factory
        assert factory is not None
        _assert_business_endpoints_work(client)
        with factory() as session:
            assert "api_requests" in inspect(session.get_bind()).get_table_names()
            assert session.scalar(select(func.count()).select_from(ApiRequest)) == 2
        assert client.get("/monitoring/requests", headers=AUTH).status_code == 200
        assert disposed_engines == []

    assert len(disposed_engines) == 1
    assert runtime.monitoring_session_factory is None
    assert runtime.monitoring_api_version is None
    assert runtime.monitoring_environment is None
    assert runtime.monitoring_api_token is None


def test_engine_creation_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Échec avant l'engine : l'API démarre, sert les métiers, `/monitoring/*` → 503."""
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)

    def failing_create_engine(config):
        raise RuntimeError("cannot create /private/monitoring/api.sqlite")

    logfire_calls = []
    monkeypatch.setattr(api_main, "create_monitoring_engine", failing_create_engine)
    monkeypatch.setattr(api_main, "configure_logfire", logfire_calls.append)

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            assert runtime.monitoring_session_factory is None
            _assert_business_endpoints_work(client)
            _assert_monitoring_endpoints_unavailable(client)

    assert _unavailable_warnings(caplog) == [
        "monitoring database unavailable (engine creation failed: RuntimeError); API "
        "continues without request archiving and /monitoring endpoints will return 503"
    ]
    # Le middleware n'a rien tenté d'écrire : aucun warning de persistance.
    assert "monitoring persistence failed" not in caplog.text
    # Ni chemin, ni secret dans les logs.
    assert "/private/monitoring" not in caplog.text
    assert TOKEN not in caplog.text
    # Logfire reste configuré, indépendamment de SQLite.
    assert len(logfire_calls) == 1
    assert runtime.monitoring_session_factory is None


def test_table_setup_failure_disposes_engine_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    disposed_engines: list,
    tmp_path,
):
    """Échec après l'engine (ici un dossier à la place du fichier SQLite) : engine disposée une fois."""
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}")

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            assert runtime.monitoring_session_factory is None
            assert len(disposed_engines) == 1
            _assert_business_endpoints_work(client)
            _assert_monitoring_endpoints_unavailable(client)

    assert len(disposed_engines) == 1
    assert _unavailable_warnings(caplog) == [
        "monitoring database unavailable (database setup failed: OperationalError); API "
        "continues without request archiving and /monitoring endpoints will return 503"
    ]
    assert str(tmp_path) not in caplog.text
    assert runtime.monitoring_session_factory is None


def test_session_factory_failure_disposes_engine_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    disposed_engines: list,
):
    """Échec de la session factory après la création de la table : même nettoyage, rien de publié."""
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)

    def failing_session_factory(engine):
        raise RuntimeError("session factory failure")

    monkeypatch.setattr(api_main, "create_session_factory", failing_session_factory)

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            assert runtime.monitoring_session_factory is None
            _assert_business_endpoints_work(client)
            _assert_monitoring_endpoints_unavailable(client)

    assert len(disposed_engines) == 1
    assert len(_unavailable_warnings(caplog)) == 1
    assert runtime.monitoring_session_factory is None


def test_previous_session_factory_is_not_kept_after_failed_startup(
    monkeypatch: pytest.MonkeyPatch,
):
    """Une factory laissée dans `runtime` n'est jamais réutilisée si SQLite échoue au démarrage."""

    def failing_create_engine(config):
        raise RuntimeError("engine failure")

    monkeypatch.setattr(runtime, "monitoring_session_factory", object())
    monkeypatch.setattr(api_main, "create_monitoring_engine", failing_create_engine)

    with TestClient(app):
        assert runtime.monitoring_session_factory is None


@pytest.mark.parametrize("loader_name", ["load_bundle", "load_recommend_context"])
def test_business_resource_failure_still_blocks_startup(
    monkeypatch: pytest.MonkeyPatch, loader_name: str
):
    """Un bundle ou le contexte `/recommend` introuvable empêche toujours l'API de démarrer."""

    def failing_loader(*args, **kwargs):
        raise FileNotFoundError("business artefact missing")

    monkeypatch.setattr(api_main, loader_name, failing_loader)

    with pytest.raises(FileNotFoundError, match="business artefact missing"):
        with TestClient(app):
            pass


# ===========================================================================
# Historique de démonstration : MONITORING_DEMO_HISTORY
# ===========================================================================


def _stored_calls() -> int:
    with runtime.monitoring_session_factory() as session:
        return session.scalar(select(func.count()).select_from(ApiRequest))


def test_demo_history_is_off_by_default():
    """Sans `MONITORING_DEMO_HISTORY` (Docker Compose local, tests) : la base reste vide."""
    with TestClient(app):
        assert _stored_calls() == 0


def test_demo_history_fills_an_empty_database_at_startup(monkeypatch: pytest.MonkeyPatch):
    """Base vide : historique créé au démarrage, puis les vraies requêtes s'y ajoutent."""
    monkeypatch.setenv("MONITORING_DEMO_HISTORY", "true")
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)

    with TestClient(app) as client:
        seeded = _stored_calls()
        totals = {
            days: client.get("/monitoring/summary", params={"days": days}, headers=AUTH).json()["total_requests"]
            for days in (7, 30, 90)
        }
        assert client.post("/predict", json=PREDICT_PAYLOAD).status_code == 200
        assert _stored_calls() == seeded + 1

    assert 0 < totals[7] < totals[30] < totals[90] <= seeded


def test_demo_history_keeps_an_existing_database(monkeypatch: pytest.MonkeyPatch):
    """Redémarrage sur une base non vide : rien n'est ajouté ni supprimé."""
    monkeypatch.setenv("MONITORING_DEMO_HISTORY", "true")
    with TestClient(app) as client:
        client.post("/predict", json=PREDICT_PAYLOAD)
        before = _stored_calls()

    with TestClient(app):
        assert _stored_calls() == before


def test_demo_history_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Échec du seed : un warning (type d'erreur seulement), puis un démarrage normal."""
    monkeypatch.setenv("MONITORING_DEMO_HISTORY", "true")
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)

    def failing_seed(session_factory, environment):
        raise RuntimeError("cannot read /private/models")

    monkeypatch.setattr(api_main, "seed_demo_history", failing_seed)

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            _assert_business_endpoints_work(client)
            assert client.get("/monitoring/requests", headers=AUTH).status_code == 200

    assert "monitoring demo history skipped (RuntimeError); API continues" in caplog.text
    assert "/private/models" not in caplog.text
