"""Fixtures partagées des tests de `agritech.monitoring`.

Toutes les fixtures sont montées sur `tmp_path` : aucun test ne peut
accidentellement écrire dans la base locale du projet
(`data/monitoring/api.sqlite`), même en cas de faute de frappe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from agritech.monitoring.config import MonitoringConfig
from agritech.monitoring.models import Base
from agritech.monitoring.session import (
    create_monitoring_engine,
    create_session_factory,
)


@pytest.fixture
def monitoring_config(tmp_path: Path) -> MonitoringConfig:
    """Config pointant vers une base SQLite jetable sous `tmp_path`."""
    return MonitoringConfig(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite'}",
        environment="test",
        logfire_token=None,
        logfire_environment="test",
        logfire_service_name="agritech-answers",
        api_token=None,
    )


@pytest.fixture
def engine(monitoring_config: MonitoringConfig) -> Iterator[Engine]:
    """Engine SQLAlchemy prêt à l'emploi avec la table `api_requests` créée."""
    engine = create_monitoring_engine(monitoring_config)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Session courte, refermée en fin de test."""
    factory = create_session_factory(engine)
    with factory() as sess:
        yield sess
