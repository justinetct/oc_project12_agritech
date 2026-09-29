"""Fixtures racine partagées par toute la suite de tests.

Le lifespan FastAPI ouvre désormais une base SQLite de monitoring au
démarrage. Sans isolation, chaque `TestClient(app)` écrirait dans la base
locale par défaut (`data/monitoring/api.sqlite`). Cette fixture autouse
redirige `DATABASE_URL` et `ENVIRONMENT` vers un dossier `tmp_path` propre
pour chaque test — y compris pour les tests non-API qui n'utilisent pas
`TestClient` (elle est alors sans effet).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_monitoring_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirige la base monitoring vers `tmp_path` et fige `ENVIRONMENT=test`."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / '_monitoring.sqlite'}")
    monkeypatch.setenv("ENVIRONMENT", "test")
