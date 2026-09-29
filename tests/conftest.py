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
    """Redirige la base monitoring vers `tmp_path`, fige l'environnement de test.

    Neutralise aussi toute variable Logfire éventuellement présente dans un
    `.env` local : les tests automatisés ne doivent JAMAIS déclencher un
    envoi réseau réel vers Logfire. Les tests qui veulent vérifier le
    branchement Logfire réactivent explicitement un token simulé.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / '_monitoring.sqlite'}")
    monkeypatch.setenv("ENVIRONMENT", "test")
    # Utilise `setenv` avec une chaîne vide plutôt que `delenv` : sinon
    # `load_dotenv(override=False)` du lifespan re-injecterait la valeur du
    # `.env` local (le token réel de l'utilisateur) dans l'environnement.
    # Une chaîne vide est traitée comme absence par `load_config()`.
    monkeypatch.setenv("LOGFIRE_TOKEN", "")
    monkeypatch.setenv("LOGFIRE_ENVIRONMENT", "")
    monkeypatch.setenv("LOGFIRE_SERVICE_NAME", "")
