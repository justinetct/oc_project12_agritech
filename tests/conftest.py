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


@pytest.fixture(scope="session", autouse=True)
def _preload_api_bundles():
    """Charge les artefacts ML une seule fois par session pytest et fait
    réutiliser ces objets par le lifespan de l'API.

    Le reste du lifespan (engine SQLite, session factory, Logfire,
    startup/shutdown) continue de s'exécuter normalement à chaque
    `TestClient(app)`. Les vrais loaders `load_bundle` et
    `load_recommend_context` restent testés directement dans
    `tests/agritech/test_serving.py`.
    """
    from agritech.api import main as api_main
    from agritech.config import PATHS
    from agritech.serving import load_bundle, load_recommend_context

    bundle_predict = load_bundle("predict")
    bundle_recommend = load_bundle("recommend")
    recommend_context = load_recommend_context(
        PATHS.root / "models" / "recommend_context.json"
    )

    original_load_bundle = api_main.load_bundle
    original_load_recommend_context = api_main.load_recommend_context

    def _cached_load_bundle(name: str):
        if name == "predict":
            return bundle_predict
        if name == "recommend":
            return bundle_recommend
        return original_load_bundle(name)

    def _cached_load_recommend_context(path: Path):
        return recommend_context

    api_main.load_bundle = _cached_load_bundle
    api_main.load_recommend_context = _cached_load_recommend_context
    try:
        yield
    finally:
        api_main.load_bundle = original_load_bundle
        api_main.load_recommend_context = original_load_recommend_context


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
