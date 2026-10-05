"""Point d'entrée de l'API Agritech Answers.

Ce module expose :

- une instance `FastAPI` avec titre, description et version ;
- un `lifespan` qui charge le modèle `/predict` au démarrage ;
- l'endpoint `GET /health` qui confirme que l'API répond et renseigne l'état
  du modèle chargé.

La version de l'API est déclarée en dur ci-dessous. Elle doit être maintenue
alignée avec `[project] version` de `pyproject.toml` lors d'un bump. La
version du modèle est lue depuis `bundle.metadata`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from agritech.api.core import runtime
from agritech.api.error_handlers import (
    model_unavailable_handler,
    monitoring_unauthorized_handler,
    monitoring_unavailable_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from agritech.api.exceptions import (
    ModelUnavailableError,
    MonitoringUnauthorizedError,
    MonitoringUnavailableError,
)
from agritech.api.middleware.request_logger import RequestLoggerMiddleware
from agritech.api.routers.monitoring import router as monitoring_router
from agritech.api.routers.predict import router as predict_router
from agritech.api.routers.recommend import router as recommend_router
from agritech.config import PATHS
from agritech.monitoring.config import load_config
from agritech.monitoring.models import Base as MonitoringBase
from agritech.monitoring.session import (
    create_monitoring_engine,
    create_session_factory,
)
from agritech.observability.logfire_setup import configure_logfire
from agritech.serving import load_bundle, load_recommend_context


API_VERSION = "1.0.0"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prépare l'état applicatif avant de servir la première requête.

    - Charge les deux bundles et le contexte `/recommend` (comportement existant).
    - Charge un éventuel `.env` local, lit la configuration monitoring, crée
      l'engine SQLite (PRAGMAs WAL appliqués), matérialise la table
      `api_requests` au premier boot, puis publie la session factory et les
      métadonnées runtime pour le middleware de persistance.
    - Publie le token des endpoints `/monitoring/*` ; s'il est absent, un
      `warning` signale au démarrage que ces endpoints seront indisponibles.

    Si un chargement échoue (`FileNotFoundError`, `ValueError`, ...), l'exception
    remonte et l'application ne démarre pas. La couche HTTP traduira
    séparément une indisponibilité en 503 via `model_unavailable_handler`.
    """
    runtime.bundle_predict = load_bundle("predict")
    runtime.bundle_recommend = load_bundle("recommend")
    runtime.recommend_context = load_recommend_context(
        PATHS.root / "models" / "recommend_context.json"
    )

    # Sans .env, `load_dotenv()` est un no-op ; il ne surcharge pas les
    # variables déjà présentes dans l'environnement (tests via monkeypatch,
    # variables Docker Compose en prod).
    load_dotenv(override=False)
    monitoring_config = load_config()
    monitoring_engine = create_monitoring_engine(monitoring_config)
    MonitoringBase.metadata.create_all(monitoring_engine)
    runtime.monitoring_session_factory = create_session_factory(monitoring_engine)
    runtime.monitoring_api_version = API_VERSION
    runtime.monitoring_environment = monitoring_config.environment
    runtime.monitoring_api_token = monitoring_config.api_token
    if monitoring_config.api_token is None:
        logger.warning(
            "MONITORING_API_TOKEN is not set; /monitoring endpoints will be unavailable"
        )

    # Logfire est strictement optionnel : sans token dans `MonitoringConfig`,
    # cet appel est un no-op silencieux. Une erreur d'initialisation est
    # loggée en `warning` et n'empêche pas l'API de démarrer.
    configure_logfire(monitoring_config)

    try:
        yield
    finally:
        monitoring_engine.dispose()
        runtime.monitoring_session_factory = None
        runtime.monitoring_api_version = None
        runtime.monitoring_environment = None
        runtime.monitoring_api_token = None


app = FastAPI(
    title="Agritech Answers API",
    description=(
        "API de prédiction agricole. Deux services :\n\n"
        "- `POST /predict` : estimation de rendement pour une parcelle ;\n"
        "- `POST /recommend` : classement des cultures pour un contexte donné.\n\n"
        "Les endpoints `GET /monitoring/*`, protégés par un token Bearer, exposent "
        "en lecture seule le suivi de ces appels.\n\n"
        "Consulter `/docs` pour la liste et le contrat des endpoints disponibles."
    ),
    version=API_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.include_router(predict_router)
app.include_router(recommend_router)
app.include_router(monitoring_router)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ModelUnavailableError, model_unavailable_handler)
app.add_exception_handler(MonitoringUnauthorizedError, monitoring_unauthorized_handler)
app.add_exception_handler(MonitoringUnavailableError, monitoring_unavailable_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Middleware de persistance : voit les 422 Pydantic générées par les handlers
# ci-dessus et toutes les réponses métier. Placé après les handlers pour que
# les 422/503/500 traversent bien le middleware avant de repartir au client.
app.add_middleware(RequestLoggerMiddleware)


@app.get("/health")
def health() -> dict:
    """État de l'API : version de l'API et état du modèle chargé.

    Après le lifespan, `model_loaded` vaut `True` et `model_version` reprend
    la clé `model_version` du metadata du bundle chargé.
    """
    bundle = runtime.bundle_predict
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "model_loaded": bundle is not None,
        "model_version": bundle.metadata["model_version"] if bundle is not None else None,
    }
