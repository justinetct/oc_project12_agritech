"""Point d'entrée de l'API Agritech Answers.

Ce module expose :

- une instance `FastAPI` avec titre, description et version ;
- un `lifespan` qui charge le modèle `/predict` au démarrage ;
- l'endpoint `GET /health` qui confirme que l'API répond et renseigne l'état
  du modèle chargé.

La version de l'API est déclarée dans `agritech.api.version`. Elle doit être
maintenue alignée avec `[project] version` de `pyproject.toml` lors d'un bump.
La version du modèle est lue depuis `bundle.metadata`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from sqlalchemy.engine import Engine

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
from agritech.api.version import API_VERSION
from agritech.config import PATHS
from agritech.monitoring.config import MonitoringConfig, load_config
from agritech.monitoring.models import Base as MonitoringBase
from agritech.monitoring.seed_history import seed_demo_history
from agritech.monitoring.session import (
    create_monitoring_engine,
    create_session_factory,
)
from agritech.observability.logfire_setup import configure_logfire
from agritech.serving import load_bundle, load_recommend_context


logger = logging.getLogger(__name__)


def _init_monitoring_database(config: MonitoringConfig) -> Engine | None:
    """Initialise la base SQLite de monitoring sans jamais bloquer le démarrage.

    En cas de succès, publie la session factory dans `runtime` et renvoie
    l'engine, que le lifespan disposera à l'arrêt.

    En cas d'échec, logge un `warning` (type d'erreur seulement : ni chemin,
    ni secret), laisse `runtime.monitoring_session_factory` à `None` et
    renvoie `None`. L'API démarre quand même : `/predict` et `/recommend`
    fonctionnent sans archivage, et `/monitoring/*` répond 503.

    - Échec à la création de l'engine : rien à nettoyer.
    - Échec après la création de l'engine (création de la table, session
      factory) : l'engine est disposée ici, une seule fois, et rien n'est
      publié.
    """
    try:
        engine = create_monitoring_engine(config)
    except Exception as exc:  # noqa: BLE001 — monitoring non bloquant
        _warn_monitoring_unavailable("engine creation", exc)
        return None

    try:
        MonitoringBase.metadata.create_all(engine)
        session_factory = create_session_factory(engine)
    except Exception as exc:  # noqa: BLE001 — monitoring non bloquant
        _warn_monitoring_unavailable("database setup", exc)
        engine.dispose()
        return None

    runtime.monitoring_session_factory = session_factory
    return engine


def _warn_monitoring_unavailable(step: str, exc: Exception) -> None:
    """Warning commun aux échecs d'initialisation du monitoring SQLite.

    Seul le type de l'exception est loggé : son message peut contenir le
    chemin du fichier SQLite ou l'URL de la base.
    """
    logger.warning(
        "monitoring database unavailable (%s failed: %s); API continues without "
        "request archiving and /monitoring endpoints will return 503",
        step,
        type(exc).__name__,
    )


def _add_demo_history(environment: str) -> None:
    """Ajoute l'historique de démonstration si la base de monitoring est vide.

    Appelée seulement avec `MONITORING_DEMO_HISTORY` activé, après une
    initialisation SQLite réussie et avant la première requête. Jamais
    bloquante : un échec est signalé par un `warning` (type d'erreur
    seulement) et l'API démarre normalement.
    """
    try:
        added = seed_demo_history(runtime.monitoring_session_factory, environment)
    except Exception as exc:  # noqa: BLE001 — historique de démonstration non bloquant
        logger.warning("monitoring demo history skipped (%s); API continues", type(exc).__name__)
        return
    if added:
        logger.info("monitoring demo history: %d requests added to the empty database", added)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prépare l'état applicatif avant de servir la première requête.

    - Charge les deux bundles et le contexte `/recommend` (comportement existant).
    - Charge un éventuel `.env` local, lit la configuration monitoring, puis
      initialise la base SQLite (engine avec PRAGMAs WAL, table `api_requests`,
      session factory) via `_init_monitoring_database`, et publie les
      métadonnées runtime pour le middleware de persistance.
    - Avec `MONITORING_DEMO_HISTORY` activé, ajoute l'historique de
      démonstration si la base est vide (`_add_demo_history`).
    - Publie le token des endpoints `/monitoring/*` ; s'il est absent, un
      `warning` signale au démarrage que ces endpoints seront indisponibles.
    - Configure Logfire, indépendamment de l'état de SQLite.

    Les ressources métier sont obligatoires : si un chargement de bundle ou
    du contexte échoue (`FileNotFoundError`, `ValueError`, ...), l'exception
    remonte et l'application ne démarre pas. Le monitoring, lui, n'est jamais
    bloquant : une base SQLite inaccessible est signalée par un `warning`.
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
    # Aucune session factory d'un démarrage précédent ne doit survivre : elle
    # n'est publiée qu'après une initialisation SQLite entièrement réussie.
    runtime.monitoring_session_factory = None
    monitoring_engine = _init_monitoring_database(monitoring_config)
    if monitoring_engine is not None and monitoring_config.demo_history:
        _add_demo_history(monitoring_config.environment)
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
        # `None` si le monitoring n'a pas pu démarrer : l'engine éventuellement
        # créée a déjà été disposée par `_init_monitoring_database`.
        if monitoring_engine is not None:
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
