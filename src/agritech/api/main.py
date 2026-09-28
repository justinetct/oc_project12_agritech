"""Point d'entrée de l'API Agritech Answers.

Ce module expose :

- une instance `FastAPI` avec titre, description et version ;
- un `lifespan` qui charge le modèle `/predict` au démarrage ;
- l'endpoint `GET /health` qui confirme que l'API répond et renseigne l'état
  du modèle chargé.

La version de l'API est lue une seule fois depuis le paquet installé
(`agritech-answers`) via `importlib.metadata` : la source de vérité reste
`pyproject.toml`. La version du modèle est lue depuis `bundle.metadata`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.metadata import version as _package_version

from fastapi import FastAPI

from agritech.api.core import runtime
from agritech.api.routers.predict import router as predict_router
from agritech.serving import load_bundle


API_VERSION = _package_version("agritech-answers")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle `/predict` avant de servir la première requête.

    Si le chargement échoue (`FileNotFoundError`, `ValueError`, ...), l'exception
    remonte et l'application ne démarre pas : la couche HTTP d'un modèle
    indisponible sera ajoutée à la sous-étape prévue pour les erreurs.
    """
    runtime.bundle_predict = load_bundle("predict")
    yield


app = FastAPI(
    title="Agritech Answers API",
    description=(
        "API de prédiction agricole. Deux services :\n\n"
        "- `POST /predict` : estimation de rendement pour une parcelle ;\n"
        "- `POST /recommend` : classement des cultures pour un contexte donné.\n\n"
        "Consulter `/docs` pour la liste et le contrat des endpoints disponibles."
    ),
    version=API_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.include_router(predict_router)


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
