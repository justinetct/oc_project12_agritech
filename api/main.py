"""Point d'entrée de l'API Agritech Answers.

Ce module expose le socle minimal :

- une instance `FastAPI` avec titre, description et version ;
- l'endpoint `GET /health` qui confirme que l'API répond.

La version de l'API est lue une seule fois depuis le paquet installé
(`agritech-answers`) via `importlib.metadata` : elle n'est jamais recopiée
ailleurs, la source de vérité reste `pyproject.toml`.
"""

from __future__ import annotations

from importlib.metadata import version as _package_version

from fastapi import FastAPI


API_VERSION = _package_version("agritech-answers")

app = FastAPI(
    title="Agritech Answers API",
    description=(
        "API de prédiction agricole. Elle exposera deux services :\n\n"
        "- `POST /predict` : estimation de rendement pour une parcelle ;\n"
        "- `POST /recommend` : classement des cultures pour un contexte donné.\n\n"
        "Cette version ne contient que le socle et l'endpoint `/health` : "
        "le chargement du modèle et les endpoints métier arrivent aux sous-étapes suivantes."
    ),
    version=API_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
)


@app.get("/health")
def health() -> dict:
    """Vérifie que l'API répond et renvoie sa version.

    À ce stade, aucun modèle n'est encore chargé : `model_loaded` vaut
    `False` et `model_version` vaut `None`. Ces deux champs seront alimentés
    en sous-étape 2, lors de l'ajout du lifespan de chargement du modèle.
    """
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "model_loaded": False,
        "model_version": None,
    }
