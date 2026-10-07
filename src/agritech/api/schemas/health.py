"""Contrat de `GET /health` : état de l'API et des deux modèles servis.

`/health` reste un endpoint de santé et de déploiement : il indique quelle
version de l'API et quels modèles tournent, pas comment ils performent. Les
métriques, domaines d'apprentissage et volumes d'appels sont dans les
metadata des modèles, les endpoints `*/context` et `/monitoring/*`.

Le lifespan charge les deux modèles et le contexte `/recommend` avant la
première requête, et l'API ne démarre pas si l'un d'eux manque : quand
`/health` répond, `loaded` et `recommend_context_loaded` valent `true`.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    """État d'un modèle servi, lu dans son bundle chargé au démarrage."""

    loaded: bool = Field(description="Le modèle est chargé en mémoire.")
    version: str | None = Field(
        description="Version du modèle servi (`model_version` de ses metadata), distincte de `api_version`.",
    )
    refit_on: date | None = Field(
        description="Date du réentraînement qui a produit le modèle servi (`created_on` de ses metadata).",
    )
    artifact_size_bytes: int | None = Field(
        description="Taille en octets du fichier `.joblib` chargé au démarrage.",
    )


class ModelsStatus(BaseModel):
    """Les deux modèles servis par l'API."""

    predict: ModelStatus = Field(description="Modèle de `POST /predict`.")
    recommend: ModelStatus = Field(description="Modèle de `POST /recommend`.")


class HealthResponse(BaseModel):
    """Réponse de `GET /health`."""

    status: Literal["ok"] = Field(description="L'API a démarré et répond.")
    api_version: str = Field(
        description="Version de l'API, c'est-à-dire du contrat HTTP, indépendante des versions des modèles.",
    )
    environment: str | None = Field(
        description="Environnement d'exécution, lu dans `ENVIRONMENT` : `local`, `staging` ou `prod`.",
    )
    models: ModelsStatus
    recommend_context_loaded: bool = Field(
        description="Le contexte pays de `/recommend` (valeurs préremplies par pays) est chargé.",
    )
