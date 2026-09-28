"""Router HTTP du service `/predict`.

Router volontairement fin : il reçoit la requête déjà validée par Pydantic, récupère le
bundle chargé au démarrage, délègue toute la logique métier à `agritech.serving`, et
sérialise la réponse. Aucune règle numérique ni aucun preprocessing ne vivent ici.
"""

from __future__ import annotations

from fastapi import APIRouter

from agritech.api.core import runtime
from agritech.api.exceptions import ModelUnavailableError
from agritech.api.schemas.common import (
    RAINFALL_PHYSICAL_MIN,
    TEMPERATURE_PHYSICAL_MAX,
    TEMPERATURE_PHYSICAL_MIN,
    ErrorResponse,
    VariableSchema,
)
from agritech.api.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictSchemaResponse,
)
from agritech.serving import Bundle, predict as serving_predict, public_training_domain


router = APIRouter(tags=["predict"])


# Réponses OpenAPI adaptées à chaque endpoint : POST peut recevoir un payload invalide
# (422), GET n'a pas de payload à valider — inutile d'y documenter un 422 qui ne peut
# pas se produire. 503 et 500 restent communs : les deux endpoints dépendent du bundle
# chargé au démarrage et peuvent subir une exception inattendue côté service.
_POST_ERROR_RESPONSES: dict = {
    422: {"model": ErrorResponse, "description": "Requête invalide"},
    503: {"model": ErrorResponse, "description": "Modèle indisponible"},
    500: {"model": ErrorResponse, "description": "Erreur interne"},
}
_GET_ERROR_RESPONSES: dict = {
    503: {"model": ErrorResponse, "description": "Modèle indisponible"},
    500: {"model": ErrorResponse, "description": "Erreur interne"},
}


def _get_bundle() -> Bundle:
    """Récupère le bundle chargé au démarrage, ou lève `ModelUnavailableError`.

    Le handler HTTP dédié traduit l'exception en 503 `model_unavailable`. Le router
    n'a pas à connaître le code HTTP : il exprime une intention métier.
    """
    bundle = runtime.bundle_predict
    if bundle is None:
        raise ModelUnavailableError(
            "predict bundle not loaded: lifespan did not initialise runtime.bundle_predict"
        )
    return bundle


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=_POST_ERROR_RESPONSES,
    summary="Predict yield from field conditions",
)
def post_predict(request: PredictRequest) -> PredictResponse:
    """Prédit le rendement d'une parcelle pour les 4 conditions reçues.

    Une valeur physiquement valide mais hors du domaine d'apprentissage est acceptée :
    la prédiction est renvoyée avec `out_of_training_domain=true` et une note par
    variable concernée. Voir `GET /predict/schema` pour connaître à l'avance les
    bornes physiques et le domaine d'apprentissage.
    """
    bundle = _get_bundle()
    return PredictResponse(**serving_predict(bundle, request.model_dump()))


@router.get(
    "/predict/schema",
    response_model=PredictSchemaResponse,
    responses=_GET_ERROR_RESPONSES,
    summary="Physical bounds and training domain of the predict model",
)
def get_predict_schema() -> PredictSchemaResponse:
    """Retourne les bornes physiques du contrat et le domaine d'apprentissage du modèle."""
    bundle = _get_bundle()
    physical_bounds = {
        "rainfall_mm": VariableSchema(min=RAINFALL_PHYSICAL_MIN, max=None, unit="mm"),
        "temperature_celsius": VariableSchema(
            min=TEMPERATURE_PHYSICAL_MIN, max=TEMPERATURE_PHYSICAL_MAX, unit="°C"
        ),
    }
    training_domain = {
        name: VariableSchema(**bounds) for name, bounds in public_training_domain(bundle).items()
    }
    return PredictSchemaResponse(
        physical_bounds=physical_bounds, training_domain=training_domain
    )
