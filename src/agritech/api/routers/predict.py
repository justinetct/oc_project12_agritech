"""Router HTTP du service `/predict`.

Router volontairement fin : il reçoit la requête déjà validée par Pydantic, récupère le
bundle chargé au démarrage, délègue toute la logique métier à `agritech.serving`, et
sérialise la réponse. Aucune règle numérique ni aucun preprocessing ne vivent ici.
"""

from __future__ import annotations

from fastapi import APIRouter

from agritech.api.core import runtime
from agritech.api.schemas.common import (
    RAINFALL_PHYSICAL_MIN,
    TEMPERATURE_PHYSICAL_MAX,
    TEMPERATURE_PHYSICAL_MIN,
)
from agritech.api.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictSchemaResponse,
    VariableSchema,
)
from agritech.serving import Bundle, predict as serving_predict, public_training_domain


router = APIRouter(tags=["predict"])


def _get_bundle() -> Bundle:
    """Récupère le bundle chargé au démarrage, ou lève une erreur interne explicite.

    Le handler HTTP qui transforme ce cas en 503 `model_unavailable` est ajouté dans la
    sous-étape 5 (gestion des erreurs). À ce stade, on lève un `RuntimeError` clair
    plutôt que de laisser un `AttributeError` accidentel remonter.
    """
    bundle = runtime.bundle_predict
    if bundle is None:
        raise RuntimeError(
            "predict bundle not loaded: lifespan startup did not initialise runtime.bundle_predict"
        )
    return bundle


@router.post("/predict", response_model=PredictResponse, summary="Predict yield from field conditions")
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
