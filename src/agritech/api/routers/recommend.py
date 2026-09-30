"""Router HTTP du service `/recommend`.

Router volontairement fin : reçoit la requête déjà validée par Pydantic,
récupère le bundle et le contexte chargés au démarrage, contrôle explicitement
l'existence de `iso3` dans le contexte, délègue toute la logique métier à
`agritech.serving.recommend`, et sérialise la réponse. Aucune règle numérique
ni preprocessing ne vivent ici.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError

from agritech.api.core import runtime
from agritech.api.exceptions import ModelUnavailableError
from agritech.api.schemas.common import (
    RAINFALL_PHYSICAL_MIN,
    TEMPERATURE_PHYSICAL_MAX,
    TEMPERATURE_PHYSICAL_MIN,
    ErrorResponse,
    VariableSchema,
)
from agritech.api.schemas.recommend import (
    PESTICIDES_PHYSICAL_MIN,
    CountryEntry,
    RecommendContextResponse,
    RecommendRequest,
    RecommendResponse,
)
from agritech.serving import (
    RECOMMEND_TARGET_YEAR,
    Bundle,
    RecommendContext,
    public_training_domain_recommend,
    recommend as serving_recommend,
)


router = APIRouter(tags=["recommend"])


# Note structurelle affichée par `GET /recommend/context`. Convention interne :
# le modèle est appris jusqu'en 2013 ; 2014 est la première année à recommander,
# construite à partir de l'historique 2011-2013. En anglais pour l'UI.
TARGET_YEAR_NOTE = (
    "The recommend model has been trained on data up to 2013. Year 2014 is a "
    "technical convention: it is the first year following the end of the "
    "historical dataset, and is used by the service to compute the 3-year "
    "history features from 2011-2013. ExtraTrees does not extrapolate beyond "
    "learned values along the year axis."
)

# Réponses OpenAPI adaptées à chaque endpoint : POST peut recevoir un payload
# invalide (422), GET n'a pas de payload à valider. 503 et 500 restent communs :
# les deux endpoints dépendent du bundle et du contexte chargés au démarrage.
_POST_ERROR_RESPONSES: dict = {
    422: {"model": ErrorResponse, "description": "Requête invalide"},
    503: {"model": ErrorResponse, "description": "Modèle ou contexte indisponible"},
    500: {"model": ErrorResponse, "description": "Erreur interne"},
}
_GET_ERROR_RESPONSES: dict = {
    503: {"model": ErrorResponse, "description": "Modèle ou contexte indisponible"},
    500: {"model": ErrorResponse, "description": "Erreur interne"},
}


def _get_bundle_and_context() -> tuple[Bundle, RecommendContext]:
    """Récupère le bundle et le contexte chargés au démarrage, ou lève `ModelUnavailableError`.

    Le handler HTTP dédié (`model_unavailable_handler`) traduit l'exception en
    503 `model_unavailable`. Le router n'a pas à connaître le code HTTP : il
    exprime une intention métier.
    """
    bundle = runtime.bundle_recommend
    context = runtime.recommend_context
    if bundle is None or context is None:
        raise ModelUnavailableError(
            "recommend bundle or context not loaded: lifespan did not initialise runtime."
        )
    return bundle, context


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    responses=_POST_ERROR_RESPONSES,
    summary="Score and rank the 10 crops for a country",
)
def post_recommend(request: RecommendRequest) -> RecommendResponse:
    """Classe les 10 cultures pour le pays demandé, à partir de son contexte historique.

    L'utilisateur peut surcharger tout ou partie des conditions préremplies via
    le bloc `conditions`. Le service renvoie les 10 recommandations triées par
    rendement prédit décroissant.

    Un `iso3` syntaxiquement valide (Pydantic accepte) mais absent du contexte
    servi renvoie 422 `validation_error` avec `type="unknown_country"` sur
    `body.iso3`, en réutilisant le handler existant. La liste des pays servis
    est exposée par `GET /recommend/context`.
    """
    bundle, context = _get_bundle_and_context()
    if request.iso3 not in context.countries:
        raise RequestValidationError([{
            "loc": ("body", "iso3"),
            "type": "unknown_country",
            "msg": f"Country not served: {request.iso3}",
        }])
    conditions = request.conditions.model_dump() if request.conditions else {}
    result = serving_recommend(bundle, context, request.iso3, conditions)
    return RecommendResponse(**result)


@router.get(
    "/recommend/context",
    response_model=RecommendContextResponse,
    responses=_GET_ERROR_RESPONSES,
    summary="Available countries, crops, physical bounds and training domain",
)
def get_recommend_context() -> RecommendContextResponse:
    """Retourne l'ensemble des informations utiles au client pour construire son formulaire.

    - `crops` : les 10 modalités connues par le modèle chargé.
    - `countries` : les pays servis, triés par nom (ordre du contexte).
    - `physical_bounds` : bornes physiques du contrat (mêmes que Pydantic).
    - `training_domain` : bornes apprises, en unités publiques (tonnes pour les
      pesticides via `expm1` depuis `log_pest_hist`).
    """
    bundle, context = _get_bundle_and_context()
    physical_bounds = {
        "average_temperature_celsius": VariableSchema(
            min=TEMPERATURE_PHYSICAL_MIN, max=TEMPERATURE_PHYSICAL_MAX, unit="°C"
        ),
        "annual_rainfall_mm": VariableSchema(
            min=RAINFALL_PHYSICAL_MIN, max=None, unit="mm"
        ),
        "average_annual_pesticides_tons": VariableSchema(
            min=PESTICIDES_PHYSICAL_MIN, max=None, unit="t"
        ),
    }
    training_domain = {
        name: VariableSchema(**bounds)
        for name, bounds in public_training_domain_recommend(bundle).items()
    }
    return RecommendContextResponse(
        year=RECOMMEND_TARGET_YEAR,
        target_year_note=TARGET_YEAR_NOTE,
        crops=bundle.metadata["categorical_values"]["crop"],
        countries=[
            CountryEntry(iso3=iso3, country=name)
            for iso3, name in context.country_entries
        ],
        physical_bounds=physical_bounds,
        training_domain=training_domain,
    )
