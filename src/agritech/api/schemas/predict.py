"""Schémas Pydantic du service /predict : requête et réponse.

`PredictRequest` importe les bornes physiques de `common.py` (source unique).
La logique du drapeau `out_of_training_domain` et le remplissage des `notes`
sont écrits en sous-étape 4 côté service ; ici on ne fait que déclarer la
forme du contrat.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from agritech.api.schemas.common import RainfallMm, TemperatureCelsius, VariableSchema


class PredictRequest(BaseModel):
    """Conditions de parcelle demandées par POST /predict.

    Les 4 champs correspondent exactement aux 4 variables du modèle final.
    Aucun champ supplémentaire n'est accepté (`extra="forbid"`). Les booléens
    sont stricts : `1`, `"true"`, `0`, `"false"` sont rejetés.
    """

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: RainfallMm
    temperature_celsius: TemperatureCelsius
    fertilizer_used: StrictBool = Field(
        description="La parcelle a-t-elle reçu un apport de fertilisant ?",
        examples=[True],
    )
    irrigation_used: StrictBool = Field(
        description="La parcelle a-t-elle bénéficié d'une irrigation ?",
        examples=[False],
    )


class PredictResponse(BaseModel):
    """Réponse renvoyée par POST /predict."""

    yield_tons_per_hectare: float = Field(
        description="Rendement estimé.",
        examples=[4.82],
    )
    unit: str = Field(
        default="t/ha",
        description="Unité du rendement (constante).",
    )
    model_version: str = Field(
        description="Version du modèle qui a produit la prédiction (voir `/health`).",
        examples=["1.1.0"],
    )
    out_of_training_domain: bool = Field(
        description=(
            "True dès qu'au moins une variable numérique est hors du domaine "
            "d'apprentissage du modèle. Le détail est dans `notes`."
        ),
        examples=[False],
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Une note par variable numérique hors domaine d'apprentissage, "
            "sous la forme `\"<champ> is out of training domain\"`. Liste vide "
            "quand toutes les variables sont dans le domaine."
        ),
        examples=[[]],
    )


class PredictContextResponse(BaseModel):
    """Réponse de `GET /predict/context` : bornes physiques du contrat + domaine d'apprentissage.

    Ce document est destiné à un client (par exemple Streamlit) qui souhaite afficher, à
    côté d'un champ de saisie, les valeurs acceptables et les valeurs que le modèle a
    réellement vues. Les clés utilisent les noms publics snake_case du contrat.
    """

    physical_bounds: dict[str, VariableSchema] = Field(
        description=(
            "Bornes physiques du contrat API. Une valeur en dehors est rejetée avec un "
            "code 422. Clé = nom public snake_case du champ."
        ),
    )
    training_domain: dict[str, VariableSchema] = Field(
        description=(
            "Domaine réellement observé pendant l'entraînement du modèle. Une valeur "
            "physique valide mais hors de ce domaine est acceptée et signalée dans "
            "`notes` avec `out_of_training_domain=true`. Clé = nom public snake_case."
        ),
    )
