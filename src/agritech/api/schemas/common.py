"""Types partagés entre les services /predict et /recommend.

Ce module porte les grandeurs physiques réutilisables (température, pluie) et
la réponse d'erreur unifiée `ErrorResponse` utilisée par les handlers 422 /
503 / 500.

Les bornes physiques (voir A5 / A9 du Plan de développement — API) sont
définies UNE SEULE FOIS ici. Tout schéma qui expose une température ou une
pluie doit importer `TemperatureCelsius` et `RainfallMm` plutôt que de
recopier les valeurs numériques.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


# Bornes physiques communes aux deux services. Elles rejettent uniquement des
# entrées manifestement invalides ; elles n'ont rien à voir avec les domaines
# d'apprentissage propres à chaque modèle (stockés dans les metadata JSON).
TEMPERATURE_PHYSICAL_MIN = -50.0
TEMPERATURE_PHYSICAL_MAX = 60.0
RAINFALL_PHYSICAL_MIN = 0.0


TemperatureCelsius = Annotated[
    float,
    Field(
        ge=TEMPERATURE_PHYSICAL_MIN,
        le=TEMPERATURE_PHYSICAL_MAX,
        description=(
            "Température moyenne saisonnière en degrés Celsius. "
            f"Doit être comprise entre {TEMPERATURE_PHYSICAL_MIN:g} et "
            f"{TEMPERATURE_PHYSICAL_MAX:g} °C (plage physique)."
        ),
        examples=[25.0],
    ),
]

RainfallMm = Annotated[
    float,
    Field(
        ge=RAINFALL_PHYSICAL_MIN,
        description=(
            "Pluie totale sur la période considérée, en millimètres. "
            "Doit être positive ou nulle."
        ),
        examples=[500.0],
    ),
]


class ValidationErrorDetail(BaseModel):
    """Détail d'une erreur de validation Pydantic exposée au client.

    Reformaté depuis `RequestValidationError.errors()` pour ne pas ré-exposer
    `input` (le payload) ni `ctx` (le contexte interne).
    """

    field: str = Field(
        description="Chemin du champ fautif, par exemple `body.rainfall_mm`.",
        examples=["body.rainfall_mm"],
    )
    type: str = Field(
        description="Code d'erreur Pydantic (`missing`, `bool_type`, `greater_than_equal`, …).",
        examples=["greater_than_equal"],
    )
    message: str = Field(
        description="Message court, lisible côté client.",
        examples=["Input should be greater than or equal to 0"],
    )


class ErrorResponse(BaseModel):
    """Réponse d'erreur unifiée retournée par les handlers 422 / 503 / 500.

    `error` est un code stable machine-readable ; `message` est une phrase courte
    constante par catégorie d'erreur ; `details` liste les erreurs de validation
    pour un 422, et vaut `null` pour un 503 ou un 500.
    """

    error: str = Field(
        description="Code d'erreur machine-readable, stable dans le temps.",
        examples=["validation_error"],
    )
    message: str = Field(
        description="Phrase humaine, constante par catégorie d'erreur.",
        examples=["Request payload is invalid."],
    )
    details: list[ValidationErrorDetail] | None = Field(
        default=None,
        description="Détails structurés pour un 422 ; `null` pour un 503 ou un 500.",
    )


class VariableSchema(BaseModel):
    """Bornes d'une variable numérique : min inclusif, max inclusif ou null si pas de borne haute.

    Utilisé par `GET /predict/context` et `GET /recommend/context` pour exposer
    `physical_bounds` et `training_domain` sous une forme commune. Les clés
    associées sont les noms publics snake_case du contrat.
    """

    min: float = Field(description="Valeur minimale acceptée (inclusive).", examples=[0.0])
    max: float | None = Field(
        description="Valeur maximale acceptée (inclusive), ou `null` si pas de borne haute.",
        examples=[1000.0],
    )
    unit: str = Field(description="Unité physique de la variable.", examples=["mm"])
