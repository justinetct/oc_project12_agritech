"""Types partagés entre les services /predict et /recommend.

Ce module ne dépend d'aucun modèle et d'aucune route : il porte uniquement
les grandeurs physiques réutilisables (température, pluie).

Les bornes physiques (voir A5 / A9 du Plan de développement — API) sont
définies UNE SEULE FOIS ici. Tout schéma qui expose une température ou une
pluie doit importer `TemperatureCelsius` et `RainfallMm` plutôt que de
recopier les valeurs numériques.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field


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
