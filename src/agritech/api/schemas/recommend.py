"""Schémas Pydantic du service /recommend : requête, réponse et description schéma.

Le contrat public reprend les conventions déjà validées pour /predict :

- bornes physiques importées depuis `schemas/common.py`, jamais recopiées ;
- `extra="forbid"` sur les modèles de requête ;
- séparation stricte validation physique (rejet 422) vs domaine d'apprentissage
  (accepté, signalé via `out_of_training_domain` + `notes`) ;
- structure `physical_bounds` / `training_domain` identique à /predict (dict de
  `VariableSchema` indexé par nom public snake_case).

Les schémas ne contiennent aucune logique métier : leur seul rôle est de figer
et documenter le contrat HTTP. La logique d'inférence vit dans
`agritech.serving` (module `recommend`), déjà validé.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agritech.api.schemas.common import (
    RainfallMm,
    TemperatureCelsius,
    VariableSchema,
)


# Borne physique propre aux pesticides : jamais négative, pas de borne haute.
# Comme la température et la pluie, la valeur doit être finie (`allow_inf_nan=False`).
PESTICIDES_PHYSICAL_MIN = 0.0


IsoAlpha3 = Annotated[
    str,
    Field(
        pattern=r"^[A-Z]{3}$",
        description=(
            "Code pays ISO 3166-1 alpha-3 (3 lettres majuscules). "
            "La liste des pays servis est exposée par `GET /recommend/context`."
        ),
        examples=["FRA"],
    ),
]


PesticidesTons = Annotated[
    float,
    Field(
        ge=PESTICIDES_PHYSICAL_MIN,
        allow_inf_nan=False,
        description=(
            "Tonnage annuel de pesticides, en tonnes. Doit être un nombre fini, "
            "positif ou nul. "
            "Le service applique `log1p` avant de nourrir la feature interne "
            "`log_pest_hist` du modèle."
        ),
        examples=[60000.0],
    ),
]


class RecommendConditions(BaseModel):
    """Bloc optionnel du corps `POST /recommend` : conditions du pays surchargeables.

    Chaque champ est optionnel. Un champ omis ou `null` utilise la valeur
    préremplie calculée par le service à partir de l'historique 2011-2013 du
    pays choisi. Un champ fourni devient la valeur effective transmise au
    modèle (voir `context.effective_conditions` de la réponse).
    """

    model_config = ConfigDict(extra="forbid")

    average_temperature_celsius: TemperatureCelsius | None = Field(
        default=None,
        description=(
            "Moyenne des températures annuelles typiques du pays. Remplace la valeur "
            "préremplie calculée sur les 3 années précédentes ; laisser vide pour "
            "utiliser la valeur du pays."
        ),
    )
    annual_rainfall_mm: RainfallMm | None = Field(
        default=None,
        description=(
            "Pluie annuelle typique du pays, en millimètres. Le dataset historique "
            "fournit une valeur unique par pays (constante dans le temps). "
            "Laisser vide pour utiliser la valeur du pays."
        ),
    )
    average_annual_pesticides_tons: PesticidesTons | None = Field(
        default=None,
        description=(
            "Tonnage annuel typique de pesticides du pays, en tonnes. Remplace la "
            "moyenne préremplie calculée sur les 3 années précédentes. Le service "
            "applique `log1p` avant d'alimenter la feature `log_pest_hist` du modèle."
        ),
    )


class RecommendRequest(BaseModel):
    """Corps de `POST /recommend`.

    Le client fournit uniquement le code pays et, optionnellement, un bloc
    partiel de conditions. Toutes les autres features consommées par le modèle
    (`temp_hist`, `log_pest_hist`, `rain_mm`, `lat_abs`, `geo_x`, `geo_y`,
    `geo_z`, `year`) sont reconstruites par le service à partir du contexte
    versionné du pays.
    """

    model_config = ConfigDict(extra="forbid")

    iso3: IsoAlpha3 = Field(
        description=(
            "Code pays ISO 3166-1 alpha-3 (3 lettres majuscules, ex. `FRA`). "
            "Doit correspondre à l'un des pays servis (voir `GET /recommend/context`)."
        ),
    )
    conditions: RecommendConditions | None = Field(
        default=None,
        description=(
            "Optionnel. Bloc de conditions caractéristiques du pays. Chaque champ "
            "omis ou `null` utilise la valeur préremplie affichée dans "
            "`context.country_defaults` de la réponse. Les valeurs fournies "
            "deviennent les conditions effectives transmises au modèle."
        ),
    )


class RecommendConditionValues(BaseModel):
    """Trois conditions publiques, toutes présentes.

    Utilisé pour représenter à la fois `country_defaults` (valeurs préremplies
    du pays) et `effective_conditions` (valeurs réellement transmises au
    modèle) dans le bloc `context` de la réponse.
    """

    average_temperature_celsius: float = Field(
        description="°C — moyenne des 3 années précédentes du pays, éventuellement surchargée.",
        examples=[11.83],
    )
    annual_rainfall_mm: float = Field(
        description="mm/an — pluie annuelle typique du pays, éventuellement surchargée.",
        examples=[867.0],
    )
    average_annual_pesticides_tons: float = Field(
        description="t/an — tonnage moyen des 3 années précédentes du pays, éventuellement surchargé.",
        examples=[65103.33],
    )


class CountryContext(BaseModel):
    """Bloc `context` de la réponse : defaults du pays et conditions effectives.

    Permet au client (par exemple Streamlit) d'afficher côte à côte la valeur
    préremplie du pays et la valeur effectivement utilisée par le modèle, sans
    refaire de logique métier.
    """

    country_defaults: RecommendConditionValues = Field(
        description="Valeurs préremplies calculées à partir de l'historique 2011-2013 du pays.",
    )
    effective_conditions: RecommendConditionValues = Field(
        description=(
            "Valeurs effectivement transmises au modèle (defaults ⊕ overrides). "
            "Utilisées pour la comparaison au domaine d'apprentissage."
        ),
    )


class Recommendation(BaseModel):
    """Une recommandation dans le classement retourné par `POST /recommend`."""

    rank: int = Field(
        ge=1,
        le=10,
        description="Rang dans le classement : 1 = meilleur rendement prédit.",
        examples=[1],
    )
    crop: str = Field(
        description="Nom de la culture, parmi les 10 modalités du modèle.",
        examples=["Potatoes"],
    )
    predicted_yield_tons_per_hectare: float = Field(
        description="Rendement estimé pour cette culture dans le contexte fourni.",
        examples=[42.24],
    )
    observed_in_country: bool = Field(
        description=(
            "True si le couple (pays, culture) figure dans l'historique disponible "
            "(1990-2013). False indique une culture jamais observée dans ce pays : "
            "recommandation moins fiable, non couverte par le test 2013."
        ),
        examples=[True],
    )


class RecommendResponse(BaseModel):
    """Réponse renvoyée par `POST /recommend`.

    La structure reprend directement le dict produit par
    `agritech.serving.recommend(...)` : mêmes clés, mêmes types.
    """

    iso3: str = Field(description="Code pays demandé.", examples=["FRA"])
    country: str = Field(description="Nom du pays.", examples=["France"])
    year: int = Field(
        description=(
            "Année cible technique de la recommandation, fixée par le service. "
            "Voir `GET /recommend/context.target_year_note` pour l'explication de "
            "cette convention interne."
        ),
        examples=[2014],
    )
    unit: str = Field(
        default="t/ha",
        description="Unité des rendements prédits (constante).",
    )
    model_version: str = Field(
        description="Version du modèle qui a produit la recommandation (voir `/health`).",
        examples=["2.0.0"],
    )
    recommendations: list[Recommendation] = Field(
        min_length=10,
        max_length=10,
        description="Les 10 cultures scorées, triées par rendement prédit décroissant.",
    )
    context: CountryContext = Field(
        description="Defaults pays et conditions effectivement utilisées par le modèle.",
    )
    out_of_training_domain: bool = Field(
        description=(
            "True si au moins une condition effective sort du domaine appris. Le "
            "détail est dans `notes`. Le fait que `year` soit hors plage "
            "d'apprentissage n'est pas signalé ici (limite structurelle documentée "
            "dans `GET /recommend/context.target_year_note`)."
        ),
        examples=[False],
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Une note par condition effective hors domaine, format "
            "`\"<champ> is out of training domain\"`. Liste vide quand toutes les "
            "conditions sont dans le domaine appris."
        ),
        examples=[[]],
    )


class CountryEntry(BaseModel):
    """Un pays disponible pour `/recommend`, exposé par `GET /recommend/context`."""

    iso3: str = Field(description="Code ISO3.", examples=["FRA"])
    country: str = Field(description="Nom du pays, pour affichage.", examples=["France"])


class SelectedCountry(BaseModel):
    """Contexte du pays sélectionné, exposé par `GET /recommend/context?iso3=<...>`.

    Permet à un client (par exemple Streamlit) de préremplir le formulaire du
    pays choisi sans reconstruire ni transformer aucune feature ML. Les
    valeurs `country_defaults` sont **exactement** celles calculées par le
    service à partir de l'historique 2011-2013 : mêmes valeurs que
    `context.country_defaults` de `POST /recommend`.
    """

    iso3: str = Field(description="Code ISO3 sélectionné.", examples=["FRA"])
    country: str = Field(description="Nom du pays sélectionné.", examples=["France"])
    country_defaults: RecommendConditionValues = Field(
        description=(
            "Valeurs préremplies calculées à partir de l'historique 2011-2013 "
            "du pays, en unités publiques. Modifiables par l'utilisateur avant "
            "d'envoyer `POST /recommend`."
        ),
    )


class RecommendContextResponse(BaseModel):
    """Réponse de `GET /recommend/context`.

    Fournit au client tout ce qu'il faut pour construire son formulaire :
    liste des pays, cultures, bornes physiques et domaine d'apprentissage.
    Aligné sur la structure de `GET /predict/context` pour `physical_bounds` et
    `training_domain` (dict de `VariableSchema` indexé par nom public).
    """

    year: int = Field(
        description="Année cible technique fixée par le service.",
        examples=[2014],
    )
    target_year_note: str = Field(
        description=(
            "Rappel structurel sur l'année cible et la plage d'apprentissage "
            "(en anglais, pour affichage dans le client)."
        ),
    )
    crops: list[str] = Field(
        min_length=10,
        max_length=10,
        description="Les 10 cultures modélisées.",
    )
    countries: list[CountryEntry] = Field(
        description=(
            "Pays servis, triés par nom de pays. Utilisé notamment par le "
            "sélecteur du client Streamlit."
        ),
    )
    physical_bounds: dict[str, VariableSchema] = Field(
        description=(
            "Bornes physiques du contrat API. Une valeur en dehors est rejetée avec "
            "un code 422. Clé = nom public snake_case du champ."
        ),
    )
    training_domain: dict[str, VariableSchema] = Field(
        description=(
            "Domaine réellement observé pendant l'entraînement, en unités publiques "
            "(les pesticides sont convertis via `expm1` depuis la feature interne "
            "`log_pest_hist`). Clé = nom public snake_case."
        ),
    )
    country: SelectedCountry | None = Field(
        default=None,
        description=(
            "Contexte du pays sélectionné, présent uniquement si le paramètre "
            "de requête `iso3` a été fourni. Vaut `null` sinon."
        ),
    )
