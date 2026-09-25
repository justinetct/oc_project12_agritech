"""Variables de `/recommend` : mêmes briques dans les notebooks, à l'entraînement final et dans l'API.

- Des **fonctions** (`add_...`) qui ajoutent des colonnes ligne par ligne. Elles n'apprennent rien et
  n'utilisent jamais la cible : on peut les appliquer avant le découpage, sans fuite, et de la même
  façon sur une requête de l'API.
- Un **transformer scikit-learn**, `CropInteractions`, placé dans le pipeline : les interactions entre
  une variable et la culture, utiles à la régression linéaire.

Chaque fonction renvoie une copie du DataFrame, sans changer ni l'ordre ni l'index des lignes. Le
module n'importe ni matplotlib ni le GeoJSON : l'API peut l'utiliser avec une simple table de
coordonnées.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from agritech.recommend_config import RECOMMEND_CROPS


def add_geographic_features(df: pd.DataFrame, coordinates: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    """Ajoute `lat_abs`, `geo_x`, `geo_y` et `geo_z` à partir du code `iso3` de chaque ligne.

    `coordinates` associe un code ISO3 à son couple (longitude, latitude) en degrés, par exemple le
    résultat de `agritech.geo.charger_coordonnees()`.

    - `lat_abs` : latitude absolue, distance à l'équateur ;
    - `geo_x = cos(lat)·cos(lon)`, `geo_y = cos(lat)·sin(lon)`, `geo_z = sin(lat)` : position sur la
      sphère. Deux pays voisins ont des valeurs voisines, y compris de part et d'autre de la
      longitude ±180°, ce que la longitude brute ne garantit pas.

    Un code absent de `coordinates` provoque une erreur : pas de coordonnées manquantes silencieuses.
    """
    missing = sorted(set(df["iso3"]) - set(coordinates))
    if missing:
        raise ValueError(f"coordonnées absentes pour : {missing}")

    longitude_degrees = df["iso3"].map(lambda code: coordinates[code][0])
    latitude_degrees = df["iso3"].map(lambda code: coordinates[code][1])
    longitude, latitude = np.radians(longitude_degrees), np.radians(latitude_degrees)
    return df.assign(
        lat_abs=latitude_degrees.abs(),
        geo_x=np.cos(latitude) * np.cos(longitude),
        geo_y=np.cos(latitude) * np.sin(longitude),
        geo_z=np.sin(latitude),
    )


def add_recommend_features(df: pd.DataFrame, coordinates: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    """Ajoute les variables calculables ligne par ligne : `log_pesticides` et la géographie.

    - `log_pesticides` = `log1p(pesticides_t)` : le tonnage national va de moins d'une tonne à plus
      d'un million ;
    - géographie : voir `add_geographic_features`.

    C'est le point d'entrée commun aux notebooks 13 à 15, entraînement final compris. Les conditions
    historiques n'en font pas partie : elles demandent les années précédentes du pays, voir
    `add_historical_conditions`. Pour une demande, `recommend_service` n'a besoin que de la géographie
    et des conditions historiques.
    """
    return add_geographic_features(df, coordinates).assign(log_pesticides=np.log1p(df["pesticides_t"]))


def add_historical_conditions(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Ajoute `temp_hist`, `pest_hist` et `log_pest_hist` : conditions moyennes des années précédentes du pays.

    Pour l'année t d'un pays : moyenne de `avg_temp` et de `pesticides_t` sur les années t−1, t−2 et
    t−3 disponibles (`lags`), puis `log1p` pour les pesticides. Le calcul se fait sur la table
    pays-année, par année calendaire : une année manquante n'est pas remplacée par la précédente.
    L'année t elle-même n'est jamais utilisée : aucune fuite temporelle. La première année d'un pays
    n'a pas d'historique : ses valeurs sont manquantes.

    `df` doit contenir toutes les années utiles de chaque pays : à l'inférence, lui fournir les
    dernières années connues du pays en plus de la ligne à prédire.
    """
    country_years = df[["iso3", "year", "avg_temp", "pesticides_t"]].drop_duplicates(["iso3", "year"])
    past = pd.concat([country_years.assign(year=country_years["year"] + lag) for lag in lags])
    history = past.groupby(["iso3", "year"])[["avg_temp", "pesticides_t"]].mean()
    history.columns = ["temp_hist", "pest_hist"]

    result = df.join(history, on=["iso3", "year"])
    return result.assign(log_pest_hist=np.log1p(result["pest_hist"]))


class CropInteractions(BaseEstimator, TransformerMixin):
    """Interactions culture × variable, pour une régression linéaire.

    Pour chaque culture et chaque variable, une colonne `"<culture> x <variable>"` qui vaut la variable
    sur les lignes de cette culture et 0 ailleurs. Une régression linéaire additive donne le même effet
    d'une variable à toutes les cultures ; avec ces colonnes, chaque culture a le sien.

    Rien n'est appris. Les colonnes créées (`created_columns`) ne dépendent que des cultures connues
    (`crops`, les 10 de `/recommend` par défaut), pas des lignes reçues. Une culture inconnue provoque
    une erreur plutôt que des colonnes nulles silencieuses.

    Elle sert aux régressions linéaires de référence et à la reproductibilité des notebooks 13 et 14 :
    le 14 recalcule avec elle la référence `crop_x_geography` et teste ces variables avec les arbres.
    Elle n'est pas utilisée par les finalistes non linéaires et ne fait pas partie du pipeline final
    de `/recommend`.
    """

    def __init__(self, variables: Sequence[str], crops: Sequence[str] = RECOMMEND_CROPS):
        self.variables = variables
        self.crops = crops

    @property
    def created_columns(self) -> list[str]:
        return [f"{crop} x {variable}" for crop in self.crops for variable in self.variables]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        unknown = sorted(set(X["crop"]) - set(self.crops))
        if unknown:
            raise ValueError(f"cultures inconnues : {unknown}")

        X = X.copy()
        for crop in self.crops:
            this_crop = X["crop"] == crop
            for variable in self.variables:
                X[f"{crop} x {variable}"] = X[variable].where(this_crop, 0.0)
        return X
