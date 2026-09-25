"""Chemin d'inférence de `/recommend` : préparer une demande, prédire les 10 cultures et les classer.

Ces fonctions sont celles que l'API utilisera. Elles ne dépendent d'aucun notebook, n'importent ni
matplotlib ni MLflow, et n'utilisent jamais de mesure de l'année demandée : les conditions historiques
viennent des années précédentes du pays, comme à l'entraînement (voir `recommend_features.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from agritech.recommend_config import RECOMMEND_CROPS
from agritech.recommend_features import add_geographic_features, add_historical_conditions

# Colonnes attendues dans l'historique d'un pays : une ligne par année connue.
HISTORIQUE_COLONNES = ["iso3", "year", "avg_temp", "rain_mm", "pesticides_t"]


def preparer_demande(
    historique: pd.DataFrame,
    iso3: str,
    year: int,
    coordinates: Mapping[str, tuple[float, float]],
    crops: Sequence[str] = RECOMMEND_CROPS,
    lags: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Les lignes d'une demande `/recommend` : un pays, une année, une culture par ligne.

    `historique` : les années connues des pays (colonnes `HISTORIQUE_COLONNES`), au moins l'une des années
    `year - lag`. Les conditions historiques (moyennes des années précédentes) et la géographie sont
    calculées comme à l'entraînement. Aucune mesure de `year` n'est utilisée : `rain_mm`, fixe par pays dans
    le dataset, est repris de la dernière année connue.

    Erreurs explicites plutôt que valeurs manquantes silencieuses : pays inconnu, culture inconnue (le
    modèle l'encoderait par des zéros sans erreur), colonne absente, ou aucune des années précédentes
    disponible.
    """
    manquantes = [colonne for colonne in HISTORIQUE_COLONNES if colonne not in historique.columns]
    if manquantes:
        raise ValueError(f"colonnes absentes de l'historique : {manquantes}")
    inconnues = sorted(set(crops) - set(RECOMMEND_CROPS))
    if inconnues:
        raise ValueError(f"cultures inconnues : {inconnues}")

    precedentes = historique.loc[(historique["iso3"] == iso3) & (historique["year"] < year), HISTORIQUE_COLONNES]
    if precedentes.empty:
        raise ValueError(f"aucune année connue avant {year} pour {iso3}")

    derniere = precedentes.sort_values("year").iloc[-1]
    demandee = pd.DataFrame([{"iso3": iso3, "year": year, "avg_temp": float("nan"),
                              "rain_mm": derniere["rain_mm"], "pesticides_t": float("nan")}])
    annees = add_historical_conditions(pd.concat([precedentes, demandee], ignore_index=True), lags=lags)
    ligne = add_geographic_features(annees.iloc[[-1]], coordinates)
    if ligne[["temp_hist", "log_pest_hist"]].isna().to_numpy().any():
        raise ValueError(f"aucune des {len(lags)} années précédant {year} n'est connue pour {iso3}")

    ligne = ligne.drop(columns=["avg_temp", "pesticides_t"])
    return pd.concat([ligne.assign(crop=culture) for culture in crops], ignore_index=True)


def recommander(pipeline, demande: pd.DataFrame, couples: Iterable[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Classement des cultures d'une demande : rang, culture et rendement prédit, du meilleur au moins bon.

    `pipeline` : modèle entraîné qui reçoit les colonnes de `preparer_demande` (elles sont sélectionnées par
    son nom, les colonnes en trop sont ignorées). `couples` : couples (pays, culture) observés à
    l'entraînement, pour ajouter la colonne `couple_observe` ; c'est un diagnostic, l'API peut s'en passer.
    """
    colonnes = getattr(pipeline, "feature_names_in_", None)
    entrees = demande[list(colonnes)] if colonnes is not None else demande
    classement = pd.DataFrame({"crop": demande["crop"].to_numpy(), "rendement_predit": pipeline.predict(entrees)})
    classement = classement.sort_values("rendement_predit", ascending=False, ignore_index=True)
    classement.insert(0, "rang", range(1, len(classement) + 1))
    if couples is not None:
        pays = demande["iso3"].iloc[0]
        couples = set(couples)
        classement["couple_observe"] = [(pays, culture) in couples for culture in classement["crop"]]
    return classement


def couples_observes(entrainement: pd.DataFrame) -> set[tuple[str, str]]:
    "Couples (pays, culture) présents dans les données d'apprentissage."
    return set(map(tuple, entrainement[["iso3", "crop"]].drop_duplicates().to_numpy()))
