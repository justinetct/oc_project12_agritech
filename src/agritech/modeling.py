"""Mécanique commune des expériences de modélisation : pipeline, validation croisée et run MLflow.

Les notebooks déclarent leurs expériences : modèle, variables, éventuelles colonnes ajoutées et
informations MLflow. `run_experiment` fait le reste : construction du pipeline, validation croisée
chronométrée, run MLflow, puis renvoie les scores par fold et le résumé. La recherche
d'hyperparamètres est dans `tuning.py` ; les analyses propres à une expérience restent dans les notebooks.

Les métriques viennent de `evaluation.py`, le preprocessing de `preprocessing.py` et l'écriture dans
MLflow de `tracking.py`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from agritech.evaluation import cross_validate_folds, format_cv_metrics, summarize_cv_folds
from agritech.preprocessing import (
    PREPROCESSING_DESCRIPTION,
    PREPROCESSING_NATIVE_DESCRIPTION,
    PREPROCESSING_SCALED_DESCRIPTION,
    build_pipeline,
    make_native_preprocessing,
    make_preprocessing,
)
from agritech.tracking import log_run


@dataclass(frozen=True)
class FeatureEngineering:
    """Colonnes ajoutées au début du pipeline, avant l'encodage : fonction d'ajout et noms des colonnes créées.

    Les colonnes créées sont numériques et suivent le même preprocessing que les autres variables
    numériques : passées telles quelles par défaut, standardisées avec `scale_numeric=True`. La fonction
    fait partie du pipeline : en validation croisée, elle est appliquée séparément à chaque fold.
    """

    add_columns: Callable[[pd.DataFrame], pd.DataFrame]
    columns: list[str]


@dataclass(frozen=True)
class RunInfo:
    """Informations MLflow d'une expérience : nom du run, jeu de variables, tags et paramètres.

    `protocol_params` : paramètres du protocole, communs à tous les runs du service ;
    `params` : paramètres propres au run (hyperparamètres, phase de tuning...) ;
    `replace=True` : à la relance d'un notebook, le nouveau run remplace l'ancien run du même nom et du
    même notebook au lieu de s'y ajouter (voir `tracking.log_run`).
    """

    name: str
    feature_set: str
    tags: dict[str, str]
    protocol_params: dict
    params: dict = field(default_factory=dict)
    replace: bool = False


def engineered_columns(engineering) -> list[str]:
    """Noms des colonnes ajoutées par `engineering` : `FeatureEngineering` ou transformer de `recommend_features.py`."""
    if engineering is None:
        return []
    if isinstance(engineering, FeatureEngineering):
        return list(engineering.columns)
    return list(engineering.created_columns)


def experiment_pipeline(
    model: BaseEstimator,
    categorical: list[str],
    numeric: list[str],
    engineering=None,
    scale_numeric: bool = False,
    native_categorical: bool = False,
) -> Pipeline:
    """Pipeline d'une expérience : colonnes ajoutées s'il y en a, preprocessing commun, puis le modèle.

    `engineering` peut être :

    - un `FeatureEngineering` : fonction d'ajout de colonnes, sans apprentissage ;
    - un transformer scikit-learn importable qui ajoute des colonnes à un DataFrame et donne leurs noms
      dans `created_columns`, comme `CropInteractions`. À préférer dès que le modèle doit être
      sauvegardé : un pipeline qui contient une fonction définie dans un notebook ne se recharge pas
      ailleurs.

    `scale_numeric=True` standardise les variables numériques, colonnes ajoutées comprises.
    `native_categorical=True` remplace le one-hot par `make_native_preprocessing` : les catégorielles
    restent des catégories, traitées par le modèle lui-même (boostings) ; incompatible avec
    `scale_numeric`.
    """
    if native_categorical and scale_numeric:
        raise ValueError("native_categorical et scale_numeric : la standardisation ne sert pas aux arbres")
    if engineering is None and not native_categorical:
        return build_pipeline(model, categorical, numeric, scale_numeric)

    columns = numeric + engineered_columns(engineering)
    if native_categorical:
        preprocessing = make_native_preprocessing(categorical, columns)
    else:
        preprocessing = make_preprocessing(categorical, columns, scale_numeric)
    steps = [("preprocessing", preprocessing), ("model", model)]
    if engineering is not None:
        step = FunctionTransformer(engineering.add_columns) if isinstance(engineering, FeatureEngineering) else engineering
        steps.insert(0, ("features", step))
    return Pipeline(steps)


def check_linear_rank(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict[str, float]:
    """Contrôle le rang d'une régression linéaire placée en fin de pipeline, sur l'apprentissage de chaque fold.

    `LinearRegression` néglige les directions dont la valeur singulière est inférieure à `tol` fois la
    plus grande : avec des variables d'échelles très différentes, des colonnes utiles disparaissent
    sans message (voir `make_preprocessing`). Le rang retenu par le modèle (`rank_`) est comparé au rang
    de la matrice des variables centrée, calculé à part avec NumPy ; un écart provoque une erreur.

    `cv` : liste de couples de positions (apprentissage, validation). Renvoie, pour le dernier fold, le
    nombre de colonnes, le rang et le conditionnement (plus grande valeur singulière / plus petite
    valeur singulière utilisée). Un rang inférieur au nombre de colonnes est normal avec des
    indicatrices : seules les directions négligées à tort sont une erreur.
    """
    for train_positions, _ in cv:
        fitted = clone(pipeline).fit(X.iloc[train_positions], y.iloc[train_positions])
        matrix = np.asarray(fitted[:-1].transform(X.iloc[train_positions]), dtype=float)
        centered = matrix - matrix.mean(axis=0)
        rank = int(np.linalg.matrix_rank(centered))
        if fitted[-1].rank_ != rank:
            raise ValueError(f"rang de la régression {fitted[-1].rank_}, rang de la matrice {rank} : colonnes négligées")

    singular_values = np.linalg.svd(centered, compute_uv=False)
    return {
        "columns": matrix.shape[1],
        "rank": rank,
        "condition_number": float(singular_values[0] / singular_values[rank - 1]),
    }


def cross_validate_model(pipeline: Pipeline, X, y, cv, verbose: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    """Validation croisée chronométrée d'un pipeline : scores par fold et résumé.

    Le résumé contient les métriques `cv_*` de `summarize_cv_folds`, temps d'ajustement compris, et
    `cv_total_time` : durée de toute la validation croisée (preprocessing, entraînements et calcul
    des scores des folds). Les scores par fold servent aux comparaisons appariées entre modèles.
    `verbose=False` coupe l'affichage des scores et des temps.
    """
    debut = time.perf_counter()
    folds = cross_validate_folds(pipeline, X, y, cv)
    metrics = summarize_cv_folds(folds, with_fit_time=True)
    metrics["cv_total_time"] = time.perf_counter() - debut

    if verbose:
        print(format_cv_metrics(metrics))
        print(
            f"Temps moyen d'entraînement : {metrics['cv_fit_time_mean']:.1f} s"
            f" | CV complète : {metrics['cv_total_time']:.0f} s"
        )
    return folds, metrics


def log_model_run(
    run: RunInfo,
    pipeline: Pipeline,
    X,
    categorical: list[str],
    numeric: list[str],
    metrics: dict[str, float],
    engineered: list[str] | None = None,
    scale_numeric: bool = False,
    native_categorical: bool = False,
) -> None:
    """Enregistre dans MLflow un pipeline évalué, avec les mêmes paramètres pour tous les runs.

    Paramètres : ceux du protocole, le nom du modèle (dernière étape du pipeline), le jeu de variables,
    le nombre de variables, les colonnes ajoutées (`none` s'il n'y en a pas), le nombre de colonnes vues
    par le modèle, la description du preprocessing, puis `run.params`. Les variables vont dans
    `features.json`. `scale_numeric` et `native_categorical` indiquent le preprocessing du pipeline
    (standardisation, catégories natives) : sa description en dépend.

    Le nombre de colonnes est calculé sur une copie non entraînée des étapes avant le modèle, apprise
    sur `X` : le pipeline passé n'est pas modifié, même s'il est déjà entraîné.
    """
    engineered = list(engineered or [])
    if native_categorical:
        description = PREPROCESSING_NATIVE_DESCRIPTION
    else:
        description = PREPROCESSING_SCALED_DESCRIPTION if scale_numeric else PREPROCESSING_DESCRIPTION

    params = run.protocol_params | {
        "model": type(pipeline[-1]).__name__,
        "feature_set": run.feature_set,
        "n_features": len(categorical) + len(numeric),
        "engineered_features": ", ".join(engineered) or "none",
        "n_engineered_features": len(engineered),
        "n_encoded_features": clone(pipeline[:-1]).fit_transform(X).shape[1],
        "preprocessing": description,
    } | run.params
    features = {"categorical": categorical, "numeric": numeric, "engineered": engineered}

    log_run(run.name, run.tags, params=params, metrics=metrics, features=features, replace=run.replace)


def run_experiment(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
    cv,
    categorical: list[str],
    numeric: list[str],
    engineering=None,
    run: RunInfo | None = None,
    verbose: bool = True,
    scale_numeric: bool = False,
    native_categorical: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Une expérience : pipeline, validation croisée, run MLflow ; renvoie les scores par fold et le résumé.

    Seules les colonnes `categorical + numeric` de `X` sont utilisées. Sans `run`, rien n'est écrit
    dans MLflow : c'est le cas d'une référence recalculée pour obtenir ses scores par fold.
    L'affichage reprend le nom du run, les scores, puis les temps. `scale_numeric=True` standardise
    les variables numériques dans le pipeline, donc séparément dans chaque fold. `engineering` et
    `native_categorical` : voir `experiment_pipeline`.
    """
    pipeline = experiment_pipeline(model, categorical, numeric, engineering, scale_numeric, native_categorical)
    X_experience = X[categorical + numeric]

    if verbose and run is not None:
        print(run.name)
    folds, metrics = cross_validate_model(pipeline, X_experience, y, cv, verbose=verbose)
    if run is not None:
        log_model_run(run, pipeline, X_experience, categorical, numeric, metrics,
                      engineered=engineered_columns(engineering) or None,
                      scale_numeric=scale_numeric, native_categorical=native_categorical)
    return folds, metrics
