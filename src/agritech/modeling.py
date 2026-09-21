"""Mécanique commune des expériences de modélisation : pipeline, validation croisée et run MLflow.

Les notebooks déclarent leurs expériences : modèle, variables, éventuelles colonnes ajoutées et
informations MLflow. `run_experiment` fait le reste : construction du pipeline, validation croisée
chronométrée, run MLflow, puis renvoie les scores par fold et le résumé. La recherche
d'hyperparamètres et les analyses propres à une expérience restent dans les notebooks.

Les métriques viennent de `evaluation.py`, le preprocessing de `preprocessing.py` et l'écriture dans
MLflow de `tracking.py`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from agritech.evaluation import cross_validate_folds, format_cv_metrics, summarize_cv_folds
from agritech.preprocessing import (
    PREPROCESSING_DESCRIPTION,
    PREPROCESSING_SCALED_DESCRIPTION,
    build_pipeline,
    make_preprocessing,
)
from agritech.tracking import log_run


@dataclass(frozen=True)
class FeatureEngineering:
    """Colonnes ajoutées au début du pipeline, avant l'encodage : fonction d'ajout et noms des colonnes créées.

    Les colonnes créées sont numériques, passées telles quelles au modèle. La fonction fait partie du
    pipeline : en validation croisée, elle est appliquée séparément à chaque fold.
    """

    add_columns: Callable[[pd.DataFrame], pd.DataFrame]
    columns: list[str]


@dataclass(frozen=True)
class RunInfo:
    """Informations MLflow d'une expérience : nom du run, jeu de variables, tags et paramètres.

    `protocol_params` : paramètres du protocole, communs à tous les runs du service ;
    `params` : paramètres propres au run (hyperparamètres, phase de tuning...).
    """

    name: str
    feature_set: str
    tags: dict[str, str]
    protocol_params: dict
    params: dict = field(default_factory=dict)


def experiment_pipeline(
    model: BaseEstimator,
    categorical: list[str],
    numeric: list[str],
    engineering: FeatureEngineering | None = None,
    scale_numeric: bool = False,
) -> Pipeline:
    """Pipeline d'une expérience : colonnes ajoutées s'il y en a, preprocessing commun, puis le modèle.

    `scale_numeric=True` standardise les variables numériques, colonnes ajoutées comprises.
    """
    if engineering is None:
        return build_pipeline(model, categorical, numeric, scale_numeric)
    return Pipeline(
        [
            ("features", FunctionTransformer(engineering.add_columns)),
            ("preprocessing", make_preprocessing(categorical, numeric + engineering.columns, scale_numeric)),
            ("model", model),
        ]
    )


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
) -> None:
    """Enregistre dans MLflow un pipeline évalué, avec les mêmes paramètres pour tous les runs.

    Paramètres : ceux du protocole, le nom du modèle (dernière étape du pipeline), le jeu de variables,
    le nombre de variables, les colonnes ajoutées (`none` s'il n'y en a pas), le nombre de colonnes vues
    par le modèle, la description du preprocessing, puis `run.params`. Les variables vont dans
    `features.json`. `scale_numeric` indique si le pipeline standardise les variables numériques : la
    description du preprocessing en dépend.

    Le nombre de colonnes est calculé sur une copie non entraînée des étapes avant le modèle, apprise
    sur `X` : le pipeline passé n'est pas modifié, même s'il est déjà entraîné.
    """
    engineered = list(engineered or [])
    params = run.protocol_params | {
        "model": type(pipeline[-1]).__name__,
        "feature_set": run.feature_set,
        "n_features": len(categorical) + len(numeric),
        "engineered_features": ", ".join(engineered) or "none",
        "n_engineered_features": len(engineered),
        "n_encoded_features": clone(pipeline[:-1]).fit_transform(X).shape[1],
        "preprocessing": PREPROCESSING_SCALED_DESCRIPTION if scale_numeric else PREPROCESSING_DESCRIPTION,
    } | run.params
    features = {"categorical": categorical, "numeric": numeric, "engineered": engineered}

    log_run(run.name, run.tags, params=params, metrics=metrics, features=features)


def run_experiment(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
    cv,
    categorical: list[str],
    numeric: list[str],
    engineering: FeatureEngineering | None = None,
    run: RunInfo | None = None,
    verbose: bool = True,
    scale_numeric: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Une expérience : pipeline, validation croisée, run MLflow ; renvoie les scores par fold et le résumé.

    Seules les colonnes `categorical + numeric` de `X` sont utilisées. Sans `run`, rien n'est écrit
    dans MLflow : c'est le cas d'une référence recalculée pour obtenir ses scores par fold.
    L'affichage reprend le nom du run, les scores, puis les temps. `scale_numeric=True` standardise
    les variables numériques dans le pipeline, donc séparément dans chaque fold.
    """
    pipeline = experiment_pipeline(model, categorical, numeric, engineering, scale_numeric)
    X_experience = X[categorical + numeric]

    if verbose and run is not None:
        print(run.name)
    folds, metrics = cross_validate_model(pipeline, X_experience, y, cv, verbose=verbose)
    if run is not None:
        log_model_run(run, pipeline, X_experience, categorical, numeric, metrics,
                      engineered=engineering.columns if engineering is not None else None,
                      scale_numeric=scale_numeric)
    return folds, metrics
