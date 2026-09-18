"""Datasets d'entraînement : lecture, contrôles, découpage et validation croisée.

Fonctions génériques, sans constante propre à un service : chaque appel reçoit le dataset, les
variables et le protocole à appliquer. Pour `/predict`, ces valeurs sont dans `predict_config.py` ;
celles de `/recommend` iront dans `recommend_config.py`.

`load_dataset`, `split` et `kfold_cv` contrôlent leur résultat et en affichent un résumé, pour que
chaque notebook n'ait pas à le réécrire ; `verbose=False` coupe l'affichage, pas les contrôles.

Le découpage aléatoire (`split`) et la validation croisée `KFold` (`kfold_cv`) conviennent à un
dataset sans ordre dans le temps, comme celui de `/predict`. Un découpage par année, pour
`/recommend`, fera l'objet de fonctions dédiées.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from agritech.config import PATHS


def load_dataset(
    path: Path, rows: int, columns: list[str], non_negative: list[str] | None = None, verbose: bool = True
) -> pd.DataFrame:
    """Lit un dataset d'entraînement, vérifie qu'il est bien celui attendu et affiche un résumé.

    Contrôles : nombre de lignes, colonnes attendues (dans n'importe quel ordre), aucune valeur
    manquante et, pour les colonnes de `non_negative`, aucune valeur négative. En cas d'écart, relancer
    le notebook qui produit le dataset.
    """
    df = pd.read_csv(path)

    if len(df) != rows or sorted(df.columns) != sorted(columns):
        raise ValueError(f"{path.name} : lignes ou colonnes inattendues")
    if df.isna().any().any():
        raise ValueError(f"{path.name} : valeurs manquantes")
    for column in non_negative or []:
        if (df[column] < 0).any():
            raise ValueError(f"{path.name} : valeurs négatives dans {column}")

    if verbose:
        # chemin relatif : pas de chemin local dans les sorties des notebooks
        chemin = path.relative_to(PATHS.root) if path.is_relative_to(PATHS.root) else path.name
        print("fichier            :", chemin)
        print("lignes x colonnes  :", df.shape)
        print("valeurs manquantes :", df.isna().sum().sum())
        for column in non_negative or []:
            print(f"valeurs < 0        : {(df[column] < 0).sum()} ({column})")
    return df


def split(
    df: pd.DataFrame, features: list[str], target: str, test_size: float, seed: int, verbose: bool = True
) -> list:
    """Découpe un dataset en `X_train, X_test, y_train, y_test`, de façon aléatoire et reproductible.

    Adapté à un dataset sans ordre dans le temps. Avec la même graine, le jeu de test est le même
    pour tous les modèles : il peut rester réservé à l'évaluation finale. Contrôle qu'aucune ligne
    n'est perdue ni commune au train et au test, puis affiche les tailles.
    """
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, random_state=seed)

    if len(X_train) + len(X_test) != len(df) or not X_train.index.intersection(X_test.index).empty:
        raise ValueError("découpage incohérent : lignes perdues ou communes au train et au test")

    if verbose:
        print(f"total   : {len(df)} lignes")
        print(f"X_train : {X_train.shape}")
        print(f"X_test  : {X_test.shape}, réservé à l'évaluation finale")
        print(f"y_train : {y_train.shape}")
        print(f"y_test  : {y_test.shape}")
    return [X_train, X_test, y_train, y_test]


def kfold_cv(n_splits: int, shuffle: bool, seed: int, verbose: bool = True) -> KFold:
    """Validation croisée en `n_splits` folds, à appliquer au seul jeu d'entraînement.

    Avec la même graine, les folds sont identiques d'un modèle à l'autre : les scores sont
    comparables.
    """
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if verbose:
        print("validation croisée :", cv)
    return cv


def feature_types(features: list[str], categorical: list[str], numeric: list[str]) -> tuple[list[str], list[str]]:
    """Sépare une sélection de variables en catégorielles et numériques, dans l'ordre de la configuration."""
    unknown = set(features) - set(categorical) - set(numeric)
    if unknown:
        raise ValueError(f"variables inconnues : {sorted(unknown)}")

    return [col for col in categorical if col in features], [col for col in numeric if col in features]


def protocol_params(
    dataset: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_size: float,
    seed: int,
    cv_folds: int,
    cv_shuffle: bool,
) -> dict:
    """Paramètres du protocole, enregistrés sous les mêmes noms dans chaque run MLflow du service."""
    return {
        "dataset": dataset.name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": seed,
        "cv_folds": cv_folds,
        "cv_shuffle": cv_shuffle,
    }
