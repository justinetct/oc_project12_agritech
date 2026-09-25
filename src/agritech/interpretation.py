"""Interprétation des modèles de rendement : importance des variables par permutation, hors apprentissage.

L'importance est mesurée sur des lignes que le modèle n'a pas vues : pour chaque fold, le modèle est
appris sur l'apprentissage, puis des variables sont permutées dans la validation seulement. Elle
décrit un modèle donné, pas une vérité générale : deux variables corrélées se partagent le signal
différemment d'un modèle à l'autre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from agritech.config import SEED


def permutation_importance_by_family(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    families: dict[str, list[str]],
    n_repeats: int = 10,
    seed: int = SEED,
    fold_labels: list | None = None,
) -> pd.DataFrame:
    """Hausse de RMSE quand une famille de variables est permutée dans la validation, fold par fold.

    - `cv` : liste de couples de positions (apprentissage, validation), comme ceux de `temporal_cv` ;
    - `families` : nom de famille → colonnes d'entrée de `X` permutées **ensemble**, avec le même
      tirage. Regrouper dans une famille toutes les colonnes tirées de la même information : les
      coordonnées d'un pays, ou une variable et ses dérivées calculées avant le pipeline. Les colonnes
      créées dans le pipeline (interactions par culture) suivent d'elles-mêmes la permutation ;
    - `fold_labels` : libellés des folds, par exemple les années de validation.

    Renvoie une ligne par famille et par fold, avec la hausse moyenne de RMSE sur `n_repeats`
    permutations. Erreurs explicites : colonne absente, colonne dans deux familles, ou famille
    constante dans un fold de validation (`year` en validation temporelle) : la permuter ne change rien.
    """
    columns = [column for family in families.values() for column in family]
    missing = sorted(set(columns) - set(X.columns))
    if missing:
        raise ValueError(f"colonnes absentes de X : {missing}")
    if len(columns) != len(set(columns)):
        raise ValueError("une colonne appartient à plusieurs familles")

    folds = list(cv)
    fold_labels = list(fold_labels) if fold_labels is not None else list(range(1, len(folds) + 1))
    if len(fold_labels) != len(folds):
        raise ValueError("un libellé par fold est attendu")

    generator = np.random.default_rng(seed)
    rows = []
    for label, (train_positions, validation_positions) in zip(fold_labels, folds):
        fitted = clone(model).fit(X.iloc[train_positions], y.iloc[train_positions])
        X_validation = X.iloc[validation_positions]
        y_validation = np.asarray(y.iloc[validation_positions], dtype=float)
        reference = _rmse(y_validation, fitted.predict(X_validation))

        for family, family_columns in families.items():
            if len(X_validation[family_columns].drop_duplicates()) == 1:
                raise ValueError(f"{family} est constante dans le fold {label} : la permuter ne change rien")

            increases = []
            for _ in range(n_repeats):
                order = generator.permutation(len(X_validation))
                permuted = X_validation.copy()
                # valeurs NumPy : une affectation pandas réalignerait les index et annulerait la permutation
                permuted[family_columns] = X_validation[family_columns].to_numpy()[order]
                increases.append(_rmse(y_validation, fitted.predict(permuted)) - reference)
            rows.append({"famille": family, "fold": label, "hausse de RMSE": float(np.mean(increases))})
    return pd.DataFrame(rows)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred, dtype=float) - y_true) ** 2)))
