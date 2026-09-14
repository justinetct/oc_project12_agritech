"""Preprocessing commun aux modèles de rendement : `/predict`, `/recommend` et modèles avancés."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Description enregistrée dans MLflow avec chaque modèle qui utilise ce preprocessing.
PREPROCESSING_DESCRIPTION = "OneHotEncoder(handle_unknown='ignore') on categorical, passthrough on numeric"


def make_preprocessing(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """One-hot sur les variables catégorielles ; variables numériques passées telles quelles.

    - `handle_unknown="ignore"` : une modalité absente des données d'apprentissage est encodée par
      des zéros au lieu de provoquer une erreur ;
    - `sparse_output=False` : matrice dense, adaptée à quelques dizaines de colonnes ;
    - pas de standardisation : elle ne change ni les prédictions d'une régression linéaire ni celles
      des arbres. Un modèle sensible à l'échelle des variables devra l'ajouter.
    """
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )


def build_pipeline(model: BaseEstimator, categorical: list[str], numeric: list[str]) -> Pipeline:
    """Enchaîne le preprocessing et le modèle dans un `Pipeline` scikit-learn.

    `fit` apprend l'encodage sur les seules données d'apprentissage, puis `predict` le réutilise tel
    quel : pas de fuite de données, y compris en validation croisée. Le pipeline ne contient que des
    objets scikit-learn : un modèle sauvegardé ne dépend pas de ce package.
    """
    return Pipeline([("preprocessing", make_preprocessing(categorical, numeric)), ("model", model)])


def count_encoded_columns(categorical: list[str], numeric: list[str], X) -> int:
    """Nombre de colonnes produites par le preprocessing, une fois appris sur `X`."""
    return len(make_preprocessing(categorical, numeric).fit(X).get_feature_names_out())
