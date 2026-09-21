"""Preprocessing commun aux modèles de rendement : `/predict`, `/recommend` et modèles avancés."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Descriptions enregistrées dans MLflow avec chaque modèle, selon le preprocessing utilisé.
PREPROCESSING_DESCRIPTION = "OneHotEncoder(handle_unknown='ignore') on categorical, passthrough on numeric"
PREPROCESSING_SCALED_DESCRIPTION = "OneHotEncoder(handle_unknown='ignore') on categorical, StandardScaler on numeric"


def make_preprocessing(categorical: list[str], numeric: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot sur les variables catégorielles ; variables numériques passées telles quelles ou standardisées.

    - `handle_unknown="ignore"` : une modalité absente des données d'apprentissage est encodée par
      des zéros au lieu de provoquer une erreur ;
    - `sparse_output=False` : matrice dense, adaptée à quelques dizaines de colonnes ;
    - `scale_numeric=False` (défaut) : pas de standardisation. Elle ne change pas les prédictions des
      arbres, ni celles d'une régression linéaire dont les variables ont des échelles comparables ;
    - `scale_numeric=True` : `StandardScaler` sur les variables numériques, appris sur les seules données
      d'apprentissage. Nécessaire quand une variable est très grande devant les colonnes 0/1, comme
      `pesticides_t` dans `/recommend` : `LinearRegression` néglige les directions dont la valeur
      singulière est inférieure à `tol` (1e-6) fois la plus grande, et ignorerait les colonnes des pays.
    """
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric),
        ]
    )


def build_pipeline(
    model: BaseEstimator, categorical: list[str], numeric: list[str], scale_numeric: bool = False
) -> Pipeline:
    """Enchaîne le preprocessing et le modèle dans un `Pipeline` scikit-learn.

    `fit` apprend l'encodage, et la standardisation si `scale_numeric=True`, sur les seules données
    d'apprentissage, puis `predict` les réutilise tels quels : pas de fuite de données, y compris en
    validation croisée. Le pipeline ne contient que des objets scikit-learn : un modèle sauvegardé ne
    dépend pas de ce package.
    """
    return Pipeline([("preprocessing", make_preprocessing(categorical, numeric, scale_numeric)), ("model", model)])


def count_encoded_columns(categorical: list[str], numeric: list[str], X) -> int:
    """Nombre de colonnes produites par le preprocessing, une fois appris sur `X`."""
    return len(make_preprocessing(categorical, numeric).fit(X).get_feature_names_out())
