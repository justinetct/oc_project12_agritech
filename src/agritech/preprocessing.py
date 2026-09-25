"""Preprocessing commun aux modèles de rendement : `/predict`, `/recommend` et modèles avancés."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# Descriptions enregistrées dans MLflow avec chaque modèle, selon le preprocessing utilisé.
PREPROCESSING_DESCRIPTION = "OneHotEncoder(handle_unknown='ignore') on categorical, passthrough on numeric"
PREPROCESSING_SCALED_DESCRIPTION = "OneHotEncoder(handle_unknown='ignore') on categorical, StandardScaler on numeric"
PREPROCESSING_NATIVE_DESCRIPTION = "NativeCategories (pandas category, handled by the model) on categorical, passthrough on numeric"


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


class NativeCategories(BaseEstimator, TransformerMixin):
    """Variables catégorielles gardées comme catégories pandas, pour les modèles qui les traitent eux-mêmes.

    HistGradientBoosting (`categorical_features="from_dtype"`, le défaut), LightGBM, XGBoost
    (`enable_categorical=True`) et CatBoost (`cat_features`) découpent directement sur une variable
    catégorielle, sans one-hot. `fit` retient les modalités vues à l'apprentissage ; `transform` renvoie
    les mêmes colonnes en type `category`, avec ces seules modalités, dans le même ordre d'un fold à
    l'autre. Une modalité inconnue provoque une erreur plutôt qu'une valeur manquante silencieuse.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {column: sorted(X[column].unique()) for column in X.columns}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column, categories in self.categories_.items():
            unknown = sorted(set(X[column]) - set(categories))
            if unknown:
                raise ValueError(f"{column} : modalités absentes de l'apprentissage {unknown}")
        return pd.DataFrame(
            {column: pd.Categorical(X[column], categories=categories) for column, categories in self.categories_.items()},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(list(self.categories_), dtype=object)


def safe_feature_names(transformer, input_features) -> list[str]:
    """Noms de colonnes sans les caractères refusés par LightGBM ou XGBoost (`[ ] < > { } " , :`), remplacés par `_`.

    Par exemple `Rice, paddy x avg_temp` devient `Rice_ paddy x avg_temp`. Deux colonnes qui
    deviendraient identiques provoquent une erreur. Fonction du module, et non d'un notebook, pour que
    le pipeline reste sérialisable.
    """
    names = [re.sub(r'[\[\]<>{}",:]', "_", str(name)) for name in input_features]
    if len(set(names)) != len(names):
        raise ValueError("noms de colonnes identiques après remplacement des caractères refusés")
    return names


def make_native_preprocessing(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Catégorielles en type `category` (`NativeCategories`), numériques telles quelles.

    La sortie est un DataFrame, avec l'index de l'entrée : un modèle qui traite les catégories lui-même
    les reconnaît à leur type, ou à leur nom pour CatBoost (`cat_features`, à donner en tuple : avec une
    liste, `clone` échoue). Les catégorielles gardent leur nom ; les numériques aussi, aux caractères
    refusés par LightGBM près (`safe_feature_names`). Pas de standardisation : elle ne change pas les
    arbres.
    """
    return ColumnTransformer(
        [
            ("categorical", NativeCategories(), categorical),
            ("numeric", FunctionTransformer(feature_names_out=safe_feature_names), numeric),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


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
