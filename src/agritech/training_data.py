"""Datasets d'entraînement produits par le notebook 06 : variables, lecture et protocole d'évaluation.

Protocole `/predict`, commun à tous les modèles :

- un découpage train/test fixe ; le jeu de test est réservé à l'évaluation finale ;
- une validation croisée sur le train pour comparer modèles, jeux de variables et hyperparamètres.

Seule la partie `/predict` existe pour l'instant ; la partie `/recommend` sera ajoutée ici lors de sa
modélisation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from agritech.config import PATHS, SEED


# ---------------------------------------------------------------------------
# /predict — Agriculture CropYield
# ---------------------------------------------------------------------------

PREDICT_DATASET = PATHS.data_processed / "predict_training_dataset.csv"
PREDICT_ROWS = 999_769
PREDICT_TARGET = "Yield_tons_per_hectare"

# Les deux variables oui/non sont traitées comme des catégories à deux modalités.
PREDICT_CATEGORICAL = [
    "Crop",
    "Soil_Type",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Region",
    "Weather_Condition",
]
PREDICT_NUMERIC = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
PREDICT_FEATURES = PREDICT_CATEGORICAL + PREDICT_NUMERIC

# Configuration métier envisagée ; la sélection finale des variables reste ouverte.
PREDICT_BUSINESS_FEATURES = [
    "Crop",
    "Soil_Type",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
]

PREDICT_TEST_SIZE = 0.2
PREDICT_CV_FOLDS = 5
PREDICT_CV_SHUFFLE = True


def load_predict_dataset() -> pd.DataFrame:
    """Lit le dataset `/predict` et vérifie qu'il est bien celui produit par le notebook 06.

    Contrôles : nombre de lignes, colonnes attendues, aucune valeur manquante et aucun rendement
    négatif. En cas d'écart, relancer le notebook 06.
    """
    df = pd.read_csv(PREDICT_DATASET)

    if len(df) != PREDICT_ROWS or sorted(df.columns) != sorted(PREDICT_FEATURES + [PREDICT_TARGET]):
        raise ValueError(f"{PREDICT_DATASET.name} : lignes ou colonnes inattendues")
    if df.isna().any().any():
        raise ValueError(f"{PREDICT_DATASET.name} : valeurs manquantes")
    if (df[PREDICT_TARGET] < 0).any():
        raise ValueError(f"{PREDICT_DATASET.name} : rendements négatifs")
    return df


def split_predict(df: pd.DataFrame) -> list:
    """Découpe le dataset `/predict` en `X_train, X_test, y_train, y_test`.

    Découpage aléatoire 80/20 : le jeu n'a ni date ni année, il n'y a pas d'ordre dans le temps à
    respecter. `random_state=SEED` rend le découpage fixe : le jeu de test est le même pour tous les
    modèles et reste réservé à l'évaluation finale.
    """
    return train_test_split(
        df[PREDICT_FEATURES],
        df[PREDICT_TARGET],
        test_size=PREDICT_TEST_SIZE,
        random_state=SEED,
    )


def predict_cv() -> KFold:
    """Validation croisée des modèles `/predict`, à appliquer au seul jeu d'entraînement.

    5 folds mélangés et reproductibles : chaque modèle est entraîné 5 fois sur 4/5 du train et évalué
    sur le cinquième restant. Les folds sont identiques d'un modèle à l'autre, donc les scores sont
    comparables.
    """
    return KFold(n_splits=PREDICT_CV_FOLDS, shuffle=PREDICT_CV_SHUFFLE, random_state=SEED)


def predict_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    """Sépare une sélection de variables `/predict` en catégorielles et numériques."""
    unknown = set(features) - set(PREDICT_FEATURES)
    if unknown:
        raise ValueError(f"variables inconnues : {sorted(unknown)}")

    categorical = [col for col in PREDICT_CATEGORICAL if col in features]
    numeric = [col for col in PREDICT_NUMERIC if col in features]
    return categorical, numeric


def predict_protocol_params(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """Paramètres du protocole, enregistrés sous les mêmes noms dans chaque run MLflow `/predict`."""
    return {
        "dataset": PREDICT_DATASET.name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": PREDICT_TEST_SIZE,
        "random_state": SEED,
        "cv_folds": PREDICT_CV_FOLDS,
        "cv_shuffle": PREDICT_CV_SHUFFLE,
    }
