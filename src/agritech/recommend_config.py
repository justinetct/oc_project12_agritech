"""Configuration du service `/recommend` : dataset, variables, jeux de features et protocole d'évaluation.

Constantes seulement. Les fonctions qui lisent le dataset et appliquent le protocole temporel sont
dans `training_data.py`. La configuration de `/predict` est dans `predict_config.py`.
"""

from __future__ import annotations

from agritech.config import PATHS


# Dataset produit par le notebook 06 : le dataset historique nettoyé, une ligne par pays, année et culture.
RECOMMEND_DATASET = PATHS.data_processed / "recommend_training_dataset.csv"
RECOMMEND_ROWS = 16_319
RECOMMEND_KEY = ["iso3", "year", "crop"]
RECOMMEND_TARGET = "yield_t_ha"

# Colonnes du fichier. `area`, le nom du pays, n'est jamais une variable du modèle.
RECOMMEND_COLUMNS = ["iso3", "area", "year", "crop", "avg_temp", "rain_mm", "pesticides_t", RECOMMEND_TARGET]

# Variables candidates, d'origine et sans transformation. `iso3` est catégorielle : un code pays n'a pas
# d'ordre. `year` est numérique : une régression linéaire en tire une tendance.
RECOMMEND_CATEGORICAL = ["crop", "iso3"]
RECOMMEND_NUMERIC = ["year", "avg_temp", "rain_mm", "pesticides_t"]
RECOMMEND_FEATURES = RECOMMEND_CATEGORICAL + RECOMMEND_NUMERIC

# Jeux de features de la baseline, comparés avec le même modèle et le même protocole. Leurs noms
# servent aussi de `feature_set` dans MLflow.
RECOMMEND_CONDITIONS = ["crop", "avg_temp", "rain_mm", "pesticides_t"]
RECOMMEND_FEATURE_SETS = {
    "conditions": RECOMMEND_CONDITIONS,
    "conditions_year": RECOMMEND_CONDITIONS + ["year"],
    "conditions_country": RECOMMEND_CONDITIONS + ["iso3"],
    "conditions_country_year": RECOMMEND_CONDITIONS + ["iso3", "year"],
}

# Les 10 cultures du dataset, par ordre alphabétique : cultures classées par `/recommend`, et modalités
# connues des transformers de `recommend_features.py`.
RECOMMEND_CROPS = [
    "Cassava",
    "Maize",
    "Plantains and others",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Sweet potatoes",
    "Wheat",
    "Yams",
]

# Variables dérivées, créées par `recommend_features.py` (notebooks 13 à 15).
RECOMMEND_LOG_CONDITIONS = ["avg_temp", "rain_mm", "log_pesticides"]
RECOMMEND_GEOGRAPHY = ["lat_abs", "geo_x", "geo_y", "geo_z"]
RECOMMEND_HISTORICAL_CONDITIONS = ["temp_hist", "rain_mm", "log_pest_hist"]

# Protocole temporel : 2013 est réservée au test final ; chaque année de validation est prédite par un
# modèle appris sur toutes les années antérieures.
RECOMMEND_TEST_YEAR = 2013
RECOMMEND_VALIDATION_YEARS = [2008, 2009, 2010, 2011, 2012]

# Modèles finalistes du notebook 14 : meilleure configuration de chaque famille sur la représentation `history`
# (tuning approfondi exhaustif des grilles recentrées), et encodage retenu pour `crop`. Le notebook 15 les reconstruit
# sans refaire de recherche, puis retient ExtraTrees réduit à 150 arbres : ce modèle final est défini dans le notebook
# 15 et sauvegardé dans `models/recommend_model.joblib`, le seul que l'API utilisera.
RECOMMEND_FINALISTS = {
    "extra_trees": {
        "native_categorical": False,
        "params": {"n_estimators": 300, "max_features": 0.9, "min_samples_split": 3},
    },
    "catboost": {
        "native_categorical": False,
        "params": {"iterations": 1000, "learning_rate": 0.2, "depth": 12, "l2_leaf_reg": 3},
    },
    "lightgbm": {
        "native_categorical": True,
        "params": {"n_estimators": 1500, "learning_rate": 0.1, "num_leaves": 127, "min_child_samples": 10,
                   "subsample": 0.7, "subsample_freq": 1, "reg_lambda": 20.0},
    },
}
