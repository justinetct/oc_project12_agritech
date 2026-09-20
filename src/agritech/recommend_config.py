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

# Référence hors jeux principaux : la culture seule.
RECOMMEND_REFERENCE_SETS = {"crop_only": ["crop"]}

# Protocole temporel : 2013 est réservée au test final ; chaque année de validation est prédite par un
# modèle appris sur toutes les années antérieures.
RECOMMEND_TEST_YEAR = 2013
RECOMMEND_VALIDATION_YEARS = [2008, 2009, 2010, 2011, 2012]
