"""Configuration du service `/predict` : dataset, variables et protocole d'évaluation.

Constantes seulement. Les fonctions qui lisent le dataset et appliquent le protocole sont dans
`training_data.py`. La configuration propre à `/recommend` ira dans `recommend_config.py`.
"""

from __future__ import annotations

from agritech.config import PATHS


# Dataset produit par le notebook 06.
PREDICT_DATASET = PATHS.data_processed / "predict_training_dataset.csv"
PREDICT_ROWS = 999_769
PREDICT_TARGET = "Yield_tons_per_hectare"

# Toutes les variables d'entrée. Les deux variables oui/non sont traitées comme des catégories à deux
# modalités.
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

# Variables retirées de `PREDICT_FEATURES` pour obtenir les variables réduites (notebook 07) : la
# comparaison des deux jeux mesure leur apport.
PREDICT_FEATURES_DROP = ["Region", "Weather_Condition", "Days_to_Harvest"]

# Variables sélectionnées par le notebook 08, utilisées pour le tuning et le modèle final.
PREDICT_SELECTED_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Fertilizer_Used", "Irrigation_Used"]

# Protocole : découpage train/test fixe, puis validation croisée sur le train.
PREDICT_TEST_SIZE = 0.2
PREDICT_CV_FOLDS = 5
PREDICT_CV_SHUFFLE = True
