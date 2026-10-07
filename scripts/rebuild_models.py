"""Reconstruit les modèles servis par l'API à partir des données préparées.

    poetry run python scripts/rebuild_models.py                       # écrit dans models/
    poetry run python scripts/rebuild_models.py --output-dir /tmp/x   # reconstruction de contrôle

Cinq fichiers sont produits, ceux que charge l'API :

- `predict_model.joblib` et `predict_model_metadata.json` : régression linéaire sur les 4 variables
  sélectionnées, celle évaluée par le notebook 11, réapprise sur toutes les lignes du dataset ;
- `recommend_model.joblib`, `recommend_model_metadata.json` et `recommend_context.json` : ExtraTrees
  réappris sur 1991-2013, et contexte pays du serving (mêmes étapes que le notebook 16). Le modèle est
  écrit par `dump_without_tree_state_memo` (voir `agritech.serialization`) : même compression lzma que
  `joblib.dump`, sans le pic mémoire au chargement par l'API.

Le script ne refait aucun choix : variables, hyperparamètres et protocoles sont ceux des notebooks,
les valeurs figées sont dans `predict_config.py` et `recommend_config.py`. Il n'y a ni validation
croisée, ni évaluation sur le test, ni MLflow : les métriques des metadata sont celles de l'évaluation
finale déjà faite (notebooks 11 et 15), recopiées telles quelles dans la partie `evaluation`, qui décrit
le modèle évalué ; la partie `refit` décrit l'apprentissage du modèle servi, qui n'est pas réévalué.
`created_on` et `versions` décrivent la reconstruction elle-même.

Il n'a pas besoin des fichiers de `models/` pour reconstruire. Contrôles, dans l'ordre :

1. contrôles des modèles reconstruits : effectifs, variables, modalités, paramètres, domaines, contexte ;
2. si `models/` contient déjà les artefacts servis : prédictions identiques (à 1e-9 près), metadata
   identiques hors `created_on` et `versions`, contexte identique. Les `.joblib` ne sont pas comparés
   octet par octet : deux fichiers différents peuvent donner les mêmes prédictions ;
3. écriture, seulement si les contrôles précédents passent ;
4. rechargement par le code de serving de l'API, puis prédictions de référence.

Prérequis : les datasets de `data/processed/` (notebooks 04 à 06) et le GeoJSON de `data/geo/`, non
versionnés (voir `data/README.md`).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from agritech.config import PATHS, SEED
from agritech.geo import GEOJSON_PAR_DEFAUT, charger_coordonnees
from agritech.modeling import experiment_pipeline
from agritech.predict_config import (
    PREDICT_CATEGORICAL,
    PREDICT_CV_FOLDS,
    PREDICT_CV_SHUFFLE,
    PREDICT_DATASET,
    PREDICT_FEATURES,
    PREDICT_FINAL_TEST_METRICS,
    PREDICT_MODEL_VERSION,
    PREDICT_NUMERIC,
    PREDICT_ROWS,
    PREDICT_SELECTED_FEATURES,
    PREDICT_TARGET,
    PREDICT_TEST_SIZE,
    PREDICT_TRAINING_DOMAIN,
)
from agritech.preprocessing import PREPROCESSING_DESCRIPTION, build_pipeline
from agritech.recommend_config import (
    RECOMMEND_COLUMNS,
    RECOMMEND_CROPS,
    RECOMMEND_DATASET,
    RECOMMEND_FINAL_PARAMS,
    RECOMMEND_FINAL_TEST_METRICS,
    RECOMMEND_GEOGRAPHY,
    RECOMMEND_HISTORICAL_CONDITIONS,
    RECOMMEND_MODEL_VERSION,
    RECOMMEND_ROWS,
    RECOMMEND_TARGET,
    RECOMMEND_TEST_YEAR,
    RECOMMEND_VALIDATION_YEARS,
)
from agritech.recommend_features import add_historical_conditions, add_recommend_features
from agritech.serialization import dump_without_tree_state_memo
from agritech.serving import (
    RECOMMEND_TARGET_YEAR,
    load_bundle,
    load_recommend_context,
    predict,
    public_training_domain,
    public_training_domain_recommend,
    recommend,
)
from agritech.training_data import feature_types, load_dataset, protocol_params, split, temporal_split

SERVED_DIR = PATHS.root / "models"

# Effectifs attendus : évaluation des notebooks 11 et 15, réapprentissages du serving (/predict ici, /recommend
# dans le notebook 16).
PREDICT_N_TRAIN, PREDICT_N_TEST, PREDICT_N_REFIT = 799_815, 199_954, 999_769
RECOMMEND_N_TRAIN, RECOMMEND_N_TEST, RECOMMEND_N_REFIT = 14_941, 695, 15_636
RECOMMEND_N_COUNTRIES = 115

# Variables du modèle `/recommend` final (notebooks 15 et 16).
RECOMMEND_FINAL_CATEGORICAL = ["crop"]
RECOMMEND_FINAL_NUMERIC = ["year", *RECOMMEND_HISTORICAL_CONDITIONS, *RECOMMEND_GEOGRAPHY]

# Historique enregistré dans le contexte : les 3 années qui précèdent l'année cible du serving.
HISTORY_YEARS = [RECOMMEND_TARGET_YEAR - lag for lag in (3, 2, 1)]

# Prédictions de référence, obtenues avec les artefacts servis (predict 1.1.0 du 06/10/2026, recommend du
# 28/09/2026) par le code de serving de l'API. Une reconstruction correcte les retrouve.
TOLERANCE = 1e-6  # t/ha
REFERENCE_PREDICT = [
    ({"rainfall_mm": 500.0, "temperature_celsius": 25.0, "fertilizer_used": True, "irrigation_used": False}, 4.501372),
    ({"rainfall_mm": 900.0, "temperature_celsius": 18.0, "fertilizer_used": False, "irrigation_used": True}, 6.059238),
]
REFERENCE_RECOMMEND_ISO3 = "FRA"  # conditions préremplies du pays
REFERENCE_RECOMMEND = [
    ("Potatoes", 42.242119),
    ("Sweet potatoes", 20.743762),
    ("Yams", 16.390216),
    ("Cassava", 13.211188),
    ("Plantains and others", 12.325997),
    ("Maize", 8.617059),
    ("Wheat", 7.145667),
    ("Sorghum", 5.605301),
    ("Rice, paddy", 4.730775),
    ("Soybeans", 2.650379),
]


def check(condition: bool, message: str) -> None:
    """Arrête la reconstruction avec un message explicite si un contrôle échoue."""
    if not condition:
        raise RuntimeError(f"contrôle échoué : {message}")


def installed_versions() -> dict[str, str]:
    """Versions réellement utilisées pour la reconstruction."""
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def categorical_values(pipeline, categorical: list[str]) -> dict[str, list]:
    """Modalités apprises par le one-hot du pipeline, par variable catégorielle."""
    encoder = pipeline["preprocessing"].named_transformers_["categorical"]
    return {name: values.tolist() for name, values in zip(categorical, encoder.categories_)}


# --- /predict ---------------------------------------------------------------------------------------


def build_predict() -> tuple:
    """Modèle `/predict` : le modèle évalué par le notebook 11, réappris sur toutes les lignes.

    Le découpage du notebook 11 n'apprend rien ici : il décrit l'évaluation dans les metadata et
    fournit 1 000 lignes fixes du test (une toutes les 200), qui servent à comparer avec le modèle
    servi. Le pipeline, avec les mêmes variables et les réglages par défaut, est appris sur les
    `PREDICT_ROWS` lignes. Il n'est pas évalué : les métriques restent celles du modèle évalué.
    """
    df = load_dataset(PREDICT_DATASET, PREDICT_ROWS, PREDICT_FEATURES + [PREDICT_TARGET],
                      non_negative=[PREDICT_TARGET], verbose=False)
    X_train, X_test, _, _ = split(df, PREDICT_FEATURES, PREDICT_TARGET, PREDICT_TEST_SIZE, SEED, verbose=False)
    features = PREDICT_SELECTED_FEATURES
    categorical, numeric = feature_types(features, PREDICT_CATEGORICAL, PREDICT_NUMERIC)

    pipeline = build_pipeline(LinearRegression(), categorical, numeric).fit(df[features], df[PREDICT_TARGET])

    protocol = protocol_params(PREDICT_DATASET, X_train, X_test, PREDICT_TEST_SIZE, SEED,
                               PREDICT_CV_FOLDS, PREDICT_CV_SHUFFLE)
    metadata = predict_metadata(pipeline, categorical, numeric, protocol, n_refit=len(df))
    return pipeline, metadata, X_test[features].iloc[::200]


def predict_metadata(pipeline, categorical: list[str], numeric: list[str], protocol: dict, n_refit: int) -> dict:
    """Metadata de `/predict`, dans l'ordre des clés du fichier servi.

    `refit` décrit l'apprentissage du modèle servi (`n_refit` lignes) ; `evaluation` décrit le modèle
    évalué par le notebook 11 (`protocol` : `training_data.protocol_params`) et porte ses métriques.
    Les métriques, la version et le domaine d'apprentissage sont les valeurs figées de `predict_config.py`.
    """
    return {
        "service": "predict",
        "model_type": type(pipeline["model"]).__name__,
        "artifact": "predict_model.joblib",
        "model_version": PREDICT_MODEL_VERSION,
        "target": PREDICT_TARGET,
        "features": PREDICT_SELECTED_FEATURES,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "categorical_values": categorical_values(pipeline, categorical),
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "training_dataset": protocol["dataset"],
        "refit": {
            "n_samples": n_refit,
            "trained_on": "all_rows_after_cleaning",
        },
        "evaluation": {
            "n_train": protocol["n_train"],
            "n_test": protocol["n_test"],
            "test_size": protocol["test_size"],
            "random_state": protocol["random_state"],
            "cv_folds": protocol["cv_folds"],
            "trained_on": "train_only",
            "test_metrics": PREDICT_FINAL_TEST_METRICS,
            "notebook": "notebooks/11_predict_final_evaluation.ipynb",
        },
        "training_domain": PREDICT_TRAINING_DOMAIN,
        "built_by": "scripts/rebuild_models.py",
        "created_on": date.today().isoformat(),
        "versions": installed_versions(),
    }


def check_predict(pipeline, metadata: dict) -> None:
    """Contrôles du modèle `/predict` reconstruit, sans les artefacts servis."""
    evaluation, refit = metadata["evaluation"], metadata["refit"]
    check((evaluation["n_train"], evaluation["n_test"]) == (PREDICT_N_TRAIN, PREDICT_N_TEST),
          "découpage de l'évaluation /predict")
    check(refit["n_samples"] == PREDICT_N_REFIT == evaluation["n_train"] + evaluation["n_test"],
          f"lignes du réapprentissage /predict : {refit['n_samples']}")
    check(evaluation["test_metrics"] == PREDICT_FINAL_TEST_METRICS, "métriques /predict recopiées")
    check(type(pipeline["model"]) is LinearRegression, "type du modèle /predict")
    check(pipeline["model"].get_params() == LinearRegression().get_params(), "hyperparamètres /predict par défaut")
    check(list(pipeline.feature_names_in_) == PREDICT_SELECTED_FEATURES, "variables /predict")
    check(metadata["categorical_values"] == {"Fertilizer_Used": [False, True], "Irrigation_Used": [False, True]},
          "modalités /predict")
    check(set(metadata["training_domain"]) <= set(metadata["numeric_features"]), "domaine /predict")
    print(f"/predict   : {refit['n_samples']} lignes réapprises (évaluation : {evaluation['n_train']} + "
          f"{evaluation['n_test']}), variables {metadata['features']}")


# --- /recommend -------------------------------------------------------------------------------------


def build_recommend() -> tuple:
    """Modèle `/recommend` : mêmes étapes que le notebook 16, réapprentissage sur 1991-2013.

    Les conditions historiques et la géographie viennent des fonctions du package, comme à
    l'entraînement des notebooks 13 à 16. Seules les lignes avec historique sont gardées, dans
    l'ordre du notebook 16 : 1991-2012 puis 2013. Renvoie le pipeline, ses metadata, le contexte de
    serving et les lignes d'apprentissage, qui servent à comparer avec le modèle servi.
    """
    df = load_dataset(RECOMMEND_DATASET, RECOMMEND_ROWS, RECOMMEND_COLUMNS,
                      non_negative=[RECOMMEND_TARGET], verbose=False)
    df = add_historical_conditions(add_recommend_features(df, charger_coordonnees()))

    categorical, numeric = RECOMMEND_FINAL_CATEGORICAL, RECOMMEND_FINAL_NUMERIC
    features = categorical + numeric
    X_train, X_test, y_train, y_test = temporal_split(df, ["iso3", *features], RECOMMEND_TARGET,
                                                      RECOMMEND_TEST_YEAR, verbose=False)
    with_history_train = X_train["temp_hist"].notna().to_numpy()
    with_history_test = X_test["temp_hist"].notna().to_numpy()
    X_hist, y_hist = X_train[with_history_train], y_train[with_history_train]
    X_test_hist, y_test_hist = X_test[with_history_test], y_test[with_history_test]
    X_refit = pd.concat([X_hist, X_test_hist], ignore_index=True)
    y_refit = pd.concat([y_hist, y_test_hist], ignore_index=True)

    check((len(X_hist), len(X_test_hist), len(X_refit)) == (RECOMMEND_N_TRAIN, RECOMMEND_N_TEST, RECOMMEND_N_REFIT),
          f"lignes /recommend {len(X_hist)} + {len(X_test_hist)} = {len(X_refit)}")
    check(with_history_test.all(), "lignes 2013 sans historique")

    pipeline = experiment_pipeline(ExtraTreesRegressor(**RECOMMEND_FINAL_PARAMS), categorical, numeric)
    pipeline.fit(X_refit[features], y_refit)

    metadata = recommend_metadata(pipeline, X_hist, X_test_hist, X_refit)
    return pipeline, metadata, build_recommend_context(df), X_refit[features]


def recommend_metadata(pipeline, X_hist: pd.DataFrame, X_test_hist: pd.DataFrame, X_refit: pd.DataFrame) -> dict:
    """Metadata de `/recommend`, dans l'ordre des clés du fichier servi.

    `X_hist` et `X_test_hist` : lignes avec historique de 1991-2012 et de 2013, celles de l'évaluation
    du notebook 15 ; `X_refit` : leur réunion, apprise par le modèle servi. Le domaine d'apprentissage
    est calculé sur `X_refit` ; les métriques et la version sont les valeurs figées de
    `recommend_config.py`.
    """
    categorical, numeric = RECOMMEND_FINAL_CATEGORICAL, RECOMMEND_FINAL_NUMERIC
    training_domain = {
        name: {"min": float(X_refit[name].min()), "max": float(X_refit[name].max()), "unit": unit}
        for name, unit in [("temp_hist", "°C"), ("rain_mm", "mm"), ("log_pest_hist", "log(t+1)")]
    }
    return {
        "service": "recommend",
        "model_type": type(pipeline["model"]).__name__,
        "artifact": "recommend_model.joblib",
        "target": RECOMMEND_TARGET,
        "features": categorical + numeric,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "categorical_values": categorical_values(pipeline, categorical),
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "model_params": {name: pipeline["model"].get_params()[name] for name in RECOMMEND_FINAL_PARAMS},
        "training_dataset": RECOMMEND_DATASET.name,
        "refit": {
            "period": f"{X_refit['year'].min()}-{X_refit['year'].max()}",
            "n_samples": len(X_refit),
            "trained_on": "all_rows_with_history_1991_to_2013",
            "target_year_for_serving": RECOMMEND_TARGET_YEAR,
        },
        "evaluation": {
            "period_train": f"{X_hist['year'].min()}-{X_hist['year'].max()}",
            "period_test": RECOMMEND_TEST_YEAR,
            "n_train": len(X_hist),
            "n_test": len(X_test_hist),
            "cv_strategy": "expanding_window_by_year",
            "cv_folds": len(RECOMMEND_VALIDATION_YEARS),
            "cv_validation_years": f"{min(RECOMMEND_VALIDATION_YEARS)}-{max(RECOMMEND_VALIDATION_YEARS)}",
            "test_metrics": RECOMMEND_FINAL_TEST_METRICS,
            "notebook": "notebooks/15_recommend_final_model.ipynb",
        },
        "notebook": "notebooks/16_recommend_final_refit.ipynb",
        "created_on": date.today().isoformat(),
        "versions": installed_versions(),
        "model_version": RECOMMEND_MODEL_VERSION,
        "training_domain": training_domain,
    }


def build_recommend_context(df: pd.DataFrame) -> dict:
    """Contexte pays de `/recommend` : de quoi préremplir les conditions et construire les 10 candidats.

    Pour chaque pays : nom, pluie, géographie déjà calculée, conditions des années `HISTORY_YEARS` et
    cultures observées. `df` : le dataset avec ses variables géographiques (notebook 16).
    """
    geography = df.drop_duplicates("iso3").set_index("iso3")[RECOMMEND_GEOGRAPHY]
    countries = {}
    for iso3 in sorted(df["iso3"].unique()):
        rows = df[df["iso3"] == iso3]
        history = rows[rows["year"].isin(HISTORY_YEARS)].drop_duplicates(["iso3", "year"]).sort_values("year")
        countries[iso3] = {
            "country": str(rows["area"].iloc[0]),
            "rain_mm": float(rows["rain_mm"].iloc[0]),
            "geography": {name: float(geography.loc[iso3, name]) for name in RECOMMEND_GEOGRAPHY},
            "history_2011_2013": [
                {"year": int(row.year), "avg_temp": float(row.avg_temp), "pesticides_t": float(row.pesticides_t)}
                for row in history.itertuples()
            ],
            "observed_crops": sorted(rows["crop"].unique().tolist()),
        }
    return {
        "generated_from": {"dataset": RECOMMEND_DATASET.name, "geojson": GEOJSON_PAR_DEFAUT.name},
        "history_years": HISTORY_YEARS,
        "target_year": RECOMMEND_TARGET_YEAR,
        "countries": countries,
    }


def check_recommend(pipeline, metadata: dict, context: dict) -> None:
    """Contrôles du modèle `/recommend` reconstruit et de son contexte, sans les artefacts servis."""
    check(type(pipeline["model"]) is ExtraTreesRegressor, "type du modèle /recommend")
    check(metadata["model_params"] == RECOMMEND_FINAL_PARAMS, "hyperparamètres /recommend")
    check(list(pipeline.feature_names_in_) == metadata["features"], "variables /recommend")
    check(metadata["categorical_values"] == {"crop": RECOMMEND_CROPS}, "cultures apprises")
    check(list(metadata["training_domain"]) == ["temp_hist", "rain_mm", "log_pest_hist"], "domaine /recommend")
    check(all(bounds["min"] < bounds["max"] for bounds in metadata["training_domain"].values()), "bornes /recommend")

    countries = context["countries"]
    check(len(countries) == RECOMMEND_N_COUNTRIES, f"{len(countries)} pays dans le contexte")
    for iso3, entry in countries.items():
        check([item["year"] for item in entry["history_2011_2013"]] == HISTORY_YEARS, f"historique de {iso3}")
        check(bool(entry["observed_crops"]), f"aucune culture observée pour {iso3}")
    observed = {crop for entry in countries.values() for crop in entry["observed_crops"]}
    check(observed == set(RECOMMEND_CROPS), "cultures observées dans le contexte")
    print(f"/recommend : {metadata['refit']['n_samples']} lignes {metadata['refit']['period']}, "
          f"{len(countries)} pays, {len(observed)} cultures")


# --- Comparaison, écriture et rechargement -----------------------------------------------------------


def without_build_info(metadata: dict) -> dict:
    """Metadata sans les champs propres à chaque reconstruction (`created_on`, `versions`)."""
    return {key: value for key, value in metadata.items() if key not in ("created_on", "versions")}


def compare_with_served(name: str, pipeline, metadata: dict, X_sample: pd.DataFrame) -> bool:
    """Compare un modèle reconstruit au modèle servi dans `models/`, s'il existe.

    Prédictions identiques à 1e-9 près sur `X_sample`, mêmes metadata hors `created_on` et `versions`,
    dans le même ordre. Renvoie False, sans rien contrôler, si le modèle servi est absent.
    """
    if not (SERVED_DIR / f"{name}_model.joblib").is_file():
        return False
    served = load_bundle(name, SERVED_DIR)
    difference = np.abs(pipeline.predict(X_sample) - served.pipeline.predict(X_sample)).max()
    check(difference <= 1e-9, f"/{name} : écart de {difference:.2e} avec le modèle servi")
    check(without_build_info(metadata) == without_build_info(served.metadata), f"/{name} : metadata différentes")
    check(list(metadata) == list(served.metadata), f"/{name} : ordre des clés des metadata")
    print(f"{'/' + name:<11}: identique au modèle servi ({len(X_sample)} prédictions, écart max {difference:.1e} t/ha)")
    return True


def compare_context_with_served(context: dict) -> bool:
    """Compare le contexte reconstruit à `models/recommend_context.json`, s'il existe."""
    path = SERVED_DIR / "recommend_context.json"
    if not path.is_file():
        return False
    check(context == json.loads(path.read_text(encoding="utf-8")), "contexte /recommend différent du contexte servi")
    print("contexte   : identique au contexte servi")
    return True


def shown_path(path: Path) -> Path:
    """Chemin relatif au projet quand c'est possible : pas de chemin local dans l'affichage."""
    return path.relative_to(PATHS.root) if path.is_relative_to(PATHS.root) else path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_reloaded(output_dir: Path) -> None:
    """Recharge les 5 fichiers avec le code de serving de l'API et retrouve les prédictions de référence."""
    # `public_training_domain*` lèvent une erreur si le domaine ne correspond pas au contrat de l'API.
    bundle_predict = load_bundle("predict", output_dir)
    public_training_domain(bundle_predict)
    for values, expected in REFERENCE_PREDICT:
        result = predict(bundle_predict, values)["yield_tons_per_hectare"]
        check(abs(result - expected) <= TOLERANCE, f"/predict {values} : {result} au lieu de {expected}")

    bundle_recommend = load_bundle("recommend", output_dir)
    context = load_recommend_context(output_dir / "recommend_context.json")
    public_training_domain_recommend(bundle_recommend)
    ranking = recommend(bundle_recommend, context, REFERENCE_RECOMMEND_ISO3)["recommendations"]
    check([item["crop"] for item in ranking] == [crop for crop, _ in REFERENCE_RECOMMEND],
          f"classement /recommend de {REFERENCE_RECOMMEND_ISO3}")
    for item, (crop, expected) in zip(ranking, REFERENCE_RECOMMEND):
        result = item["predicted_yield_tons_per_hectare"]
        check(abs(result - expected) <= TOLERANCE, f"/recommend {crop} : {result} au lieu de {expected}")

    print(f"serving    : 5 fichiers rechargés, {len(REFERENCE_PREDICT)} prédictions /predict et "
          f"classement /recommend de {REFERENCE_RECOMMEND_ISO3} retrouvés")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruit les modèles servis par l'API.")
    parser.add_argument("--output-dir", type=Path, default=SERVED_DIR,
                        help="dossier des fichiers produits (défaut : models/)")
    output_dir = parser.parse_args(argv).output_dir.resolve()

    missing = [path for path in (PREDICT_DATASET, RECOMMEND_DATASET, GEOJSON_PAR_DEFAUT) if not path.is_file()]
    if missing:
        names = ", ".join(str(shown_path(path)) for path in missing)
        print(f"Données absentes : {names}. Voir data/README.md.", file=sys.stderr)
        return 1

    # 1. reconstruction et contrôles, tout en mémoire
    predict_pipeline, predict_metadata, predict_sample = build_predict()
    check_predict(predict_pipeline, predict_metadata)
    recommend_pipeline, recommend_metadata, context, recommend_sample = build_recommend()
    check_recommend(recommend_pipeline, recommend_metadata, context)

    # 2. comparaison avec les artefacts servis, s'ils existent, avant toute écriture
    compared = [
        compare_with_served("predict", predict_pipeline, predict_metadata, predict_sample),
        compare_with_served("recommend", recommend_pipeline, recommend_metadata, recommend_sample),
        compare_context_with_served(context),
    ]
    if not all(compared):
        print("models/    : artefacts servis absents ou incomplets, comparaison partielle ou ignorée")

    # 3. écriture
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(predict_pipeline, output_dir / "predict_model.joblib")
    write_json(output_dir / "predict_model_metadata.json", predict_metadata)
    dump_without_tree_state_memo(recommend_pipeline, output_dir / "recommend_model.joblib")
    write_json(output_dir / "recommend_model_metadata.json", recommend_metadata)
    write_json(output_dir / "recommend_context.json", context)

    # 4. rechargement par le code de serving
    check_reloaded(output_dir)
    print(f"5 fichiers écrits dans {shown_path(output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
