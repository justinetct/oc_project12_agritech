"""Scripts de reconstruction des modèles (`scripts/rebuild_models.py` et `scripts/prepare_data.py`), sans
les vrais entraînements.

Les vrais modèles ne sont pas reconstruits ici : il faudrait les données brutes de `data/`, non
versionnées, et la reconstruction complète se lance à la main. Les tests utilisent de petits modèles
entraînés sur des données synthétiques, et lisent les artefacts servis de `models/` pour vérifier que
les valeurs figées du script leur correspondent. La préparation des données est testée sur de petits
fichiers bruts écrits dans un dossier temporaire.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from agritech.modeling import experiment_pipeline
from agritech.predict_config import (
    PREDICT_DATASET,
    PREDICT_FEATURES,
    PREDICT_FINAL_TEST_METRICS,
    PREDICT_MODEL_VERSION,
    PREDICT_SELECTED_FEATURES,
    PREDICT_TARGET,
    PREDICT_TEST_SIZE,
    PREDICT_TRAINING_DOMAIN,
)
from agritech.preprocessing import build_pipeline
from agritech.recommend_config import (
    RECOMMEND_CROPS,
    RECOMMEND_FINAL_PARAMS,
    RECOMMEND_FINAL_TEST_METRICS,
    RECOMMEND_MODEL_VERSION,
)
from agritech.serialization import dump_without_tree_state_memo
from agritech.serving import _REQUIRED_METADATA_KEYS, load_bundle, load_recommend_context
from agritech.training_data import protocol_params, split

# `rebuild_models.py` importe `prepare_data.py` comme un script voisin : `scripts/` doit être dans les chemins.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data as prepare  # noqa: E402
from scripts import rebuild_models as rebuild  # noqa: E402

FIVE_FILES = {
    "predict_model.joblib",
    "predict_model_metadata.json",
    "recommend_model.joblib",
    "recommend_model_metadata.json",
    "recommend_context.json",
}


def _served_metadata(name: str) -> dict:
    return json.loads((rebuild.SERVED_DIR / f"{name}_model_metadata.json").read_text(encoding="utf-8"))


# --- Petits modèles synthétiques ------------------------------------------------------------------


def _predict_frame(rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Rainfall_mm": rng.uniform(100, 1000, rows),
            "Temperature_Celsius": rng.uniform(15, 40, rows),
            "Fertilizer_Used": [i % 2 == 0 for i in range(rows)],
            "Irrigation_Used": [i % 3 == 0 for i in range(rows)],
        }
    )


def _predict_pipeline(scale: float = 1.0):
    """Régression linéaire `/predict` sur 12 lignes ; `scale` change la cible, donc les prédictions."""
    X = _predict_frame()
    y = scale * (0.005 * X["Rainfall_mm"] + 0.6 * X["Fertilizer_Used"] + 0.5 * X["Irrigation_Used"])
    return build_pipeline(LinearRegression(), ["Fertilizer_Used", "Irrigation_Used"],
                          ["Rainfall_mm", "Temperature_Celsius"]).fit(X, y)


def _predict_metadata(pipeline, protocol: dict | None = None, n_refit: int | None = None) -> dict:
    """Metadata `/predict` ; par défaut, évaluation 9 + 3 lignes et réapprentissage sur les 12."""
    X = _predict_frame()
    protocol = protocol or protocol_params(PREDICT_DATASET, X.iloc[:9], X.iloc[9:], 0.2, 42, 5, True)
    return rebuild.predict_metadata(pipeline, ["Fertilizer_Used", "Irrigation_Used"],
                                    ["Rainfall_mm", "Temperature_Celsius"], protocol,
                                    n_refit=len(X) if n_refit is None else n_refit)


def _predict_dataset(rows: int = 40) -> pd.DataFrame:
    """Toutes les colonnes du dataset `/predict`, avec une cible bruitée : appris sur le train seul ou
    sur toutes les lignes, le modèle n'est pas le même."""
    rng = np.random.default_rng(1)
    df = _predict_frame(rows).assign(Crop="Wheat", Soil_Type="Loam", Region="North", Weather_Condition="Sunny",
                                     Days_to_Harvest=rng.integers(60, 150, rows))
    df[PREDICT_TARGET] = (0.005 * df["Rainfall_mm"] + 0.02 * df["Temperature_Celsius"] + 1.5 * df["Fertilizer_Used"]
                          + 1.2 * df["Irrigation_Used"] + rng.normal(0, 0.5, rows))
    return df[PREDICT_FEATURES + [PREDICT_TARGET]]


def _recommend_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Les 10 cultures sur 2011-2013 : `X_hist` (2011-2012), `X_test_hist` (2013), `X_refit` et sa cible."""
    rng = np.random.default_rng(0)
    rows = [{"crop": crop, "year": year} for year in (2011, 2012, 2013) for crop in RECOMMEND_CROPS]
    X = pd.DataFrame(rows).assign(
        temp_hist=rng.uniform(5, 30, len(rows)),
        rain_mm=rng.uniform(100, 3000, len(rows)),
        log_pest_hist=rng.uniform(0, 14, len(rows)),
        lat_abs=rng.uniform(0, 60, len(rows)),
        geo_x=rng.uniform(-1, 1, len(rows)),
        geo_y=rng.uniform(-1, 1, len(rows)),
        geo_z=rng.uniform(-1, 1, len(rows)),
    )[rebuild.RECOMMEND_FINAL_CATEGORICAL + rebuild.RECOMMEND_FINAL_NUMERIC]
    y = pd.Series(rng.uniform(1, 40, len(rows)))
    return X[X["year"] < 2013], X[X["year"] == 2013], X, y


@pytest.fixture(scope="module")
def recommend_model():
    """ExtraTrees avec les réglages finaux, appris sur 30 lignes synthétiques, ses metadata et ses lignes."""
    X_hist, X_test_hist, X_refit, y = _recommend_frames()
    pipeline = experiment_pipeline(ExtraTreesRegressor(**RECOMMEND_FINAL_PARAMS),
                                   rebuild.RECOMMEND_FINAL_CATEGORICAL, rebuild.RECOMMEND_FINAL_NUMERIC)
    pipeline.fit(X_refit, y)
    return pipeline, rebuild.recommend_metadata(pipeline, X_hist, X_test_hist, X_refit), X_refit


def _context_dataset() -> pd.DataFrame:
    """Deux pays, 2010-2013, deux cultures par pays et par année, avec leur géographie."""
    rows = []
    for iso3, area, crops, rain, geo in [
        ("BBB", "Bravo", ["Wheat", "Maize"], 600.0, (45.0, 0.5, 0.1, 0.7)),
        ("AAA", "Alpha", ["Yams", "Cassava"], 1500.0, (8.0, 0.9, 0.3, 0.1)),
    ]:
        for year in (2010, 2011, 2012, 2013):
            for crop in crops:
                rows.append({"iso3": iso3, "area": area, "year": year, "crop": crop, "avg_temp": 10.0 + year - 2010,
                             "rain_mm": rain, "pesticides_t": 100.0 * (year - 2009), "yield_t_ha": 1.0,
                             "lat_abs": geo[0], "geo_x": geo[1], "geo_y": geo[2], "geo_z": geo[3]})
    return pd.DataFrame(rows)


# --- Préparation des données (`prepare_data.py`) ---------------------------------------------------


def _write_raw_sources(directory: Path, pesticides: str | None = None) -> Path:
    """Les quatre fichiers annuels et un GeoJSON minimal, avec un cas pour chaque règle du nettoyage.

    Restent : France 1990-1991 et `China, mainland` 1990. Disparaissent : 1989 (hors période), l'agrégat
    `China`, Atlantis (sans code ISO3), le Kenya (sans température), le Chili (sans pesticides) et le
    Monténégro (pluie erronée). Renvoie le chemin du GeoJSON.
    """
    (directory / "yield.csv").write_text(
        "Area,Item,Year,Value\nFrance,Wheat,1989,10000\nFrance,Wheat,1990,70000\nFrance,Wheat,1991,72000\n"
        'China,Wheat,1990,50000\n"China, mainland",Wheat,1990,52000\nAtlantis,Wheat,1990,1000\n'
        "Kenya,Maize,1990,20000\nChile,Maize,1990,30000\nMontenegro,Maize,1990,40000\n"
    )
    # France 1990 : un doublon exact, puis une autre valeur
    (directory / "temp.csv").write_text(
        "year,country,avg_temp\n1990,France,11.0\n1990,France,11.0\n1990,France,12.0\n1991,France,12.5\n"
        "1990,China,15.0\n1990,Chile,10.0\n1990,Montenegro,9.0\n"
    )
    # espace initiale dans l'en-tête et valeur `..`, comme dans le vrai fichier
    (directory / "rainfall.csv").write_text(
        " Area,Year,average_rain_fall_mm_per_year\nFrance,1990,800\nFrance,1991,..\nChina,1990,600\n"
        "Kenya,1990,630\nChile,1990,700\nMontenegro,1990,241\n"
    )
    (directory / "pesticides.csv").write_text(
        pesticides or "Area,Year,Value\nFrance,1990,100\nFrance,1991,110\nChina,1990,1000\nKenya,1990,50\n"
                      "Montenegro,1990,5\n"
    )
    geojson = directory / "countries.geojson"
    features = [{"type": "Feature", "properties": {"NAME": name, "ISO_A3": code}, "geometry": None}
                for name, code in [("France", "FRA"), ("China", "CHN"), ("Kenya", "KEN"), ("Chile", "CHL"),
                                   ("Montenegro", "MNE")]]
    geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return geojson


def test_build_crop_yield_clean_applies_the_cleaning_rules(tmp_path: Path):
    geojson = _write_raw_sources(tmp_path)

    clean = prepare.build_crop_yield_clean(tmp_path, geojson)

    expected = pd.DataFrame({
        "iso3": ["FRA", "FRA", "CHN"],
        "area": ["France", "France", "China, mainland"],
        "year": [1990, 1991, 1990],
        "crop": ["Wheat", "Wheat", "Wheat"],
        "yield_t_ha": [7.0, 7.2, 5.2],
        "avg_temp": [11.5, 12.5, 15.0],  # France 1990 : doublon retiré, moyenne de 11 et 12
        "rain_mm": [800.0, 800.0, 600.0],  # France 1991 : `..` remplacé par la valeur du pays
        "pesticides_t": [100.0, 110.0, 1000.0],
    })
    pd.testing.assert_frame_equal(clean, expected, check_exact=True)


def test_build_crop_yield_clean_rejects_a_duplicated_key(tmp_path: Path):
    geojson = _write_raw_sources(tmp_path, pesticides="Area,Year,Value\nFrance,1990,100\nFrance,1990,120\n")

    with pytest.raises(ValueError, match="pesticides.csv"):
        prepare.build_crop_yield_clean(tmp_path, geojson)


def test_prepare_predict_dataset_sets_negative_yields_apart():
    raw = _predict_dataset(4).assign(**{PREDICT_TARGET: [1.5, 0.0, -0.2, np.nan]})

    training, negative = prepare.prepare_predict_dataset(raw)

    assert sorted(prepare.PREDICT_COLUMNS) == sorted(PREDICT_FEATURES + [PREDICT_TARGET])
    assert training.columns.tolist() == prepare.PREDICT_COLUMNS
    assert training[PREDICT_TARGET].tolist() == [1.5, 0.0]  # rendement nul gardé, rendement manquant écarté
    pd.testing.assert_frame_equal(negative, raw.iloc[[2]])  # toutes les colonnes d'origine


def test_through_csv_reads_values_back_like_a_processed_file():
    df = pd.DataFrame({"avg_temp": [18.240000000000002, 11.5]}, index=[7, 3])

    result = prepare.through_csv(df)

    assert result["avg_temp"].tolist() == [18.24, 11.5]  # valeur relue, comme par les notebooks
    assert isinstance(result.index, pd.RangeIndex)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.iloc[1:], "lignes"),
        (lambda df: df.assign(y=[1.0, None]), "valeurs manquantes"),
        (lambda df: df.assign(y=[1.0, -0.5]), "valeurs négatives"),
    ],
)
def test_check_training_dataset_rejects_an_unexpected_dataset(change, message):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 0.0]})
    rebuild.check_training_dataset("data.csv", df, 2, ["y", "x"], "y")  # colonnes dans un autre ordre : accepté

    with pytest.raises(RuntimeError, match=message):
        rebuild.check_training_dataset("data.csv", change(df), 2, ["x", "y"], "y")


# --- Valeurs figées et artefacts servis ------------------------------------------------------------


def test_predict_frozen_constants_match_served_metadata():
    served = _served_metadata("predict")

    assert served["model_version"] == PREDICT_MODEL_VERSION
    assert served["evaluation"]["test_metrics"] == PREDICT_FINAL_TEST_METRICS
    assert served["training_domain"] == PREDICT_TRAINING_DOMAIN
    assert served["features"] == PREDICT_SELECTED_FEATURES


def test_recommend_frozen_constants_match_served_metadata():
    served = _served_metadata("recommend")

    assert served["model_version"] == RECOMMEND_MODEL_VERSION
    assert served["evaluation"]["test_metrics"] == RECOMMEND_FINAL_TEST_METRICS
    assert served["model_params"] == RECOMMEND_FINAL_PARAMS
    assert served["features"] == rebuild.RECOMMEND_FINAL_CATEGORICAL + rebuild.RECOMMEND_FINAL_NUMERIC


def test_reference_predictions_match_served_artifacts():
    """Les prédictions de référence du script sont celles des 5 fichiers servis, lus par le serving."""
    rebuild.check_reloaded(rebuild.SERVED_DIR)


# --- Contexte /recommend ---------------------------------------------------------------------------


def test_recommend_context_keeps_the_three_previous_years_of_each_country(tmp_path: Path):
    context = rebuild.build_recommend_context(_context_dataset())

    assert list(context["countries"]) == ["AAA", "BBB"]
    assert context["history_years"] == [2011, 2012, 2013]
    assert context["target_year"] == 2014
    alpha = context["countries"]["AAA"]
    assert alpha["country"] == "Alpha"
    assert alpha["rain_mm"] == 1500.0
    assert alpha["geography"] == {"lat_abs": 8.0, "geo_x": 0.9, "geo_y": 0.3, "geo_z": 0.1}
    # une ligne par année malgré deux cultures, 2010 écartée
    assert alpha["history_2011_2013"] == [
        {"year": 2011, "avg_temp": 11.0, "pesticides_t": 200.0},
        {"year": 2012, "avg_temp": 12.0, "pesticides_t": 300.0},
        {"year": 2013, "avg_temp": 13.0, "pesticides_t": 400.0},
    ]
    assert alpha["observed_crops"] == ["Cassava", "Yams"]

    # le fichier écrit est accepté par le serving
    path = tmp_path / "recommend_context.json"
    rebuild.write_json(path, context)
    assert load_recommend_context(path).country_entries == [("AAA", "Alpha"), ("BBB", "Bravo")]


# --- Structure des metadata -----------------------------------------------------------------------


def test_predict_metadata_has_the_served_structure_and_build_info():
    metadata = _predict_metadata(_predict_pipeline())
    served = _served_metadata("predict")

    assert list(metadata) == list(served)
    assert list(metadata["refit"]) == list(served["refit"])
    assert list(metadata["evaluation"]) == list(served["evaluation"])
    assert set(_REQUIRED_METADATA_KEYS) <= set(metadata)
    assert metadata["categorical_values"] == {"Fertilizer_Used": [False, True], "Irrigation_Used": [False, True]}
    assert metadata["refit"] == {"n_samples": 12, "trained_on": "all_rows_after_cleaning"}
    assert metadata["built_by"] == "scripts/rebuild_models.py"
    assert metadata["created_on"] == date.today().isoformat()
    assert metadata["versions"]["scikit-learn"] == sklearn.__version__


def test_predict_metadata_keeps_the_metrics_with_the_evaluated_model():
    """Les métriques ne décrivent que le modèle évalué (train seul) : rien ne les rattache au refit."""
    metadata = _predict_metadata(_predict_pipeline())

    assert metadata["evaluation"] == {
        "n_train": 9,
        "n_test": 3,
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "trained_on": "train_only",
        "test_metrics": PREDICT_FINAL_TEST_METRICS,
        "notebook": "notebooks/11_predict_final_evaluation.ipynb",
    }
    assert "final_test_metrics" not in metadata
    assert set(metadata["refit"]) == {"n_samples", "trained_on"}


# --- Réapprentissage /predict ---------------------------------------------------------------------


def test_build_predict_refits_the_evaluated_model_on_all_rows():
    df = _predict_dataset()

    pipeline, metadata, sample = rebuild.build_predict(df)

    features = PREDICT_SELECTED_FEATURES
    categorical, numeric = ["Fertilizer_Used", "Irrigation_Used"], ["Rainfall_mm", "Temperature_Celsius"]
    X_train, X_test, y_train, _ = split(df, PREDICT_FEATURES, PREDICT_TARGET, PREDICT_TEST_SIZE, rebuild.SEED,
                                        verbose=False)
    on_all_rows = build_pipeline(LinearRegression(), categorical, numeric).fit(df[features], df[PREDICT_TARGET])
    on_train = build_pipeline(LinearRegression(), categorical, numeric).fit(X_train[features], y_train)

    assert np.allclose(pipeline.predict(df[features]), on_all_rows.predict(df[features]))
    assert not np.allclose(pipeline.predict(df[features]), on_train.predict(df[features]))
    assert metadata["refit"] == {"n_samples": 40, "trained_on": "all_rows_after_cleaning"}
    assert (metadata["evaluation"]["n_train"], metadata["evaluation"]["n_test"]) == (32, 8)
    assert metadata["evaluation"]["test_metrics"] == PREDICT_FINAL_TEST_METRICS
    assert sample.index.isin(X_test.index).all()


def _real_protocol() -> dict:
    """Effectifs de l'évaluation du notebook 11, sans le dataset."""
    return {"dataset": PREDICT_DATASET.name, "n_train": rebuild.PREDICT_N_TRAIN, "n_test": rebuild.PREDICT_N_TEST,
            "test_size": 0.2, "random_state": 42, "cv_folds": 5, "cv_shuffle": True}


def test_check_predict_accepts_the_refit_on_all_rows():
    pipeline = _predict_pipeline()

    rebuild.check_predict(pipeline, _predict_metadata(pipeline, _real_protocol(), rebuild.PREDICT_N_REFIT))


def test_check_predict_rejects_a_model_learned_on_the_train_only():
    pipeline = _predict_pipeline()
    metadata = _predict_metadata(pipeline, _real_protocol(), rebuild.PREDICT_N_TRAIN)

    with pytest.raises(RuntimeError, match="réapprentissage /predict"):
        rebuild.check_predict(pipeline, metadata)


def test_recommend_metadata_has_the_served_structure_and_computed_values(recommend_model):
    _, metadata, X_refit = recommend_model
    served = _served_metadata("recommend")

    assert list(metadata) == list(served)
    assert list(metadata["refit"]) == list(served["refit"])
    assert list(metadata["evaluation"]) == list(served["evaluation"])
    assert set(_REQUIRED_METADATA_KEYS) <= set(metadata)
    assert metadata["categorical_values"] == {"crop": RECOMMEND_CROPS}
    assert metadata["model_params"] == RECOMMEND_FINAL_PARAMS
    assert metadata["refit"] == {
        "period": "2011-2013",
        "n_samples": 30,
        "trained_on": "all_rows_with_history_1991_to_2013",
        "target_year_for_serving": 2014,
    }
    assert metadata["evaluation"]["period_train"] == "2011-2012"
    assert (metadata["evaluation"]["n_train"], metadata["evaluation"]["n_test"]) == (20, 10)
    assert metadata["evaluation"]["test_metrics"] == RECOMMEND_FINAL_TEST_METRICS
    assert metadata["training_domain"]["temp_hist"] == {
        "min": X_refit["temp_hist"].min(), "max": X_refit["temp_hist"].max(), "unit": "°C",
    }
    assert metadata["created_on"] == date.today().isoformat()


def test_without_build_info_drops_only_created_on_and_versions():
    metadata = {"service": "predict", "created_on": "2026-01-01", "versions": {}, "model_version": "1.0.0"}

    assert rebuild.without_build_info(metadata) == {"service": "predict", "model_version": "1.0.0"}


# --- Comparaison avec les artefacts servis --------------------------------------------------------


@pytest.fixture
def served_dir(tmp_path: Path, monkeypatch) -> Path:
    """Dossier des artefacts servis, vide au départ, à la place de `models/`."""
    path = tmp_path / "models"
    path.mkdir()
    monkeypatch.setattr(rebuild, "SERVED_DIR", path)
    return path


def _serve(directory: Path, name: str, pipeline, metadata: dict) -> None:
    joblib.dump(pipeline, directory / f"{name}_model.joblib")
    rebuild.write_json(directory / f"{name}_model_metadata.json", metadata)


def test_compare_with_served_is_skipped_without_served_model(served_dir: Path):
    pipeline = _predict_pipeline()

    assert rebuild.compare_with_served("predict", pipeline, _predict_metadata(pipeline), _predict_frame()) is False


def test_compare_with_served_accepts_same_model_built_another_day(served_dir: Path):
    pipeline = _predict_pipeline()
    metadata = _predict_metadata(pipeline)
    _serve(served_dir, "predict", pipeline, metadata | {"created_on": "2026-09-18", "versions": {"numpy": "0"}})

    assert rebuild.compare_with_served("predict", pipeline, metadata, _predict_frame()) is True


def test_compare_with_served_rejects_different_predictions(served_dir: Path):
    served = _predict_pipeline()
    _serve(served_dir, "predict", served, _predict_metadata(served))
    rebuilt = _predict_pipeline(scale=2.0)

    with pytest.raises(RuntimeError, match="écart"):
        rebuild.compare_with_served("predict", rebuilt, _predict_metadata(rebuilt), _predict_frame())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda metadata: metadata | {"model_version": "9.9.9"}, "metadata différentes"),
        (lambda metadata: dict(reversed(list(metadata.items()))), "ordre des clés"),
    ],
)
def test_compare_with_served_rejects_different_metadata(served_dir: Path, change, message):
    pipeline = _predict_pipeline()
    metadata = _predict_metadata(pipeline)
    _serve(served_dir, "predict", pipeline, change(metadata))

    with pytest.raises(RuntimeError, match=message):
        rebuild.compare_with_served("predict", pipeline, metadata, _predict_frame())


def test_compare_context_with_served(served_dir: Path):
    context = rebuild.build_recommend_context(_context_dataset())
    assert rebuild.compare_context_with_served(context) is False

    rebuild.write_json(served_dir / "recommend_context.json", context)
    assert rebuild.compare_context_with_served(context) is True

    context["countries"]["AAA"]["rain_mm"] = 1501.0
    with pytest.raises(RuntimeError, match="contexte"):
        rebuild.compare_context_with_served(context)


# --- main : dossier de sortie et arrêt avant écriture ---------------------------------------------


@pytest.fixture
def fake_rebuild(served_dir: Path, tmp_path: Path, monkeypatch, recommend_model) -> list[Path]:
    """`main` avec les petits modèles synthétiques à la place des vrais entraînements.

    Les contrôles propres aux vraies données et aux vrais modèles (datasets préparés, effectifs, 115 pays,
    prédictions de référence) sont neutralisés ; la comparaison avec `served_dir`, l'écriture et le format
    des fichiers restent réels. Renvoie la liste des dossiers passés au rechargement final.
    """
    predict_pipeline = _predict_pipeline()
    recommend_pipeline, recommend_metadata, X_refit = recommend_model
    context = rebuild.build_recommend_context(_context_dataset())

    data_file = tmp_path / "data.csv"
    data_file.write_text("")
    monkeypatch.setattr(rebuild, "RAW_FILES", [data_file])
    monkeypatch.setattr(rebuild, "GEOJSON_PAR_DEFAUT", data_file)

    monkeypatch.setattr(rebuild, "prepare_training_datasets", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(rebuild, "check_training_dataset", lambda *args: None)
    monkeypatch.setattr(rebuild, "build_predict",
                        lambda df: (predict_pipeline, _predict_metadata(predict_pipeline), _predict_frame()))
    monkeypatch.setattr(rebuild, "build_recommend",
                        lambda df: (recommend_pipeline, recommend_metadata, context, X_refit))
    monkeypatch.setattr(rebuild, "check_predict", lambda pipeline, metadata: None)
    monkeypatch.setattr(rebuild, "check_recommend", lambda pipeline, metadata, context: None)
    reloaded: list[Path] = []
    monkeypatch.setattr(rebuild, "check_reloaded", reloaded.append)
    return reloaded


def test_main_writes_the_five_files_in_output_dir_only(fake_rebuild: list[Path], served_dir: Path, tmp_path: Path):
    output_dir = tmp_path / "rebuild"

    assert rebuild.main(["--output-dir", str(output_dir)]) == 0

    assert {path.name for path in output_dir.iterdir()} == FIVE_FILES
    assert list(served_dir.iterdir()) == []
    assert fake_rebuild == [output_dir.resolve()]
    assert load_bundle("predict", output_dir).metadata["model_version"] == PREDICT_MODEL_VERSION
    assert load_bundle("recommend", output_dir).metadata["model_version"] == RECOMMEND_MODEL_VERSION
    assert len(load_recommend_context(output_dir / "recommend_context.json").countries) == 2


def test_main_writes_in_served_dir_by_default(fake_rebuild: list[Path], served_dir: Path):
    assert rebuild.main([]) == 0

    assert {path.name for path in served_dir.iterdir()} == FIVE_FILES


def test_main_writes_the_recommend_model_without_tree_state_memo(fake_rebuild, recommend_model, monkeypatch,
                                                                 tmp_path: Path):
    """Le modèle `/recommend` passe par `dump_without_tree_state_memo` et se relit avec les mêmes prédictions."""
    written: list[str] = []

    def spy(model, path: Path) -> None:
        written.append(path.name)
        dump_without_tree_state_memo(model, path)

    monkeypatch.setattr(rebuild, "dump_without_tree_state_memo", spy)
    output_dir = tmp_path / "rebuild"

    assert rebuild.main(["--output-dir", str(output_dir)]) == 0

    assert written == ["recommend_model.joblib"]
    pipeline, _, X_refit = recommend_model
    reloaded = load_bundle("recommend", output_dir).pipeline
    assert np.array_equal(reloaded.predict(X_refit), pipeline.predict(X_refit))


def test_main_stops_before_writing_when_served_model_diverges(fake_rebuild, served_dir: Path, tmp_path: Path):
    other = _predict_pipeline(scale=2.0)
    _serve(served_dir, "predict", other, _predict_metadata(other))
    output_dir = tmp_path / "rebuild"

    with pytest.raises(RuntimeError, match="écart"):
        rebuild.main(["--output-dir", str(output_dir)])

    assert not output_dir.exists()
    assert fake_rebuild == []


@pytest.mark.parametrize("missing", ["RAW_FILES", "GEOJSON_PAR_DEFAUT"])
def test_main_without_data_returns_1_and_writes_nothing(fake_rebuild, monkeypatch, tmp_path: Path, missing: str):
    absent = tmp_path / "absent.csv"
    monkeypatch.setattr(rebuild, missing, [absent] if missing == "RAW_FILES" else absent)
    output_dir = tmp_path / "rebuild"

    assert rebuild.main(["--output-dir", str(output_dir)]) == 1

    assert not output_dir.exists()
