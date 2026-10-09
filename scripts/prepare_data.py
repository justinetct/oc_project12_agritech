"""Prépare en mémoire les datasets d'entraînement, à partir des données brutes.

Appelé par `scripts/rebuild_models.py`. Mêmes étapes que les notebooks 04 et 06, qui restent la trace
commentée de la préparation et écrivent les fichiers de `data/processed/`. Ici, rien n'est écrit :
`prepare_training_datasets` renvoie les deux datasets d'entraînement tels que les modèles les lisent
dans `data/processed/`, ce qui permet de reconstruire les modèles servis sans ces fichiers.

- `build_crop_yield_clean` : le dataset historique nettoyé, une ligne par pays, année et culture
  (notebook 04) ;
- `prepare_predict_dataset` et `prepare_recommend_dataset` : les datasets d'entraînement de `/predict`
  et de `/recommend` (notebook 06) ;
- `through_csv` : le passage par le format CSV que font les notebooks entre ces étapes.

Les contrôles sont génériques : clés uniques, une seule valeur de pluie par pays, aucune valeur
manquante à la fin. Les effectifs attendus sur les vraies données sont contrôlés par `rebuild_models.py`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from agritech import geo
from agritech.config import AGRICULTURE_CROP_YIELD_FILENAME, PATHS
from agritech.predict_config import PREDICT_TARGET
from agritech.recommend_config import RECOMMEND_COLUMNS, RECOMMEND_KEY

# Données brutes : le dataset parcellaire de `/predict` et les quatre fichiers annuels par pays de
# `/recommend`. `yield_df.csv`, la fusion fournie avec ces derniers, n'est pas utilisé.
PREDICT_RAW_DATASET = PATHS.data_agriculture_crop_yield / AGRICULTURE_CROP_YIELD_FILENAME
CROP_YIELD_SOURCES = ["yield.csv", "temp.csv", "rainfall.csv", "pesticides.csv"]
RAW_FILES = [PREDICT_RAW_DATASET, *(PATHS.data_crop_yield_prediction / name for name in CROP_YIELD_SOURCES)]

# Période commune aux quatre fichiers annuels.
FIRST_YEAR, LAST_YEAR = 1990, 2013

# Pays dont la valeur de pluie est recopiée depuis un autre pays dans `rainfall.csv` (voir
# `data/README.md`) : retirés plutôt que corrigés.
WRONG_RAINFALL_COUNTRIES = ["MNE", "SDN"]

# Colonnes du dataset `/predict`, dans l'ordre du fichier écrit par le notebook 06 : les 9 variables
# d'entrée, puis la cible.
PREDICT_COLUMNS = [
    "Crop",
    "Soil_Type",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Region",
    "Weather_Condition",
    "Days_to_Harvest",
    PREDICT_TARGET,
]


def _with_iso3(df: pd.DataFrame, country_column: str, geojson: Path | None) -> pd.DataFrame:
    """Ajoute le code `iso3` de chaque nom de pays ; il reste manquant pour un nom sans code."""
    codes = geo.vers_iso3(df[country_column].astype(str).unique(), geojson)
    return df.assign(iso3=df[country_column].map(codes))


def build_crop_yield_clean(
    directory: Path = PATHS.data_crop_yield_prediction, geojson: Path | None = None
) -> pd.DataFrame:
    """Dataset historique nettoyé : une ligne par pays, année et culture, sans valeur manquante (notebook 04).

    `directory` : dossier des quatre fichiers annuels ; `geojson` : contours Natural Earth, qui
    donnent le code ISO3 de chaque nom de pays (fichier de `data/geo/` par défaut).

    1. Chaque source est ramenée à une ligne par clé sur 1990-2013 : l'agrégat `China` du rendement
       est écarté (`China, mainland` est gardé), les doublons exacts de température sont retirés puis
       les relevés moyennés par pays et par année, les valeurs de pluie `..` deviennent manquantes.
    2. Jointures à gauche sur le rendement, par `iso3` et `year` ; les lignes sans code ISO3 sont écartées.
    3. Nettoyage : une année sans pluie reprend la valeur du pays ; les pays sans aucune température ou
       sans aucun pesticide sont retirés, puis ceux de `WRONG_RAINFALL_COUNTRIES`.
    """
    yield_raw, temp_raw, rainfall_raw, pesticides_raw = (pd.read_csv(directory / name) for name in CROP_YIELD_SOURCES)
    # `rainfall.csv` a une espace initiale dans l'en-tête de sa colonne pays
    rainfall_raw.columns = rainfall_raw.columns.str.strip()

    crop_yield = (
        yield_raw[(yield_raw["Area"] != "China") & yield_raw["Year"].between(FIRST_YEAR, LAST_YEAR)]
        .pipe(_with_iso3, "Area", geojson)
        .rename(columns={"Area": "area", "Year": "year", "Item": "crop"})
        .assign(yield_t_ha=lambda d: d["Value"] / 10_000)  # hg/ha -> t/ha
        [["iso3", "area", "year", "crop", "yield_t_ha"]]
    )
    temperature = (
        temp_raw.drop_duplicates()
        .loc[lambda d: d["year"].between(FIRST_YEAR, LAST_YEAR)]
        .pipe(_with_iso3, "country", geojson)
        .groupby(["iso3", "year"], as_index=False)["avg_temp"].mean()
    )
    rainfall = (
        rainfall_raw.assign(rain_mm=lambda d: pd.to_numeric(d["average_rain_fall_mm_per_year"], errors="coerce"))
        .loc[lambda d: d["Year"].between(FIRST_YEAR, LAST_YEAR)]
        .pipe(_with_iso3, "Area", geojson)
        .rename(columns={"Year": "year"})
        [["iso3", "year", "rain_mm"]]
    )
    pesticides = (
        pesticides_raw.loc[lambda d: d["Year"].between(FIRST_YEAR, LAST_YEAR)]
        .pipe(_with_iso3, "Area", geojson)
        .rename(columns={"Year": "year", "Value": "pesticides_t"})
        [["iso3", "year", "pesticides_t"]]
    )

    # une seule ligne par clé dans chaque source : les jointures ne dupliquent aucune ligne
    country_year = ["iso3", "year"]
    sources = {"yield.csv": (crop_yield, RECOMMEND_KEY), "temp.csv": (temperature, country_year),
               "rainfall.csv": (rainfall, country_year), "pesticides.csv": (pesticides, country_year)}
    for name, (source, key) in sources.items():
        if source.dropna(subset=["iso3"]).duplicated(key).any():
            raise ValueError(f"{name} : la clé {'+'.join(key)} n'est pas unique")

    df = crop_yield.dropna(subset=["iso3"])
    for source in (temperature, rainfall, pesticides):
        df = df.merge(source.dropna(subset=["iso3"]), on=country_year, how="left")

    # `rainfall.csv` donne une seule valeur par pays : une année sans valeur reprend celle du pays
    if df.groupby("iso3")["rain_mm"].nunique().max() > 1:
        raise ValueError("rainfall.csv : plusieurs valeurs de pluie pour un même pays")
    country_rain = df.groupby("iso3")["rain_mm"].first()  # first() ignore les valeurs manquantes
    missing_rain = df["rain_mm"].isna() & df["iso3"].map(country_rain).notna()
    df.loc[missing_rain, "rain_mm"] = df.loc[missing_rain, "iso3"].map(country_rain)

    # pays sans aucune température ou sans aucun pesticide : aucune valeur à reprendre
    counts = df.groupby("iso3")[["avg_temp", "pesticides_t"]].count()
    without_values = counts.index[(counts["avg_temp"] == 0) | (counts["pesticides_t"] == 0)]
    clean = df[~df["iso3"].isin([*without_values, *WRONG_RAINFALL_COUNTRIES])].reset_index(drop=True)

    if clean.isna().any().any():
        raise ValueError("dataset historique nettoyé : valeurs manquantes")
    return clean


def prepare_predict_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset d'entraînement `/predict` et rendements négatifs mis à part (notebook 06).

    `raw` : `crop_yield.csv` tel quel. Un rendement négatif est impossible : ces lignes sont écartées
    de l'entraînement et renvoyées à part, avec toutes leurs colonnes ; un rendement nul est gardé.
    Le dataset d'entraînement garde les colonnes de `PREDICT_COLUMNS`, dans cet ordre.
    """
    target = raw[PREDICT_TARGET]
    return raw.loc[target >= 0, PREDICT_COLUMNS], raw[target < 0]


def prepare_recommend_dataset(clean: pd.DataFrame) -> pd.DataFrame:
    """Dataset d'entraînement `/recommend` (notebook 06) : tout le dataset historique nettoyé, sans ligne
    retirée, avec les colonnes de `RECOMMEND_COLUMNS` dans cet ordre (`area` reste hors modèle)."""
    return clean[RECOMMEND_COLUMNS]


def through_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Écrit `df` au format CSV en mémoire, puis le relit avec les options par défaut de `pd.read_csv`.

    C'est le passage que font les notebooks par les fichiers de `data/processed/`, et il ne redonne
    pas toujours exactement le nombre écrit : une moyenne de températures écrite `18.240000000000002`
    est relue `18.24`. Les modèles servis ont été appris sur les valeurs relues ; sans ce passage,
    677 valeurs de `avg_temp` sur 16 319 diffèrent d'environ 1e-15, assez pour changer le contexte
    `/recommend`. L'index repart de 0, comme à la lecture d'un fichier.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def prepare_training_datasets(
    predict_raw: Path = PREDICT_RAW_DATASET,
    crop_yield_directory: Path = PATHS.data_crop_yield_prediction,
    geojson: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Datasets d'entraînement `/predict` et `/recommend`, identiques à leur lecture dans `data/processed/`.

    Reprend la chaîne des notebooks, passages par le CSV compris : le notebook 04 écrit le dataset
    historique nettoyé, que le notebook 06 relit pour écrire le dataset `/recommend` ; les modèles
    relisent ensuite les fichiers du notebook 06. Rien n'est écrit sur le disque.
    """
    predict, _ = prepare_predict_dataset(pd.read_csv(predict_raw))
    clean = through_csv(build_crop_yield_clean(crop_yield_directory, geojson))
    return through_csv(predict), through_csv(prepare_recommend_dataset(clean))
