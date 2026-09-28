"""Génère `models/recommend_context.json` et complète `models/recommend_model_metadata.json`.

Le script reproduit strictement la préparation du notebook 15 pour calculer le
`training_domain` sur les mêmes lignes que celles apprises par le modèle final.
Il extrait ensuite, pour chacun des 115 pays, les informations minimales dont
l'API a besoin pour recommander en 2014 sans dépendre du CSV et du GeoJSON.

Exécution : `poetry run python scripts/build_recommend_context.py`

Contrôles obligatoires :
- `len(X_hist) == metadata["n_train"]` (invariant strict) ;
- 115 pays, 3 années d'historique par pays, ≥ 1 culture observée par pays ;
- union des cultures observées = les 10 cultures du modèle ;
- aucun code ISO3 sans coordonnées ;
- déterminisme : deux exécutions successives produisent des fichiers identiques
  (vérifié par le lanceur, pas par le script).

Le script rapporte, à titre informatif seulement, les pays dont les defaults
2014 sortent du domaine d'apprentissage : ce n'est pas un invariant du script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agritech.config import PATHS
from agritech.geo import charger_coordonnees
from agritech.recommend_config import (
    RECOMMEND_COLUMNS,
    RECOMMEND_CROPS,
    RECOMMEND_DATASET,
    RECOMMEND_GEOGRAPHY,
    RECOMMEND_HISTORICAL_CONDITIONS,
    RECOMMEND_ROWS,
    RECOMMEND_TARGET,
    RECOMMEND_TEST_YEAR,
)
from agritech.recommend_features import add_historical_conditions, add_recommend_features
from agritech.training_data import load_dataset, temporal_split


MODEL_VERSION = "1.0.0"
HISTORY_YEARS = [2011, 2012, 2013]
TARGET_YEAR = 2014


def build_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduit strictement la préparation de `X_hist` du notebook 15."""
    df = load_dataset(
        RECOMMEND_DATASET, RECOMMEND_ROWS, RECOMMEND_COLUMNS,
        non_negative=[RECOMMEND_TARGET], verbose=False,
    )
    coord = charger_coordonnees()
    df = add_historical_conditions(add_recommend_features(df, coord))

    variables = ["iso3", "crop", "year", *RECOMMEND_HISTORICAL_CONDITIONS, *RECOMMEND_GEOGRAPHY]
    X_train, _, _, _ = temporal_split(
        df, variables, RECOMMEND_TARGET, RECOMMEND_TEST_YEAR, verbose=False,
    )
    X_hist = X_train[X_train["temp_hist"].notna()]
    return df, X_hist


def compute_training_domain_model_units(X_hist: pd.DataFrame, n_train_expected: int) -> dict:
    """Extrema des 3 features numériques modifiables sur `X_hist`, grain `(iso3, year, crop)`."""
    if len(X_hist) != n_train_expected:
        raise RuntimeError(
            f"invariant cassé : len(X_hist)={len(X_hist)} != metadata['n_train']={n_train_expected}. "
            "Les données ou le feature engineering ne correspondent plus à l'artefact modèle : investiguer."
        )
    return {
        "temp_hist": {
            "min": float(X_hist["temp_hist"].min()),
            "max": float(X_hist["temp_hist"].max()),
            "unit": "°C",
        },
        "rain_mm": {
            "min": float(X_hist["rain_mm"].min()),
            "max": float(X_hist["rain_mm"].max()),
            "unit": "mm",
        },
        "log_pest_hist": {
            "min": float(X_hist["log_pest_hist"].min()),
            "max": float(X_hist["log_pest_hist"].max()),
            "unit": "log(t+1)",
        },
    }


def build_country_context(df: pd.DataFrame) -> dict:
    """Construit le mapping `iso3 → contexte pays` pour les 115 pays du dataset.

    `df` doit déjà porter les colonnes ajoutées par `add_recommend_features`
    (`lat_abs`, `geo_x`, `geo_y`, `geo_z`) : on les reprend telles quelles.
    """
    geo = df.drop_duplicates("iso3").set_index("iso3")[["lat_abs", "geo_x", "geo_y", "geo_z"]]

    countries: dict[str, dict] = {}
    for iso3 in sorted(df["iso3"].unique()):
        sub = df[df["iso3"] == iso3]
        country_name = str(sub["area"].iloc[0])
        rain_mm = float(sub["rain_mm"].iloc[0])

        hist = (
            sub[sub["year"].isin(HISTORY_YEARS)]
            .drop_duplicates(["iso3", "year"])
            .sort_values("year")
        )
        history = [
            {
                "year": int(row.year),
                "avg_temp": float(row.avg_temp),
                "pesticides_t": float(row.pesticides_t),
            }
            for row in hist.itertuples()
        ]

        observed_crops = sorted(sub["crop"].unique().tolist())

        countries[iso3] = {
            "country": country_name,
            "rain_mm": rain_mm,
            "geography": {
                "lat_abs": float(geo.loc[iso3, "lat_abs"]),
                "geo_x": float(geo.loc[iso3, "geo_x"]),
                "geo_y": float(geo.loc[iso3, "geo_y"]),
                "geo_z": float(geo.loc[iso3, "geo_z"]),
            },
            "history_2011_2013": history,
            "observed_crops": observed_crops,
        }
    return countries


def assert_country_invariants(countries: dict, df: pd.DataFrame) -> None:
    """Contrôles de cohérence sur le contexte pays généré, échec bloquant."""
    expected_iso3 = set(df["iso3"].unique())
    got_iso3 = set(countries)
    if got_iso3 != expected_iso3:
        raise RuntimeError(
            f"iso3 manquants : {sorted(expected_iso3 - got_iso3)} | "
            f"iso3 en trop : {sorted(got_iso3 - expected_iso3)}"
        )

    all_observed: set[str] = set()
    for iso3, entry in countries.items():
        for key in ("country", "rain_mm", "geography", "history_2011_2013", "observed_crops"):
            if key not in entry:
                raise RuntimeError(f"{iso3} : clé manquante « {key} »")
        if not entry["country"]:
            raise RuntimeError(f"{iso3} : nom de pays vide")
        years = [item["year"] for item in entry["history_2011_2013"]]
        if years != HISTORY_YEARS:
            raise RuntimeError(f"{iso3} : historique attendu {HISTORY_YEARS}, obtenu {years}")
        if not entry["observed_crops"]:
            raise RuntimeError(f"{iso3} : aucune culture observée")
        all_observed.update(entry["observed_crops"])

    expected_crops = set(RECOMMEND_CROPS)
    if all_observed != expected_crops:
        raise RuntimeError(
            f"union des cultures observées ≠ 10 cultures attendues : "
            f"manquantes {sorted(expected_crops - all_observed)}, "
            f"inconnues {sorted(all_observed - expected_crops)}"
        )


def report_defaults_out_of_domain(countries: dict, training_domain: dict) -> list[tuple[str, str, float, float, float]]:
    """Rapport informatif : defaults 2014 hors domaine (pas un invariant)."""
    bounds = {
        "temp_hist": (training_domain["temp_hist"]["min"], training_domain["temp_hist"]["max"]),
        "rain_mm": (training_domain["rain_mm"]["min"], training_domain["rain_mm"]["max"]),
        "log_pest_hist": (training_domain["log_pest_hist"]["min"], training_domain["log_pest_hist"]["max"]),
    }
    out: list[tuple[str, str, float, float, float]] = []
    for iso3, entry in countries.items():
        temps = [item["avg_temp"] for item in entry["history_2011_2013"]]
        pests = [item["pesticides_t"] for item in entry["history_2011_2013"]]
        defaults = {
            "temp_hist": float(np.mean(temps)),
            "rain_mm": entry["rain_mm"],
            "log_pest_hist": float(np.log1p(np.mean(pests))),
        }
        for name, value in defaults.items():
            lo, hi = bounds[name]
            if value < lo or value > hi:
                out.append((iso3, name, value, lo, hi))
    return out


def update_metadata(metadata_path: Path, training_domain: dict, model_version: str) -> None:
    """Ajoute `model_version` et `training_domain` au metadata sans toucher aux autres clés."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_version"] = model_version
    metadata["training_domain"] = training_domain
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    metadata_path = PATHS.root / "models" / "recommend_model_metadata.json"
    context_path = PATHS.root / "models" / "recommend_context.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_train_expected = int(metadata["n_train"])

    df, X_hist = build_training_frame()

    training_domain = compute_training_domain_model_units(X_hist, n_train_expected)

    countries = build_country_context(df)
    assert_country_invariants(countries, df)

    payload = {
        "generated_from": {
            "dataset": RECOMMEND_DATASET.name,
            "geojson": "ne_110m_admin_0_countries.geojson",
        },
        "history_years": HISTORY_YEARS,
        "target_year": TARGET_YEAR,
        "countries": countries,
    }
    context_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    update_metadata(metadata_path, training_domain, MODEL_VERSION)

    print(f"Contexte pays : {len(countries)} pays écrits dans {context_path.relative_to(PATHS.root)}")
    print(f"Metadata modèle : {metadata_path.relative_to(PATHS.root)} — model_version={MODEL_VERSION} et training_domain ajoutés")
    print(f"  temp_hist     : [{training_domain['temp_hist']['min']:.6f}, {training_domain['temp_hist']['max']:.6f}] °C")
    print(f"  rain_mm       : [{training_domain['rain_mm']['min']:.4f}, {training_domain['rain_mm']['max']:.4f}] mm")
    print(f"  log_pest_hist : [{training_domain['log_pest_hist']['min']:.6f}, {training_domain['log_pest_hist']['max']:.6f}] (log tonnes)")

    out_of_domain = report_defaults_out_of_domain(countries, training_domain)
    if out_of_domain:
        print(f"\n{len(out_of_domain)} default(s) 2014 hors domaine d'apprentissage (informatif) :")
        for iso3, name, value, lo, hi in out_of_domain:
            print(f"  {iso3} {name}={value:.4f} hors [{lo:.4f}, {hi:.4f}]")
    else:
        print("\nAucun default 2014 hors domaine.")


if __name__ == "__main__":
    main()
