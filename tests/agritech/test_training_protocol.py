"""Invariants du protocole d'entraînement `/recommend` : pas de fuite temporelle.

Petits DataFrames synthétiques : aucun dataset réel, aucun réseau, aucun MLflow.
On vérifie que les conditions historiques n'utilisent que les années passées, que
le serving reconstruit les mêmes variables qu'à l'entraînement, et que le
découpage temporel garde l'année de test à part.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agritech.recommend_features import add_historical_conditions
from agritech.serving import RecommendContext, _assemble_candidates, _country_defaults
from agritech.training_data import temporal_cv, temporal_split


def _country(iso3: str, rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """Une ligne par année : (année, température moyenne, tonnage de pesticides)."""
    return pd.DataFrame(
        [{"iso3": iso3, "year": year, "avg_temp": temp, "pesticides_t": pest} for year, temp, pest in rows]
    )


def _row(df: pd.DataFrame, iso3: str, year: int) -> pd.Series:
    return df[(df["iso3"] == iso3) & (df["year"] == year)].iloc[0]


# --- add_historical_conditions -------------------------------------------------


def test_historical_conditions_use_only_the_three_previous_years():
    df = _country(
        "AAA",
        [(2009, 10.0, 1.0), (2010, 11.0, 2.0), (2011, 12.0, 3.0), (2012, 13.0, 4.0), (2013, 99.0, 99.0)],
    )

    row = _row(add_historical_conditions(df), "AAA", 2013)

    # 2010-2012 seulement : ni 2009 (trop ancienne), ni 2013 (l'année elle-même).
    assert row["temp_hist"] == pytest.approx((11.0 + 12.0 + 13.0) / 3)
    assert row["pest_hist"] == pytest.approx((2.0 + 3.0 + 4.0) / 3)


def test_changing_the_current_year_never_changes_its_history():
    """Absence de fuite : les valeurs de l'année t n'entrent pas dans l'historique de t."""
    rows = [(2010, 11.0, 2.0), (2011, 12.0, 3.0), (2012, 13.0, 4.0)]
    before = _row(add_historical_conditions(_country("AAA", rows)), "AAA", 2012)
    after = _row(add_historical_conditions(_country("AAA", rows[:-1] + [(2012, 50.0, 5000.0)])), "AAA", 2012)

    assert after["temp_hist"] == before["temp_hist"]
    assert after["log_pest_hist"] == before["log_pest_hist"]


def test_first_year_has_no_history_and_a_missing_year_is_not_filled():
    df = _country("AAA", [(2008, 10.0, 1.0), (2012, 20.0, 9.0)])

    result = add_historical_conditions(df)

    # Première année : aucun historique.
    assert _row(result, "AAA", 2008)[["temp_hist", "pest_hist", "log_pest_hist"]].isna().all()
    # 2009-2011 absentes : 2008 est trop ancienne pour 2012, rien ne la remplace.
    assert _row(result, "AAA", 2012)[["temp_hist", "pest_hist", "log_pest_hist"]].isna().all()


def test_history_never_mixes_two_countries():
    df = pd.concat(
        [
            _country("AAA", [(2011, 10.0, 1.0), (2012, 0.0, 0.0)]),
            _country("BBB", [(2011, 30.0, 9.0), (2012, 0.0, 0.0)]),
        ]
    )

    result = add_historical_conditions(df)

    assert _row(result, "AAA", 2012)["temp_hist"] == 10.0
    assert _row(result, "BBB", 2012)["temp_hist"] == 30.0


def test_log_pest_hist_is_the_log_of_the_mean_not_the_mean_of_logs():
    pesticides = [10.0, 1_000.0, 100_000.0]
    df = _country("AAA", [(2011, 1.0, pesticides[0]), (2012, 1.0, pesticides[1]), (2013, 1.0, pesticides[2])])
    df = pd.concat([df, _country("AAA", [(2014, 1.0, 0.0)])])

    row = _row(add_historical_conditions(df), "AAA", 2014)

    log_of_mean = math.log1p(np.mean(pesticides))
    mean_of_logs = np.mean(np.log1p(pesticides))
    assert row["log_pest_hist"] == pytest.approx(log_of_mean)
    assert abs(log_of_mean - mean_of_logs) > 1  # les deux calculs diffèrent nettement ici


# --- Cohérence entraînement / serving ------------------------------------------


def test_serving_rebuilds_the_same_history_features_as_training():
    """L'historique 2011-2013 d'un pays donne les mêmes `temp_hist` / `log_pest_hist` pour 2014."""
    history = [(2011, 14.2, 120.0), (2012, 15.9, 3_400.0), (2013, 17.3, 56_000.0)]
    with_2014 = _country("AAA", history + [(2014, np.nan, np.nan)])
    training = _row(add_historical_conditions(with_2014), "AAA", 2014)

    entry = {
        "country": "Alpha",
        "rain_mm": 500.0,
        "geography": {"lat_abs": 30.0, "geo_x": 0.7, "geo_y": 0.2, "geo_z": 0.5},
        "history_2011_2013": [{"year": y, "avg_temp": t, "pesticides_t": p} for y, t, p in history],
        "observed_crops": [],
    }
    context = RecommendContext(
        countries={"AAA": entry}, observed_couples=frozenset(), country_entries=[("AAA", "Alpha")]
    )
    served = _assemble_candidates(context, "AAA", _country_defaults(entry))

    assert (served["year"] == 2014).all()
    assert served["temp_hist"].iloc[0] == pytest.approx(training["temp_hist"])
    assert served["log_pest_hist"].iloc[0] == pytest.approx(training["log_pest_hist"])


# --- temporal_split ------------------------------------------------------------


def _yearly(years: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"year": years, "x": range(len(years)), "y": range(len(years))})


def test_temporal_split_keeps_the_test_year_apart():
    df = _yearly([2010, 2011, 2012, 2013, 2013, 2012])

    X_train, X_test, y_train, y_test = temporal_split(df, ["year", "x"], "y", test_year=2013, verbose=False)

    assert set(X_train["year"]) == {2010, 2011, 2012}
    assert set(X_test["year"]) == {2013}
    assert len(X_train) + len(X_test) == len(df)
    assert list(y_train.index) == list(X_train.index) and list(y_test.index) == list(X_test.index)


@pytest.mark.parametrize(
    "years",
    [[2010, 2011, 2012], [2011, 2012, 2013, 2014]],
    ids=["annee-de-test-absente", "annee-posterieure-au-test"],
)
def test_temporal_split_refuses_an_inconsistent_calendar(years):
    with pytest.raises(ValueError):
        temporal_split(_yearly(years), ["x"], "y", test_year=2013, verbose=False)


# --- temporal_cv ---------------------------------------------------------------


def test_each_temporal_fold_learns_only_from_earlier_years():
    years = np.array([2006, 2007, 2008, 2008, 2009, 2010, 2011, 2012, 2012])
    validation_years = [2008, 2009, 2010, 2011, 2012]

    folds = temporal_cv(years, validation_years, test_year=2013, verbose=False)

    assert len(folds) == len(validation_years)
    for year, (train_positions, validation_positions) in zip(validation_years, folds):
        assert set(years[validation_positions]) == {year}
        assert years[train_positions].max() < year
        # Fenêtre qui s'agrandit : toutes les années antérieures, pas seulement la précédente.
        assert set(train_positions) == set(np.flatnonzero(years < year))


def test_temporal_cv_refuses_test_year_rows_in_the_training_years():
    with pytest.raises(ValueError):
        temporal_cv(np.array([2010, 2011, 2012, 2013]), [2011, 2012], test_year=2013, verbose=False)


@pytest.mark.parametrize(
    "validation_years",
    [[2010], [2011, 2012]],
    ids=["aucune-annee-anterieure", "annee-validee-absente"],
)
def test_temporal_cv_refuses_an_empty_fold(validation_years):
    with pytest.raises(ValueError):
        temporal_cv(np.array([2010, 2011, 2010]), validation_years, test_year=2013, verbose=False)
