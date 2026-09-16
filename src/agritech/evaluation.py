"""Évaluation des modèles de rendement, commune à `/predict` et `/recommend`.

Métriques principales retenues : RMSE et R². La MAE est ajoutée en complément : elle donne
l'erreur moyenne directement en t/ha.

- `cross_validate_folds` fait la validation croisée et garde le détail par fold : scores et temps ;
- `summarize_cv_folds` en tire les moyennes et écarts-types, sous les noms des métriques MLflow ;
- `cross_validate_regressor` enchaîne les deux, pour comparer modèles, jeux de variables et
  hyperparamètres par validation croisée sur le jeu d'entraînement ;
- `regression_metrics` est réservée à l'évaluation finale du modèle retenu sur le jeu de test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_validate


# Scorers scikit-learn : RMSE et MAE y sont négatives, car un score plus grand doit être meilleur.
CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def cross_validate_folds(model, X, y, cv) -> pd.DataFrame:
    """Validation croisée : une ligne par fold, avec les scores et les temps.

    `cross_validate` entraîne une copie du modèle sur chaque fold. Un preprocessing placé dans le
    pipeline est donc réappris sans voir le fold de validation : pas de fuite de données.

    Colonnes : `rmse`, `mae` et `r2` sur le fold de validation ; `fit_time`, durée de `fit` du
    pipeline complet, preprocessing compris ; `score_time`, durée du calcul des scores. Temps en
    secondes. Le détail par fold permet les comparaisons appariées : mêmes folds, écart fold à fold.
    """
    scores = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING, error_score="raise")

    # dans `cross_validate`, « test » désigne le fold de validation, pas le jeu de test du projet
    folds = pd.DataFrame(
        {
            "rmse": -scores["test_rmse"],
            "mae": -scores["test_mae"],
            "r2": scores["test_r2"],
            "fit_time": scores["fit_time"],
            "score_time": scores["score_time"],
        }
    )
    folds.index = pd.RangeIndex(start=1, stop=len(folds) + 1, name="fold")
    return folds


def summarize_cv_folds(folds: pd.DataFrame, with_fit_time: bool = False) -> dict[str, float]:
    """Moyenne et écart-type par fold de RMSE, MAE et R², et du temps d'ajustement sur demande.

    Les clés servent de noms de métriques MLflow : `cv_rmse_mean`, `cv_rmse_std`, etc.
    `with_fit_time=True` ajoute `cv_fit_time_mean` et `cv_fit_time_std`. Écart-type calculé comme
    NumPy (`ddof=0`).
    """
    columns = ["rmse", "mae", "r2"]
    if with_fit_time:
        columns.append("fit_time")

    metrics = {}
    for name in columns:
        values = folds[name].to_numpy()
        metrics[f"cv_{name}_mean"] = float(values.mean())
        metrics[f"cv_{name}_std"] = float(values.std())
    return metrics


def cross_validate_regressor(model, X, y, cv) -> dict[str, float]:
    """Validation croisée : moyenne et écart-type de RMSE, MAE et R² sur les folds.

    Raccourci de `summarize_cv_folds(cross_validate_folds(...))` : les six métriques historiques,
    sans les temps.
    """
    return summarize_cv_folds(cross_validate_folds(model, X, y, cv))


def format_cv_metrics(metrics: dict[str, float]) -> str:
    """Une ligne lisible, par exemple `RMSE 0.4993 ± 0.0006 t/ha | MAE ... | R² ...`."""
    return (
        f"RMSE {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f} t/ha"
        f" | MAE {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f} t/ha"
        f" | R² {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"
    )


def regression_metrics(y_true, y_pred, prefix: str = "test") -> dict[str, float]:
    """RMSE, MAE et R² sur un jeu donné, pour l'évaluation finale sur le jeu de test.

    Clés : `test_rmse`, `test_mae`, `test_r2` ; `prefix` distingue un autre jeu, par exemple `train`.
    """
    return {
        f"{prefix}_rmse": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }
