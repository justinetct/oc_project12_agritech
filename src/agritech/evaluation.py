"""Évaluation des modèles de rendement, commune à `/predict` et `/recommend`.

Métriques principales retenues : RMSE et R². La MAE est ajoutée en complément : elle donne
l'erreur moyenne directement en t/ha.

- `cross_validate_regressor` compare modèles, jeux de variables et hyperparamètres par validation
  croisée sur le jeu d'entraînement ;
- `regression_metrics` est réservée à l'évaluation finale du modèle retenu sur le jeu de test.
"""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_validate


# Scorers scikit-learn : RMSE et MAE y sont négatives, car un score plus grand doit être meilleur.
CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def cross_validate_regressor(model, X, y, cv) -> dict[str, float]:
    """Validation croisée : moyenne et écart-type de RMSE, MAE et R² sur les folds.

    `cross_validate` entraîne une copie du modèle sur chaque fold. Un preprocessing placé dans le
    pipeline est donc réappris sans voir le fold de validation : pas de fuite de données.
    Les clés servent aussi de noms de métriques MLflow : `cv_rmse_mean`, `cv_rmse_std`, etc.
    """
    scores = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING, error_score="raise")

    # dans `cross_validate`, « test » désigne le fold de validation, pas le jeu de test du projet
    folds = {
        "rmse": -scores["test_rmse"],
        "mae": -scores["test_mae"],
        "r2": scores["test_r2"],
    }
    metrics = {}
    for name, values in folds.items():
        metrics[f"cv_{name}_mean"] = float(values.mean())
        metrics[f"cv_{name}_std"] = float(values.std())
    return metrics


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
