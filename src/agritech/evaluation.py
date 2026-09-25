"""Évaluation des modèles de rendement, commune à `/predict` et `/recommend`.

Métriques principales retenues : RMSE et R². La MAE est ajoutée en complément : elle donne
l'erreur moyenne directement en t/ha.

- `cross_validate_folds` fait la validation croisée et garde le détail par fold : scores et temps ;
- `summarize_cv_folds` en tire les moyennes et écarts-types, sous les noms des métriques MLflow ;
- `cross_validate_regressor` enchaîne les deux, pour comparer modèles, jeux de variables et
  hyperparamètres par validation croisée sur le jeu d'entraînement ;
- `compare_cv_folds` compare plusieurs évaluations fold par fold, et `predict_cv_folds` donne les
  prédictions hors apprentissage, y compris quand les folds ne couvrent pas toutes les lignes ;
- `errors_by_crop` détaille les erreurs culture par culture ;
- `regression_metrics` est réservée à l'évaluation finale du modèle retenu sur le jeu de test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
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


def compare_cv_folds(results: dict[str, pd.DataFrame], reference: str, fold_labels: list | None = None) -> pd.DataFrame:
    """Comparaison appariée de plusieurs évaluations faites sur les mêmes folds, fold par fold.

    `results` associe un nom aux scores par fold renvoyés par `cross_validate_folds` ou
    `run_experiment`. Pour chaque évaluation : RMSE moyenne, écart moyen de RMSE avec `reference`
    calculé fold par fold, nombre de folds où la RMSE est meilleure que celle de la référence, et MAE
    moyenne. L'écart-type entre folds n'est pas utilisé : en validation temporelle, il mesure surtout
    la différence entre années, pas l'incertitude sur l'écart entre deux modèles.

    `fold_labels`, par exemple les années de validation : ajoute le R² moyen puis la RMSE de chaque
    fold, une colonne par libellé, pour voir si un écart tient sur tous les folds ou vient d'un seul.
    """
    reference_rmse = results[reference]["rmse"].to_numpy()
    if fold_labels is not None and len(fold_labels) != len(reference_rmse):
        raise ValueError("un libellé par fold est attendu")

    rows = {}
    for name, folds in results.items():
        if len(folds) != len(reference_rmse):
            raise ValueError(f"{name} : {len(folds)} folds, {len(reference_rmse)} pour la référence")
        gaps = folds["rmse"].to_numpy() - reference_rmse
        rows[name] = {
            "RMSE": folds["rmse"].mean(),
            "écart moyen": gaps.mean(),
            "folds améliorés": f"{int((gaps < 0).sum())}/{len(gaps)}",
            "MAE": folds["mae"].mean(),
        }
        if fold_labels is not None:
            rows[name]["R²"] = folds["r2"].mean()
            rows[name] |= dict(zip(fold_labels, folds["rmse"].to_numpy()))
    return pd.DataFrame.from_dict(rows, orient="index")


def predict_cv_folds(model, X: pd.DataFrame, y: pd.Series, cv) -> pd.Series:
    """Prédictions hors apprentissage : chaque fold est prédit par une copie du modèle apprise sur son seul apprentissage.

    `cv` est une liste de couples de positions (apprentissage, validation), comme ceux de
    `temporal_cv`, ou un découpage scikit-learn. Contrairement à `cross_val_predict`, les folds n'ont
    pas à couvrir toutes les lignes : en validation temporelle, seules les dernières années sont
    prédites. Renvoie une Series indexée comme `X`, limitée aux lignes évaluées, dans l'ordre des
    folds. Une ligne évaluée dans deux folds provoque une erreur : une seule prédiction par ligne.
    """
    folds = cv.split(X, y) if hasattr(cv, "split") else cv

    predictions = []
    for train_positions, validation_positions in folds:
        fitted = clone(model).fit(X.iloc[train_positions], y.iloc[train_positions])
        validation = X.iloc[validation_positions]
        predictions.append(pd.Series(fitted.predict(validation), index=validation.index, name="prediction"))

    predictions = pd.concat(predictions)
    if predictions.index.has_duplicates:
        raise ValueError("une ligne est évaluée dans plusieurs folds, ou l'index de X n'est pas unique")
    return predictions


def errors_by_crop(y_true, y_pred, crops) -> pd.DataFrame:
    """Erreurs d'un modèle culture par culture, à partir des rendements observés, prédits et des cultures.

    Les trois entrées sont alignées par position et doivent avoir la même longueur. Pour chaque
    culture : nombre de lignes, rendement moyen observé, RMSE, MAE, biais (moyenne de prédit − observé :
    positif si le modèle surestime) et RMSE relative (RMSE / rendement moyen), qui rend comparables des
    cultures aux niveaux de rendement très différents. Cultures triées par rendement moyen.
    """
    table = pd.DataFrame({"crop": list(crops), "observed": list(y_true), "predicted": list(y_pred)})
    if not len(table) or table.isna().any().any():
        raise ValueError("erreurs par culture : entrées vides ou valeurs manquantes")
    table["error"] = table["predicted"] - table["observed"]

    by_crop = table.groupby("crop")
    result = pd.DataFrame(
        {
            "n": by_crop.size(),
            "rendement moyen": by_crop["observed"].mean(),
            "RMSE": by_crop["error"].apply(lambda errors: float((errors**2).mean() ** 0.5)),
            "MAE": by_crop["error"].apply(lambda errors: float(errors.abs().mean())),
            "biais": by_crop["error"].mean(),
        }
    )
    result["RMSE relative"] = result["RMSE"] / result["rendement moyen"]
    return result.sort_values("rendement moyen")


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
