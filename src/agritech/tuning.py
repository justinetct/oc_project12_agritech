"""Recherche d'hyperparamètres sur des folds fixés, commune aux notebooks de tuning.

`random_search` tire des configurations au hasard dans une grille de valeurs, les évalue toutes sur
les mêmes folds (validation temporelle pour `/recommend`) et renvoie un essai par ligne :
hyperparamètres, métriques `cv_*` et RMSE de chaque fold. `position_in_range` et `value_effects`
aident à décider s'il faut élargir la recherche : meilleure valeur au bord de la grille, ou plateau.
`log_trials` écrit un run MLflow par essai.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from agritech.config import SEED
from agritech.evaluation import CV_SCORING
from agritech.modeling import RunInfo, log_model_run


def random_search(
    pipeline,
    space: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    n_iter: int,
    seed: int = SEED,
    fold_labels: list | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, float]:
    """Recherche aléatoire parmi les valeurs de `space`, évaluée sur les folds de `cv` ; renvoie les essais et la durée.

    - `pipeline` : pipeline dont la dernière étape s'appelle `model`, comme ceux de `experiment_pipeline` ;
    - `space` : hyperparamètre du modèle → liste de valeurs. Les configurations sont tirées sans remise
      parmi toutes les combinaisons, et toutes sont évaluées si `n_iter` dépasse leur nombre ;
    - `fold_labels` : libellés des folds, par exemple les années de validation ;
    - `n_jobs` : configurations évaluées en parallèle. 1 par défaut, le modèle utilisant déjà
      plusieurs cœurs.

    Une ligne par configuration, triée par RMSE moyenne : valeurs des hyperparamètres, `params` (dict,
    pour recréer le modèle), métriques `cv_*` au format de `summarize_cv_folds` (moyenne et écart-type
    de RMSE, MAE, R², temps d'ajustement), puis `rmse_<libellé>` pour chaque fold. L'index garde l'ordre
    de tirage. Aucun modèle n'est réentraîné sur toutes les données (`refit=False`).
    """
    search = RandomizedSearchCV(
        pipeline,
        {f"model__{name}": values for name, values in space.items()},
        n_iter=n_iter,
        scoring=CV_SCORING,
        refit=False,
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        error_score="raise",
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        # joblib relance parfois un processus de calcul en cours de recherche : message sans effet sur les résultats
        warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")
        search.fit(X, y)
    duration = time.perf_counter() - start

    raw = pd.DataFrame(search.cv_results_)
    n_folds = sum(1 for column in raw.columns if column.startswith("split") and column.endswith("_test_rmse"))
    labels = list(fold_labels) if fold_labels is not None else list(range(1, n_folds + 1))
    if len(labels) != n_folds:
        raise ValueError("un libellé par fold est attendu")

    trials = pd.DataFrame({name: raw[f"param_model__{name}"] for name in space})
    trials["params"] = [{key.removeprefix("model__"): value for key, value in params.items()} for params in raw["params"]]
    # scikit-learn renvoie la RMSE et la MAE en négatif : un score plus grand doit y être meilleur
    for metric, sign in [("rmse", -1), ("mae", -1), ("r2", 1)]:
        trials[f"cv_{metric}_mean"] = sign * raw[f"mean_test_{metric}"]
        trials[f"cv_{metric}_std"] = raw[f"std_test_{metric}"]
    trials["cv_fit_time_mean"] = raw["mean_fit_time"]
    trials["cv_fit_time_std"] = raw["std_fit_time"]
    for position, label in enumerate(labels):
        trials[f"rmse_{label}"] = -raw[f"split{position}_test_rmse"]
    return trials.sort_values("cv_rmse_mean"), duration


def position_in_range(trials: pd.DataFrame, space: dict[str, list]) -> pd.DataFrame:
    """Meilleure configuration : valeur de chaque hyperparamètre et position dans les valeurs testées.

    « début de plage » ou « fin de plage » : la meilleure valeur est la première ou la dernière de la
    liste de `space` ; la recherche peut être élargie dans cette direction.
    """
    best = trials.iloc[0]["params"]
    rows = []
    for name, values in space.items():
        value = best[name]
        if len(values) == 1:
            position = "fixé"
        elif value == values[0]:
            position = "début de plage"
        elif value == values[-1]:
            position = "fin de plage"
        else:
            position = "intérieur"
        # texte plutôt que nombre : pandas afficherait sinon les entiers d'une colonne mixte avec une décimale
        rows.append({"hyperparamètre": name, "meilleure valeur": str(value), "valeurs testées": str(values), "position": position})
    return pd.DataFrame(rows).set_index("hyperparamètre")


def value_effects(trials: pd.DataFrame, space: dict[str, list]) -> pd.DataFrame:
    """Pour chaque hyperparamètre et chaque valeur testée : nombre d'essais, meilleure RMSE et RMSE médiane.

    Lecture des plateaux : si plusieurs valeurs donnent presque la même meilleure RMSE, l'hyperparamètre
    compte peu dans cette zone ; si la meilleure RMSE baisse jusqu'au bord de la grille, la recherche
    peut être élargie de ce côté.
    """
    rows = []
    for name, values in space.items():
        for value in values:
            selected = trials[[params[name] == value for params in trials["params"]]]
            rows.append(
                {
                    "hyperparamètre": name,
                    "valeur": str(value),
                    "essais": len(selected),
                    "meilleure RMSE": selected["cv_rmse_mean"].min(),
                    "RMSE médiane": selected["cv_rmse_mean"].median(),
                }
            )
    return pd.DataFrame(rows).set_index(["hyperparamètre", "valeur"])


def log_trials(
    trials: pd.DataFrame,
    run: RunInfo,
    pipeline,
    X: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    engineered: list[str] | None = None,
    native_categorical: bool = False,
) -> None:
    """Un run MLflow par essai de `random_search`, via `log_model_run`.

    `run` sert de modèle : chaque essai s'appelle `<run.name>_<numéro de tirage>` et ajoute ses
    hyperparamètres à `run.params`. Métriques : les colonnes `cv_*` de l'essai. `X` est celui de la
    recherche ; `engineered` et `native_categorical` décrivent le pipeline, comme pour `log_model_run`.
    """
    metrics = [column for column in trials.columns if column.startswith("cv_")]
    for number in sorted(trials.index):
        trial = trials.loc[number]
        trial_run = RunInfo(
            f"{run.name}_{number + 1:03d}", run.feature_set, run.tags, run.protocol_params,
            run.params | trial["params"], replace=run.replace,
        )
        log_model_run(
            trial_run, pipeline, X, categorical, numeric, {metric: float(trial[metric]) for metric in metrics},
            engineered=engineered, native_categorical=native_categorical,
        )
