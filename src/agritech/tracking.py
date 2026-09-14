"""Suivi des expériences dans MLflow, commun à `/predict` et `/recommend`.

La configuration vient de l'environnement (fichier `.env` du dépôt, non versionné, modèle dans
`.env.example`) :

- `MLFLOW_TRACKING_URI` renseignée : les runs sont envoyés à ce serveur, avec les identifiants
  `MLFLOW_TRACKING_USERNAME` et `MLFLOW_TRACKING_PASSWORD`. L'adresse n'apparaît ni dans le code ni
  dans les sorties des notebooks ;
- sinon, par exemple lors d'une réexécution par un tiers : les runs sont enregistrés localement dans
  `mlruns/`, à la racine du dépôt (non versionné). Les résultats sont les mêmes.
"""

from __future__ import annotations

import contextlib
import io
import os

import mlflow
from dotenv import load_dotenv

from agritech.config import PATHS


EXPERIMENT_PREFIX = "oc_p12_agritech"
SERVICES = ("predict", "recommend")
LOCAL_TRACKING_DIR = PATHS.root / "mlruns"


def setup_mlflow(service: str):
    """Configure MLflow depuis l'environnement, puis active l'expérience du service.

    Une expérience par service : `oc_p12_agritech_predict` et `oc_p12_agritech_recommend`, créée au
    premier appel si elle n'existe pas. Tous les runs d'un service y sont comparés sur les mêmes
    colonnes (`cv_rmse_mean`, `cv_r2_mean`...).
    """
    if service not in SERVICES:
        raise ValueError(f"service inconnu : {service!r}, attendu parmi {SERVICES}")

    load_dotenv(PATHS.root / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip() or LOCAL_TRACKING_DIR.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(f"{EXPERIMENT_PREFIX}_{service}")
    print("expérience MLflow :", experiment.name)
    return experiment


def run_tags(service: str, stage: str, notebook: str) -> dict[str, str]:
    """Tags communs à tous les runs : service, étape (`baseline`, `comparison`, `tuning`...) et source.

    Deux tags remplacent des valeurs que MLflow remplirait autrement : `mlflow.source.name`, qui
    contiendrait le chemin local du noyau Jupyter, et `mlflow.user`, qui contiendrait l'identifiant de
    connexion au serveur et apparaîtrait donc dans l'interface et les captures.
    """
    return {
        "service": service,
        "stage": stage,
        "mlflow.source.name": f"notebooks/{notebook}",
        "mlflow.user": "agritech",
    }


def log_run(
    run_name: str,
    tags: dict[str, str],
    params: dict,
    metrics: dict[str, float],
    features: dict[str, list[str]] | None = None,
) -> None:
    """Enregistre un modèle évalué : paramètres, métriques et variables utilisées.

    Les variables vont dans un fichier `features.json` plutôt que dans un tag, pour rester lisibles.
    À la fin du run, MLflow écrit un lien vers le serveur : cette sortie est masquée, pour que
    l'adresse du serveur n'apparaisse pas dans les notebooks versionnés. Rien n'est retourné, donc
    rien ne s'affiche sous la cellule.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        with mlflow.start_run(run_name=run_name, tags=tags):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if features is not None:
                mlflow.log_dict(features, "features.json")
