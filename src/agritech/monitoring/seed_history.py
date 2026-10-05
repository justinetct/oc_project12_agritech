"""Génère un historique de démonstration dans la base de monitoring.

Outil manuel de développement et de démonstration : il remplit la table
`api_requests` avec des appels `/predict` et `/recommend` plausibles sur
plusieurs semaines, pour que le dashboard de monitoring (vues 7 / 30 / 90
jours) montre des volumes, des latences et quelques erreurs sans attendre
des semaines d'utilisation réelle. Le script n'est jamais lancé
automatiquement. Seule exception, explicite : avec `MONITORING_DEMO_HISTORY`
activé, l'API appelle `seed_demo_history` à son démarrage, qui ajoute le même
historique uniquement si la base est vide.

    poetry run python -m agritech.monitoring.seed_history            # aperçu, n'écrit rien
    poetry run python -m agritech.monitoring.seed_history --write    # ajoute les lignes

Fonctionnement :

- **aperçu par défaut** : sans `--write`, le script affiche ce qu'il
  ajouterait (volume, services, erreurs, période) et ne fait que lire la
  base ; une base absente n'est pas créée ;
- **ajout seulement** : `--write` insère les lignes en une transaction, sans
  jamais supprimer ni modifier une ligne existante. Relancé, il ajoute un
  second jeu : c'est à l'utilisateur de ne lancer `--write` qu'une fois ;
- **reproductible** : à `--seed`, `--days` et fin de période identiques,
  l'historique généré est identique ;
- **fin de période** : par défaut, juste avant l'appel le plus récent déjà
  archivé (sinon maintenant), pour que les derniers appels réels restent les
  plus récents ; `--end` fixe une autre fin ;
- **garde-fou prod** : refus si `ENVIRONMENT` vaut `prod`, sauf `--allow-prod`.

Ce qui est généré, et ce qui ne l'est pas :

- erreurs : uniquement des `422 validation_error`, sur des corps réellement
  rejetés par les schémas de l'API ; la réponse d'erreur a le même corps que
  celle du handler 422, comme pour un appel réel ;
- succès : corps valides (domaine d'apprentissage pour l'essentiel, quelques
  valeurs en dehors), pays réels de `models/recommend_context.json`.
  Aucune prédiction n'est inventée : `response_payload` reste vide (`NULL`),
  comme `logfire_trace_id` ;
- versions : `api_version` de l'API, `model_version` lue dans les
  métadonnées des modèles, `environment` = valeur réelle de `ENVIRONMENT` ;
- aucune donnée personnelle, aucun token, aucun en-tête, aucune IP.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, inspect, select
from sqlalchemy.engine import make_url

from agritech.api.error_handlers import validation_error_body, validation_summary
from agritech.api.schemas.predict import PredictRequest
from agritech.api.schemas.recommend import RecommendRequest
from agritech.api.version import API_VERSION
from agritech.config import PATHS
from agritech.monitoring.config import load_config
from agritech.monitoring.models import ApiRequest, Base
from agritech.monitoring.session import create_monitoring_engine, create_session_factory


DEFAULT_DAYS = 90
DEFAULT_SEED = 42

# Valeurs de `ENVIRONMENT` qui désignent la production (voir docker-compose.yml).
PROD_ENVIRONMENTS = {"prod", "production"}

# Volume quotidien : jours ouvrés et week-ends, bornes incluses.
WEEKDAY_REQUESTS = (15, 35)
WEEKEND_REQUESTS = (5, 15)
# Quelques jours ouvrés plus chargés : probabilité et multiplicateur.
BUSY_DAY_PROBABILITY = 0.08
BUSY_DAY_FACTOR = (1.3, 1.6)

# Part de /predict, tirée chaque jour autour de cette moyenne.
PREDICT_SHARE = 0.55
PREDICT_SHARE_SPREAD = 0.06

# Erreurs 422 : taux de fond, et quelques jours où il monte (essais ratés d'une
# intégration cliente, par exemple), sans jamais devenir majoritaire.
ERROR_RATE = 0.035
ERROR_BURST_PROBABILITY = 0.05
ERROR_BURST_RATE = (0.10, 0.18)

# Latences (ms) : plage habituelle, plage lente et part d'appels lents.
LATENCY_MS = {
    "predict": {"usual": (4, 12), "slow": (20, 50)},
    "recommend": {"usual": (8, 20), "slow": (20, 60)},
}
SLOW_CALL_PROBABILITY = 0.07
ERROR_LATENCY_MS = (1, 3)

# Heures d'activité (UTC) : surtout en journée, un peu le soir, rarement la nuit.
HOUR_WEIGHTS = [
    1, 1, 1, 1, 1, 2,      # 00 h - 05 h
    4, 8, 12, 14, 14, 12,  # 06 h - 11 h
    9, 12, 14, 14, 12, 9,  # 12 h - 17 h
    6, 4, 3, 2, 2, 1,      # 18 h - 23 h
]

# Domaine d'apprentissage de /predict (voir `GET /predict/context`) et part des
# appels réussis volontairement en dehors (acceptés, signalés par l'API).
PREDICT_RAINFALL_MM = (100.0, 1000.0)
PREDICT_TEMPERATURE_C = (15.0, 40.0)
OUT_OF_DOMAIN_PROBABILITY = 0.05

# Part des appels /recommend sans conditions (valeurs par défaut du pays).
RECOMMEND_DEFAULTS_ONLY_PROBABILITY = 0.3

SERVICES = {
    "predict": {"endpoint": "/predict", "request_model": PredictRequest},
    "recommend": {"endpoint": "/recommend", "request_model": RecommendRequest},
}


class SeedRefusedError(RuntimeError):
    """Exécution refusée par un garde-fou (production sans `--allow-prod`)."""


@dataclass(frozen=True)
class Catalog:
    """Ce qui rend les appels générés cohérents avec le projet actuel."""

    api_version: str
    model_versions: dict[str, str]
    # iso3 → conditions par défaut du pays (unités publiques de /recommend).
    country_defaults: dict[str, dict[str, float]]
    environment: str


def load_catalog(environment: str) -> Catalog:
    """Versions et pays réels, lus dans les fichiers de modèles du dépôt."""
    models_dir = PATHS.root / "models"
    model_versions = {
        service: json.loads((models_dir / f"{service}_model_metadata.json").read_text())[
            "model_version"
        ]
        for service in SERVICES
    }
    context = json.loads((models_dir / "recommend_context.json").read_text())
    country_defaults = {
        iso3: {
            "average_temperature_celsius": mean(
                float(item["avg_temp"]) for item in entry["history_2011_2013"]
            ),
            "annual_rainfall_mm": float(entry["rain_mm"]),
            "average_annual_pesticides_tons": mean(
                float(item["pesticides_t"]) for item in entry["history_2011_2013"]
            ),
        }
        for iso3, entry in sorted(context["countries"].items())
    }
    return Catalog(API_VERSION, model_versions, country_defaults, environment)


# --- Génération ------------------------------------------------------------------


def generate_history(
    catalog: Catalog, *, days: int, end: datetime, seed: int
) -> list[dict[str, Any]]:
    """Les lignes `api_requests` de la période, dans l'ordre chronologique.

    La période couvre les `days` jours calendaires UTC qui se terminent le
    jour de `end` ; aucun appel n'est placé à `end` ou après. Le résultat ne
    dépend que des arguments : même `seed`, même historique.
    """
    if end.tzinfo is None:
        raise ValueError("end doit être un datetime avec fuseau (UTC)")
    end = end.astimezone(timezone.utc)
    rng = random.Random(seed)
    first_day = end.date() - timedelta(days=days - 1)
    rows: list[dict[str, Any]] = []
    for offset in range(days):
        day = first_day + timedelta(days=offset)
        rows.extend(_day_rows(rng, catalog, day, end))
    return rows


def _day_rows(
    rng: random.Random, catalog: Catalog, day: date, end: datetime
) -> list[dict[str, Any]]:
    """Les appels d'une journée : volume, répartition, erreurs et heures."""
    weekend = day.weekday() >= 5
    low, high = WEEKEND_REQUESTS if weekend else WEEKDAY_REQUESTS
    count = rng.randint(low, high)
    if not weekend and rng.random() < BUSY_DAY_PROBABILITY:
        count = round(count * rng.uniform(*BUSY_DAY_FACTOR))

    predict_share = min(0.75, max(0.35, rng.gauss(PREDICT_SHARE, PREDICT_SHARE_SPREAD)))
    error_rate = ERROR_RATE
    if rng.random() < ERROR_BURST_PROBABILITY:
        error_rate = rng.uniform(*ERROR_BURST_RATE)

    rows = []
    for moment in sorted(_random_moment(rng, day) for _ in range(count)):
        if moment >= end:
            continue
        service = "predict" if rng.random() < predict_share else "recommend"
        success = rng.random() >= error_rate
        rows.append(_row(rng, catalog, service, success, moment))
    return rows


def _random_moment(rng: random.Random, day: date) -> datetime:
    """Un instant UTC de la journée, surtout aux heures ouvrées."""
    hour = rng.choices(range(24), weights=HOUR_WEIGHTS)[0]
    start = datetime.combine(day, time(hour), tzinfo=timezone.utc)
    return start + timedelta(microseconds=rng.randrange(3_600_000_000))


def _row(
    rng: random.Random, catalog: Catalog, service: str, success: bool, moment: datetime
) -> dict[str, Any]:
    """Une ligne complète, au format écrit par le middleware de l'API."""
    request_model = SERVICES[service]["request_model"]
    if success:
        payload = (
            _predict_payload(rng) if service == "predict" else _recommend_payload(rng, catalog)
        )
        response_payload = None
        duration_ms = _latency(rng, service)
    else:
        payload = (
            _invalid_predict_payload(rng)
            if service == "predict"
            else _invalid_recommend_payload(rng, catalog)
        )
        response_payload = validation_error_response(request_model, payload)
        duration_ms = rng.randint(*ERROR_LATENCY_MS)
    return {
        "timestamp": moment,
        "service": service,
        "endpoint": SERVICES[service]["endpoint"],
        "method": "POST",
        "status_code": 200 if success else 422,
        "success": success,
        "duration_ms": duration_ms,
        "api_version": catalog.api_version,
        "model_version": catalog.model_versions[service],
        "request_payload": payload,
        "response_payload": response_payload,
        "error_type": None if success else response_payload["error"],
        "error_message": None
        if success
        else validation_summary(response_payload["details"]) or response_payload["message"],
        "logfire_trace_id": None,
        "environment": catalog.environment,
    }


def _latency(rng: random.Random, service: str) -> int:
    """Durée d'un appel réussi : le plus souvent habituelle, parfois lente."""
    ranges = LATENCY_MS[service]
    if rng.random() < SLOW_CALL_PROBABILITY:
        return rng.randint(*ranges["slow"])
    low, high = ranges["usual"]
    # Distribution penchée vers le bas de la plage, comme les mesures réelles.
    return round(rng.triangular(low, high, low + (high - low) / 3))


def _predict_payload(rng: random.Random) -> dict[str, Any]:
    """Corps /predict valide, dans le domaine d'apprentissage le plus souvent."""
    if rng.random() < OUT_OF_DOMAIN_PROBABILITY:
        rainfall = rng.choice([rng.uniform(20, 99), rng.uniform(1001, 1400)])
    else:
        rainfall = rng.uniform(*PREDICT_RAINFALL_MM)
    return {
        "rainfall_mm": float(round(rainfall)),
        "temperature_celsius": round(rng.uniform(*PREDICT_TEMPERATURE_C), 1),
        "fertilizer_used": rng.random() < 0.5,
        "irrigation_used": rng.random() < 0.4,
    }


def _recommend_payload(rng: random.Random, catalog: Catalog) -> dict[str, Any]:
    """Corps /recommend valide : un pays réel, avec ou sans conditions ajustées."""
    iso3 = rng.choice(list(catalog.country_defaults))
    if rng.random() < RECOMMEND_DEFAULTS_ONLY_PROBABILITY:
        return {"iso3": iso3}
    defaults = catalog.country_defaults[iso3]
    return {
        "iso3": iso3,
        "conditions": {
            "average_temperature_celsius": round(
                defaults["average_temperature_celsius"] + rng.uniform(-2, 2), 1
            ),
            "annual_rainfall_mm": float(
                round(defaults["annual_rainfall_mm"] * rng.uniform(0.6, 1.4))
            ),
            "average_annual_pesticides_tons": round(
                defaults["average_annual_pesticides_tons"] * rng.uniform(0.7, 1.3), 2
            ),
        },
    }


def _invalid_predict_payload(rng: random.Random) -> dict[str, Any]:
    """Corps /predict que l'API rejette (422) : une seule faute par appel."""
    payload = _predict_payload(rng)
    mistake = rng.choice(["negative_rain", "temperature", "missing", "extra", "not_bool"])
    if mistake == "negative_rain":
        payload["rainfall_mm"] = -float(rng.randint(1, 50))
    elif mistake == "temperature":
        payload["temperature_celsius"] = float(rng.choice([rng.randint(61, 95), rng.randint(-80, -51)]))
    elif mistake == "missing":
        del payload[rng.choice(["fertilizer_used", "irrigation_used", "temperature_celsius"])]
    elif mistake == "extra":
        payload["crop"] = rng.choice(["Wheat", "Maize", "Rice"])
    else:
        payload["fertilizer_used"] = rng.choice(["yes", "true", 1])
    return payload


def _invalid_recommend_payload(rng: random.Random, catalog: Catalog) -> dict[str, Any]:
    """Corps /recommend que l'API rejette (422) : une seule faute par appel."""
    iso3 = rng.choice(list(catalog.country_defaults))
    mistake = rng.choice(["extra", "lowercase", "negative_rain", "temperature"])
    if mistake == "extra":
        return {"iso3": iso3, "crop": rng.choice(["Wheat", "Maize", "Potatoes"])}
    if mistake == "lowercase":
        return {"iso3": iso3.lower()}
    payload = _recommend_payload(rng, catalog)
    conditions = payload.setdefault("conditions", {})
    if mistake == "negative_rain":
        conditions["annual_rainfall_mm"] = -float(rng.randint(1, 200))
    else:
        conditions["average_temperature_celsius"] = float(rng.randint(61, 90))
    return payload


def validation_error_response(request_model: type[BaseModel], payload: Any) -> dict[str, Any]:
    """Réponse 422 que l'API renverrait pour ce corps, construite comme par son handler.

    Lève `ValueError` si le corps est valide : une ligne d'erreur générée
    correspond toujours à un corps réellement rejeté par le contrat.
    """
    try:
        request_model.model_validate(payload)
    except ValidationError as exc:
        errors = [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        return validation_error_body(RequestValidationError(errors))
    raise ValueError("corps valide : l'API ne renverrait pas d'erreur 422")


# --- Résumé et écriture ----------------------------------------------------------


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Ce que le script va ajouter : volumes, services, erreurs et période."""
    services = Counter(row["service"] for row in rows)
    errors = sum(not row["success"] for row in rows)
    per_day = Counter(row["timestamp"].date() for row in rows)
    weekday = [count for day, count in per_day.items() if day.weekday() < 5]
    weekend = [count for day, count in per_day.items() if day.weekday() >= 5]
    return {
        "total": len(rows),
        "predict": services["predict"],
        "recommend": services["recommend"],
        "success": len(rows) - errors,
        "errors": errors,
        "error_rate": errors / len(rows) if rows else 0.0,
        "first": rows[0]["timestamp"] if rows else None,
        "last": rows[-1]["timestamp"] if rows else None,
        "weekday_mean": mean(weekday) if weekday else 0.0,
        "weekend_mean": mean(weekend) if weekend else 0.0,
    }


def check_environment(environment: str, *, allow_prod: bool) -> None:
    """Refuse d'écrire l'historique en production sans demande explicite."""
    if environment.strip().lower() in PROD_ENVIRONMENTS and not allow_prod:
        raise SeedRefusedError(
            f"ENVIRONMENT={environment} : refus d'ajouter un historique de démonstration "
            "en production. Relancer avec --allow-prod seulement si c'est voulu."
        )


def insert_rows(session_factory, rows: list[dict[str, Any]]) -> int:
    """Ajoute toutes les lignes en une transaction (tout ou rien), sans rien supprimer."""
    with session_factory() as session:
        session.add_all(ApiRequest(**values) for values in rows)
        session.commit()
    return len(rows)


def seed_demo_history(session_factory, environment: str, *, now: datetime | None = None) -> int:
    """Ajoute l'historique de démonstration si la table `api_requests` est vide.

    Appelée au démarrage de l'API quand `MONITORING_DEMO_HISTORY` est activé :
    90 jours (graine 42) qui se terminent à `now`, maintenant par défaut, pour
    que les vues 7 / 30 / 90 jours du dashboard soient remplies dès
    l'ouverture. Une table qui contient déjà des appels n'est jamais modifiée.
    Renvoie le nombre de lignes ajoutées (0 si la table n'était pas vide).
    """
    existing, _ = _count_and_latest(session_factory)
    if existing:
        return 0
    rows = generate_history(
        load_catalog(environment), days=DEFAULT_DAYS, end=now or datetime.now(timezone.utc), seed=DEFAULT_SEED
    )
    return insert_rows(session_factory, rows)


def _count_and_latest(session_factory) -> tuple[int, datetime | None]:
    with session_factory() as session:
        return session.execute(select(func.count(), func.max(ApiRequest.timestamp))).one()


def _database_label(database_url: str) -> str:
    """Base ciblée, sans identifiants ni chemin complet : moteur et nom du fichier."""
    url = make_url(database_url)
    name = (url.database or "").rsplit("/", 1)[-1] or "(mémoire)"
    return f"{url.get_backend_name()} · {name}"


def _print_summary(summary: dict[str, Any], *, environment: str, database: str, existing: int) -> None:
    def share(part: int) -> str:
        return f"{part / summary['total']:.0%}" if summary["total"] else "—"

    first = summary["first"].strftime("%Y-%m-%d %H:%M") if summary["first"] else "—"
    last = summary["last"].strftime("%Y-%m-%d %H:%M") if summary["last"] else "—"
    print(f"Base ciblée      : {database} (environnement : {environment})")
    print(f"Lignes existantes: {existing} (conservées, rien n'est supprimé)")
    print(f"Lignes à ajouter : {summary['total']}")
    print(f"  /predict       : {summary['predict']} ({share(summary['predict'])})")
    print(f"  /recommend     : {summary['recommend']} ({share(summary['recommend'])})")
    print(f"  succès         : {summary['success']}")
    print(f"  erreurs 422    : {summary['errors']} ({summary['error_rate']:.1%})")
    print(f"Période (UTC)    : {first} → {last}")
    print(
        f"Moyenne par jour : {summary['weekday_mean']:.1f} en semaine, "
        f"{summary['weekend_mean']:.1f} le week-end"
    )


# --- Ligne de commande -----------------------------------------------------------


def _parse_end(text: str) -> datetime:
    """`--end` : date ou date-heure ISO, lue en UTC si aucun fuseau n'est donné."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"date invalide : {text!r}") from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ajoute un historique de démonstration à la base de monitoring. "
            "Sans --write, affiche seulement ce qui serait ajouté."
        )
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="jours d'historique (défaut : 90)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="graine aléatoire (défaut : 42)")
    parser.add_argument(
        "--end",
        type=_parse_end,
        help="fin de période, exclue (défaut : appel le plus récent déjà archivé, sinon maintenant)",
    )
    parser.add_argument("--write", action="store_true", help="écrit réellement les lignes dans la base")
    parser.add_argument(
        "--allow-prod", action="store_true", help="autorise l'écriture quand ENVIRONMENT=prod"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : aperçu par défaut, écriture avec `--write`."""
    args = _build_parser().parse_args(argv)
    if args.days < 1:
        print("--days doit être au moins 1.", file=sys.stderr)
        return 2

    load_dotenv(override=False)
    config = load_config()
    try:
        check_environment(config.environment, allow_prod=args.allow_prod)
    except SeedRefusedError as exc:
        print(exc, file=sys.stderr)
        return 1

    catalog = load_catalog(config.environment)
    database = _database_label(config.database_url)
    if not args.write and _sqlite_file_missing(config.database_url):
        # Aperçu sur une base qui n'existe pas encore : ne pas la créer.
        rows = generate_history(catalog, days=args.days, end=_end(args.end, None), seed=args.seed)
        _print_summary(summarize(rows), environment=config.environment, database=database, existing=0)
        print(PREVIEW_ONLY)
        return 0

    engine = create_monitoring_engine(config)
    try:
        session_factory = create_session_factory(engine)
        if args.write:
            Base.metadata.create_all(engine)
        existing, latest = (
            _count_and_latest(session_factory)
            if inspect(engine).has_table(ApiRequest.__tablename__)
            else (0, None)
        )
        rows = generate_history(catalog, days=args.days, end=_end(args.end, latest), seed=args.seed)
        _print_summary(
            summarize(rows), environment=config.environment, database=database, existing=existing
        )
        if not args.write:
            print(PREVIEW_ONLY)
            return 0
        inserted = insert_rows(session_factory, rows)
        total, _ = _count_and_latest(session_factory)
        print(f"\n{inserted} lignes ajoutées. La table contient maintenant {total} lignes.")
        return 0
    finally:
        engine.dispose()


PREVIEW_ONLY = "\nAperçu seulement : rien n'a été écrit. Ajouter --write pour insérer ces lignes."


def _end(requested: datetime | None, latest: datetime | None) -> datetime:
    """Fin de période : `--end`, sinon l'appel le plus récent archivé, sinon maintenant."""
    return requested or latest or datetime.now(timezone.utc)


def _sqlite_file_missing(database_url: str) -> bool:
    """Vrai pour une base SQLite fichier qui n'existe pas encore."""
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return False
    # Chemin relatif : résolu depuis le dossier courant, comme le fait SQLite.
    return not Path(url.database).exists()


if __name__ == "__main__":
    sys.exit(main())
