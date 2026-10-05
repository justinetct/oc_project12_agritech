"""Repository de la table `api_requests` : insertion et lectures dédiées.

Le repository ne connaît ni HTTP, ni la sémantique métier d'une ligne :

- `insert_api_request` prend un dict de valeurs, construit un `ApiRequest`
  et l'écrit. Toute la politique « une panne de persistance ne casse jamais
  une requête HTTP » vit dans le middleware, qui appelle cette fonction dans
  un `try/except` ;
- `list_recent_requests` lit les dernières lignes archivées, avec des
  filtres simples ;
- `summarize_requests` calcule les agrégats d'une période (volumes, taux de
  succès, latences, erreurs, volume quotidien par service).

Les deux lectures ne font que des SELECT : aucune écriture, aucun commit.

Les lectures restent des fonctions dédiées aux besoins du monitoring, pas un
accès générique à la table. La validation des paramètres (bornes de `limit`,
valeurs de `service`) appartient à la couche HTTP.
"""

from __future__ import annotations

import statistics
from datetime import datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agritech.monitoring.models import ApiRequest


# Services métier archivés dans `api_requests`. Le résumé les renvoie
# toujours tous les deux, même sans requête, pour garder une forme stable.
MONITORED_SERVICES = ("predict", "recommend")


def insert_api_request(session: Session, values: dict[str, Any]) -> int:
    """Insère une ligne `api_requests` et renvoie son `id` auto-incrémenté.

    `values` reprend les noms des colonnes du modèle `ApiRequest` — le
    middleware construit ce dict à partir de la requête HTTP, de la réponse
    et du contexte (api_version, model_version, environment, trace id).
    Toute clé inconnue lèvera un `TypeError` immédiat de SQLAlchemy, ce qui
    fait remonter l'erreur au plus tôt.
    """
    row = ApiRequest(**values)
    session.add(row)
    session.commit()
    return row.id


def list_recent_requests(
    session: Session,
    *,
    limit: int,
    service: str | None = None,
    success: bool | None = None,
) -> list[ApiRequest]:
    """Renvoie les `limit` lignes les plus récentes, éventuellement filtrées.

    Tri du plus récent au plus ancien sur `timestamp`, puis sur `id` pour
    départager deux lignes au même timestamp : l'ordre reste déterministe.

    Filtres facultatifs et combinables :
        service : `"predict"` ou `"recommend"` ; `None` garde les deux.
        success : `True` (succès) ou `False` (erreurs) ; `None` garde tout.

    Aucune validation de `limit` ici : les bornes sont imposées par
    l'endpoint HTTP qui appelle cette fonction.
    """
    query = select(ApiRequest)
    if service is not None:
        query = query.where(ApiRequest.service == service)
    if success is not None:
        query = query.where(ApiRequest.success.is_(success))
    query = query.order_by(ApiRequest.timestamp.desc(), ApiRequest.id.desc()).limit(limit)
    return list(session.execute(query).scalars())


def summarize_requests(
    session: Session,
    *,
    days: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Calcule les agrégats du monitoring sur les `days` derniers jours calendaires UTC.

    La période couvre aujourd'hui et les `days - 1` jours précédents : elle
    commence à `start_date 00:00:00 UTC` et s'arrête à la fin de la journée
    en cours. Ainsi, la somme de `requests_per_day` est toujours égale à
    `total_requests`. `now` ne sert qu'à rendre les tests déterministes ;
    par défaut, c'est l'heure UTC actuelle.

    Renvoie un dict avec :
        total_requests, error_count : compteurs de la période ;
        success_rate : part des requêtes réussies (entre 0 et 1), `None` si
            aucune requête ;
        last_request_at : timestamp UTC de la dernière requête, ou `None` ;
        services : une entrée par service de `MONITORED_SERVICES`, avec ses
            compteurs, son taux de succès et ses latences (`mean`, `median`,
            `max`) calculées sur les seules requêtes réussies ;
        errors_by_type : nombre d'erreurs par `error_type` ;
        requests_per_day : une entrée par jour de la période, dans l'ordre
            chronologique, avec le volume de chaque service (0 si aucun appel).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    start_date = today - timedelta(days=days - 1)
    period_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    period_end = datetime.combine(today + timedelta(days=1), time.min, tzinfo=timezone.utc)
    # Filtre commun à tous les agrégats : la période et les services suivis.
    in_period = (
        ApiRequest.timestamp >= period_start,
        ApiRequest.timestamp < period_end,
        ApiRequest.service.in_(MONITORED_SERVICES),
    )

    # Compteurs par service et par résultat (succès / échec).
    counts = {service: {"total": 0, "errors": 0} for service in MONITORED_SERVICES}
    count_query = (
        select(ApiRequest.service, ApiRequest.success, func.count())
        .where(*in_period)
        .group_by(ApiRequest.service, ApiRequest.success)
    )
    for service, success, count in session.execute(count_query):
        counts[service]["total"] += count
        if not success:
            counts[service]["errors"] += count

    # Durées des seules requêtes réussies, regroupées par service.
    durations: dict[str, list[int]] = {service: [] for service in MONITORED_SERVICES}
    duration_query = (
        select(ApiRequest.service, ApiRequest.duration_ms)
        .where(*in_period)
        .where(ApiRequest.success.is_(True))
    )
    for service, duration_ms in session.execute(duration_query):
        durations[service].append(duration_ms)

    services = [
        {
            "service": service,
            "total_requests": counts[service]["total"],
            "error_count": counts[service]["errors"],
            "success_rate": _success_rate(counts[service]["total"], counts[service]["errors"]),
            "latency_ms": _latency_stats(durations[service]),
        }
        for service in MONITORED_SERVICES
    ]

    total_requests = sum(entry["total_requests"] for entry in services)
    error_count = sum(entry["error_count"] for entry in services)

    last_request_at = session.scalar(select(func.max(ApiRequest.timestamp)).where(*in_period))

    # Erreurs par type. Une erreur sans `error_type` (réponse qui n'était pas
    # une `ErrorResponse`) n'est rangée dans aucune catégorie.
    errors_query = (
        select(ApiRequest.error_type, func.count())
        .where(*in_period)
        .where(ApiRequest.success.is_(False))
        .where(ApiRequest.error_type.is_not(None))
        .group_by(ApiRequest.error_type)
        .order_by(ApiRequest.error_type)
    )
    errors_by_type = {error_type: count for error_type, count in session.execute(errors_query)}

    # Volume quotidien par service. `date(timestamp)` renvoie la date UTC,
    # puisque les timestamps sont stockés en UTC.
    day = func.date(ApiRequest.timestamp)
    daily_query = (
        select(day, ApiRequest.service, func.count())
        .where(*in_period)
        .group_by(day, ApiRequest.service)
    )
    daily_counts = {
        (day_text, service): count for day_text, service, count in session.execute(daily_query)
    }
    requests_per_day = []
    for offset in range(days):
        current_date = start_date + timedelta(days=offset)
        entry: dict[str, Any] = {"date": current_date}
        for service in MONITORED_SERVICES:
            entry[service] = daily_counts.get((current_date.isoformat(), service), 0)
        requests_per_day.append(entry)

    return {
        "total_requests": total_requests,
        "error_count": error_count,
        "success_rate": _success_rate(total_requests, error_count),
        "last_request_at": last_request_at,
        "services": services,
        "errors_by_type": errors_by_type,
        "requests_per_day": requests_per_day,
    }


def _success_rate(total: int, errors: int) -> float | None:
    """Part des requêtes réussies, entre 0 et 1 ; `None` s'il n'y a aucune requête."""
    if total == 0:
        return None
    return (total - errors) / total


def _latency_stats(durations: list[int]) -> dict[str, float | int | None]:
    """Moyenne, médiane et maximum des durées en ms ; `None` partout si la liste est vide."""
    if not durations:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": float(statistics.mean(durations)),
        "median": float(statistics.median(durations)),
        "max": max(durations),
    }
