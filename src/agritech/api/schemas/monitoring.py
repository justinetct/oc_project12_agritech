"""Schémas Pydantic des endpoints de monitoring : résumé et requêtes récentes.

Ces modèles décrivent uniquement des réponses en lecture seule :

- `MonitoringSummaryResponse` reprend exactement le dict produit par
  `agritech.monitoring.repository.summarize_requests` ; aucun calcul n'est
  refait ici ;
- `MonitoringRequestsResponse` expose une version volontairement réduite des
  lignes `ApiRequest` : ni réponse archivée, ni identifiant de trace, ni
  métadonnées techniques de déploiement.

Les bornes des paramètres de requête (`days`, `limit`) appartiennent au
router, pas à ces modèles.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


# Services métier suivis par le monitoring. Doit rester aligné avec
# `agritech.monitoring.repository.MONITORED_SERVICES` (vérifié par un test).
MonitoringService = Literal["predict", "recommend"]


class LatencySummary(BaseModel):
    """Latences d'un service, en millisecondes, calculées sur les requêtes réussies.

    Les trois valeurs valent `null` quand le service n'a aucune requête réussie
    sur la période.
    """

    mean: float | None = Field(description="Durée moyenne.", examples=[13.3])
    median: float | None = Field(description="Durée médiane.", examples=[11.0])
    max: int | None = Field(description="Durée maximale.", examples=[48])


class ServiceSummary(BaseModel):
    """Agrégats d'un service sur la période demandée."""

    service: MonitoringService = Field(description="Service métier concerné.", examples=["predict"])
    total_requests: int = Field(description="Nombre d'appels archivés.", examples=[29])
    error_count: int = Field(description="Nombre d'appels en erreur (statut ≥ 400).", examples=[1])
    success_rate: float | None = Field(
        description="Part des appels réussis, entre 0 et 1 ; `null` sans aucun appel.",
        examples=[0.966],
    )
    latency_ms: LatencySummary = Field(description="Latences des appels réussis.")


class DailyVolume(BaseModel):
    """Volume d'appels d'une journée UTC, par service, et nombre d'erreurs."""

    date: dt.date = Field(description="Jour calendaire UTC.", examples=["2026-10-02"])
    predict: int = Field(description="Nombre d'appels `/predict` ce jour-là.", examples=[3])
    recommend: int = Field(description="Nombre d'appels `/recommend` ce jour-là.", examples=[5])
    errors: int = Field(
        description=(
            "Nombre d'appels en erreur ce jour-là, tous services confondus "
            "(déjà comptés dans `predict` et `recommend`)."
        ),
        examples=[1],
    )


class MonitoringSummaryResponse(BaseModel):
    """Réponse de `GET /monitoring/summary` : agrégats sur les derniers jours."""

    total_requests: int = Field(description="Nombre total d'appels sur la période.", examples=[82])
    error_count: int = Field(description="Nombre total d'appels en erreur.", examples=[2])
    success_rate: float | None = Field(
        description="Part des appels réussis, entre 0 et 1 ; `null` sans aucun appel.",
        examples=[0.976],
    )
    last_request_at: dt.datetime | None = Field(
        description="Date et heure UTC du dernier appel de la période ; `null` sans aucun appel.",
        examples=["2026-10-01T13:45:06.279869Z"],
    )
    services: list[ServiceSummary] = Field(
        description="Une entrée par service, toujours `predict` puis `recommend`.",
    )
    errors_by_type: dict[str, int] = Field(
        description="Nombre d'erreurs par code d'erreur (`validation_error`, ...).",
        examples=[{"validation_error": 2}],
    )
    requests_per_day: list[DailyVolume] = Field(
        description="Un élément par jour de la période, dans l'ordre chronologique.",
    )


class MonitoringRequestItem(BaseModel):
    """Vue publique d'une ligne `api_requests`.

    `from_attributes=True` permet de construire ce modèle directement depuis un
    objet ORM `ApiRequest` (`MonitoringRequestItem.model_validate(row)`). Seuls
    les champs déclarés ici sont exposés : `response_payload`,
    `logfire_trace_id`, `environment`, `api_version`, `endpoint` et `method`
    restent internes.

    `request_payload` est le corps JSON reçu, tel qu'archivé : un objet pour
    les appels `/predict` et `/recommend`, mais n'importe quelle valeur JSON
    si le client a envoyé un corps invalide (rejeté en 422).
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Identifiant de la ligne archivée.", examples=[82])
    timestamp: dt.datetime = Field(
        description="Date et heure UTC de l'appel.", examples=["2026-10-01T13:45:06.279869Z"]
    )
    service: MonitoringService = Field(description="Service métier appelé.", examples=["recommend"])
    status_code: int = Field(description="Statut HTTP renvoyé.", examples=[200])
    success: bool = Field(description="`true` si le statut est inférieur à 400.", examples=[True])
    duration_ms: int = Field(description="Durée de traitement, en millisecondes.", examples=[14])
    model_version: str | None = Field(
        description="Version du modèle déployé au moment de l'appel.", examples=["2.0.0"]
    )
    request_payload: JsonValue = Field(
        description="Corps JSON de la requête reçue.", examples=[{"iso3": "FRA"}]
    )
    error_type: str | None = Field(
        description="Code d'erreur pour un appel en échec ; `null` sinon.", examples=[None]
    )
    error_message: str | None = Field(
        description="Message d'erreur public pour un appel en échec ; `null` sinon.",
        examples=[None],
    )


class MonitoringRequestsResponse(BaseModel):
    """Réponse de `GET /monitoring/requests` : appels les plus récents d'abord."""

    items: list[MonitoringRequestItem] = Field(
        description="Appels archivés, du plus récent au plus ancien.",
    )
