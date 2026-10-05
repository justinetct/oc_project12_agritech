"""Router HTTP du monitoring : `GET /monitoring/summary` et `GET /monitoring/requests`.

Router volontairement fin et en lecture seule : il vérifie le token Bearer,
ouvre une session sur la base de monitoring, délègue les SELECT au
repository `agritech.monitoring.repository` et sérialise la réponse. Aucun
calcul ni aucune écriture ne vivent ici.

`require_monitoring_token` est déclaré au niveau du router : FastAPI le
résout avant les paramètres des endpoints, donc avant l'ouverture de la
session. Un appel refusé (401 ou 503) n'ouvre jamais la base.

Ces GET ne sont pas archivés dans `api_requests` : le middleware de
persistance ne suit que `POST /predict` et `POST /recommend`. Le dashboard
ne mesure donc pas ses propres lectures.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from agritech.api.dependencies import get_monitoring_session, require_monitoring_token
from agritech.api.schemas.common import ErrorResponse
from agritech.api.schemas.monitoring import (
    MonitoringRequestItem,
    MonitoringRequestsResponse,
    MonitoringService,
    MonitoringSummaryResponse,
)
from agritech.monitoring.repository import list_recent_requests, summarize_requests


router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    dependencies=[Depends(require_monitoring_token)],
)


# Réponses d'erreur communes aux deux endpoints : token absent ou invalide
# (401), paramètre de requête invalide (422), monitoring non configuré ou base
# indisponible (503), exception inattendue (500).
_ERROR_RESPONSES: dict = {
    401: {"model": ErrorResponse, "description": "Token Bearer absent ou invalide"},
    422: {"model": ErrorResponse, "description": "Paramètre de requête invalide"},
    503: {"model": ErrorResponse, "description": "Monitoring indisponible"},
    500: {"model": ErrorResponse, "description": "Erreur interne"},
}


@router.get(
    "/summary",
    response_model=MonitoringSummaryResponse,
    responses=_ERROR_RESPONSES,
    summary="Aggregated API usage over the last days",
)
def get_monitoring_summary(
    session: Annotated[Session, Depends(get_monitoring_session)],
    days: Annotated[
        int,
        Query(
            ge=1,
            le=365,
            description=(
                "Nombre de jours calendaires UTC couverts, jour courant inclus "
                "(entre 1 et 365)."
            ),
        ),
    ] = 30,
) -> MonitoringSummaryResponse:
    """Retourne les agrégats des appels `/predict` et `/recommend` sur la période.

    Volumes, taux de succès, latences des appels réussis, erreurs par type et
    volume quotidien par service (un élément par jour, jours vides à 0).
    """
    return MonitoringSummaryResponse(**summarize_requests(session, days=days))


@router.get(
    "/requests",
    response_model=MonitoringRequestsResponse,
    responses=_ERROR_RESPONSES,
    summary="Most recent archived API calls",
)
def get_monitoring_requests(
    session: Annotated[Session, Depends(get_monitoring_session)],
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Nombre maximal d'appels renvoyés (entre 1 et 100)."),
    ] = 20,
    service: Annotated[
        MonitoringService | None,
        Query(description="Optionnel. Ne garde que les appels de ce service."),
    ] = None,
    success: Annotated[
        bool | None,
        Query(description="Optionnel. `true` : appels réussis ; `false` : appels en erreur."),
    ] = None,
) -> MonitoringRequestsResponse:
    """Retourne les derniers appels archivés, du plus récent au plus ancien."""
    rows = list_recent_requests(session, limit=limit, service=service, success=success)
    return MonitoringRequestsResponse(
        items=[MonitoringRequestItem.model_validate(row) for row in rows]
    )
