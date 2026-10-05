"""Handlers d'erreur globaux de l'API : 401, 422, 503, 500.

Ils traduisent en `ErrorResponse` unifiée :

- `RequestValidationError` (Pydantic)          → 422 `validation_error`
- `ModelUnavailableError`  (applicative)       → 503 `model_unavailable`
- `MonitoringUnauthorizedError` (monitoring)   → 401 `unauthorized`
- `MonitoringUnavailableError`  (monitoring)   → 503 `monitoring_unavailable`
- toute autre `Exception` non prévue          → 500 `internal_error`

Aucun handler n'expose de traceback, de chemin local ni le payload d'origine au
client. Les 500 sont loggés côté serveur via `logger.exception(...)` pour
diagnostic ; la couche observabilité détaillée (Logfire) est traitée dans la
tâche 19, pas ici.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agritech.api.exceptions import (
    ModelUnavailableError,
    MonitoringUnauthorizedError,
    MonitoringUnavailableError,
)
from agritech.api.schemas.common import ErrorResponse, ValidationErrorDetail


logger = logging.getLogger(__name__)


# Messages publics constants par catégorie d'erreur (voir Plan API — sous-étape 5).
_MESSAGES: dict[str, str] = {
    "validation_error": "Request payload is invalid.",
    "model_unavailable": "Model is unavailable.",
    "internal_error": "Internal server error.",
    "unauthorized": "Authentication required.",
    "monitoring_unavailable": "Monitoring is unavailable.",
}


def format_validation_errors(exc: RequestValidationError) -> list[ValidationErrorDetail]:
    """Reformate les erreurs Pydantic en `ValidationErrorDetail` publiques.

    Ne recopie ni `input` (le payload) ni `ctx` (le contexte interne) : seuls
    `loc` (transformé en chemin `body.<champ>`), `type` et `msg` (renommé
    `message`) sont exposés.
    """
    details: list[ValidationErrorDetail] = []
    for error in exc.errors():
        loc_parts = [str(part) for part in error.get("loc", ())]
        details.append(
            ValidationErrorDetail(
                field=".".join(loc_parts),
                type=str(error.get("type", "")),
                message=str(error.get("msg", "")),
            )
        )
    return details


def validation_summary(details: Any) -> str | None:
    """Résumé court d'une 422 pour le monitoring : « champ: message Pydantic » par détail.

    Lit les `details` publics de `ErrorResponse` (rien d'autre : ni payload, ni
    contexte interne). Plusieurs champs sont séparés par « · ». `None` si aucun
    détail exploitable.
    """
    if not isinstance(details, list):
        return None
    parts = [
        f"{str(detail['field']).rsplit('.', 1)[-1]}: {detail['message']}"
        for detail in details
        if isinstance(detail, dict) and detail.get("field") and detail.get("message")
    ]
    return " · ".join(parts) or None


def _error_body(error_code: str, details: list[ValidationErrorDetail] | None = None) -> dict[str, Any]:
    """Corps JSON d'une `ErrorResponse` pour un code donné."""
    return ErrorResponse(error=error_code, message=_MESSAGES[error_code], details=details).model_dump()


def validation_error_body(exc: RequestValidationError) -> dict[str, Any]:
    """Corps d'une réponse 422, sans requête HTTP.

    C'est le contenu renvoyé par `validation_exception_handler` ; l'historique
    de démonstration du monitoring s'en sert pour archiver des erreurs
    identiques à celles de l'API.
    """
    return _error_body("validation_error", format_validation_errors(exc))


def _error_response(
    error_code: str,
    status_code: int,
    details: list[ValidationErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Construit une `JSONResponse` conforme à `ErrorResponse` pour un code donné."""
    return JSONResponse(status_code=status_code, content=_error_body(error_code, details), headers=headers)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handler 422 : reformate les erreurs Pydantic en `ErrorResponse`."""
    return _error_response("validation_error", 422, details=format_validation_errors(exc))


async def model_unavailable_handler(request: Request, exc: ModelUnavailableError) -> JSONResponse:
    """Handler 503 : le modèle attendu par l'endpoint n'est pas chargé."""
    return _error_response("model_unavailable", 503)


async def monitoring_unauthorized_handler(
    request: Request, exc: MonitoringUnauthorizedError
) -> JSONResponse:
    """Handler 401 : appel `/monitoring/*` sans token Bearer valide.

    L'en-tête `WWW-Authenticate: Bearer` indique au client le schéma attendu.
    """
    return _error_response("unauthorized", 401, headers={"WWW-Authenticate": "Bearer"})


async def monitoring_unavailable_handler(
    request: Request, exc: MonitoringUnavailableError
) -> JSONResponse:
    """Handler 503 : monitoring non configuré ou base de monitoring indisponible.

    La réponse reste générique : elle ne dit pas si c'est le token ou la base
    qui manque.
    """
    return _error_response("monitoring_unavailable", 503)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler 500 : catch-all des exceptions non prévues.

    L'exception complète est loggée côté serveur (`logger.exception`) pour
    diagnostic ; aucune information technique n'est exposée dans la réponse.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return _error_response("internal_error", 500)
