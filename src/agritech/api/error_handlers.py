"""Handlers d'erreur globaux de l'API : 422, 503, 500.

Ils traduisent en `ErrorResponse` unifiée :

- `RequestValidationError` (Pydantic)  → 422 `validation_error`
- `ModelUnavailableError`  (applicative) → 503 `model_unavailable`
- toute autre `Exception` non prévue    → 500 `internal_error`

Aucun handler n'expose de traceback, de chemin local ni le payload d'origine au
client. Les 500 sont loggés côté serveur via `logger.exception(...)` pour
diagnostic ; la couche observabilité détaillée (Logfire) est traitée dans la
tâche 19, pas ici.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agritech.api.exceptions import ModelUnavailableError
from agritech.api.schemas.common import ErrorResponse, ValidationErrorDetail


logger = logging.getLogger(__name__)


# Messages publics constants par catégorie d'erreur (voir Plan API — sous-étape 5).
_MESSAGES: dict[str, str] = {
    "validation_error": "Request payload is invalid.",
    "model_unavailable": "Model is unavailable.",
    "internal_error": "Internal server error.",
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


def _error_response(error_code: str, status_code: int, details: list[ValidationErrorDetail] | None = None) -> JSONResponse:
    """Construit une `JSONResponse` conforme à `ErrorResponse` pour un code donné."""
    body = ErrorResponse(error=error_code, message=_MESSAGES[error_code], details=details).model_dump()
    return JSONResponse(status_code=status_code, content=body)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handler 422 : reformate les erreurs Pydantic en `ErrorResponse`."""
    return _error_response("validation_error", 422, details=format_validation_errors(exc))


async def model_unavailable_handler(request: Request, exc: ModelUnavailableError) -> JSONResponse:
    """Handler 503 : le modèle attendu par l'endpoint n'est pas chargé."""
    return _error_response("model_unavailable", 503)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler 500 : catch-all des exceptions non prévues.

    L'exception complète est loggée côté serveur (`logger.exception`) pour
    diagnostic ; aucune information technique n'est exposée dans la réponse.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return _error_response("internal_error", 500)
