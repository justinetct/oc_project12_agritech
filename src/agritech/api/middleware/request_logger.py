"""Middleware ASGI qui persiste chaque appel `/predict` et `/recommend`.

Le middleware voit toutes les réponses de ces endpoints — y compris les 422
générées par la validation Pydantic et les 500/503 émises par les handlers
d'erreur — parce qu'il enveloppe l'application ASGI et intercepte les
messages `http.response.start` / `http.response.body` bruts. Il ne modifie
ni les handlers existants, ni le contenu des réponses.

Choix d'un ASGI middleware pur plutôt que `BaseHTTPMiddleware` : ce dernier
souffre du bug documenté Starlette #2049. Une exception qui traverse
`call_next` fait échapper la Response 500 déjà émise par
`unhandled_exception_handler` : le middleware ne peut plus la persister.
L'ASGI pur voit le vrai flux de messages, capture la 500 formatée, puis
re-lève l'exception pour laisser `ServerErrorMiddleware` faire son travail
en amont (qui verra que la réponse est déjà partie).

Politique de non-blocage : l'écriture SQLite est enveloppée dans un
`try/except` unique dans `_persist_safely`. Toute exception de la couche
persistance est loggée en `warning` et la réponse HTTP originale est
renvoyée telle quelle. Cette règle vit ici, pas dans le repository.

Endpoints persistés (POST) : `/predict`, `/recommend`.
Endpoints jamais persistés : `/health`, `/predict/schema`,
`/recommend/schema`, `/docs`, `/openapi.json`, `/redoc`, ainsi que toute
autre route non listée dans `_PERSISTED_ROUTES`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agritech.api.core import runtime
from agritech.monitoring.repository import insert_api_request


logger = logging.getLogger(__name__)


# Mapping path → nom de service persisté dans la colonne `service`.
_PERSISTED_ROUTES: dict[str, str] = {
    "/predict": "predict",
    "/recommend": "recommend",
}

# Longueur maximum de `error_message` en base : évite qu'une traceback verbeuse
# fasse gonfler une ligne. Valeur retenue par la tâche 19.
_ERROR_MESSAGE_LIMIT = 2000


class RequestLoggerMiddleware:
    """Middleware ASGI qui persiste une ligne `api_requests` par appel POST métier.

    La classe ne prend pas d'argument au-delà de l'application ASGI enveloppée :
    `api_version` et `environment` sont lus depuis `runtime` à chaque requête,
    peuplés par le lifespan FastAPI.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        method = scope["method"]
        service = _PERSISTED_ROUTES.get(path)
        if service is None or method != "POST":
            await self.app(scope, receive, send)
            return

        # Consomme le body de la requête et le rejoue tel quel à l'application.
        request_body = await _read_body(receive)

        async def replay_receive() -> Message:
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }

        # Intercepte le flux de réponse en préservant l'ordre et le contenu :
        # `http.response.start` porte le status, `http.response.body` porte
        # les bytes. Une réponse peut être envoyée en plusieurs chunks.
        response_status: int | None = None
        response_body = b""

        async def capture_send(message: Message) -> None:
            nonlocal response_status, response_body
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)

        start = time.perf_counter()
        exception: BaseException | None = None
        try:
            await self.app(scope, replay_receive, capture_send)
        except Exception as exc:  # noqa: BLE001 — capturé pour persistance
            exception = exc
        # Arrondi vers le haut : garantit `duration_ms >= 1` pour tout endpoint
        # ayant effectivement tourné.
        elapsed_ms = max(1, math.ceil((time.perf_counter() - start) * 1000))

        # Cas exception non gérée par `ExceptionMiddleware` : le handler pour
        # `Exception` est routé par Starlette vers `ServerErrorMiddleware`
        # (plus externe que nous). Aucun message n'a donc été envoyé côté
        # `capture_send`. On reflète ce que ServerErrorMiddleware enverra au
        # client via `unhandled_exception_handler` : status 500 et l'enveloppe
        # `ErrorResponse` unifiée, sans détails.
        if exception is not None and response_status is None:
            response_status = 500
            response_body = (
                b'{"error":"internal_error",'
                b'"message":"Internal server error.",'
                b'"details":null}'
            )

        if response_status is not None:
            _persist_safely(
                service=service,
                method=method,
                path=path,
                raw_request_body=request_body,
                response_body=response_body,
                status_code=response_status,
                duration_ms=elapsed_ms,
            )

        # Si l'application a levé, on re-lève : `ServerErrorMiddleware`
        # produit la 500 formatée au client, avec exactement le même contenu
        # que celui persisté ci-dessus.
        if exception is not None:
            raise exception


async def _read_body(receive: Receive) -> bytes:
    """Lit le body complet de la requête, éventuellement multi-chunks."""
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)
    return body


def _persist_safely(
    *,
    service: str,
    method: str,
    path: str,
    raw_request_body: bytes,
    response_body: bytes,
    status_code: int,
    duration_ms: int,
) -> None:
    """Compose la ligne à persister et l'insère, en absorbant toute erreur.

    Un `try/except` global attrape tout ce qui pourrait aller de travers
    (session factory absente, panne SQLite, JSON incompréhensible, valeurs
    inattendues) et se contente de logger un `warning`.
    """
    try:
        factory = runtime.monitoring_session_factory
        if factory is None:
            return

        request_payload = _parse_json_or_empty(raw_request_body)
        response_payload = _parse_json_or_none(response_body)
        success = status_code < 400
        error_type, error_message = _classify_error(success, response_payload)
        model_version = _model_version_for(service)

        values = {
            "timestamp": datetime.now(timezone.utc),
            "service": service,
            "endpoint": path,
            "method": method,
            "status_code": status_code,
            "success": success,
            "duration_ms": duration_ms,
            "api_version": runtime.monitoring_api_version or "unknown",
            "model_version": model_version,
            "request_payload": request_payload,
            "response_payload": response_payload,
            "error_type": error_type,
            "error_message": error_message,
            "logfire_trace_id": None,
            "environment": runtime.monitoring_environment or "unknown",
        }

        with factory() as session:
            insert_api_request(session, values)
    except Exception:  # noqa: BLE001 — non-blocage explicite
        logger.warning(
            "monitoring persistence failed for %s %s (status=%s); response returned as-is",
            method,
            path,
            status_code,
            exc_info=True,
        )


def _parse_json_or_empty(raw: bytes) -> dict | list:
    """Retourne le JSON décodé ou `{}` si le body est vide / non-JSON.

    La colonne `request_payload` est `NOT NULL` ; on préfère un dict vide à
    une pirouette de sérialisation. Un body binaire ou invalide reste rare
    (Pydantic renvoie déjà 422 avant même d'atteindre le middleware).
    """
    parsed = _parse_json_or_none(raw)
    return parsed if parsed is not None else {}


def _parse_json_or_none(raw: bytes) -> dict | list | None:
    """Retourne le JSON décodé, ou `None` si non-JSON / vide / non structurable."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _classify_error(
    success: bool, response_payload: Any
) -> tuple[str | None, str | None]:
    """Extrait `error_type` et `error_message` depuis l'`ErrorResponse` unifiée.

    Réutilise le contrat déjà exposé par les handlers 422 / 503 / 500 :
    `{"error": "<code>", "message": "<phrase>", "details": ...}`. Pour une
    réponse à succès, les deux champs restent `None`.
    """
    if success:
        return None, None
    if not isinstance(response_payload, dict):
        return None, None

    error_type = response_payload.get("error")
    error_message = response_payload.get("message")
    if isinstance(error_message, str) and len(error_message) > _ERROR_MESSAGE_LIMIT:
        error_message = error_message[:_ERROR_MESSAGE_LIMIT]
    return (
        error_type if isinstance(error_type, str) else None,
        error_message if isinstance(error_message, str) else None,
    )


def _model_version_for(service: str) -> str | None:
    """Renvoie `model_version` du bundle correspondant au service, ou `None`.

    Même pour une 422 (le modèle n'a pas tourné), on archive la version
    déployée au moment de la requête : c'est la version que l'appelant
    aurait obtenue si son payload avait été valide.
    """
    bundle = (
        runtime.bundle_predict if service == "predict" else runtime.bundle_recommend
    )
    if bundle is None:
        return None
    return bundle.metadata.get("model_version")
