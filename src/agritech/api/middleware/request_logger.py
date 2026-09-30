"""Middleware ASGI qui trace chaque appel `/health` et persiste `/predict` / `/recommend`.

Le middleware traite deux catégories d'endpoints :

- **Tracés + persistés** : `POST /predict`, `POST /recommend`. Un span
  Logfire nommé `POST /predict` (ou `POST /recommend`) porte la requête et
  la réponse en attributs structurés. Une ligne est écrite dans la table
  `api_requests` de SQLite, avec le `trace_id` du span pour corrélation.
- **Tracé seulement** : `GET /health`. Un span Logfire nommé `GET /health`
  porte le statut et la réponse — utile pour surveiller l'état de l'API
  côté Logfire — mais aucune ligne n'est écrite en SQLite. La table
  `api_requests` reste réservée aux appels métier.

Toutes les autres routes (`/predict/context`, `/recommend/context`, `/docs`,
`/openapi.json`, `/redoc`, etc.) passent en direct : ni span, ni ligne.

Choix d'un ASGI middleware pur plutôt que `BaseHTTPMiddleware` : ce dernier
souffre du bug documenté Starlette #2049. Une exception qui traverse
`call_next` fait échapper la Response 500 déjà émise par
`unhandled_exception_handler` : le middleware ne peut plus la persister.
L'ASGI pur voit le vrai flux de messages, capture la 500 formatée, puis
re-lève l'exception pour laisser `ServerErrorMiddleware` faire son travail
en amont (qui verra que la réponse est déjà partie).

Corrélation Logfire : le middleware ouvre lui-même un `logfire.span`. Sans
token Logfire, ce span est un no-op silencieux et `_current_trace_id()`
renvoie `None`. Avec token, le span alimente le contexte OTel — le
`trace_id` capturé dans `capture_send` est celui qu'on retrouve côté
Logfire. On préfère ce span "maison" à `logfire.instrument_fastapi(app)`
parce que ce dernier fait `app.add_middleware(...)` après le démarrage, ce
que Starlette refuse.

Politique de non-blocage : l'écriture SQLite est enveloppée dans un
`try/except` unique dans `_persist_safely`. Toute exception de la couche
persistance est loggée en `warning` et la réponse HTTP originale est
renvoyée telle quelle. Cette règle vit ici, pas dans le repository.

Sécurité : le middleware n'ajoute jamais d'attribut portant un en-tête
(Authorization, Cookie), une IP, un identifiant de client ou un token. Les
attributs de span reprennent uniquement le corps métier de la requête et
la réponse, sous une forme structurée lisible dans Logfire.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import logfire
from opentelemetry import trace
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agritech.api.core import runtime
from agritech.monitoring.repository import insert_api_request


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RouteConfig:
    """Configuration d'observabilité d'une route (method, path).

    Attributs :
        service : nom pour la colonne `api_requests.service` ; `None` pour
            un endpoint tracé sans persistance.
        persist : si True, écrit une ligne `api_requests` par appel.
    """

    service: str | None
    persist: bool


# Table de routage : chaque (méthode, path) tracé par le middleware y figure.
# Une entrée absente = ni span Logfire, ni ligne SQLite (endpoint passe en direct).
_ROUTES: dict[tuple[str, str], _RouteConfig] = {
    ("POST", "/predict"): _RouteConfig(service="predict", persist=True),
    ("POST", "/recommend"): _RouteConfig(service="recommend", persist=True),
    ("GET", "/health"): _RouteConfig(service=None, persist=False),
}

# Longueur maximum de `error_message` en base : évite qu'une traceback verbeuse
# fasse gonfler une ligne. Valeur retenue par la tâche 19.
_ERROR_MESSAGE_LIMIT = 2000

# Enveloppe `ErrorResponse` utilisée quand une exception traverse le middleware
# sans qu'aucune réponse n'ait été émise (voir bloc `exception is not None and
# response_status is None` plus bas). Reflète strictement ce que
# `ServerErrorMiddleware` va envoyer au client via `unhandled_exception_handler`.
_INTERNAL_ERROR_BODY = (
    b'{"error":"internal_error",'
    b'"message":"Internal server error.",'
    b'"details":null}'
)


class RequestLoggerMiddleware:
    """Middleware ASGI qui trace les appels observés et persiste les appels métier.

    Sans argument au-delà de l'application ASGI enveloppée : `api_version`
    et `environment` sont lus depuis `runtime` à chaque requête, peuplés
    par le lifespan FastAPI.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        route = _ROUTES.get((method, path))
        if route is None:
            await self.app(scope, receive, send)
            return

        # Body de requête lu uniquement quand on va persister — c'est-à-dire
        # pour les POST métier. Pour `/health` (GET), pas de body à lire.
        request_body = b""
        actual_receive: Receive = receive
        if route.persist:
            request_body = await _read_body(receive)

            async def replay_receive() -> Message:
                return {
                    "type": "http.request",
                    "body": request_body,
                    "more_body": False,
                }

            actual_receive = replay_receive

        # Intercepte le flux de réponse en préservant l'ordre et le contenu.
        # `http.response.start` porte le status, `http.response.body` porte
        # les bytes (potentiellement en plusieurs chunks). On capture aussi
        # le trace_id OTel à ce moment-là : on est encore dans le `logfire.span`
        # ouvert plus bas, donc si Logfire est configuré, le trace_id est valide.
        response_status: int | None = None
        response_body = b""
        trace_id: str | None = None

        async def capture_send(message: Message) -> None:
            nonlocal response_status, response_body, trace_id
            if message["type"] == "http.response.start":
                response_status = message["status"]
                trace_id = _current_trace_id()
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)

        span_name = f"{method} {path}"
        exception: BaseException | None = None
        start = time.perf_counter()

        with logfire.span(span_name) as span:
            try:
                await self.app(scope, actual_receive, capture_send)
            except Exception as exc:  # noqa: BLE001 — capturé pour persistance
                exception = exc

            # Arrondi vers le haut : garantit `duration_ms >= 1` pour tout
            # endpoint ayant effectivement tourné.
            elapsed_ms = max(1, math.ceil((time.perf_counter() - start) * 1000))

            # Cas exception non gérée par `ExceptionMiddleware` : le handler
            # pour `Exception` est routé par Starlette vers
            # `ServerErrorMiddleware` (plus externe que nous). Aucun message
            # n'a donc été envoyé côté `capture_send`. On reflète ce que
            # ServerErrorMiddleware enverra au client.
            if exception is not None and response_status is None:
                response_status = 500
                response_body = _INTERNAL_ERROR_BODY
                if trace_id is None:
                    trace_id = _current_trace_id()

            # Enrichit le span avec les attributs métier lisibles côté Logfire.
            _annotate_span(
                span=span,
                persist=route.persist,
                request_body=request_body,
                response_body=response_body,
                response_status=response_status,
            )

            # Persistance SQLite + événement `api_request_persisted` — dans
            # le contexte du span pour que l'event lui soit rattaché.
            if route.persist and route.service and response_status is not None:
                _persist_safely(
                    service=route.service,
                    method=method,
                    path=path,
                    raw_request_body=request_body,
                    response_body=response_body,
                    status_code=response_status,
                    duration_ms=elapsed_ms,
                    trace_id=trace_id,
                )

        # Si l'application a levé, on re-lève : `ServerErrorMiddleware` produit
        # la 500 formatée au client, avec exactement le même contenu que celui
        # persisté ci-dessus.
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


def _annotate_span(
    *,
    span: Any,
    persist: bool,
    request_body: bytes,
    response_body: bytes,
    response_status: int | None,
) -> None:
    """Ajoute les attributs métier au span Logfire et son niveau.

    Attributs ajoutés :
    - `request.body` (uniquement si `persist=True` et body non vide) : dict
      métier envoyé par le client, sérialisé en JSON string.
    - `response.status_code` : entier du statut HTTP.
    - `response.body` : dict métier renvoyé par l'API, sérialisé en JSON string.
    - `error_type` et `error_message` : présents uniquement si la réponse est
      une `ErrorResponse` unifiée (contient les clés `error` / `message`).
      `error_message` est tronqué à `_ERROR_MESSAGE_LIMIT` caractères, comme
      la colonne SQLite correspondante.

    Niveau du span :
    - 5xx → `error` (panne serveur).
    - 4xx → `warning` (requête client invalide, l'API fonctionne).
    - 2xx / 3xx → `info` (parcours nominal).
    Cf. `_level_for_status`. On ne touche pas au statut OTel : une 422 ne
    doit pas apparaître comme une exception dans Logfire.

    Rien d'autre : pas d'en-tête, pas d'IP, pas de token. Une exception dans
    cette fonction ne doit jamais casser la requête — le try/except global
    la protège.
    """
    try:
        if persist and request_body:
            request_json = _parse_json_or_none(request_body)
            if request_json is not None:
                span.set_attribute("request.body", json.dumps(request_json))

        if response_status is not None:
            span.set_attribute("response.status_code", response_status)
            span.set_level(_level_for_status(response_status))

        if response_body:
            response_json = _parse_json_or_none(response_body)
            if response_json is not None:
                span.set_attribute("response.body", json.dumps(response_json))
                if isinstance(response_json, dict):
                    error_type = response_json.get("error")
                    if isinstance(error_type, str):
                        span.set_attribute("error_type", error_type)
                    error_message = response_json.get("message")
                    if isinstance(error_message, str):
                        if len(error_message) > _ERROR_MESSAGE_LIMIT:
                            error_message = error_message[:_ERROR_MESSAGE_LIMIT]
                        span.set_attribute("error_message", error_message)
    except Exception:  # noqa: BLE001 — observabilité non bloquante
        logger.warning("logfire span annotation failed", exc_info=True)


def _level_for_status(status_code: int) -> str:
    """Retourne le niveau Logfire correspondant au status HTTP.

    - 5xx → `error` : panne serveur, mérite un tag distinctif.
    - 4xx → `warning` : la requête client est invalide, mais l'API fonctionne.
    - autres (2xx, 3xx) → `info` : parcours nominal.

    On ne change PAS le statut OTel ERROR pour un 4xx : une 422 est une
    requête invalide, pas une exception serveur — Logfire ne doit pas la
    présenter comme une panne.
    """
    if status_code >= 500:
        return "error"
    if status_code >= 400:
        return "warning"
    return "info"


def _persist_safely(
    *,
    service: str,
    method: str,
    path: str,
    raw_request_body: bytes,
    response_body: bytes,
    status_code: int,
    duration_ms: int,
    trace_id: str | None,
) -> None:
    """Compose la ligne à persister, l'insère, puis émet l'événement Logfire.

    Un `try/except` global attrape tout ce qui pourrait aller de travers
    (session factory absente, panne SQLite, JSON incompréhensible, valeurs
    inattendues) et se contente de logger un `warning`. L'événement Logfire
    n'est émis qu'après une insertion réussie : c'est ce qui permet à
    l'événement de porter le vrai `api_request_id` renvoyé par la base.
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
        api_version = runtime.monitoring_api_version or "unknown"
        environment = runtime.monitoring_environment or "unknown"

        values = {
            "timestamp": datetime.now(timezone.utc),
            "service": service,
            "endpoint": path,
            "method": method,
            "status_code": status_code,
            "success": success,
            "duration_ms": duration_ms,
            "api_version": api_version,
            "model_version": model_version,
            "request_payload": request_payload,
            "response_payload": response_payload,
            "error_type": error_type,
            "error_message": error_message,
            "logfire_trace_id": trace_id,
            "environment": environment,
        }

        with factory() as session:
            row_id = insert_api_request(session, values)

        _emit_persisted_event(
            api_request_id=row_id,
            service=service,
            endpoint=path,
            status_code=status_code,
            success=success,
            duration_ms=duration_ms,
            api_version=api_version,
            model_version=model_version,
            environment=environment,
        )
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


def _current_trace_id() -> str | None:
    """Retourne le trace_id OpenTelemetry actif en hex (32 caractères), ou `None`.

    Sans instrumentation OTel (Logfire non configuré), aucun span n'est
    actif : la fonction renvoie `None` et la colonne `logfire_trace_id`
    reste `NULL` en base. Ce contrat rend la corrélation Logfire ↔ SQLite
    triviale : si `trace_id` est renseigné, la même valeur est indexée dans
    Logfire.
    """
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


def _emit_persisted_event(**attributes: Any) -> None:
    """Émet l'événement Logfire `api_request_persisted` sans jamais lever.

    Les attributs se limitent aux métadonnées utiles à l'observabilité :
    pas de payload de requête, pas de payload de réponse, pas de
    `error_message`, pas d'en-têtes, pas d'IP. SQLite reste l'archive
    détaillée ; Logfire sert à observer et corréler.

    Sans configuration Logfire, `logfire.info` est un no-op silencieux —
    aucun risque de fuite ni d'envoi réseau.
    """
    try:
        logfire.info("api_request_persisted", **attributes)
    except Exception:  # noqa: BLE001 — observabilité non bloquante
        logger.warning("logfire event emission failed", exc_info=True)


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
