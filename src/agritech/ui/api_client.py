"""Client HTTP minimal pour appeler l'API Agritech Answers.

Fonctions publiques, qui retournent chacune le dict JSON décodé :

- ``get_predict_context()`` : ``GET /predict/context`` ;
- ``post_predict(payload)`` : ``POST /predict`` ;
- ``get_recommend_context(iso3=None)`` : ``GET /recommend/context``,
  avec ``?iso3=`` pour obtenir les valeurs par défaut d'un pays ;
- ``post_recommend(payload)`` : ``POST /recommend``.

Deux fonctions lisent le monitoring, protégé par un token Bearer, et
retournent la réponse validée par les schémas Pydantic de l'API :

- ``get_monitoring_summary(days)`` : ``GET /monitoring/summary`` ;
- ``get_monitoring_requests(limit=..., service=..., success=...)`` :
  ``GET /monitoring/requests``.

Les fonctions s'appuient sur ``settings.api_base_url()``,
``settings.request_timeout_seconds()`` et
``settings.monitoring_api_token()`` ; le module ``errors`` définit les
exceptions remontées au front.

Aucune logique d'interface ni logique ML n'est faite ici : le module
se contente de sérialiser/désérialiser du JSON et de traduire les
erreurs ``requests`` en exceptions dédiées. La primitive interne
``_request`` est commune à tous les appels.
"""

from __future__ import annotations

from typing import Any, TypeVar
from urllib.parse import quote

import requests
from pydantic import BaseModel, ValidationError

from agritech.api.schemas.monitoring import (
    MonitoringRequestsResponse,
    MonitoringService,
    MonitoringSummaryResponse,
)
from agritech.ui import settings
from agritech.ui.errors import (
    ApiConfigurationError,
    ApiConnectionError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
)


# Type du schéma Pydantic validé par `_validate_monitoring` (résumé ou requêtes).
_ResponseModel = TypeVar("_ResponseModel", bound=BaseModel)


def get_predict_context() -> dict[str, Any]:
    """Récupère les bornes physiques et le domaine d'entraînement de /predict."""
    return _request("GET", "/predict/context")


def post_predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Envoie un payload à ``POST /predict`` et retourne la réponse décodée."""
    return _request("POST", "/predict", json=payload)


def get_recommend_context(iso3: str | None = None) -> dict[str, Any]:
    """Contexte de /recommend ; avec ``iso3``, ajoute les valeurs par défaut du pays."""
    path = "/recommend/context" + (f"?iso3={quote(iso3)}" if iso3 else "")
    return _request("GET", path)


def post_recommend(payload: dict[str, Any]) -> dict[str, Any]:
    """Envoie un payload à ``POST /recommend`` et retourne la réponse décodée."""
    return _request("POST", "/recommend", json=payload)


def get_monitoring_summary(days: int) -> MonitoringSummaryResponse:
    """Agrégats du monitoring sur les ``days`` derniers jours (``GET /monitoring/summary``).

    Lève ``ApiConfigurationError`` sans appel HTTP si le token est absent,
    et ``ApiInvalidResponseError`` si la réponse ne respecte pas le contrat.
    Les bornes de ``days`` sont vérifiées par l'API (422 sinon).
    """
    data = _request(
        "GET",
        "/monitoring/summary",
        params={"days": days},
        headers=_monitoring_headers(),
    )
    return _validate_monitoring(MonitoringSummaryResponse, data)


def get_monitoring_requests(
    *,
    limit: int,
    service: MonitoringService | None = None,
    success: bool | None = None,
) -> MonitoringRequestsResponse:
    """Derniers appels archivés, du plus récent au plus ancien (``GET /monitoring/requests``).

    ``limit`` est toujours envoyé ; ``service`` et ``success`` ne sont
    envoyés que s'ils sont renseignés (``success=False`` est une vraie
    valeur : il filtre les appels en erreur).
    """
    params: dict[str, Any] = {"limit": limit}
    if service is not None:
        params["service"] = service
    if success is not None:
        params["success"] = "true" if success else "false"
    data = _request(
        "GET",
        "/monitoring/requests",
        params=params,
        headers=_monitoring_headers(),
    )
    return _validate_monitoring(MonitoringRequestsResponse, data)


def _monitoring_headers() -> dict[str, str]:
    """En-tête ``Authorization: Bearer <token>`` des endpoints ``/monitoring/*``.

    Lève ``ApiConfigurationError`` si le token n'est pas configuré : on ne
    fait pas un appel voué au refus. Le token n'apparaît dans aucun message.
    """
    token = settings.monitoring_api_token()
    if token is None:
        raise ApiConfigurationError("token de monitoring non défini (MONITORING_API_TOKEN)")
    return {"Authorization": f"Bearer {token}"}


def _validate_monitoring(model: type[_ResponseModel], data: dict[str, Any]) -> _ResponseModel:
    """Valide ``data`` avec le schéma Pydantic ``model`` de l'API.

    Une réponse JSON valide mais hors contrat devient une
    ``ApiInvalidResponseError`` : la ``ValidationError`` Pydantic ne remonte
    jamais telle quelle jusqu'à l'interface.
    """
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ApiInvalidResponseError(
            "réponse non conforme au contrat de monitoring"
        ) from exc


def _request(
    method: str,
    path: str,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Effectue une requête HTTP simple et retourne le JSON décodé.

    ``params`` devient la query string et ``headers`` des en-têtes HTTP
    supplémentaires ; tous deux sont facultatifs.

    Traduit chaque type de défaillance en une exception ``ApiError``
    dédiée. Aucune retry, aucun cache : la simplicité prime.
    """
    url = settings.api_base_url().rstrip("/") + path
    timeout = settings.request_timeout_seconds()

    try:
        response = requests.request(
            method, url, json=json, params=params, headers=headers, timeout=timeout
        )
    except requests.Timeout as exc:
        raise ApiTimeoutError(str(exc)) from exc
    except requests.ConnectionError as exc:
        raise ApiConnectionError(str(exc)) from exc
    except requests.RequestException as exc:
        # Filet de sécurité pour les erreurs `requests` non spécifiques.
        raise ApiConnectionError(str(exc)) from exc

    if response.status_code >= 400:
        raise _http_error(response)

    try:
        data = response.json()
    except ValueError as exc:
        raise ApiInvalidResponseError("réponse JSON illisible") from exc

    # Nos contrats API renvoient toujours un objet JSON : une valeur `null`,
    # une liste, un nombre, etc. ne correspond à aucun contrat connu.
    if not isinstance(data, dict):
        raise ApiInvalidResponseError(
            "réponse JSON inattendue (objet JSON attendu)"
        )
    return data


def _http_error(response: requests.Response) -> ApiHttpError:
    """Construit une ``ApiHttpError`` à partir d'une réponse d'erreur.

    Tente de lire le corps ``ErrorResponse`` de l'API ; si le JSON est
    invalide ou incomplet, retombe sur des valeurs génériques.
    """
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}

    code = str(body.get("error") or "unknown_error")
    message = str(body.get("message") or "Erreur inconnue.")
    raw_details = body.get("details")
    details = raw_details if isinstance(raw_details, list) else None
    return ApiHttpError(response.status_code, code, message, details)
