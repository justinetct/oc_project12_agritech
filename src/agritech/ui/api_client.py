"""Client HTTP minimal pour appeler l'API Agritech Answers.

Deux fonctions publiques pour la sous-étape 21.2 :

- ``get_predict_context()`` : appelle ``GET /predict/context`` et
  retourne le dict décodé.
- ``post_predict(payload)`` : appelle ``POST /predict`` et retourne
  le dict décodé.

Les fonctions s'appuient sur ``settings.api_base_url()`` et
``settings.request_timeout_seconds()`` ; le module ``errors``
définit les exceptions remontées au front.

Aucune logique Streamlit ni logique ML n'est faite ici : le module
se contente de sérialiser/désérialiser du JSON et de traduire les
erreurs ``requests`` en exceptions dédiées. La primitive interne
``_request`` sera réutilisée pour ``/recommend`` sans modification.
"""

from __future__ import annotations

from typing import Any

import requests

from agritech.ui import settings
from agritech.ui.errors import (
    ApiConnectionError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
)


def get_predict_context() -> dict[str, Any]:
    """Récupère les bornes physiques et le domaine d'entraînement de /predict."""
    return _request("GET", "/predict/context")


def post_predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Envoie un payload à ``POST /predict`` et retourne la réponse décodée."""
    return _request("POST", "/predict", json=payload)


def _request(
    method: str, path: str, json: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Effectue une requête HTTP simple et retourne le JSON décodé.

    Traduit chaque type de défaillance en une exception ``ApiError``
    dédiée. Aucune retry, aucun cache : la simplicité prime.
    """
    url = settings.api_base_url().rstrip("/") + path
    timeout = settings.request_timeout_seconds()

    try:
        response = requests.request(method, url, json=json, timeout=timeout)
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
