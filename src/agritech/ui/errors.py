"""Exceptions et formatage pour le client HTTP des interfaces.

Chaque type d'erreur porte un sens précis pour le front (configuration
absente, timeout, connexion, réponse HTTP invalide, JSON illisible). Deux
helpers fournissent un message lisible destiné à l'utilisateur :

- ``format_api_error(exc)`` pour les pages Streamlit ``/predict`` et
  ``/recommend`` (``st.error(format_api_error(exc))``) ;
- ``format_monitoring_error(exc)`` pour l'interface de monitoring.

Aucun message ne reprend un token, une trace d'erreur ou un corps de réponse.
"""

from __future__ import annotations

from typing import Any

# Message affiché quand le contexte reçu de l'API n'a pas la structure attendue.
INVALID_CONTEXT_MESSAGE = "Le service a renvoyé des informations inattendues."

# Message affiché pour chaque code d'erreur de l'API (champ ``error`` de la réponse).
HTTP_ERROR_MESSAGES = {
    "model_unavailable": "Le modèle est momentanément indisponible. Réessayez dans un instant.",
    "internal_error": "Le service a rencontré une erreur. Réessayez dans un instant.",
    "validation_error": "Les valeurs envoyées n’ont pas été acceptées.",
}


class ApiError(Exception):
    """Base de toutes les erreurs remontées par le client HTTP."""


class ApiConfigurationError(ApiError):
    """Configuration locale incomplète : aucun appel HTTP n'a été tenté.

    Exemple : ``MONITORING_API_TOKEN`` absent pour appeler ``/monitoring/*``.
    """


class ApiTimeoutError(ApiError):
    """L'API n'a pas répondu dans le délai imparti."""


class ApiConnectionError(ApiError):
    """Impossible d'établir une connexion avec l'API."""


class ApiInvalidResponseError(ApiError):
    """Réponse HTTP reçue mais illisible (JSON invalide, corps inattendu, ...)."""


class ApiHttpError(ApiError):
    """L'API a répondu avec un code HTTP >= 400.

    Porte les champs de la réponse ``ErrorResponse`` de l'API quand ils
    sont disponibles.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(f"HTTP {status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


def format_api_error(exc: ApiError) -> str:
    """Retourne un message lisible destiné à l'utilisateur.

    Message court, en français. Les messages et détails techniques renvoyés
    par l'API ne sont pas réaffichés : seul le code d'erreur choisit le message.
    """
    if isinstance(exc, ApiTimeoutError):
        return "Le service de calcul n'a pas répondu à temps. Réessayez dans un instant."
    if isinstance(exc, ApiConnectionError):
        return "Impossible de joindre le service de calcul. Réessayez dans un instant."
    if isinstance(exc, ApiInvalidResponseError):
        return "Le service a renvoyé une réponse inattendue."
    if isinstance(exc, ApiHttpError):
        return HTTP_ERROR_MESSAGES.get(
            exc.code, f"Le service a renvoyé une erreur inattendue (code {exc.status_code})."
        )
    return "Erreur inattendue lors de l'appel API."


# Message affiché par l'interface de monitoring pour chaque statut HTTP connu.
# Le statut est utilisé plutôt que le code d'erreur : un 503 renvoyé par un
# proxy ou un hébergeur n'a pas forcément le corps ``ErrorResponse`` de l'API.
MONITORING_HTTP_MESSAGES = {
    401: "Accès refusé : le token de monitoring n'est pas accepté par l'API.",
    503: "Le monitoring est momentanément indisponible côté API.",
}


def format_monitoring_error(exc: ApiError) -> str:
    """Retourne un message lisible pour l'interface de monitoring.

    Message court, en français, sans token, trace d'erreur ni détail renvoyé
    par l'API : seul le type d'erreur (ou le statut HTTP) choisit le message.
    """
    if isinstance(exc, ApiConfigurationError):
        return "Monitoring non configuré : le token d'accès n'est pas défini pour cette interface."
    if isinstance(exc, ApiTimeoutError):
        return "L'API de monitoring n'a pas répondu à temps. Réessayez dans un instant."
    if isinstance(exc, ApiConnectionError):
        return "Impossible de joindre l'API de monitoring. Vérifiez qu'elle est démarrée."
    if isinstance(exc, ApiInvalidResponseError):
        return "L'API de monitoring a renvoyé une réponse inattendue."
    if isinstance(exc, ApiHttpError):
        return MONITORING_HTTP_MESSAGES.get(
            exc.status_code,
            f"L'API de monitoring a renvoyé une erreur inattendue (code {exc.status_code}).",
        )
    return "Erreur inattendue lors de l'appel à l'API de monitoring."
