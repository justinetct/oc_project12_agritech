"""Exceptions et formatage pour le client HTTP de l'interface Streamlit.

Chaque type d'erreur porte un sens précis pour le front (timeout,
connexion, réponse HTTP invalide, JSON illisible). Un helper
``format_api_error(exc)`` fournit un message lisible destiné à
l'utilisateur ; il centralise la traduction pour que les pages
Streamlit se contentent de ``st.error(format_api_error(exc))``.
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
