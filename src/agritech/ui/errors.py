"""Exceptions et formatage pour le client HTTP de l'interface Streamlit.

Chaque type d'erreur porte un sens précis pour le front (timeout,
connexion, réponse HTTP invalide, JSON illisible). Un helper
``format_api_error(exc)`` fournit un message lisible destiné à
l'utilisateur ; il centralise la traduction pour que les pages
Streamlit se contentent de ``st.error(format_api_error(exc))``.
"""

from __future__ import annotations

from typing import Any


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

    Message court, en français, sans jargon HTTP inutile. Les détails
    de validation d'une 422 sont listés à la suite.
    """
    if isinstance(exc, ApiTimeoutError):
        return "L'API n'a pas répondu à temps. Réessayez dans un instant."
    if isinstance(exc, ApiConnectionError):
        return "Impossible de joindre l'API. Vérifiez qu'elle est démarrée."
    if isinstance(exc, ApiInvalidResponseError):
        return "L'API a renvoyé une réponse inattendue."
    if isinstance(exc, ApiHttpError):
        base = f"L'API a renvoyé une erreur ({exc.status_code}) : {exc.message}"
        if exc.details:
            lignes = [
                f"- {d.get('field', '?')} : {d.get('message', '')}"
                for d in exc.details
            ]
            base += "\n" + "\n".join(lignes)
        return base
    return "Erreur inattendue lors de l'appel API."
