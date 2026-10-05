"""Exceptions et formatage pour le client HTTP des interfaces.

Chaque type d'erreur porte un sens précis pour le front (configuration
absente, timeout, connexion, réponse HTTP invalide, JSON illisible). Deux
helpers fournissent un message lisible destiné à l'utilisateur :

- ``format_api_error(exc, labels, bounds)`` pour les pages Streamlit
  ``/predict`` et ``/recommend`` (``st.error(format_api_error(exc))``) ;
- ``format_monitoring_error(exc)`` pour l'interface de monitoring.

Aucun message ne reprend un token, une trace d'erreur ou un corps de réponse.
"""

from __future__ import annotations

from typing import Any

from agritech.ui.components import fr_number

# Message affiché quand le contexte reçu de l'API n'a pas la structure attendue.
INVALID_CONTEXT_MESSAGE = "Le service a renvoyé des informations inattendues."

# Message affiché pour chaque code d'erreur de l'API (champ ``error`` de la réponse).
HTTP_ERROR_MESSAGES = {
    "model_unavailable": "Le modèle est momentanément indisponible. Réessayez dans un instant.",
    "internal_error": "Le service a rencontré une erreur. Réessayez dans un instant.",
    "validation_error": "Les valeurs envoyées n’ont pas été acceptées.",
}


# Types d'erreur Pydantic d'une valeur hors bornes physiques ou non finie.
RANGE_ERROR_TYPES = {"greater_than_equal", "less_than_equal", "finite_number"}


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


def format_api_error(
    exc: ApiError,
    labels: dict[str, str] | None = None,
    bounds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Retourne un message lisible destiné à l'utilisateur.

    Message court, en français. Les messages techniques renvoyés par l'API ne
    sont pas réaffichés. Pour une 422, ``labels`` (nom public → libellé) et
    ``bounds`` (bornes physiques de ``/context``) donnent une ligne par champ
    refusé ; sans détail exploitable, le message générique est conservé.
    """
    if isinstance(exc, ApiTimeoutError):
        return "Le service de calcul n'a pas répondu à temps. Réessayez dans un instant."
    if isinstance(exc, ApiConnectionError):
        return "Impossible de joindre le service de calcul. Réessayez dans un instant."
    if isinstance(exc, ApiInvalidResponseError):
        return "Le service a renvoyé une réponse inattendue."
    if isinstance(exc, ApiHttpError):
        if exc.code == "validation_error" and labels:
            lines = _invalid_field_lines(exc.details, labels, bounds or {})
            if lines:
                return "Certaines valeurs sont invalides :\n" + "\n".join(f"- {line}" for line in lines)
        return HTTP_ERROR_MESSAGES.get(
            exc.code, f"Le service a renvoyé une erreur inattendue (code {exc.status_code})."
        )
    return "Erreur inattendue lors de l'appel API."


def _invalid_field_lines(
    details: list[Any], labels: dict[str, str], bounds: dict[str, dict[str, Any]]
) -> list[str]:
    """Une ligne par champ refusé et connu de la page, dans l'ordre de l'API.

    Le champ vient du dernier segment de ``field`` (``body.conditions.annual_rainfall_mm``).
    Les détails inconnus ou mal formés sont ignorés.
    """
    lines: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        field = str(detail.get("field", "")).rsplit(".", 1)[-1]
        if field not in labels:
            continue
        reason = (
            _accepted_values(bounds.get(field))
            if detail.get("type") in RANGE_ERROR_TYPES
            else "la valeur n’a pas été acceptée."
        )
        lines.append(f"{labels[field]} : {reason}")
    return lines


def _accepted_values(bound: dict[str, Any] | None) -> str:
    """Valeurs acceptées, d'après les bornes physiques fournies par l'API."""
    low, high = (bound or {}).get("min"), (bound or {}).get("max")
    unit = (bound or {}).get("unit", "")
    if low is not None and high is not None:
        return f"la valeur doit être comprise entre {_bound(low)} et {_bound(high)}\u00a0{unit}."
    if low is not None:
        return f"la valeur doit être supérieure ou égale à {_bound(low)}\u00a0{unit}."
    return "la valeur n’a pas été acceptée."


def _bound(value: float) -> str:
    """Borne à la française : −50, 0, 60 ; au plus deux décimales."""
    return fr_number(value, 2).rstrip("0").rstrip(",").replace("-", "−")


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
