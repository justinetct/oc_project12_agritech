"""Configuration des interfaces (Streamlit et monitoring).

Une seule source pour l'URL de l'API, le timeout HTTP et le token des
endpoints de monitoring. Aucune logique d'appel réseau ici : le client
HTTP (``api_client``) lit ces valeurs.
"""

from __future__ import annotations

import os


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0


def api_base_url() -> str:
    """Retourne l'URL de base de l'API.

    Lit la variable d'environnement ``AGRITECH_API_URL``. Absente ou
    vide (après ``strip``), on retombe sur ``DEFAULT_API_URL``.
    """
    value = os.getenv("AGRITECH_API_URL", "").strip()
    return value or DEFAULT_API_URL


def request_timeout_seconds() -> float:
    """Retourne le timeout HTTP en secondes (constante pour l'instant)."""
    return DEFAULT_TIMEOUT_SECONDS


def monitoring_api_token() -> str | None:
    """Retourne le token Bearer des endpoints ``/monitoring/*``, ou ``None``.

    Lit la variable d'environnement ``MONITORING_API_TOKEN``, la même que
    celle attendue par l'API. Absente, vide ou faite seulement d'espaces :
    ``None``, sans valeur par défaut. La valeur n'est jamais loggée.
    """
    value = os.getenv("MONITORING_API_TOKEN", "").strip()
    return value or None
