"""Configuration de l'interface Streamlit.

Une seule source pour l'URL de l'API et le timeout HTTP. Aucune
logique d'appel réseau ici : le client HTTP arrivera à la
sous-étape suivante et lira ces valeurs.
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
