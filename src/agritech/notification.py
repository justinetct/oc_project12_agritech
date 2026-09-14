"""Notification de fin d'entraînement (API Pushover), désactivée par défaut.

Elle n'est envoyée que si l'environnement contient `NOTIFICATIONS_ENABLED=true`, avec
`PUSHOVER_USER_KEY` et `PUSHOVER_APP_TOKEN`. Sans configuration, `notify` ne fait rien et n'affiche
rien ; un envoi réussi n'affiche rien non plus. Un échec affiche une ligne courte, sans jamais
interrompre l'appelant.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

from agritech.config import PATHS


PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"
TRUE_VALUES = {"1", "true", "yes", "on"}


def notify(title: str, message: str) -> None:
    """Envoie la notification si elle est activée ; ne lève jamais d'exception."""
    load_dotenv(PATHS.root / ".env")
    if os.getenv("NOTIFICATIONS_ENABLED", "").strip().lower() not in TRUE_VALUES:
        return

    try:
        response = requests.post(
            PUSHOVER_API_URL,
            data={
                "token": os.environ["PUSHOVER_APP_TOKEN"],
                "user": os.environ["PUSHOVER_USER_KEY"],
                "title": title,
                "message": message,
            },
            timeout=10,
        )
        response.raise_for_status()
    except Exception as error:  # accessoire : ne doit jamais interrompre le notebook
        # type d'erreur seulement : aucun détail de la requête ni des clés
        print(f"notification non envoyée ({type(error).__name__})")
