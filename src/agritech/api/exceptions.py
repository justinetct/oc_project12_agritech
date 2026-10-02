"""Exceptions applicatives de l'API Agritech Answers.

Les exceptions métier vivent ici pour que le code des routers reste indépendant
des codes HTTP : les handlers de `error_handlers.py` les traduisent en
`ErrorResponse` unifiées.
"""

from __future__ import annotations


class ModelUnavailableError(Exception):
    """Un modèle attendu par un endpoint n'est pas chargé.

    Est levée quand `runtime.bundle_predict` (ou l'équivalent `/recommend`) vaut
    `None` alors qu'un endpoint en a besoin. Le handler HTTP la traduit en 503
    avec le code `model_unavailable`.
    """


class MonitoringUnauthorizedError(Exception):
    """L'appelant d'un endpoint `/monitoring/*` n'est pas authentifié.

    Est levée quand l'en-tête `Authorization: Bearer <token>` est absent, mal
    formé ou porte un token incorrect. Le handler HTTP la traduit en 401
    `unauthorized`, avec l'en-tête `WWW-Authenticate: Bearer`.
    """


class MonitoringUnavailableError(Exception):
    """Les endpoints `/monitoring/*` ne peuvent pas être servis.

    Est levée quand le token de monitoring n'est pas configuré côté serveur
    ou quand la session factory de la base de monitoring n'est pas
    disponible. Le message de l'exception reste interne : le handler HTTP
    renvoie un 503 `monitoring_unavailable` générique, sans la cause exacte.
    """
