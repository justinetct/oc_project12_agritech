"""Dépendances FastAPI des endpoints `/monitoring/*`.

Deux dépendances, prévues pour être déclarées dans cet ordre :

1. `require_monitoring_token` : vérifie l'en-tête
   `Authorization: Bearer <token>` contre `runtime.monitoring_api_token` ;
2. `get_monitoring_session` : ouvre une session de lecture sur la base de
   monitoring, à partir de la session factory créée par le lifespan.

Ainsi, un appel non authentifié est refusé avant toute ouverture de session.
Aucune des deux ne journalise de token ni d'en-tête `Authorization`.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from agritech.api.core import runtime
from agritech.api.exceptions import MonitoringUnauthorizedError, MonitoringUnavailableError


# `auto_error=False` : sans en-tête, avec un schéma autre que `Bearer` ou un
# token vide, `HTTPBearer` renvoie `None` au lieu de lever sa propre erreur.
# On garde ainsi la main sur la réponse (`ErrorResponse` unifiée).
_bearer_scheme = HTTPBearer(auto_error=False)


def require_monitoring_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Vérifie le token Bearer des endpoints `/monitoring/*`.

    - Token serveur non configuré → `MonitoringUnavailableError` (503).
    - En-tête absent, mal formé ou token incorrect → `MonitoringUnauthorizedError` (401).

    La comparaison se fait avec `secrets.compare_digest`, sur les octets UTF-8
    des deux tokens (une chaîne non ASCII ne fait donc pas planter la
    comparaison).
    """
    expected_token = runtime.monitoring_api_token
    if not expected_token:
        raise MonitoringUnavailableError("monitoring API token is not configured")

    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise MonitoringUnauthorizedError()


def get_monitoring_session() -> Iterator[Session]:
    """Fournit une session SQLAlchemy de lecture sur la base de monitoring.

    Réutilise la session factory publiée par le lifespan : aucune nouvelle
    engine n'est créée. La session est toujours fermée après la requête,
    même si l'endpoint lève une exception.

    Session factory absente → `MonitoringUnavailableError` (503).
    """
    factory = runtime.monitoring_session_factory
    if factory is None:
        raise MonitoringUnavailableError("monitoring session factory is not initialised")

    with factory() as session:
        yield session
