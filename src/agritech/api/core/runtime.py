"""État de l'application, initialisé au démarrage par le lifespan.

Chaque attribut vaut `None` avant le démarrage, puis contient l'objet chargé
une fois pour toutes. Les endpoints les lisent sans les modifier.

`monitoring_*` alimentent le middleware de persistance : ils sont peuplés
par le lifespan à partir de `MonitoringConfig`. `monitoring_session_factory`
reste `None` si la base SQLite ne peut pas être initialisée — dans ce cas le
middleware n'écrit rien mais laisse la requête passer normalement, et les
endpoints `/monitoring/*` répondent 503.

`monitoring_api_token` est le secret attendu par les endpoints
`/monitoring/*`. Il vaut `None` tant que `MONITORING_API_TOKEN` n'est pas
configuré : ces endpoints restent alors indisponibles.
"""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from agritech.serving import Bundle, RecommendContext


bundle_predict: Bundle | None = None
bundle_recommend: Bundle | None = None
recommend_context: RecommendContext | None = None
monitoring_session_factory: sessionmaker | None = None
monitoring_api_version: str | None = None
monitoring_environment: str | None = None
monitoring_api_token: str | None = None
