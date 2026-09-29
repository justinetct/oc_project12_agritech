"""Configuration Logfire optionnelle et non bloquante.

Une seule fonction publique, `configure_logfire`, appelée depuis le lifespan
FastAPI. Politique :

- Sans token (`config.logfire_token` vide ou `None`) : no-op silencieux. Aucun
  envoi réseau, aucune modification de l'application, aucune erreur logguée.
- Avec token : configure Logfire avec le nom de service et l'environnement de
  la `MonitoringConfig`. Aucun appel à `logfire.instrument_fastapi()` :
  celui-ci fait `app.add_middleware(...)`, refusé par Starlette après le
  démarrage. À la place, notre middleware ouvre lui-même un `logfire.span`
  autour de chaque appel métier — sans configure Logfire, ce span est un
  no-op silencieux et le contexte OTel reste vide.
- Toute erreur est absorbée : un `warning` est loggé côté serveur, mais
  l'API démarre et sert normalement.

Rappel des consignes tâche 19 : pas de wrapper OpenTelemetry maison, pas de
provider custom. `logfire` est utilisé tel qu'il est fourni par la librairie.
"""

from __future__ import annotations

import logging

from agritech.monitoring.config import MonitoringConfig


logger = logging.getLogger(__name__)


def configure_logfire(config: MonitoringConfig) -> None:
    """Configure Logfire selon `config` ; strictement no-op si aucun token.

    `logfire.configure()` est idempotent : peut être appelé à chaque
    démarrage sans effets de bord cumulatifs.
    """
    if not config.logfire_token:
        return

    try:
        import logfire

        logfire.configure(
            token=config.logfire_token,
            service_name=config.logfire_service_name,
            environment=config.logfire_environment,
            send_to_logfire="if-token-present",
            console=False,
        )
    except Exception:  # noqa: BLE001 — observabilité non bloquante
        logger.warning(
            "logfire configuration failed; API continues without external tracing",
            exc_info=True,
        )
