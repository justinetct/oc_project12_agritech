"""Configuration de la couche monitoring.

Lit les variables d'environnement nécessaires à la persistance SQLite et à
Logfire, avec des défauts adaptés à un développement local. Le module ne
charge PAS `.env` lui-même : ce chargement est fait par la couche qui utilise
la configuration (`agritech.api.main` au démarrage de l'application, script
CLI de rejeu). Cela garde `config` pur et facilement testable.

Sans variable d'environnement fournie :

- `DATABASE_URL` = `sqlite:///data/monitoring/api.sqlite` (chemin relatif au
  projet) : cohérent avec la production Docker qui pose
  `sqlite:////app/data/monitoring/api.sqlite` (chemin absolu, 4 slashes) ;
- `ENVIRONMENT` = `local` ;
- `LOGFIRE_TOKEN` = `None` (aucun envoi réseau, mode silencieux) ;
- `LOGFIRE_ENVIRONMENT` = valeur de `ENVIRONMENT` ;
- `LOGFIRE_SERVICE_NAME` = `agritech-answers`.

Une chaîne vide (`LOGFIRE_TOKEN=`) est traitée comme absence : elle ne
produit pas un token vide envoyé à Logfire.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DATABASE_URL = "sqlite:///data/monitoring/api.sqlite"
DEFAULT_ENVIRONMENT = "local"
DEFAULT_LOGFIRE_SERVICE_NAME = "agritech-answers"


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    """Configuration lue au chargement du module.

    Les attributs sont immuables (`frozen=True`) : la configuration ne peut
    plus changer après construction, ce qui protège l'appelant contre des
    modifications accidentelles pendant l'exécution.

    Attributs :
        database_url : URL SQLAlchemy vers la base de monitoring.
        environment : nom court de l'environnement d'exécution, persisté dans
            la colonne `environment` de la table `api_requests`.
        logfire_token : token Logfire ; `None` désactive tout envoi réseau.
        logfire_environment : `environment_name` transmis à Logfire ; reprend
            `environment` s'il n'est pas explicitement renseigné.
        logfire_service_name : nom de service transmis à Logfire.
    """

    database_url: str
    environment: str
    logfire_token: str | None
    logfire_environment: str
    logfire_service_name: str


def load_config() -> MonitoringConfig:
    """Construit une `MonitoringConfig` à partir de l'environnement courant.

    Ne lit qu'`os.environ` : le chargement d'un éventuel `.env` est de la
    responsabilité de l'application ou du script CLI qui appelle
    `load_config`. Une valeur absente ou une chaîne vide utilise le défaut
    documenté.
    """
    environment = os.environ.get("ENVIRONMENT") or DEFAULT_ENVIRONMENT
    return MonitoringConfig(
        database_url=os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
        environment=environment,
        logfire_token=os.environ.get("LOGFIRE_TOKEN") or None,
        logfire_environment=os.environ.get("LOGFIRE_ENVIRONMENT") or environment,
        logfire_service_name=(
            os.environ.get("LOGFIRE_SERVICE_NAME") or DEFAULT_LOGFIRE_SERVICE_NAME
        ),
    )
