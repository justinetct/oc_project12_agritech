"""Construction de l'engine SQLAlchemy et de la fabrique de sessions.

Deux fonctions, rien de plus :

- `create_monitoring_engine(config)` : construit un engine à partir de
  `config.database_url`. Si le backend est SQLite, applique les PRAGMAs
  attendus (WAL, busy_timeout, synchronous, foreign_keys) à chaque
  connexion, et crée le dossier parent du fichier de base au besoin.
- `create_session_factory(engine)` : renvoie un `sessionmaker` prêt à
  ouvrir des sessions courtes.

Aucune singleton global : l'engine et la factory sont construits une fois
par l'appelant (lifespan FastAPI au lot 3, script CLI de rejeu au lot 5) et
passés là où on en a besoin.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine as _sa_create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from agritech.monitoring.config import MonitoringConfig


def create_monitoring_engine(config: MonitoringConfig) -> Engine:
    """Construit l'engine SQLAlchemy à partir de la configuration monitoring.

    Pour un backend SQLite avec un chemin fichier, crée le dossier parent
    (`data/monitoring/` en local, `/app/data/monitoring/` en Docker) avant
    d'ouvrir la base, puis enregistre les PRAGMAs à chaque connexion.

    Pour tout autre backend (PostgreSQL par exemple), aucun PRAGMA spécifique
    n'est appliqué : c'est ce qui rend le module portable sans code de
    branchement dans les couches supérieures.
    """
    _ensure_sqlite_directory(config.database_url)
    engine = _sa_create_engine(config.database_url, future=True)
    if engine.dialect.name == "sqlite":
        _register_sqlite_pragmas(engine)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker:
    """Fabrique `sessionmaker` attachée à l'engine donné.

    `expire_on_commit=False` : après `commit()`, les attributs de l'instance
    restent utilisables (notamment `row.id` fraîchement peuplé), sans SELECT
    supplémentaire. `autoflush=False` : le middleware contrôle explicitement
    quand la ligne est écrite, sans flush intempestif au milieu du dispatch.
    """
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def _ensure_sqlite_directory(database_url: str) -> None:
    """Crée le dossier parent du fichier SQLite s'il n'existe pas.

    Sans effet pour un backend non-SQLite, pour une base en mémoire, ou pour
    une URL sans fichier — dans ces cas il n'y a rien à créer.
    """
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return
    db_path = url.database
    if not db_path or db_path == ":memory:":
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _register_sqlite_pragmas(engine: Engine) -> None:
    """Applique les PRAGMAs SQLite documentés à chaque nouvelle connexion.

    `journal_mode=WAL` : lectures et écritures concurrentes sans blocage
    mutuel. `busy_timeout=5000` : attend jusqu'à 5 s en cas de verrou
    concurrent avant d'échouer. `synchronous=NORMAL` : compromis durabilité
    / vitesse recommandé avec WAL. `foreign_keys=ON` : contraintes de clés
    étrangères effectivement vérifiées (désactivées par défaut en SQLite).
    """

    @event.listens_for(engine, "connect")
    def _apply_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
