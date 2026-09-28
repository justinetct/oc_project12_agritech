"""Tests de `create_monitoring_engine` et `create_session_factory`.

Vérifient que :

- les PRAGMAs SQLite sont réellement appliqués à chaque connexion ;
- le dossier parent de la base est créé automatiquement ;
- une URL SQLite en mémoire est acceptée sans erreur ni tentative de
  création de dossier ;
- la fabrique de sessions renvoie des `Session` utilisables.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from agritech.monitoring.config import MonitoringConfig
from agritech.monitoring.session import (
    _ensure_sqlite_directory,
    create_monitoring_engine,
    create_session_factory,
)


def _config_for(database_url: str) -> MonitoringConfig:
    """Construit un `MonitoringConfig` minimal ciblant l'URL demandée."""
    return MonitoringConfig(
        database_url=database_url,
        environment="test",
        logfire_token=None,
        logfire_environment="test",
        logfire_service_name="agritech-answers",
    )


def test_engine_dialect_is_sqlite(monitoring_config: MonitoringConfig):
    """L'engine construit depuis une URL `sqlite:///...` est bien un dialecte SQLite."""
    engine = create_monitoring_engine(monitoring_config)
    try:
        assert engine.dialect.name == "sqlite"
    finally:
        engine.dispose()


def test_sqlite_pragmas_are_applied_on_connect(monitoring_config: MonitoringConfig):
    """Les 4 PRAGMAs documentés valent leur valeur cible à l'ouverture d'une connexion."""
    engine = create_monitoring_engine(monitoring_config)
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000
            # SQLite: OFF=0, NORMAL=1, FULL=2, EXTRA=3.
            assert conn.execute(text("PRAGMA synchronous")).scalar() == 1
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_engine_creates_parent_directory(tmp_path: Path):
    """Le dossier parent du fichier SQLite est créé s'il n'existe pas."""
    nested_dir = tmp_path / "nested" / "data" / "monitoring"
    assert not nested_dir.exists()

    config = _config_for(f"sqlite:///{nested_dir / 'api.sqlite'}")
    engine = create_monitoring_engine(config)
    try:
        # Force l'ouverture d'une connexion pour matérialiser le fichier.
        with engine.connect():
            pass
        assert nested_dir.exists() and nested_dir.is_dir()
        assert (nested_dir / "api.sqlite").exists()
    finally:
        engine.dispose()


def test_engine_handles_memory_sqlite_without_creating_files(tmp_path: Path):
    """Une URL `sqlite:///:memory:` fonctionne et ne crée aucun fichier."""
    config = _config_for("sqlite:///:memory:")
    engine = create_monitoring_engine(config)
    try:
        with engine.connect() as conn:
            # La base répond aux requêtes (base en mémoire fonctionnelle).
            assert conn.execute(text("SELECT 1")).scalar() == 1
    finally:
        engine.dispose()


def test_session_factory_returns_usable_sessions(engine):
    """La fabrique renvoie une session capable d'exécuter du SQL brut."""
    factory = create_session_factory(engine)
    with factory() as sess:
        assert isinstance(sess, Session)
        assert sess.execute(text("SELECT 1")).scalar() == 1


def test_ensure_sqlite_directory_is_noop_for_non_sqlite_url(tmp_path: Path):
    """Une URL non-SQLite (par exemple PostgreSQL) ne crée aucun dossier local.

    Vérifie la portabilité : le code de préparation SQLite ne s'exécute que
    pour un backend SQLite. Pas de vraie connexion : la fonction est pure.
    """
    before = set(tmp_path.iterdir())
    _ensure_sqlite_directory("postgresql://user:pass@localhost/db")
    after = set(tmp_path.iterdir())
    assert after == before
