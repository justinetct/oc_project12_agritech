"""Modèle SQLAlchemy de la table `api_requests`.

Une seule table, un modèle unique : chaque appel HTTP à un endpoint métier
(`POST /predict` ou `POST /recommend`) est archivé en une ligne. Le middleware
qui viendra au lot 3 est le seul point d'écriture. Le script de rejeu du lot
5 est le seul lecteur applicatif ; sinon la table peut être inspectée à la
main via `sqlite3` ou un client PostgreSQL.

Les types choisis sont volontairement portables SQLite → PostgreSQL : `JSON`
pour les payloads (`TEXT` sous SQLite, `JSONB`/`JSON` sous PostgreSQL, sans
changer le code d'écriture ni de lecture) et `DateTime(timezone=True)` pour
un timestamp toujours timezone-aware. `Base.metadata.create_all(engine)`
suffira à créer la table au démarrage de l'application ; pas d'Alembic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UtcDateTime(TypeDecorator):
    """`DateTime` portable SQLite ↔ PostgreSQL, toujours timezone-aware en UTC.

    SQLite ignore le drapeau `timezone=True` de SQLAlchemy : sans ce
    décorateur, un `datetime` tz-aware inséré ressort naïf à la lecture,
    ce qui casserait toute comparaison ou tout affichage cohérent côté
    replay. Le décorateur normalise :

    - à l'écriture : refuse un `datetime` naïf (l'appelant doit explicitement
      dire dans quelle timezone il travaille), convertit vers UTC ;
    - à la lecture : si la valeur revient naïve (cas SQLite), retagge UTC ;
      si la valeur revient déjà aware (cas PostgreSQL), la laisse telle
      quelle.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "naive datetime is not accepted; pass a timezone-aware datetime"
            )
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    """Base déclarative commune des modèles SQLAlchemy de la couche monitoring."""


class ApiRequest(Base):
    """Une ligne de l'historique d'appels des endpoints métier.

    Colonnes obligatoires (`nullable=False`) : `timestamp`, `service`,
    `endpoint`, `method`, `status_code`, `success`, `duration_ms`,
    `api_version`, `request_payload`, `environment`.

    Colonnes optionnelles (`nullable=True`) : `model_version` (peut manquer
    quand aucun bundle n'est chargé), `response_payload` (absente si
    l'endpoint n'a produit aucun corps), `error_type` et `error_message` (que
    pour les échecs), `logfire_trace_id` (absent si Logfire est désactivé).
    """

    __tablename__ = "api_requests"
    __table_args__ = (
        Index("api_requests_service_ts_idx", "service", "timestamp"),
        Index("api_requests_success_ts_idx", "success", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    service: Mapped[str] = mapped_column(String(20), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    api_version: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    logfire_trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    environment: Mapped[str] = mapped_column(String(20), nullable=False)
