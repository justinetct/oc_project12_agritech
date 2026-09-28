"""Tests structurels du modèle `ApiRequest` : colonnes, nullabilité, index, PK.

Ces tests inspectent le mapping SQLAlchemy directement, sans ouvrir de base
de données : ils vérifient uniquement la déclaration du schéma. Les tests
d'intégration (base réelle) vivent dans `test_repository.py` et
`test_session.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agritech.monitoring.models import ApiRequest, Base, UtcDateTime


EXPECTED_COLUMNS = (
    "id",
    "timestamp",
    "service",
    "endpoint",
    "method",
    "status_code",
    "success",
    "duration_ms",
    "api_version",
    "model_version",
    "request_payload",
    "response_payload",
    "error_type",
    "error_message",
    "logfire_trace_id",
    "environment",
)

NOT_NULL_COLUMNS = {
    "id",
    "timestamp",
    "service",
    "endpoint",
    "method",
    "status_code",
    "success",
    "duration_ms",
    "api_version",
    "request_payload",
    "environment",
}

NULLABLE_COLUMNS = {
    "model_version",
    "response_payload",
    "error_type",
    "error_message",
    "logfire_trace_id",
}


def test_tablename_is_api_requests():
    """Le nom de table est stable dans le temps : `api_requests`."""
    assert ApiRequest.__tablename__ == "api_requests"


def test_table_has_expected_columns_in_expected_order():
    """La liste et l'ordre des colonnes correspondent à la spécification."""
    table = ApiRequest.__table__
    actual = tuple(column.name for column in table.columns)
    assert actual == EXPECTED_COLUMNS


def test_not_null_columns_are_not_nullable():
    """Les colonnes obligatoires sont bien `nullable=False`."""
    table = ApiRequest.__table__
    for name in NOT_NULL_COLUMNS:
        assert table.columns[name].nullable is False, name


def test_nullable_columns_are_nullable():
    """Les colonnes optionnelles sont bien `nullable=True`."""
    table = ApiRequest.__table__
    for name in NULLABLE_COLUMNS:
        assert table.columns[name].nullable is True, name


def test_id_is_primary_key_autoincrement():
    """`id` est la clé primaire auto-incrémentée."""
    id_col = ApiRequest.__table__.columns["id"]
    assert id_col.primary_key is True
    # SQLAlchemy représente autoincrement par `True` ou par la valeur par défaut "auto".
    assert id_col.autoincrement in (True, "auto")


def test_table_has_expected_indexes():
    """Deux index attendus : `(service, timestamp)` et `(success, timestamp)`."""
    indexes = ApiRequest.__table__.indexes
    by_name = {ix.name: tuple(col.name for col in ix.columns) for ix in indexes}
    assert by_name == {
        "api_requests_service_ts_idx": ("service", "timestamp"),
        "api_requests_success_ts_idx": ("success", "timestamp"),
    }


def test_base_metadata_contains_only_api_requests():
    """Un seul modèle est déclaré dans le package monitoring."""
    assert set(Base.metadata.tables) == {"api_requests"}


# --- UtcDateTime : branches défensives (aucune base réelle nécessaire) ---


def test_utc_datetime_bind_param_none_returns_none():
    """`None` en écriture reste `None` : évite d'imposer un timestamp sur les nullable."""
    assert UtcDateTime().process_bind_param(None, dialect=None) is None


def test_utc_datetime_result_value_none_returns_none():
    """`None` en lecture reste `None` (colonnes nullable)."""
    assert UtcDateTime().process_result_value(None, dialect=None) is None


def test_utc_datetime_result_value_already_aware_is_untouched():
    """Sur un backend qui préserve la tz (PostgreSQL), la valeur n'est pas retouchée."""
    aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert UtcDateTime().process_result_value(aware, dialect=None) is aware
