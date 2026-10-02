"""Tests d'intégration du repository `api_requests` contre une base SQLite jetable.

Couvre l'écriture (`insert_api_request`) et la lecture des requêtes récentes
(`list_recent_requests`).

Toutes les fixtures viennent de `conftest.py` : la base vit sous `tmp_path`,
ce qui garantit qu'aucun test ne peut écrire dans la base locale du projet.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from agritech.monitoring.models import ApiRequest
from agritech.monitoring.repository import insert_api_request, list_recent_requests


def _full_row(**overrides) -> dict:
    """Renvoie un dict complet valide pour `ApiRequest`, surchargé par `overrides`."""
    base = {
        "timestamp": datetime(2026, 9, 28, 15, 30, tzinfo=timezone.utc),
        "service": "predict",
        "endpoint": "/predict",
        "method": "POST",
        "status_code": 200,
        "success": True,
        "duration_ms": 42,
        "api_version": "0.1.0",
        "model_version": "1.0.0",
        "request_payload": {
            "rainfall_mm": 500.0,
            "temperature_celsius": 25.0,
            "fertilizer_used": True,
            "irrigation_used": False,
        },
        "response_payload": {
            "yield_tons_per_hectare": 4.82,
            "unit": "t/ha",
            "model_version": "1.0.0",
            "out_of_training_domain": False,
            "notes": [],
        },
        "error_type": None,
        "error_message": None,
        "logfire_trace_id": "0123456789abcdef0123456789abcdef",
        "environment": "test",
    }
    base.update(overrides)
    return base


def test_insert_api_request_returns_positive_id(session: Session):
    """L'insertion renvoie un `id` entier strictement positif."""
    row_id = insert_api_request(session, _full_row())
    assert isinstance(row_id, int)
    assert row_id > 0


def test_insert_api_request_persists_all_fields(session: Session):
    """La ligne relue depuis la base contient exactement les valeurs insérées."""
    payload = _full_row()
    row_id = insert_api_request(session, payload)

    fetched = session.scalar(select(ApiRequest).where(ApiRequest.id == row_id))
    assert fetched is not None

    for key, expected in payload.items():
        assert getattr(fetched, key) == expected, key


def test_insert_api_request_persists_json_payloads(session: Session):
    """Les dicts passés à `request_payload` / `response_payload` sont relus comme dicts."""
    row_id = insert_api_request(session, _full_row())

    fetched = session.scalar(select(ApiRequest).where(ApiRequest.id == row_id))
    assert isinstance(fetched.request_payload, dict)
    assert fetched.request_payload["rainfall_mm"] == 500.0
    assert isinstance(fetched.response_payload, dict)
    assert fetched.response_payload["yield_tons_per_hectare"] == 4.82


def test_insert_api_request_nullable_fields_can_be_none(session: Session):
    """Les colonnes optionnelles acceptent `None` et sont relues comme `None`."""
    row_id = insert_api_request(
        session,
        _full_row(
            model_version=None,
            response_payload=None,
            error_type="validation_error",
            error_message="Request payload is invalid.",
            logfire_trace_id=None,
        ),
    )

    fetched = session.scalar(select(ApiRequest).where(ApiRequest.id == row_id))
    assert fetched.model_version is None
    assert fetched.response_payload is None
    assert fetched.logfire_trace_id is None
    # Les champs d'erreur, eux, ont bien été enregistrés :
    assert fetched.error_type == "validation_error"
    assert fetched.error_message == "Request payload is invalid."


def test_insert_api_request_ids_are_incremental(session: Session):
    """Deux insertions consécutives produisent des `id` strictement croissants."""
    first = insert_api_request(session, _full_row())
    second = insert_api_request(session, _full_row(service="recommend", endpoint="/recommend"))

    assert second > first


def test_insert_api_request_preserves_utc_timezone(session: Session):
    """Le timestamp relu est aware et en UTC, même sur SQLite (via `UtcDateTime`)."""
    aware_utc = datetime(2026, 9, 28, 15, 30, tzinfo=timezone.utc)
    row_id = insert_api_request(session, _full_row(timestamp=aware_utc))

    fetched = session.scalar(select(ApiRequest).where(ApiRequest.id == row_id))
    assert fetched.timestamp == aware_utc
    assert fetched.timestamp.tzinfo is not None


def test_insert_api_request_rejects_naive_timestamp(session: Session):
    """Un `datetime` naïf (sans tzinfo) est explicitement refusé par le TypeDecorator."""
    naive = datetime(2026, 9, 28, 15, 30)

    with pytest.raises(StatementError):
        insert_api_request(session, _full_row(timestamp=naive))


def test_insert_api_request_writes_only_under_tmp_path(
    tmp_path, monitoring_config, session: Session
):
    """La base réellement créée par la fixture est bien sous `tmp_path`.

    Contrôle explicite d'isolation : aucun test ne doit pouvoir toucher
    `data/monitoring/api.sqlite` du dépôt.
    """
    # L'URL utilisée par la fixture pointe vers un fichier de `tmp_path`.
    assert str(tmp_path) in monitoring_config.database_url

    # Une insertion réussit et matérialise bien le fichier là où on l'attend.
    insert_api_request(session, _full_row())
    expected_file = tmp_path / "test.sqlite"
    assert expected_file.exists()


# ===========================================================================
# list_recent_requests — lecture des requêtes récentes
# ===========================================================================


def _at(minute: int) -> datetime:
    """Timestamp UTC fixe du 28/09/2026 à 15 h, décalé de `minute` minutes."""
    return datetime(2026, 9, 28, 15, minute, tzinfo=timezone.utc)


def _error_row(**overrides) -> dict:
    """Ligne `_full_row` transformée en 422 de validation."""
    values = {
        "status_code": 422,
        "success": False,
        "response_payload": {"error": "validation_error"},
        "error_type": "validation_error",
        "error_message": "Request payload is invalid.",
    }
    values.update(overrides)
    return _full_row(**values)


def test_list_recent_requests_on_empty_database_returns_empty_list(session: Session):
    """Sans aucune ligne archivée, la lecture renvoie une liste vide."""
    assert list_recent_requests(session, limit=20) == []


def test_list_recent_requests_orders_from_most_recent(session: Session):
    """Les lignes sortent de la plus récente à la plus ancienne, quel que soit l'ordre d'insertion."""
    middle_id = insert_api_request(session, _full_row(timestamp=_at(10)))
    oldest_id = insert_api_request(session, _full_row(timestamp=_at(5)))
    newest_id = insert_api_request(session, _full_row(timestamp=_at(20)))

    rows = list_recent_requests(session, limit=20)

    assert [row.id for row in rows] == [newest_id, middle_id, oldest_id]


def test_list_recent_requests_breaks_timestamp_ties_with_id_desc(session: Session):
    """À timestamp identique, l'`id` le plus grand sort en premier : ordre déterministe."""
    ids = [insert_api_request(session, _full_row(timestamp=_at(0))) for _ in range(3)]

    rows = list_recent_requests(session, limit=20)

    assert [row.id for row in rows] == sorted(ids, reverse=True)


def test_list_recent_requests_respects_limit(session: Session):
    """`limit` borne le nombre de lignes et garde les plus récentes."""
    ids = [insert_api_request(session, _full_row(timestamp=_at(minute))) for minute in range(5)]

    rows = list_recent_requests(session, limit=2)

    assert [row.id for row in rows] == [ids[4], ids[3]]


def test_list_recent_requests_filters_by_service(session: Session):
    """`service` ne garde que les lignes du service demandé."""
    insert_api_request(session, _full_row(timestamp=_at(0)))
    recommend_id = insert_api_request(
        session,
        _full_row(timestamp=_at(1), service="recommend", endpoint="/recommend"),
    )

    rows = list_recent_requests(session, limit=20, service="recommend")

    assert [row.id for row in rows] == [recommend_id]


def test_list_recent_requests_filters_by_success(session: Session):
    """`success=False` ne garde que les erreurs, `success=True` que les succès."""
    success_id = insert_api_request(session, _full_row(timestamp=_at(0)))
    error_id = insert_api_request(session, _error_row(timestamp=_at(1)))

    errors = list_recent_requests(session, limit=20, success=False)
    successes = list_recent_requests(session, limit=20, success=True)

    assert [row.id for row in errors] == [error_id]
    assert [row.id for row in successes] == [success_id]


def test_list_recent_requests_combines_service_and_success(session: Session):
    """Les deux filtres se combinent : seules les erreurs `/recommend` restent."""
    insert_api_request(session, _full_row(timestamp=_at(0)))
    insert_api_request(session, _error_row(timestamp=_at(1)))
    insert_api_request(
        session,
        _full_row(timestamp=_at(2), service="recommend", endpoint="/recommend"),
    )
    recommend_error_id = insert_api_request(
        session,
        _error_row(timestamp=_at(3), service="recommend", endpoint="/recommend"),
    )

    rows = list_recent_requests(session, limit=20, service="recommend", success=False)

    assert [row.id for row in rows] == [recommend_error_id]


def test_list_recent_requests_does_not_modify_the_database(session: Session):
    """La lecture n'ajoute, ne modifie ni ne supprime aucune ligne."""
    insert_api_request(session, _full_row(timestamp=_at(0)))
    insert_api_request(session, _error_row(timestamp=_at(1)))

    list_recent_requests(session, limit=1, service="predict", success=False)

    assert not session.new
    assert not session.dirty
    assert not session.deleted
    assert len(session.execute(select(ApiRequest)).scalars().all()) == 2
