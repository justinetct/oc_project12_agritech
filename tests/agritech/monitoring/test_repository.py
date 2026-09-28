"""Tests d'intégration de `insert_api_request` contre une base SQLite jetable.

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
from agritech.monitoring.repository import insert_api_request


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
