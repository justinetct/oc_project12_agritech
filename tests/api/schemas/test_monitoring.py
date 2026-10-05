"""Tests des schémas Pydantic du monitoring : résumé et requêtes récentes.

Ces tests vérifient les modèles seuls (validation et sérialisation JSON),
sans passer par FastAPI : les endpoints `/monitoring/*` n'existent pas encore.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import get_args

import pytest
from pydantic import ValidationError

from agritech.api.schemas.monitoring import (
    LatencySummary,
    MonitoringRequestItem,
    MonitoringRequestsResponse,
    MonitoringService,
    MonitoringSummaryResponse,
    ServiceSummary,
)
from agritech.monitoring.config import MonitoringConfig
from agritech.monitoring.models import ApiRequest, Base
from agritech.monitoring.repository import (
    MONITORED_SERVICES,
    insert_api_request,
    summarize_requests,
)
from agritech.monitoring.session import create_monitoring_engine, create_session_factory


NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)

PREDICT_PAYLOAD = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}
RECOMMEND_PAYLOAD = {
    "iso3": "FRA",
    "conditions": {
        "average_temperature_celsius": 12.5,
        "annual_rainfall_mm": 867.0,
        "average_annual_pesticides_tons": None,
    },
}

# Champs de `ApiRequest` qui ne doivent jamais sortir dans la vue publique.
PRIVATE_FIELDS = (
    "response_payload",
    "logfire_trace_id",
    "environment",
    "api_version",
    "endpoint",
    "method",
)


def _empty_latency() -> dict:
    return {"mean": None, "median": None, "max": None}


def _full_summary() -> dict:
    """Dict ayant la forme exacte renvoyée par `summarize_requests`."""
    return {
        "total_requests": 7,
        "error_count": 1,
        "success_rate": 6 / 7,
        "last_request_at": datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc),
        "services": [
            {
                "service": "predict",
                "total_requests": 4,
                "error_count": 1,
                "success_rate": 0.75,
                "latency_ms": {"mean": 30.0, "median": 20.0, "max": 60},
            },
            {
                "service": "recommend",
                "total_requests": 3,
                "error_count": 0,
                "success_rate": 1.0,
                "latency_ms": {"mean": 23.0, "median": 15.0, "max": 40},
            },
        ],
        "errors_by_type": {"validation_error": 1},
        "requests_per_day": [
            {"date": date(2026, 9, 30), "predict": 3, "recommend": 2, "errors": 1},
            {"date": date(2026, 10, 1), "predict": 0, "recommend": 0, "errors": 0},
            {"date": date(2026, 10, 2), "predict": 1, "recommend": 1, "errors": 0},
        ],
    }


def _row_values(**overrides) -> dict:
    """Valeurs complètes d'une ligne `api_requests`, surchargées par `overrides`."""
    values = {
        "id": 82,
        "timestamp": datetime(2026, 10, 1, 13, 45, 6, 279869, tzinfo=timezone.utc),
        "service": "predict",
        "endpoint": "/predict",
        "method": "POST",
        "status_code": 200,
        "success": True,
        "duration_ms": 14,
        "api_version": "1.0.0",
        "model_version": "1.0.0",
        "request_payload": PREDICT_PAYLOAD,
        "response_payload": {"yield_tons_per_hectare": 4.5, "unit": "t/ha"},
        "error_type": None,
        "error_message": None,
        "logfire_trace_id": "0123456789abcdef0123456789abcdef",
        "environment": "prod",
    }
    values.update(overrides)
    return values


def _api_request(**overrides) -> ApiRequest:
    """Objet ORM `ApiRequest` complet, non persisté."""
    return ApiRequest(**_row_values(**overrides))


# --- Alignement avec le repository ---


def test_monitoring_service_literal_matches_repository_services():
    """Le `Literal` des schémas reste aligné avec `MONITORED_SERVICES` du repository."""
    assert get_args(MonitoringService) == MONITORED_SERVICES


def test_summary_accepts_real_summarize_requests_output(tmp_path):
    """La sortie réelle de `summarize_requests` se valide telle quelle."""
    config = MonitoringConfig(
        database_url=f"sqlite:///{tmp_path / 'schemas.sqlite'}",
        environment="test",
        logfire_token=None,
        logfire_environment="test",
        logfire_service_name="agritech-answers",
        api_token=None,
    )
    engine = create_monitoring_engine(config)
    Base.metadata.create_all(engine)
    try:
        with create_session_factory(engine)() as session:
            insert_api_request(session, _row_values(timestamp=NOW))
            summary = summarize_requests(session, days=3, now=NOW)
    finally:
        engine.dispose()

    response = MonitoringSummaryResponse(**summary)

    assert response.total_requests == 1
    assert [entry.service for entry in response.services] == ["predict", "recommend"]
    assert len(response.requests_per_day) == 3


# --- MonitoringSummaryResponse ---


def test_latency_summary_exposes_only_mean_median_and_max():
    """Contrat V1 des latences : `mean`, `median` et `max`, rien d'autre."""
    assert set(LatencySummary.model_fields) == {"mean", "median", "max"}


def test_full_summary_is_valid():
    """Un résumé complet se valide et conserve toutes les valeurs."""
    response = MonitoringSummaryResponse(**_full_summary())

    assert response.total_requests == 7
    assert response.error_count == 1
    assert response.success_rate == pytest.approx(6 / 7)
    assert response.services[0].latency_ms.mean == 30.0
    assert response.services[1].latency_ms.max == 40
    assert response.errors_by_type == {"validation_error": 1}
    assert response.requests_per_day[1].predict == 0


def test_empty_summary_accepts_none_values():
    """Résumé vide : taux, dernier appel et latences à `None`, aucune erreur."""
    response = MonitoringSummaryResponse(
        total_requests=0,
        error_count=0,
        success_rate=None,
        last_request_at=None,
        services=[
            {
                "service": service,
                "total_requests": 0,
                "error_count": 0,
                "success_rate": None,
                "latency_ms": _empty_latency(),
            }
            for service in ("predict", "recommend")
        ],
        errors_by_type={},
        requests_per_day=[{"date": date(2026, 10, 2), "predict": 0, "recommend": 0, "errors": 0}],
    )

    body = response.model_dump(mode="json")
    assert body["success_rate"] is None
    assert body["last_request_at"] is None
    assert body["services"][0]["latency_ms"] == _empty_latency()
    assert body["errors_by_type"] == {}


def test_summary_serializes_dates_and_datetimes_in_iso_format():
    """`date` devient `YYYY-MM-DD` et `last_request_at` une date-heure ISO UTC."""
    body = MonitoringSummaryResponse(**_full_summary()).model_dump(mode="json")

    assert body["last_request_at"] == "2026-10-02T14:00:00Z"
    assert [day["date"] for day in body["requests_per_day"]] == [
        "2026-09-30",
        "2026-10-01",
        "2026-10-02",
    ]


def test_service_summary_rejects_unknown_service():
    """Seuls `predict` et `recommend` sont des services valides."""
    with pytest.raises(ValidationError):
        ServiceSummary(
            service="other",
            total_requests=0,
            error_count=0,
            success_rate=None,
            latency_ms=_empty_latency(),
        )


# --- MonitoringRequestItem / MonitoringRequestsResponse ---


def test_request_item_is_built_from_orm_object():
    """Un `ApiRequest` ORM se convertit directement avec `model_validate`."""
    item = MonitoringRequestItem.model_validate(_api_request())

    assert item.id == 82
    assert item.service == "predict"
    assert item.status_code == 200
    assert item.success is True
    assert item.duration_ms == 14
    assert item.model_version == "1.0.0"
    assert item.request_payload == PREDICT_PAYLOAD


def test_request_item_serialization_hides_private_fields():
    """La sérialisation publique ne contient que les 10 champs prévus."""
    body = MonitoringRequestItem.model_validate(_api_request()).model_dump(mode="json")

    assert set(body) == {
        "id",
        "timestamp",
        "service",
        "status_code",
        "success",
        "duration_ms",
        "model_version",
        "request_payload",
        "error_type",
        "error_message",
    }
    for field in PRIVATE_FIELDS:
        assert field not in body
    assert body["timestamp"] == "2026-10-01T13:45:06.279869Z"


def test_request_item_accepts_recommend_payload():
    """Le payload `/recommend` (avec bloc `conditions` imbriqué) est conservé tel quel."""
    item = MonitoringRequestItem.model_validate(
        _api_request(service="recommend", endpoint="/recommend", request_payload=RECOMMEND_PAYLOAD)
    )

    assert item.request_payload == RECOMMEND_PAYLOAD


def test_request_item_accepts_any_json_body_archived_for_a_422():
    """Un corps invalide archivé pour une 422 (ici une liste JSON) reste représentable."""
    item = MonitoringRequestItem.model_validate(
        _api_request(
            status_code=422,
            success=False,
            request_payload=[1, 2],
            error_type="validation_error",
            error_message="Request payload is invalid.",
        )
    )

    assert item.request_payload == [1, 2]
    assert item.error_type == "validation_error"
    assert item.error_message == "Request payload is invalid."


def test_request_item_optional_fields_can_be_none():
    """`model_version`, `error_type` et `error_message` acceptent `None`."""
    body = MonitoringRequestItem.model_validate(
        _api_request(model_version=None, error_type=None, error_message=None)
    ).model_dump(mode="json")

    assert body["model_version"] is None
    assert body["error_type"] is None
    assert body["error_message"] is None


def test_request_item_rejects_unknown_service():
    """Une ligne dont le service n'est pas suivi n'est pas un item valide."""
    with pytest.raises(ValidationError):
        MonitoringRequestItem.model_validate(_api_request(service="other"))


def test_requests_response_wraps_items_in_order():
    """La réponse liste les items dans `items`, sans changer leur ordre."""
    rows = [_api_request(id=2), _api_request(id=1, service="recommend", endpoint="/recommend")]

    response = MonitoringRequestsResponse(
        items=[MonitoringRequestItem.model_validate(row) for row in rows]
    )

    body = response.model_dump(mode="json")
    assert list(body) == ["items"]
    assert [item["id"] for item in body["items"]] == [2, 1]


def test_requests_response_accepts_empty_list():
    """Sans aucun appel, la réponse contient une liste vide."""
    assert MonitoringRequestsResponse(items=[]).model_dump(mode="json") == {"items": []}
