"""Tests de ``agritech.ui.monitoring_view`` (mise en forme du monitoring).

Les fonctions testées sont pures : elles reçoivent des réponses Pydantic
déjà validées et renvoient des valeurs ou des tableaux prêts à afficher.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from agritech.api.schemas.monitoring import (
    MonitoringRequestsResponse,
    MonitoringSummaryResponse,
)
from agritech.ui import monitoring_view as view
from agritech.ui.monitoring_view import Kpi


NNBSP = " "  # espace fine insécable (milliers, %, ms)


def _summary(**overrides) -> MonitoringSummaryResponse:
    """Résumé complet, tel que renvoyé par l'API, surchargé par `overrides`."""
    values = {
        "total_requests": 1234,
        "error_count": 30,
        "success_rate": 0.9756888,
        "last_request_at": "2026-10-01T13:45:06.279869Z",
        "services": [
            {
                "service": "predict",
                "total_requests": 700,
                "error_count": 20,
                "success_rate": 680 / 700,
                "latency_ms": {"mean": 13.321428, "median": 11.0, "max": 48},
            },
            {
                "service": "recommend",
                "total_requests": 534,
                "error_count": 10,
                "success_rate": 524 / 534,
                "latency_ms": {"mean": 7.0, "median": 7.5, "max": 1250},
            },
        ],
        "errors_by_type": {"validation_error": 28, "internal_error": 2},
        "requests_per_day": [
            {"date": "2026-09-30", "predict": 3, "recommend": 5},
            {"date": "2026-10-01", "predict": 0, "recommend": 4},
            {"date": "2026-10-02", "predict": 2, "recommend": 0},
        ],
    }
    values.update(overrides)
    return MonitoringSummaryResponse.model_validate(values)


def _empty_summary() -> MonitoringSummaryResponse:
    """Résumé d'une période sans aucun appel."""
    return _summary(
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
                "latency_ms": {"mean": None, "median": None, "max": None},
            }
            for service in ("predict", "recommend")
        ],
        errors_by_type={},
        requests_per_day=[
            {"date": "2026-10-01", "predict": 0, "recommend": 0},
            {"date": "2026-10-02", "predict": 0, "recommend": 0},
        ],
    )


def _item(**overrides) -> dict:
    """Un appel archivé réussi, au format JSON de l'API."""
    values = {
        "id": 82,
        "timestamp": "2026-10-01T13:45:06.279869Z",
        "service": "recommend",
        "status_code": 200,
        "success": True,
        "duration_ms": 14,
        "model_version": "2.0.0",
        "request_payload": {"iso3": "FRA"},
        "error_type": None,
        "error_message": None,
    }
    values.update(overrides)
    return values


def _requests(*items: dict) -> MonitoringRequestsResponse:
    return MonitoringRequestsResponse.model_validate({"items": list(items)})


# --- Indicateurs -------------------------------------------------------------


def test_summary_kpis_values_and_labels():
    """Quatre indicateurs, dans l'ordre du bandeau, valeurs déjà formatées."""
    assert view.summary_kpis(_summary()) == [
        Kpi("Requêtes", f"1{NNBSP}234"),
        Kpi("Taux de succès", f"97,6{NNBSP}%"),
        Kpi("Erreurs", "30"),
        Kpi("Dernière requête", "2026-10-01 13:45 UTC"),
    ]


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(1.0, f"100,0{NNBSP}%"), (0.0, f"0,0{NNBSP}%"), (0.5, f"50,0{NNBSP}%"), (2 / 3, f"66,7{NNBSP}%")],
)
def test_format_rate_as_percentage(rate: float, expected: str):
    assert view.format_rate(rate) == expected


def test_summary_kpis_without_any_request():
    """Période vide : 0 pour les compteurs, « — » pour le taux et la dernière requête."""
    assert view.summary_kpis(_empty_summary()) == [
        Kpi("Requêtes", "0"),
        Kpi("Taux de succès", "—"),
        Kpi("Erreurs", "0"),
        Kpi("Dernière requête", "—"),
    ]


def test_format_utc_keeps_utc_without_local_conversion():
    """Une date dans un autre fuseau est ramenée en UTC, jamais en heure locale."""
    paris = timezone(timedelta(hours=2))
    moment = datetime(2026, 10, 1, 15, 45, 6, tzinfo=paris)
    assert view.format_utc(moment) == "2026-10-01 13:45 UTC"
    assert view.format_utc(moment, with_seconds=True, with_suffix=False) == "2026-10-01 13:45:06"


def test_format_utc_treats_naive_datetime_as_utc():
    assert view.format_utc(datetime(2026, 10, 1, 13, 45)) == "2026-10-01 13:45 UTC"


# --- Volume quotidien ----------------------------------------------------------


def test_daily_volume_is_long_format_with_two_rows_per_day():
    frame = view.daily_volume_frame(_summary())

    assert list(frame.columns) == ["date", "service", "requests"]
    assert frame.to_dict("records") == [
        {"date": "2026-09-30", "service": "predict", "requests": 3},
        {"date": "2026-09-30", "service": "recommend", "requests": 5},
        {"date": "2026-10-01", "service": "predict", "requests": 0},
        {"date": "2026-10-01", "service": "recommend", "requests": 4},
        {"date": "2026-10-02", "service": "predict", "requests": 2},
        {"date": "2026-10-02", "service": "recommend", "requests": 0},
    ]


def test_daily_volume_keeps_zero_days_in_chronological_order():
    """Une période sans appel garde tous ses jours, à zéro, dans l'ordre."""
    frame = view.daily_volume_frame(_empty_summary())

    assert frame["date"].tolist() == ["2026-10-01", "2026-10-01", "2026-10-02", "2026-10-02"]
    assert frame["service"].tolist() == ["predict", "recommend", "predict", "recommend"]
    assert frame["requests"].tolist() == [0, 0, 0, 0]
    assert frame["date"].tolist() == sorted(frame["date"].tolist())


def test_daily_volume_dates_are_iso_strings_and_counts_integers():
    frame = view.daily_volume_frame(_summary())

    assert all(isinstance(value, str) and len(value) == 10 for value in frame["date"])
    assert pd.api.types.is_integer_dtype(frame["requests"])


def test_daily_volume_without_days_is_an_empty_frame_with_columns():
    frame = view.daily_volume_frame(_summary(requests_per_day=[]))

    assert frame.empty
    assert list(frame.columns) == ["date", "service", "requests"]


# --- Tableau par service -------------------------------------------------------


def test_services_frame_has_one_formatted_row_per_service():
    frame = view.services_frame(_summary())

    assert list(frame.columns) == view.SERVICES_COLUMNS
    assert frame.to_dict("records") == [
        {
            "Service": "predict",
            "Requêtes": "700",
            "Erreurs": "20",
            "Taux de succès": f"97,1{NNBSP}%",
            "Latence moyenne": f"13,3{NNBSP}ms",
            "Latence médiane": f"11,0{NNBSP}ms",
            "Latence max": f"48,0{NNBSP}ms",
        },
        {
            "Service": "recommend",
            "Requêtes": "534",
            "Erreurs": "10",
            "Taux de succès": f"98,1{NNBSP}%",
            "Latence moyenne": f"7,0{NNBSP}ms",
            "Latence médiane": f"7,5{NNBSP}ms",
            "Latence max": f"1{NNBSP}250,0{NNBSP}ms",
        },
    ]


def test_services_frame_shows_dash_when_no_measure_exists():
    """Sans appel réussi, les latences et le taux s'affichent « — », jamais « 0 ms »."""
    frame = view.services_frame(_empty_summary())

    for row in frame.to_dict("records"):
        assert row["Requêtes"] == "0"
        assert row["Taux de succès"] == "—"
        assert row["Latence moyenne"] == "—"
        assert row["Latence médiane"] == "—"
        assert row["Latence max"] == "—"


def test_services_frame_zero_latency_is_not_confused_with_missing():
    """Une vraie mesure à 0 reste « 0,0 ms » : seul `None` devient « — »."""
    summary = _summary()
    summary.services[0].latency_ms.max = 0
    row = view.services_frame(summary).iloc[0]

    assert row["Latence max"] == f"0,0{NNBSP}ms"


# --- Erreurs par type ----------------------------------------------------------


def test_errors_frame_sorted_by_count_descending():
    frame = view.errors_frame(_summary())

    assert list(frame.columns) == ["Type d'erreur", "Nombre"]
    assert frame.to_dict("records") == [
        {"Type d'erreur": "validation_error", "Nombre": 28},
        {"Type d'erreur": "internal_error", "Nombre": 2},
    ]


def test_errors_frame_ties_are_sorted_by_name():
    summary = _summary(
        errors_by_type={"validation_error": 3, "internal_error": 3, "model_unavailable": 5}
    )

    assert view.errors_frame(summary)["Type d'erreur"].tolist() == [
        "model_unavailable",
        "internal_error",
        "validation_error",
    ]


def test_errors_frame_without_error_is_empty():
    frame = view.errors_frame(_empty_summary())

    assert frame.empty
    assert list(frame.columns) == ["Type d'erreur", "Nombre"]


# --- Requêtes récentes ---------------------------------------------------------


def test_recent_requests_success_row():
    frame = view.recent_requests_frame(_requests(_item()))

    assert list(frame.columns) == view.REQUESTS_COLUMNS
    assert frame.to_dict("records") == [
        {
            "Date (UTC)": "2026-10-01 13:45:06",
            "Service": "recommend",
            "Statut": 200,
            "Durée": f"14{NNBSP}ms",
            "Modèle": "2.0.0",
            "Erreur": "—",
            "Entrées": '{"iso3":"FRA"}',
        }
    ]


def test_recent_requests_422_row_shows_short_error_and_invalid_input():
    item = _item(
        service="predict",
        status_code=422,
        success=False,
        duration_ms=1,
        model_version="1.0.0",
        request_payload={"rainfall_mm": -1, "temperature_celsius": 25.0},
        error_type="validation_error",
        error_message="Request payload is invalid.",
    )
    row = view.recent_requests_frame(_requests(item)).iloc[0]

    assert row["Statut"] == 422
    assert row["Erreur"] == "validation_error"
    assert row["Entrées"] == '{"rainfall_mm":-1,"temperature_celsius":25.0}'


def test_recent_requests_error_falls_back_to_message_then_dash():
    message_only = _item(status_code=500, success=False, error_message="Internal server error.")
    nothing = _item(status_code=500, success=False)
    frame = view.recent_requests_frame(_requests(message_only, nothing))

    assert frame["Erreur"].tolist() == ["Internal server error.", "—"]


def test_recent_requests_missing_model_version_is_dash():
    row = view.recent_requests_frame(_requests(_item(model_version=None))).iloc[0]
    assert row["Modèle"] == "—"


def test_recent_requests_keep_received_order():
    frame = view.recent_requests_frame(
        _requests(
            _item(id=3, timestamp="2026-10-02T09:00:00Z"),
            _item(id=2, timestamp="2026-10-01T09:00:00Z"),
        )
    )
    assert frame["Date (UTC)"].tolist() == ["2026-10-02 09:00:00", "2026-10-01 09:00:00"]


def test_recent_requests_without_items_is_an_empty_frame():
    frame = view.recent_requests_frame(_requests())

    assert frame.empty
    assert list(frame.columns) == view.REQUESTS_COLUMNS


def test_recent_requests_timestamps_are_rendered_in_utc():
    """Un timestamp reçu avec un décalage horaire est affiché en UTC."""
    row = view.recent_requests_frame(_requests(_item(timestamp="2026-10-01T15:45:06+02:00"))).iloc[0]
    assert row["Date (UTC)"] == "2026-10-01 13:45:06"


# --- JSON compact --------------------------------------------------------------


def test_compact_json_is_single_line_without_spaces():
    payload = {"iso3": "FRA", "conditions": {"annual_rainfall_mm": 867.0}}
    assert view.compact_json(payload) == '{"iso3":"FRA","conditions":{"annual_rainfall_mm":867.0}}'


def test_compact_json_keeps_non_ascii_characters():
    assert view.compact_json({"pays": "Côte d’Ivoire"}) == '{"pays":"Côte d’Ivoire"}'


def test_compact_json_is_deterministic_and_keeps_field_order():
    """Même payload → même texte ; l'ordre des champs envoyés est conservé."""
    payload = {"temperature_celsius": 25.0, "rainfall_mm": 500.0}
    first = view.compact_json(payload)

    assert first == view.compact_json(dict(payload))
    assert first == '{"temperature_celsius":25.0,"rainfall_mm":500.0}'
    assert json.loads(first) == payload


def test_compact_json_accepts_non_object_payloads():
    """Corps invalides archivés pour une 422 : liste ou valeur seule."""
    assert view.compact_json([1, 2]) == "[1,2]"
    assert view.compact_json(None) == "null"


# --- Immutabilité --------------------------------------------------------------


def test_transformations_do_not_modify_received_objects():
    summary = _summary()
    payload = {"iso3": "FRA", "conditions": {"annual_rainfall_mm": 867.0}}
    requests_response = _requests(_item(request_payload=payload))
    summary_before = summary.model_dump()
    requests_before = requests_response.model_dump()

    view.summary_kpis(summary)
    view.daily_volume_frame(summary)
    view.services_frame(summary)
    view.errors_frame(summary)
    view.recent_requests_frame(requests_response)

    assert summary.model_dump() == summary_before
    assert requests_response.model_dump() == requests_before
    assert requests_response.items[0].request_payload == {
        "iso3": "FRA",
        "conditions": {"annual_rainfall_mm": 867.0},
    }
