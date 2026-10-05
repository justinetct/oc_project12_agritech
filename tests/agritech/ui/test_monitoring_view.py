"""Tests de ``agritech.ui.monitoring_view`` (mise en forme du monitoring).

Les fonctions testées sont pures : elles reçoivent des réponses Pydantic
déjà validées et renvoient des valeurs ou des tableaux prêts à afficher.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

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
            {"date": "2026-09-30", "predict": 3, "recommend": 5, "errors": 1},
            {"date": "2026-10-01", "predict": 0, "recommend": 4, "errors": 0},
            {"date": "2026-10-02", "predict": 2, "recommend": 0, "errors": 1},
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
            {"date": "2026-10-01", "predict": 0, "recommend": 0, "errors": 0},
            {"date": "2026-10-02", "predict": 0, "recommend": 0, "errors": 0},
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
        Kpi("Requêtes réussies", f"97,6{NNBSP}%"),
        Kpi("Erreurs", "30"),
        Kpi("Dernière requête", "01 oct. 2026 · 15:45"),
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
        Kpi("Requêtes réussies", "—"),
        Kpi("Erreurs", "0"),
        Kpi("Dernière requête", "—"),
    ]


def test_empty_kpis_keep_labels_without_values():
    """Résumé indisponible : mêmes libellés, toutes les valeurs à « — »."""
    assert view.empty_kpis() == [
        Kpi("Requêtes", "—"),
        Kpi("Requêtes réussies", "—"),
        Kpi("Erreurs", "—"),
        Kpi("Dernière requête", "—"),
    ]
    assert [kpi.label for kpi in view.empty_kpis()] == [
        kpi.label for kpi in view.summary_kpis(_summary())
    ]


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        # Heure d'été (UTC+2) et heure d'hiver (UTC+1).
        (datetime(2026, 7, 14, 13, 45, 6, tzinfo=timezone.utc), "2026-07-14 15:45:06"),
        (datetime(2026, 12, 25, 13, 45, 6, tzinfo=timezone.utc), "2026-12-25 14:45:06"),
        # Changements d'heure 2026 : 29 mars et 25 octobre à 01:00 UTC.
        (datetime(2026, 3, 29, 0, 59, 59, tzinfo=timezone.utc), "2026-03-29 01:59:59"),
        (datetime(2026, 3, 29, 1, 0, 0, tzinfo=timezone.utc), "2026-03-29 03:00:00"),
        (datetime(2026, 10, 25, 0, 59, 59, tzinfo=timezone.utc), "2026-10-25 02:59:59"),
        (datetime(2026, 10, 25, 1, 0, 0, tzinfo=timezone.utc), "2026-10-25 02:00:00"),
        # Date sans fuseau : lue comme UTC.
        (datetime(2026, 10, 1, 13, 45, 6), "2026-10-01 15:45:06"),
    ],
)
def test_format_datetime_shows_paris_time_with_daylight_saving(moment: datetime, expected: str):
    assert view.format_datetime(moment, with_seconds=True) == expected


def test_format_datetime_without_seconds_and_missing_value():
    assert view.format_datetime(datetime(2026, 10, 1, 13, 45, tzinfo=timezone.utc)) == "2026-10-01 15:45"
    assert view.format_datetime(None) == "—"


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (datetime(2026, 10, 5, 8, 51, 30, tzinfo=timezone.utc), "05 oct. 2026 · 10:51"),
        (datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), "01 janv. 2026 · 01:00"),
        (datetime(2026, 2, 14, 9, 5, tzinfo=timezone.utc), "14 févr. 2026 · 10:05"),
        (datetime(2026, 8, 31, 23, 59, tzinfo=timezone.utc), "01 sept. 2026 · 01:59"),
        (datetime(2026, 12, 25, 12, 0, tzinfo=timezone.utc), "25 déc. 2026 · 13:00"),
    ],
)
def test_format_readable_datetime_in_french_paris_time(moment: datetime, expected: str):
    """Mois en français ; le jour suit l'heure de Paris (31 août 23:59 UTC → 1er septembre)."""
    assert view.format_readable_datetime(moment) == expected


def test_format_readable_datetime_handles_other_zones_and_missing_value():
    new_york = timezone(timedelta(hours=-4))
    assert view.format_readable_datetime(datetime(2026, 10, 5, 4, 51, tzinfo=new_york)) == (
        "05 oct. 2026 · 10:51"
    )
    assert view.format_readable_datetime(None) == "—"


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


# --- Indicateurs par service ---------------------------------------------------


def test_service_kpis_give_five_formatted_values_per_service():
    """Requêtes, réussies, erreurs, latences médiane et max, pour predict puis recommend."""
    kpis = view.service_kpis(_summary())

    assert list(kpis) == ["predict", "recommend"]
    assert kpis["predict"] == [
        Kpi("Requêtes", "700"),
        Kpi("Réussies", f"97,1{NNBSP}%"),
        Kpi("Erreurs", "20"),
        Kpi("Latence médiane", f"11{NNBSP}ms"),
        Kpi("Latence max", f"48{NNBSP}ms"),
    ]
    assert kpis["recommend"] == [
        Kpi("Requêtes", "534"),
        Kpi("Réussies", f"98,1{NNBSP}%"),
        Kpi("Erreurs", "10"),
        Kpi("Latence médiane", f"7,5{NNBSP}ms"),
        Kpi("Latence max", f"1{NNBSP}250{NNBSP}ms"),
    ]


def test_service_kpis_without_any_call_show_zero_and_dash():
    """Service sans appel : 0 pour les compteurs, « — » pour la réussite et les latences."""
    for kpis in view.service_kpis(_empty_summary()).values():
        assert [kpi.value for kpi in kpis] == ["0", "—", "0", "—", "—"]


def test_service_kpis_missing_service_stays_empty():
    """Un service absent du résumé garde ses libellés, avec « — » partout."""
    summary = _summary()
    summary.services = [entry for entry in summary.services if entry.service == "predict"]

    assert view.service_kpis(summary)["recommend"] == view.empty_service_kpis()["recommend"]


def test_empty_service_kpis_keep_labels_without_values():
    empty = view.empty_service_kpis()

    assert list(empty) == ["predict", "recommend"]
    for kpis in empty.values():
        assert [kpi.label for kpi in kpis] == list(view.SERVICE_KPI_LABELS)
        assert all(kpi.value == "—" for kpi in kpis)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12.0, f"12{NNBSP}ms"), (5.5, f"5,5{NNBSP}ms"), (40.44, f"40,4{NNBSP}ms"), (0.0, f"0{NNBSP}ms"), (None, "—")],
)
def test_format_latency(value, expected: str):
    """Une décimale sans « ,0 » inutile ; une vraie mesure à 0 n'est pas confondue avec « — »."""
    assert view.format_latency(value) == expected


# --- Erreurs quotidiennes --------------------------------------------------------


def test_daily_errors_has_one_row_per_day_with_zero_days():
    frame = view.daily_errors_frame(_summary())

    assert list(frame.columns) == ["date", "errors"]
    assert frame.to_dict("records") == [
        {"date": "2026-09-30", "errors": 1},
        {"date": "2026-10-01", "errors": 0},
        {"date": "2026-10-02", "errors": 1},
    ]
    assert pd.api.types.is_integer_dtype(frame["errors"])


def test_daily_errors_of_empty_period_are_all_zero():
    assert view.daily_errors_frame(_empty_summary())["errors"].tolist() == [0, 0]


def test_daily_errors_without_days_is_an_empty_frame_with_columns():
    frame = view.daily_errors_frame(_summary(requests_per_day=[]))

    assert frame.empty
    assert list(frame.columns) == ["date", "errors"]


# --- Étiquettes de l'axe des dates -----------------------------------------------


def _days(count: int) -> list[str]:
    """`count` jours consécutifs au format ISO, à partir du 2026-07-08."""
    first = date(2026, 7, 8)
    return [(first + timedelta(days=offset)).isoformat() for offset in range(count)]


def test_date_axis_labels_every_day_over_7_days():
    days = _days(7)
    assert view.date_axis_labels(days) == days


def test_date_axis_labels_every_3_days_over_30_days_with_first_and_last():
    days = _days(30)
    labels = view.date_axis_labels(days)

    assert labels == [days[i] for i in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29)]


def test_date_axis_labels_every_week_over_90_days_with_first_and_last():
    days = _days(90)
    labels = view.date_axis_labels(days)

    assert labels == [days[i] for i in range(0, 85, 7)] + [days[89]]


def test_date_axis_labels_drop_the_label_too_close_to_the_last_day():
    """Le dernier jour, toujours affiché, remplace l'étiquette trop proche de lui."""
    days = _days(29)  # écart 3 : 0, 3, ..., 27, puis 28 à 1 jour seulement de 27
    labels = view.date_axis_labels(days)

    assert labels[-2:] == [days[24], days[28]]


def test_date_axis_labels_count_each_day_once():
    """Le volume quotidien a une ligne par service : chaque jour ne compte qu'une fois."""
    days = _days(7)
    repeated = [day for day in days for _ in ("predict", "recommend")]

    assert view.date_axis_labels(repeated) == days


@pytest.mark.parametrize("count, expected", [(1, 1), (7, 1), (8, 3), (31, 3), (32, 7), (365, 7)])
def test_date_label_step_depends_on_period_length(count: int, expected: int):
    assert view.date_label_step(count) == expected


def test_date_axis_labels_without_days_is_empty():
    assert view.date_axis_labels([]) == []


# --- Dernières erreurs et derniers succès ---------------------------------------


def _error_item(**overrides) -> dict:
    values = {
        "service": "predict",
        "status_code": 422,
        "success": False,
        "duration_ms": 1,
        "model_version": "1.0.0",
        "request_payload": {"rainfall_mm": -1, "temperature_celsius": 25.0},
        "error_type": "validation_error",
        "error_message": "Request payload is invalid.",
    }
    values.update(overrides)
    return _item(**values)


def test_recent_errors_row_shows_status_error_message_and_input():
    frame = view.recent_errors_frame(_requests(_error_item()))

    assert list(frame.columns) == view.RECENT_ERRORS_COLUMNS
    assert frame.to_dict("records") == [
        {
            "Date": "2026-10-01 15:45:06",
            "Service": "predict",
            "Statut": 422,
            "Erreur": "validation_error",
            "Message": "Request payload is invalid.",
            "Durée": f"1{NNBSP}ms",
            "Entrées": '{"rainfall_mm":-1,"temperature_celsius":25.0}',
        }
    ]


def test_recent_errors_missing_type_or_message_is_dash():
    row = view.recent_errors_frame(
        _requests(_error_item(status_code=500, error_type=None, error_message=None))
    ).iloc[0]

    assert row["Erreur"] == "—"
    assert row["Message"] == "—"


def test_recent_successes_row_shows_duration_model_and_input():
    frame = view.recent_successes_frame(_requests(_item()))

    assert list(frame.columns) == view.RECENT_SUCCESSES_COLUMNS
    assert frame.to_dict("records") == [
        {
            "Date": "2026-10-01 15:45:06",
            "Service": "recommend",
            "Durée": f"14{NNBSP}ms",
            "Modèle": "2.0.0",
            "Entrées": '{"iso3":"FRA"}',
        }
    ]


def test_recent_successes_missing_model_version_is_dash():
    row = view.recent_successes_frame(_requests(_item(model_version=None))).iloc[0]
    assert row["Modèle"] == "—"


@pytest.mark.parametrize("build", [view.recent_errors_frame, view.recent_successes_frame])
def test_recent_tables_keep_received_order_and_render_paris_time(build):
    frame = build(
        _requests(
            _item(id=3, timestamp="2026-10-02T11:00:00+02:00"),
            _item(id=2, timestamp="2026-10-01T09:00:00Z"),
        )
    )
    assert frame["Date"].tolist() == ["2026-10-02 11:00:00", "2026-10-01 11:00:00"]


@pytest.mark.parametrize(
    ("build", "columns"),
    [
        (view.recent_errors_frame, view.RECENT_ERRORS_COLUMNS),
        (view.recent_successes_frame, view.RECENT_SUCCESSES_COLUMNS),
    ],
)
def test_recent_tables_without_items_are_empty_frames(build, columns):
    frame = build(_requests())

    assert frame.empty
    assert list(frame.columns) == columns


def test_recent_tables_never_show_private_fields():
    """Seules les colonnes prévues sortent : rien sur la réponse, la trace ou le déploiement."""
    errors = view.recent_errors_frame(_requests(_error_item()))
    successes = view.recent_successes_frame(_requests(_item()))

    for frame in (errors, successes):
        text = frame.to_csv()
        for private in ("response_payload", "logfire_trace_id", "environment", "Authorization"):
            assert private not in text


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
    view.service_kpis(summary)
    view.daily_errors_frame(summary)
    view.recent_errors_frame(requests_response)
    view.recent_successes_frame(requests_response)

    assert summary.model_dump() == summary_before
    assert requests_response.model_dump() == requests_before
    assert requests_response.items[0].request_payload == {
        "iso3": "FRA",
        "conditions": {"annual_rainfall_mm": 867.0},
    }
