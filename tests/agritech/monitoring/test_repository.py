"""Tests d'intégration du repository `api_requests` contre une base SQLite jetable.

Couvre l'écriture (`insert_api_request`), la lecture des requêtes récentes
(`list_recent_requests`) et les agrégats du résumé (`summarize_requests`).

Toutes les fixtures viennent de `conftest.py` : la base vit sous `tmp_path`,
ce qui garantit qu'aucun test ne peut écrire dans la base locale du projet.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from agritech.monitoring import repository
from agritech.monitoring.models import ApiRequest
from agritech.monitoring.repository import (
    insert_api_request,
    list_recent_requests,
    summarize_requests,
)


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


# ===========================================================================
# summarize_requests — agrégats du résumé
# ===========================================================================

# Instant de référence des tests : 2 octobre 2026, 15 h UTC. Avec `days=3`,
# la période couvre le 30/09, le 01/10 et le 02/10 (à partir du 30/09 à 0 h UTC).
NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


def _utc(
    month: int,
    day: int,
    hour: int = 12,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    """Timestamp UTC en 2026, à midi par défaut."""
    return datetime(2026, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)


def _recommend_row(**overrides) -> dict:
    """Ligne `_full_row` du service `/recommend`."""
    return _full_row(service="recommend", endpoint="/recommend", **overrides)


def _service(summary: dict, name: str) -> dict:
    """Entrée du service `name` dans `summary["services"]`."""
    return next(entry for entry in summary["services"] if entry["service"] == name)


def test_summarize_requests_on_empty_database(session: Session):
    """Base vide : compteurs à 0, taux et latences à `None`, jours complétés de zéros."""
    summary = summarize_requests(session, days=3, now=NOW)

    assert summary["total_requests"] == 0
    assert summary["error_count"] == 0
    assert summary["success_rate"] is None
    assert summary["last_request_at"] is None
    assert summary["errors_by_type"] == {}
    assert summary["services"] == [
        {
            "service": name,
            "total_requests": 0,
            "error_count": 0,
            "success_rate": None,
            "latency_ms": {"mean": None, "median": None, "p95": None, "max": None},
        }
        for name in ("predict", "recommend")
    ]
    assert summary["requests_per_day"] == [
        {"date": date(2026, 9, 30), "predict": 0, "recommend": 0},
        {"date": date(2026, 10, 1), "predict": 0, "recommend": 0},
        {"date": date(2026, 10, 2), "predict": 0, "recommend": 0},
    ]


def test_summarize_requests_counts_requests_errors_and_success_rate(session: Session):
    """Totaux globaux et par service, taux de succès entre 0 et 1."""
    for _ in range(3):
        insert_api_request(session, _full_row(timestamp=_utc(10, 1)))
    insert_api_request(session, _error_row(timestamp=_utc(10, 1)))
    for _ in range(2):
        insert_api_request(session, _recommend_row(timestamp=_utc(10, 2)))

    summary = summarize_requests(session, days=3, now=NOW)

    assert summary["total_requests"] == 6
    assert summary["error_count"] == 1
    assert summary["success_rate"] == pytest.approx(5 / 6)

    predict = _service(summary, "predict")
    assert predict["total_requests"] == 4
    assert predict["error_count"] == 1
    assert predict["success_rate"] == pytest.approx(0.75)

    recommend = _service(summary, "recommend")
    assert recommend["total_requests"] == 2
    assert recommend["error_count"] == 0
    assert recommend["success_rate"] == pytest.approx(1.0)


def test_summarize_requests_always_returns_both_services(session: Session):
    """Même si un seul service a été appelé, `predict` et `recommend` sont présents, dans cet ordre."""
    insert_api_request(session, _recommend_row(timestamp=_utc(10, 2)))

    summary = summarize_requests(session, days=3, now=NOW)

    assert [entry["service"] for entry in summary["services"]] == ["predict", "recommend"]
    assert _service(summary, "predict")["total_requests"] == 0
    assert _service(summary, "predict")["success_rate"] is None


def test_summarize_requests_latency_uses_only_successful_requests(session: Session):
    """Moyenne, médiane et max sur les seuls succès : la 422 à 1 ms est ignorée."""
    for duration in (10, 20, 60):
        insert_api_request(session, _full_row(timestamp=_utc(10, 1), duration_ms=duration))
    insert_api_request(session, _error_row(timestamp=_utc(10, 1), duration_ms=1))

    latency = _service(summarize_requests(session, days=3, now=NOW), "predict")["latency_ms"]

    assert latency == {"mean": 30.0, "median": 20.0, "p95": 56.0, "max": 60}


def test_summarize_requests_latency_is_none_without_success(session: Session):
    """Un service qui n'a que des erreurs n'a pas de latence calculée."""
    insert_api_request(
        session,
        _error_row(timestamp=_utc(10, 1), service="recommend", endpoint="/recommend", duration_ms=1),
    )

    recommend = _service(summarize_requests(session, days=3, now=NOW), "recommend")

    assert recommend["total_requests"] == 1
    assert recommend["success_rate"] == 0.0
    assert recommend["latency_ms"] == {"mean": None, "median": None, "p95": None, "max": None}


def test_summarize_requests_counts_errors_by_type(session: Session):
    """Les erreurs sont regroupées par `error_type` ; les succès n'y figurent pas."""
    insert_api_request(session, _full_row(timestamp=_utc(10, 1)))
    insert_api_request(session, _error_row(timestamp=_utc(10, 1)))
    insert_api_request(session, _error_row(timestamp=_utc(10, 2)))
    insert_api_request(
        session,
        _error_row(
            timestamp=_utc(10, 2),
            status_code=500,
            error_type="internal_error",
            error_message="Internal server error.",
        ),
    )

    summary = summarize_requests(session, days=3, now=NOW)

    assert summary["errors_by_type"] == {"internal_error": 1, "validation_error": 2}


def test_summarize_requests_excludes_rows_outside_period(session: Session):
    """Une ligne antérieure au premier jour ou postérieure au jour courant n'est pas comptée."""
    insert_api_request(session, _full_row(timestamp=_utc(10, 1)))
    insert_api_request(session, _full_row(timestamp=_utc(9, 20)))
    insert_api_request(session, _error_row(timestamp=_utc(9, 29)))
    insert_api_request(session, _full_row(timestamp=_utc(10, 3, hour=0)))

    summary = summarize_requests(session, days=3, now=NOW)

    assert summary["total_requests"] == 1
    assert summary["error_count"] == 0
    assert summary["errors_by_type"] == {}
    assert summary["last_request_at"] == _utc(10, 1)
    assert sum(day["predict"] + day["recommend"] for day in summary["requests_per_day"]) == 1


def test_summarize_requests_period_starts_exactly_at_midnight_utc(session: Session):
    """Le 30/09 à 00:00:00 UTC est inclus ; une microseconde avant est exclue."""
    insert_api_request(session, _full_row(timestamp=_utc(9, 30, hour=0)))
    insert_api_request(
        session,
        _full_row(timestamp=_utc(9, 29, hour=23, minute=59, second=59, microsecond=999999)),
    )

    summary = summarize_requests(session, days=3, now=NOW)

    assert summary["total_requests"] == 1
    assert summary["requests_per_day"][0] == {
        "date": date(2026, 9, 30), "predict": 1, "recommend": 0,
    }


def test_summarize_requests_last_request_at_is_latest_timestamp_in_utc(session: Session):
    """`last_request_at` est le timestamp le plus récent de la période, en UTC."""
    insert_api_request(session, _full_row(timestamp=_utc(10, 2, hour=9)))
    insert_api_request(session, _recommend_row(timestamp=_utc(10, 2, hour=14, minute=30)))
    insert_api_request(session, _full_row(timestamp=_utc(10, 1)))

    last_request_at = summarize_requests(session, days=3, now=NOW)["last_request_at"]

    assert last_request_at == _utc(10, 2, hour=14, minute=30)
    assert last_request_at.tzinfo is not None
    assert last_request_at.utcoffset().total_seconds() == 0


def test_summarize_requests_daily_volume_by_service_with_zero_days(session: Session):
    """Volume par jour et par service, jours vides à 0, ordre chronologique."""
    insert_api_request(session, _full_row(timestamp=_utc(9, 30, hour=8)))
    insert_api_request(session, _error_row(timestamp=_utc(9, 30, hour=23, minute=59)))
    insert_api_request(session, _recommend_row(timestamp=_utc(9, 30, hour=10)))
    insert_api_request(session, _recommend_row(timestamp=_utc(10, 2, hour=0)))

    summary = summarize_requests(session, days=3, now=NOW)

    assert summary["requests_per_day"] == [
        {"date": date(2026, 9, 30), "predict": 2, "recommend": 1},
        {"date": date(2026, 10, 1), "predict": 0, "recommend": 0},
        {"date": date(2026, 10, 2), "predict": 0, "recommend": 1},
    ]


def test_summarize_requests_days_one_covers_only_today(session: Session):
    """`days=1` ne couvre que le jour courant, depuis 0 h UTC."""
    insert_api_request(session, _full_row(timestamp=_utc(10, 2, hour=1)))
    insert_api_request(session, _full_row(timestamp=_utc(10, 1, hour=23)))

    summary = summarize_requests(session, days=1, now=NOW)

    assert summary["total_requests"] == 1
    assert summary["requests_per_day"] == [
        {"date": date(2026, 10, 2), "predict": 1, "recommend": 0},
    ]


def test_summarize_requests_does_not_modify_the_database(session: Session):
    """Le résumé ne fait que lire : aucune ligne ajoutée, modifiée ou supprimée."""
    insert_api_request(session, _full_row(timestamp=_utc(10, 1)))

    summarize_requests(session, days=3, now=NOW)

    assert not session.new
    assert not session.dirty
    assert not session.deleted
    assert len(session.execute(select(ApiRequest)).scalars().all()) == 1


class _FrozenDatetime(datetime):
    """`datetime` dont `now()` renvoie toujours `NOW` : horloge figée pour les tests."""

    @classmethod
    def now(cls, tz=None):
        return NOW


def test_summarize_requests_uses_current_utc_time_by_default(
    session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Sans `now`, la période se termine au jour courant donné par l'horloge UTC."""
    monkeypatch.setattr(repository, "datetime", _FrozenDatetime)
    insert_api_request(session, _full_row(timestamp=_utc(10, 2, hour=1)))
    insert_api_request(session, _full_row(timestamp=_utc(10, 1, hour=23)))

    summary = summarize_requests(session, days=1)

    assert summary["total_requests"] == 1
    assert summary["requests_per_day"] == [
        {"date": date(2026, 10, 2), "predict": 1, "recommend": 0},
    ]


# ===========================================================================
# 95e centile des latences
# ===========================================================================


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (list(range(1, 21)), 19.05),  # position 19 × 0,95 = 18,05 → 19 + 0,05 × (20 − 19)
        ([10, 20], 19.5),  # position 0,95 → 10 + 0,95 × (20 − 10)
        ([20, 10, 60], 56.0),  # ordre d'arrivée sans importance
        ([7], 7.0),  # une seule valeur
        ([5, 5, 5], 5.0),
    ],
)
def test_percentile_interpolates_between_neighbour_ranks(values: list[int], expected: float):
    assert repository._percentile(values, 95) == pytest.approx(expected)


@pytest.mark.parametrize("values", [[3], [3, 9], [1, 4, 9, 12], [5, 1, 30, 7, 2, 48, 6]])
def test_percentile_50_is_the_median(values: list[int]):
    """Même convention que `statistics.median` : le 50e centile redonne la médiane."""
    assert repository._percentile(values, 50) == pytest.approx(statistics.median(values))


def test_summarize_requests_p95_ignores_failed_requests(session: Session):
    """Une erreur très lente ne pèse pas sur le p95 des requêtes réussies."""
    for _ in range(3):
        insert_api_request(session, _full_row(timestamp=_utc(10, 1), duration_ms=5))
    insert_api_request(session, _error_row(timestamp=_utc(10, 1), duration_ms=999))

    latency = _service(summarize_requests(session, days=3, now=NOW), "predict")["latency_ms"]

    assert latency["p95"] == 5.0
    assert latency["max"] == 5
