"""Tests du script d'historique de démonstration (`agritech.monitoring.seed_history`).

Aucune écriture dans la vraie base : la fixture racine `_isolate_monitoring_database`
pointe `DATABASE_URL` vers un fichier SQLite sous `tmp_path`, et les tests
d'insertion utilisent la base jetable de `conftest.py`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from agritech.api.main import API_VERSION
from agritech.api.schemas.predict import PredictRequest
from agritech.api.schemas.recommend import RecommendRequest
from agritech.monitoring.models import ApiRequest
from agritech.monitoring.repository import insert_api_request, summarize_requests
from agritech.monitoring.session import create_session_factory
from agritech.monitoring import seed_history as seed


END = datetime(2026, 10, 5, 8, 51, 58, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def catalog() -> seed.Catalog:
    return seed.load_catalog("local")


@pytest.fixture(scope="module")
def rows(catalog: seed.Catalog) -> list[dict]:
    return seed.generate_history(catalog, days=90, end=END, seed=42)


def _database_path() -> Path:
    return Path(make_url(os.environ["DATABASE_URL"]).database)


# --- Génération -------------------------------------------------------------------


def test_same_seed_gives_the_same_history(catalog, rows):
    assert seed.generate_history(catalog, days=90, end=END, seed=42) == rows


def test_another_seed_gives_another_history(catalog, rows):
    assert seed.generate_history(catalog, days=90, end=END, seed=7) != rows


def test_history_covers_the_period_in_utc_before_the_end(rows):
    first_day = (END - timedelta(days=89)).date()

    assert all(row["timestamp"].tzinfo == timezone.utc for row in rows)
    assert all(first_day <= row["timestamp"].date() <= END.date() for row in rows)
    assert all(row["timestamp"] < END for row in rows)
    assert rows[0]["timestamp"].date() == first_day
    assert [row["timestamp"] for row in rows] == sorted(row["timestamp"] for row in rows)


def test_timestamps_are_spread_over_the_day(rows):
    hours = {row["timestamp"].hour for row in rows}
    minutes = {row["timestamp"].minute for row in rows}

    assert len(hours) >= 18
    assert len(minutes) == 60
    assert len({row["timestamp"] for row in rows}) == len(rows)


def test_volume_is_realistic_with_quieter_weekends(rows):
    summary = seed.summarize(rows)

    assert 1500 <= summary["total"] <= 2500
    assert summary["weekend_mean"] < summary["weekday_mean"] / 2


def test_both_services_with_predict_slightly_ahead(rows):
    summary = seed.summarize(rows)
    predict_share = summary["predict"] / summary["total"]

    assert summary["recommend"] > 0
    assert 0.50 <= predict_share <= 0.60


@pytest.mark.parametrize("seed_value", [42, 1, 2, 3, 7])
def test_error_rate_stays_between_2_and_6_percent(catalog, seed_value):
    summary = seed.summarize(seed.generate_history(catalog, days=90, end=END, seed=seed_value))

    assert 0.02 <= summary["error_rate"] <= 0.06


def test_errors_are_only_422_validation_errors(rows):
    errors = [row for row in rows if not row["success"]]

    assert errors
    assert {row["status_code"] for row in errors} == {422}
    assert {row["error_type"] for row in errors} == {"validation_error"}
    # Même message que les lignes réelles : les champs refusés, pas la phrase générique.
    for row in errors:
        refused = [detail["field"].rsplit(".", 1)[-1] for detail in row["response_payload"]["details"]]
        assert all(f"{field}: " in row["error_message"] for field in refused)
    assert {row["status_code"] for row in rows if row["success"]} == {200}


def test_every_error_payload_is_really_rejected_by_the_api_schema(rows):
    """Chaque ligne en erreur correspond à un corps que le contrat rejette vraiment."""
    models = {"predict": PredictRequest, "recommend": RecommendRequest}
    for row in rows:
        model = models[row["service"]]
        if row["success"]:
            model.model_validate(row["request_payload"])
        else:
            with pytest.raises(ValueError):
                model.model_validate(row["request_payload"])
            assert row["response_payload"]["details"]
            assert all(d["field"].startswith("body") for d in row["response_payload"]["details"])


def test_validation_error_response_refuses_a_valid_payload():
    valid = {"rainfall_mm": 500.0, "temperature_celsius": 25.0, "fertilizer_used": True, "irrigation_used": False}

    with pytest.raises(ValueError):
        seed.validation_error_response(PredictRequest, valid)


def test_latencies_stay_in_expected_ranges(rows):
    for row in rows:
        if not row["success"]:
            assert 1 <= row["duration_ms"] <= 3
        elif row["service"] == "predict":
            assert 4 <= row["duration_ms"] <= 50
        else:
            assert 8 <= row["duration_ms"] <= 60

    predict = [r["duration_ms"] for r in rows if r["service"] == "predict" and r["success"]]
    recommend = [r["duration_ms"] for r in rows if r["service"] == "recommend" and r["success"]]
    assert mean(predict) < 15 and mean(recommend) < 22
    assert len(set(predict)) > 10 and len(set(recommend)) > 10


def test_predict_successes_mostly_inside_training_domain(rows):
    payloads = [r["request_payload"] for r in rows if r["service"] == "predict" and r["success"]]
    inside = [p for p in payloads if 100 <= p["rainfall_mm"] <= 1000 and 15 <= p["temperature_celsius"] <= 40]

    assert len(inside) / len(payloads) > 0.9


def test_recommend_uses_many_real_countries(catalog, rows):
    countries = {r["request_payload"]["iso3"] for r in rows if r["service"] == "recommend" and r["success"]}

    assert len(countries) > 30
    assert countries <= set(catalog.country_defaults)


def test_versions_and_environment_match_the_project(catalog, rows):
    assert {r["api_version"] for r in rows} == {API_VERSION}
    assert {r["model_version"] for r in rows if r["service"] == "predict"} == {"1.0.0"}
    assert {r["model_version"] for r in rows if r["service"] == "recommend"} == {"2.0.0"}
    assert {r["environment"] for r in rows} == {"local"}
    assert {(r["endpoint"], r["method"]) for r in rows} == {("/predict", "POST"), ("/recommend", "POST")}


def test_rows_carry_no_sensitive_data(rows):
    """Ni token, ni en-tête, ni IP, ni trace ; aucune prédiction inventée."""
    text = json.dumps([r["request_payload"] for r in rows]).lower()

    for forbidden in ("authorization", "bearer", "token", "ip", "email", "password", "user"):
        assert f'"{forbidden}' not in text
    assert all(r["logfire_trace_id"] is None for r in rows)
    assert all(r["response_payload"] is None for r in rows if r["success"])


# --- Garde-fou et écriture ----------------------------------------------------------


@pytest.mark.parametrize("environment", ["prod", "PROD", "production"])
def test_prod_is_refused_without_allow_prod(environment):
    with pytest.raises(seed.SeedRefusedError):
        seed.check_environment(environment, allow_prod=False)
    seed.check_environment(environment, allow_prod=True)


def test_main_refuses_prod_and_writes_nothing(monkeypatch, capsys):
    monkeypatch.setenv("ENVIRONMENT", "prod")

    assert seed.main(["--write", "--end", END.isoformat()]) == 1
    assert "--allow-prod" in capsys.readouterr().err
    assert not _database_path().exists()


def test_preview_by_default_writes_nothing(capsys):
    assert seed.main(["--end", END.isoformat()]) == 0

    output = capsys.readouterr().out
    assert "Lignes à ajouter" in output and "rien n'a été écrit" in output
    assert not _database_path().exists()


def test_write_appends_rows_and_keeps_existing_ones(capsys):
    """Les lignes existantes sont conservées ; l'historique se termine avant la plus récente."""
    from agritech.monitoring.config import load_config
    from agritech.monitoring.models import Base
    from agritech.monitoring.session import create_monitoring_engine

    engine = create_monitoring_engine(load_config())
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    real = {
        "timestamp": END, "service": "predict", "endpoint": "/predict", "method": "POST",
        "status_code": 200, "success": True, "duration_ms": 6, "api_version": API_VERSION,
        "model_version": "1.0.0", "request_payload": {"rainfall_mm": 550.0}, "environment": "test",
    }
    with factory() as session:
        real_id = insert_api_request(session, real)

    assert seed.main(["--write", "--days", "30"]) == 0

    with factory() as session:
        total = session.scalar(select(func.count()).select_from(ApiRequest))
        latest = session.scalars(select(ApiRequest).order_by(ApiRequest.timestamp.desc())).first()
        summary = summarize_requests(session, days=30, now=END)
    engine.dispose()
    added = int(capsys.readouterr().out.split("lignes ajoutées")[0].split()[-1])

    assert total == added + 1
    assert latest.id == real_id
    assert summary["total_requests"] == total
    assert summary["error_count"] > 0


# --- Historique de démonstration au démarrage de l'API ------------------------------


def test_demo_history_fills_an_empty_database_with_recent_calls(engine):
    """Base vide : 90 jours d'appels qui se terminent au démarrage, vues 7 / 30 / 90 jours remplies."""
    factory = create_session_factory(engine)

    added = seed.seed_demo_history(factory, "staging", now=END)

    with factory() as session:
        stored = session.scalars(select(ApiRequest)).all()
        totals = {days: summarize_requests(session, days=days, now=END)["total_requests"] for days in (7, 30, 90)}
    assert added == len(stored) > 0
    assert all(END - timedelta(days=90) < row.timestamp < END for row in stored)
    assert 0 < totals[7] < totals[30] < totals[90] == added
    assert {row.environment for row in stored} == {"staging"}


def test_demo_history_is_the_reference_history_for_this_start(engine):
    """Déterministe : pour une même date de démarrage, les lignes de `generate_history` (graine 42)."""
    factory = create_session_factory(engine)
    seed.seed_demo_history(factory, "staging", now=END)

    expected = seed.generate_history(
        seed.load_catalog("staging"), days=seed.DEFAULT_DAYS, end=END, seed=seed.DEFAULT_SEED
    )
    with factory() as session:
        stored = session.scalars(select(ApiRequest).order_by(ApiRequest.id)).all()
    assert [(r.timestamp, r.service, r.status_code, r.request_payload) for r in stored] == [
        (e["timestamp"], e["service"], e["status_code"], e["request_payload"]) for e in expected
    ]


def test_demo_history_never_touches_a_database_with_calls(engine):
    """Base non vide : rien n'est ajouté, modifié ni supprimé."""
    factory = create_session_factory(engine)
    real = {
        "timestamp": END, "service": "predict", "endpoint": "/predict", "method": "POST",
        "status_code": 200, "success": True, "duration_ms": 6, "api_version": API_VERSION,
        "model_version": "1.0.0", "request_payload": {"rainfall_mm": 550.0}, "environment": "test",
    }
    with factory() as session:
        real_id = insert_api_request(session, real)

    assert seed.seed_demo_history(factory, "staging", now=END) == 0

    with factory() as session:
        stored = session.scalars(select(ApiRequest)).all()
    assert [(row.id, row.request_payload) for row in stored] == [(real_id, {"rainfall_mm": 550.0})]
