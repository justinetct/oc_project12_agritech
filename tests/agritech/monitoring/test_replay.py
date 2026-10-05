"""Tests du CLI de rejeu `agritech.monitoring.replay`.

Isolation stricte :

- La configuration commune des tests (`tests/conftest.py`) redirige
  automatiquement DATABASE_URL vers une base temporaire (`tmp_path`).
- Les tests unitaires monkeypatchent `load_bundle` / `serving_predict` /
  `serving_recommend` pour ne pas charger les vrais artefacts. Un unique
  test d'intégration charge le vrai bundle `predict` (LinearRegression 4
  variables, très léger) pour prouver le chemin bout-en-bout.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agritech.monitoring import replay as replay_module
from agritech.monitoring.config import load_config
from agritech.monitoring.models import Base
from agritech.monitoring.repository import insert_api_request
from agritech.monitoring.session import (
    create_monitoring_engine,
    create_session_factory,
)


PREDICT_REQUEST = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}
PREDICT_RESPONSE = {
    "yield_tons_per_hectare": 4.82,
    "unit": "t/ha",
    "model_version": "1.0.0",
    "out_of_training_domain": False,
    "notes": [],
}


def _now() -> datetime:
    return datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)


def _row_values(**overrides: Any) -> dict:
    """Renvoie un dict complet pour `ApiRequest`, surchargé par `overrides`."""
    base = {
        "timestamp": _now(),
        "service": "predict",
        "endpoint": "/predict",
        "method": "POST",
        "status_code": 200,
        "success": True,
        "duration_ms": 5,
        "api_version": "0.1.0",
        "model_version": "1.0.0",
        "request_payload": PREDICT_REQUEST,
        "response_payload": PREDICT_RESPONSE,
        "error_type": None,
        "error_message": None,
        "logfire_trace_id": None,
        "environment": "test",
    }
    base.update(overrides)
    return base


@pytest.fixture
def factory():
    """Session factory pointant vers la base créée par `DATABASE_URL` autouse."""
    engine = create_monitoring_engine(load_config())
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


@pytest.fixture
def stub_services(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Neutralise les loaders et services pour éviter de charger les artefacts.

    Renvoie un dict `{"predict": <ref>, "recommend": <ref>}` que les tests
    peuvent muter pour changer la réponse simulée.
    """
    state = {
        "predict_response": dict(PREDICT_RESPONSE),
        "recommend_response": {"iso3": "FRA", "country": "France", "recommendations": []},
        "current_model_version": "1.0.0",
    }

    class _FakeBundle:
        def __init__(self) -> None:
            self.metadata = {"model_version": state["current_model_version"]}

    def _load_bundle(name: str):
        # metadata dynamique pour tester les mismatches
        b = _FakeBundle()
        b.metadata = {"model_version": state["current_model_version"]}
        return b

    monkeypatch.setattr(replay_module, "load_bundle", _load_bundle)
    monkeypatch.setattr(replay_module, "load_recommend_context", lambda path: None)
    monkeypatch.setattr(
        replay_module,
        "serving_predict",
        lambda bundle, values: state["predict_response"],
    )
    monkeypatch.setattr(
        replay_module,
        "serving_recommend",
        lambda bundle, context, iso3, conditions: state["recommend_response"],
    )
    return state


# ---------------------------------------------------------------------------
# _diff_json : test unitaire pur
# ---------------------------------------------------------------------------


def test_diff_json_identical_returns_empty():
    assert replay_module._diff_json({"a": 1}, {"a": 1}) == []


def test_diff_json_scalar_difference():
    lines = replay_module._diff_json({"a": 1}, {"a": 2})
    body = "\n".join(lines)
    assert "a:" in body
    assert "archived: 1" in body
    assert "current:  2" in body


def test_diff_json_nested_dict():
    lines = replay_module._diff_json(
        {"nested": {"x": 1, "y": 2}}, {"nested": {"x": 1, "y": 9}}
    )
    body = "\n".join(lines)
    assert "nested.y" in body


def test_diff_json_list_length_difference():
    lines = replay_module._diff_json([1, 2, 3], [1, 2])
    assert any("list length differs" in line for line in lines)


def test_diff_json_type_mismatch():
    lines = replay_module._diff_json({"a": 1}, [1])
    body = "\n".join(lines)
    assert "root" in body


# ---------------------------------------------------------------------------
# _parse_iso_date
# ---------------------------------------------------------------------------


def test_parse_iso_date_valid():
    dt = replay_module._parse_iso_date("2026-09-29")
    assert dt == datetime(2026, 9, 29, tzinfo=timezone.utc)


def test_parse_iso_date_invalid_raises_argument_error():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="invalid date"):
        replay_module._parse_iso_date("29/09/2026")


# ---------------------------------------------------------------------------
# Replay simple (id unique)
# ---------------------------------------------------------------------------


def test_replay_predict_identical_response(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Un predict archivé rejoué avec le même modèle : `No difference.`"""
    with factory() as sess:
        row_id = insert_api_request(sess, _row_values())

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert f"REPLAY api_request #{row_id}" in out
    assert "Service: predict" in out
    assert "Model version archived: 1.0.0" in out
    assert "Model version current:  1.0.0" in out
    assert "No difference." in out
    assert "WARNING" not in out


def test_replay_predict_with_difference(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Une réponse différente aujourd'hui : le diff pointe le champ modifié."""
    stub_services["predict_response"] = {
        **PREDICT_RESPONSE,
        "yield_tons_per_hectare": 4.90,
    }

    with factory() as sess:
        row_id = insert_api_request(sess, _row_values())

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "yield_tons_per_hectare" in out
    assert "archived: 4.82" in out
    assert "current:  4.9" in out
    assert "No difference." not in out


def test_replay_recommend_identical_response(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Un recommend archivé se rejoue via `serving_recommend`."""
    recommend_payload = {"iso3": "FRA"}
    recommend_response = {"iso3": "FRA", "country": "France", "recommendations": []}
    stub_services["recommend_response"] = recommend_response

    with factory() as sess:
        row_id = insert_api_request(
            sess,
            _row_values(
                service="recommend",
                endpoint="/recommend",
                model_version="2.0.0",
                request_payload=recommend_payload,
                response_payload=recommend_response,
            ),
        )
    stub_services["current_model_version"] = "2.0.0"

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Service: recommend" in out
    assert "No difference." in out


def test_replay_unknown_id_returns_2(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Un id inconnu produit un message clair sur stderr et un exit code 2."""
    exit_code = replay_module.main(["99999"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "No api_request found with id=99999" in captured.err


def test_replay_unsupported_service(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Un `service` inconnu ne tente pas le replay et l'annonce explicitement."""
    with factory() as sess:
        row_id = insert_api_request(
            sess, _row_values(service="unknown", endpoint="/unknown")
        )

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Replay not applicable" in out
    assert "unknown" in out


def test_replay_model_version_mismatch_warns(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Model version archivée ≠ actuelle : un WARNING clair est affiché."""
    stub_services["current_model_version"] = "1.1.0"

    with factory() as sess:
        row_id = insert_api_request(sess, _row_values(model_version="1.0.0"))

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Model version archived: 1.0.0" in out
    assert "Model version current:  1.1.0" in out
    assert "WARNING" in out
    assert "NOT a strict reproduction" in out


def test_replay_archived_error_not_rejoined(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Une 422 archivée n'est pas rejouée : le CLI l'affiche et s'arrête."""
    error_body = {
        "error": "validation_error",
        "message": "Request payload is invalid.",
        "details": [{"field": "body.rainfall_mm", "type": "greater_than_equal",
                     "message": "Input should be >= 0"}],
    }
    with factory() as sess:
        row_id = insert_api_request(
            sess,
            _row_values(
                status_code=422,
                success=False,
                error_type="validation_error",
                error_message="Request payload is invalid.",
                request_payload={"rainfall_mm": -1.0},
                response_payload=error_body,
            ),
        )

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Archived response was an error" in out
    assert "status=422" in out
    assert "error_type=validation_error" in out
    assert "Replay not applicable" in out
    # La réponse archivée doit être visible pour analyse.
    assert "validation_error" in out
    # Aucun replay n'a été tenté :
    assert "CURRENT RESPONSE" not in out


# ---------------------------------------------------------------------------
# Mode batch --failed --since
# ---------------------------------------------------------------------------


def test_failed_since_selects_only_recent_errors(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """`--failed --since` prend les erreurs à partir de la date, pas avant."""
    old = _now() - timedelta(days=5)
    recent = _now()
    with factory() as sess:
        # Vieille erreur : à ignorer.
        insert_api_request(
            sess,
            _row_values(
                timestamp=old,
                status_code=500,
                success=False,
                error_type="internal_error",
            ),
        )
        # Succès récent : à ignorer.
        insert_api_request(sess, _row_values(timestamp=recent))
        # Erreur récente : à sélectionner.
        row_id = insert_api_request(
            sess,
            _row_values(
                timestamp=recent,
                status_code=422,
                success=False,
                error_type="validation_error",
            ),
        )

    since = (recent - timedelta(hours=1)).date().isoformat()
    exit_code = replay_module.main(["--failed", "--since", since])
    out = capsys.readouterr().out

    assert exit_code == 0
    # Une seule entrée : le row_id de l'erreur récente.
    assert f"REPLAY api_request #{row_id}" in out
    # Les erreurs anciennes et succès ne doivent PAS apparaître.
    assert out.count("REPLAY api_request #") == 1


def test_failed_since_no_result(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Base sans erreur récente : message clair, exit 0."""
    with factory() as sess:
        insert_api_request(sess, _row_values())  # 1 succès seulement

    exit_code = replay_module.main(["--failed", "--since", "2026-01-01"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "No failed api_request since 2026-01-01" in out


def test_failed_requires_since(capsys: pytest.CaptureFixture):
    """`--failed` sans `--since` : erreur CLI claire, exit non nul."""
    with pytest.raises(SystemExit) as exc_info:
        replay_module.main(["--failed"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--since" in err


def test_invalid_date_exits_with_argument_error(capsys: pytest.CaptureFixture):
    """`--since 29/09/2026` : argparse rejette au parsing."""
    with pytest.raises(SystemExit):
        replay_module.main(["--failed", "--since", "29/09/2026"])
    err = capsys.readouterr().err
    assert "invalid date" in err


def test_no_arguments_shows_usage_error(capsys: pytest.CaptureFixture):
    """Aucun argument : argparse indique qu'il en faut au moins un."""
    with pytest.raises(SystemExit) as exc_info:
        replay_module.main([])
    assert exc_info.value.code == 2


def test_empty_database_and_unknown_id(
    factory, stub_services: dict, capsys: pytest.CaptureFixture
):
    """Base vide + id demandé : exit 2 avec message d'absence."""
    exit_code = replay_module.main(["1"])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "No api_request found with id=1" in err


# ---------------------------------------------------------------------------
# Intégration : rejoue un vrai predict via `agritech.serving`
# ---------------------------------------------------------------------------


def test_replay_real_predict_bundle_end_to_end(
    factory, capsys: pytest.CaptureFixture
):
    """Charge le vrai bundle `predict` et rejoue une prédiction précédemment calculée.

    Test d'intégration : aucun monkeypatch de `agritech.serving`. On calcule
    la réponse via `serving_predict` puis on l'archive comme si l'API l'avait
    faite ; le CLI doit renvoyer `No difference.` en la rejouant.
    """
    from agritech.serving import load_bundle, predict as serving_predict

    bundle = load_bundle("predict")
    response = serving_predict(bundle, PREDICT_REQUEST)

    with factory() as sess:
        row_id = insert_api_request(
            sess,
            _row_values(
                model_version=bundle.metadata["model_version"],
                response_payload=response,
            ),
        )

    exit_code = replay_module.main([str(row_id)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "No difference." in out
    assert "WARNING" not in out
