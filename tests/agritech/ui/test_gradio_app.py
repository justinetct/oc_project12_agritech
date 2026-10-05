"""Tests de l'interface Gradio de monitoring (``gradio_app/app.py``).

On teste la logique Python de la page — ``load_dashboard`` et les petits
blocs HTML — sans lancer de serveur ni inspecter le HTML interne de Gradio.
L'appel HTTP est remplacé par une fonction de test.
"""

from __future__ import annotations

import dotenv
import gradio as gr
import pandas as pd
import pytest

from agritech.api.schemas.monitoring import MonitoringSummaryResponse
from agritech.ui import api_client
from agritech.ui.errors import (
    ApiConfigurationError,
    ApiConnectionError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
    format_monitoring_error,
)
from agritech.ui.monitoring_view import (
    DAILY_COLUMNS,
    ERRORS_COLUMNS,
    daily_volume_frame,
    empty_kpis,
    empty_service_kpis,
    errors_frame,
    service_kpis,
    summary_kpis,
)
from agritech.ui.theme import COLORS
from gradio_app import app


def _summary(total: int = 6, errors: dict[str, int] | None = None) -> MonitoringSummaryResponse:
    """Résumé de 3 jours ; `total=0` donne une période sans appel."""
    errors = {"validation_error": 1} if errors is None and total else (errors or {})
    empty = total == 0
    return MonitoringSummaryResponse.model_validate(
        {
            "total_requests": total,
            "error_count": sum(errors.values()),
            "success_rate": None if empty else 5 / 6,
            "last_request_at": None if empty else "2026-10-02T14:00:00Z",
            "services": [
                {
                    "service": service,
                    "total_requests": 0 if empty else 3,
                    "error_count": 0,
                    "success_rate": None if empty else 1.0,
                    "latency_ms": (
                        {"mean": None, "median": None, "max": None}
                        if empty
                        else {"mean": 12.0, "median": 11.0, "max": 20}
                    ),
                }
                for service in ("predict", "recommend")
            ],
            "errors_by_type": errors,
            "requests_per_day": [
                {"date": "2026-09-30", "predict": 0 if empty else 2, "recommend": 0 if empty else 1},
                {"date": "2026-10-01", "predict": 0, "recommend": 0},
                {"date": "2026-10-02", "predict": 0 if empty else 1, "recommend": 0 if empty else 2},
            ],
        }
    )


@pytest.fixture
def summary_calls(monkeypatch: pytest.MonkeyPatch):
    """Remplace l'appel HTTP : renvoie `summary_calls.response`, note les `days` reçus."""

    class Calls(list):
        response: MonitoringSummaryResponse = _summary()

    calls = Calls()

    def fake_get_monitoring_summary(days: int) -> MonitoringSummaryResponse:
        calls.append(days)
        return calls.response

    monkeypatch.setattr(api_client, "get_monitoring_summary", fake_get_monitoring_summary)
    return calls


def _assert_page_is_reset(view: app.DashboardView) -> None:
    """Après une erreur : indicateurs à « — », tableaux vides, aucun message d'état vide."""
    assert view.kpis_html == app.kpi_cards_html(empty_kpis())
    assert view.daily.empty and list(view.daily.columns) == DAILY_COLUMNS
    assert view.services_html == app.service_rows_html(empty_service_kpis())
    assert view.errors.empty and list(view.errors.columns) == ERRORS_COLUMNS
    assert view.daily_note_html == ""
    assert view.errors_note_html == ""


# --- Chargement réussi -----------------------------------------------------------


@pytest.mark.parametrize("days", [7, 30, 90])
def test_load_dashboard_sends_selected_period(summary_calls, days: int):
    app.load_dashboard(days)
    assert summary_calls == [days]


def test_periods_map_labels_to_numbers_of_days():
    assert app.PERIODS == [("7 jours", 7), ("30 jours", 30), ("90 jours", 90)]
    assert app.DEFAULT_DAYS == 30


def test_load_dashboard_uses_presentation_layer(summary_calls):
    """Tout l'affichage vient de `monitoring_view`, sans calcul dans l'application."""
    summary = summary_calls.response
    view = app.load_dashboard(30)

    assert view.status_html == ""
    assert view.kpis_html == app.kpi_cards_html(summary_kpis(summary))
    pd.testing.assert_frame_equal(view.daily, daily_volume_frame(summary))
    assert view.services_html == app.service_rows_html(service_kpis(summary))
    pd.testing.assert_frame_equal(view.errors, errors_frame(summary))
    assert view.daily_note_html == ""
    assert view.errors_note_html == ""


def test_load_dashboard_without_errors_shows_no_error_message(summary_calls):
    summary_calls.response = _summary(total=6, errors={})
    view = app.load_dashboard(30)

    assert view.errors.empty
    assert view.errors_note_html == app.note_html("Aucune erreur sur cette période.")
    assert view.daily_note_html == ""


# --- Période vide ----------------------------------------------------------------


def test_load_dashboard_with_empty_period(summary_calls):
    summary_calls.response = _summary(total=0)
    view = app.load_dashboard(7)

    assert view.status_html == ""
    assert view.kpis_html == app.kpi_cards_html(summary_kpis(summary_calls.response))
    assert "—" in view.kpis_html
    assert view.daily["requests"].tolist() == [0] * 6
    assert view.services_html == app.service_rows_html(service_kpis(summary_calls.response))
    assert view.errors.empty
    assert view.daily_note_html == app.note_html("Aucune requête sur cette période.")
    assert view.errors_note_html == app.note_html("Aucune erreur sur cette période.")


# --- Erreurs API -----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ApiConfigurationError("token de monitoring non défini"),
        ApiConnectionError("connection refused"),
        ApiTimeoutError("read timed out"),
        ApiHttpError(401, "unauthorized", "Authentication required."),
        ApiHttpError(503, "monitoring_unavailable", "Monitoring is unavailable."),
        ApiInvalidResponseError("réponse non conforme au contrat de monitoring"),
    ],
    ids=["configuration", "connection", "timeout", "401", "503", "invalid-response"],
)
def test_load_dashboard_api_error_shows_message_and_resets_page(
    monkeypatch: pytest.MonkeyPatch, error: Exception
):
    def failing_get_monitoring_summary(days: int):
        raise error

    monkeypatch.setattr(api_client, "get_monitoring_summary", failing_get_monitoring_summary)
    view = app.load_dashboard(30)

    assert view.status_html == app.alert_html(format_monitoring_error(error))
    _assert_page_is_reset(view)


def test_error_after_success_keeps_no_previous_data(summary_calls, monkeypatch: pytest.MonkeyPatch):
    """Un échec après un chargement réussi n'affiche plus aucune donnée ancienne."""
    first = app.load_dashboard(30)
    assert not first.daily.empty

    def failing_get_monitoring_summary(days: int):
        raise ApiConnectionError("connection refused")

    monkeypatch.setattr(api_client, "get_monitoring_summary", failing_get_monitoring_summary)
    second = app.load_dashboard(30)

    _assert_page_is_reset(second)


def test_error_message_never_contains_token_or_api_body(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MONITORING_API_TOKEN", "secret-token-value")

    def failing_get_monitoring_summary(days: int):
        raise ApiHttpError(500, "internal_error", "secret internal detail")

    monkeypatch.setattr(api_client, "get_monitoring_summary", failing_get_monitoring_summary)
    view = app.load_dashboard(30)

    page = "".join(str(value) for value in view)
    assert "secret-token-value" not in page
    assert "secret internal detail" not in page


# --- Blocs HTML et construction --------------------------------------------------


def test_kpi_cards_show_each_label_and_value_escaped():
    html = app.kpi_cards_html(summary_kpis(_summary()))

    assert html.count('class="mon-kpi"') == 4
    for kpi in summary_kpis(_summary()):
        assert kpi.label in html
    assert "<script>" not in app.alert_html("<script>alert(1)</script>")


def test_header_shows_logo_and_title():
    html = app.header_html()

    assert 'src="data:image/png;base64,' in html
    assert "Monitoring Agritech" in html


def test_build_app_returns_gradio_blocks():
    assert isinstance(app.build_app(), gr.Blocks)


def test_service_colors_come_from_palette_without_warning_gold():
    """Deux services de même niveau : bleu ardoise et bleu-vert ; le doré reste aux alertes."""
    assert app.SERVICE_COLORS == {"predict": COLORS["slate"], "recommend": COLORS["teal"]}
    assert not {COLORS["warn-line"], COLORS["warn-bg"], COLORS["warn-ink"]} & set(app.SERVICE_COLORS.values())


# --- Lancement local : chargement de `.env` --------------------------------------


@pytest.fixture
def launch_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Remplace la construction et le lancement de la page : rien ne démarre vraiment."""
    calls = []

    class FakeDemo:
        def launch(self, **kwargs):
            calls.append(("launch", kwargs))

    def fake_build_app():
        calls.append(("build", None))
        return FakeDemo()

    monkeypatch.setattr(app, "build_app", fake_build_app)
    return calls


def _use_dotenv_file(monkeypatch: pytest.MonkeyPatch, path) -> list:
    """`load_dotenv` réel, mais sur `path` au lieu du `.env` du dépôt ; note les arguments."""
    received = []

    def load_test_dotenv(**kwargs):
        received.append(kwargs)
        return dotenv.load_dotenv(dotenv_path=path, **kwargs)

    monkeypatch.setattr(app, "load_dotenv", load_test_dotenv)
    return received


def test_main_loads_dotenv_without_override_before_launching(
    monkeypatch: pytest.MonkeyPatch, launch_calls: list
):
    monkeypatch.setattr(app, "load_dotenv", lambda **kwargs: launch_calls.append(("dotenv", kwargs)))

    app.main()

    assert [name for name, _ in launch_calls] == ["dotenv", "build", "launch"]
    assert launch_calls[0][1] == {"override": False}
    assert launch_calls[2][1] == {"theme": app.THEME, "css": app.CSS}


def test_main_reads_token_from_dotenv_when_environment_has_none(
    monkeypatch: pytest.MonkeyPatch, launch_calls: list, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text("MONITORING_API_TOKEN=test-token-from-dotenv\n")
    monkeypatch.delenv("MONITORING_API_TOKEN", raising=False)
    _use_dotenv_file(monkeypatch, env_file)

    app.main()

    assert api_client.settings.monitoring_api_token() == "test-token-from-dotenv"


def test_main_never_overrides_a_token_already_in_environment(
    monkeypatch: pytest.MonkeyPatch, launch_calls: list, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text("MONITORING_API_TOKEN=test-token-from-dotenv\n")
    monkeypatch.setenv("MONITORING_API_TOKEN", "test-token-from-environment")
    _use_dotenv_file(monkeypatch, env_file)

    app.main()

    assert api_client.settings.monitoring_api_token() == "test-token-from-environment"


def test_main_without_dotenv_or_token_still_starts_and_shows_not_configured(
    monkeypatch: pytest.MonkeyPatch, launch_calls: list, tmp_path
):
    """Sans `.env` ni token : la page démarre et annonce « Monitoring non configuré »."""
    monkeypatch.delenv("MONITORING_API_TOKEN", raising=False)
    _use_dotenv_file(monkeypatch, tmp_path / "absent.env")

    app.main()
    view = app.load_dashboard(30)

    assert [name for name, _ in launch_calls] == ["build", "launch"]
    assert view.status_html == app.alert_html(
        "Monitoring non configuré : le token d'accès n'est pas défini pour cette interface."
    )


# --- Indicateurs par service -----------------------------------------------------


def test_service_rows_show_predict_then_recommend_with_five_cards_each():
    html = app.service_rows_html(service_kpis(_summary()))

    assert html.index(">Predict<") < html.index(">Recommend<")
    assert html.count('class="mon-service"') == 2
    assert html.count('class="mon-kpi"') == 10
    for label in ("Requêtes", "Réussies", "Erreurs", "Latence médiane", "Latence max"):
        assert html.count(f">{label}<") == 2


def test_service_rows_use_chart_colors_as_accents():
    """Chaque rangée reprend la couleur de son service dans le graphique."""
    html = app.service_rows_html(empty_service_kpis())

    assert f'--accent:{COLORS["slate"]}' in html
    assert f'--accent:{COLORS["teal"]}' in html
    assert COLORS["warn-line"] not in html


def test_service_rows_show_latency_median_and_max_values(summary_calls):
    view = app.load_dashboard(30)

    assert "11\u202fms" in view.services_html  # médiane
    assert "20\u202fms" in view.services_html  # max


def test_dashboard_no_longer_has_a_per_service_table():
    """Le tableau « Par service » est remplacé : un seul tableau reste, celui des erreurs."""
    demo = app.build_app()
    tables = [block for block in demo.blocks.values() if isinstance(block, gr.Dataframe)]

    assert len(tables) == 1
    assert "Par service" not in str([getattr(block, "value", "") for block in demo.blocks.values()])


def test_service_rows_come_right_after_global_kpis_before_the_chart():
    """Ordre de la page : indicateurs globaux, Predict / Recommend, puis volume quotidien."""
    blocks = list(app.build_app().blocks.values())
    position = {
        "global": next(i for i, b in enumerate(blocks) if 'class="mon-kpis"' in str(getattr(b, "value", ""))),
        "services": next(i for i, b in enumerate(blocks) if 'class="mon-services"' in str(getattr(b, "value", ""))),
        "chart": next(i for i, b in enumerate(blocks) if isinstance(b, gr.BarPlot)),
        "errors": next(i for i, b in enumerate(blocks) if isinstance(b, gr.Dataframe)),
    }

    assert position["global"] < position["services"] < position["chart"] < position["errors"]
