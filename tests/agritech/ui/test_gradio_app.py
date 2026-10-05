"""Tests de l'interface Gradio de monitoring (``gradio_app/app.py``).

On teste la logique Python de la page — ``load_dashboard`` et les petits
blocs HTML — sans lancer de serveur ni inspecter le HTML interne de Gradio.
L'appel HTTP est remplacé par une fonction de test.
"""

from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import dotenv
import gradio as gr
import pandas as pd
import pytest

from agritech.api.schemas.monitoring import MonitoringRequestsResponse, MonitoringSummaryResponse
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
    DAILY_ERRORS_COLUMNS,
    RECENT_ERRORS_COLUMNS,
    RECENT_SUCCESSES_COLUMNS,
    daily_errors_frame,
    daily_volume_frame,
    date_axis_labels,
    empty_kpis,
    empty_service_kpis,
    recent_errors_frame,
    recent_successes_frame,
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
                {"date": "2026-09-30", "predict": 0 if empty else 2, "recommend": 0 if empty else 1, "errors": 0 if empty else 1},
                {"date": "2026-10-01", "predict": 0, "recommend": 0, "errors": 0},
                {"date": "2026-10-02", "predict": 0 if empty else 1, "recommend": 0 if empty else 2, "errors": 0},
            ],
        }
    )


def _call(**overrides) -> dict:
    """Un appel archivé, au format JSON de `/monitoring/requests` (succès par défaut)."""
    values = {
        "id": 82,
        "timestamp": "2026-10-01T13:45:06Z",
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


ERROR_CALL = _call(
    id=81,
    service="predict",
    status_code=422,
    success=False,
    duration_ms=1,
    model_version="1.0.0",
    request_payload={"rainfall_mm": -1},
    error_type="validation_error",
    error_message="Request payload is invalid.",
)


@pytest.fixture
def summary_calls(monkeypatch: pytest.MonkeyPatch):
    """Remplace les appels HTTP du dashboard et note leurs paramètres.

    `summary_calls` contient les `days` reçus par le résumé ; `requests_calls`
    les paramètres de `/monitoring/requests`. Les réponses sont modifiables.
    """

    class Calls(list):
        response: MonitoringSummaryResponse = _summary()
        errors_response: MonitoringRequestsResponse = _requests(ERROR_CALL)
        successes_response: MonitoringRequestsResponse = _requests(_call())
        requests_calls: list = []

    calls = Calls()
    calls.requests_calls = []

    def fake_get_monitoring_summary(days: int) -> MonitoringSummaryResponse:
        calls.append(days)
        return calls.response

    def fake_get_monitoring_requests(*, limit, service=None, success=None):
        calls.requests_calls.append({"limit": limit, "service": service, "success": success})
        return calls.successes_response if success else calls.errors_response

    monkeypatch.setattr(api_client, "get_monitoring_summary", fake_get_monitoring_summary)
    monkeypatch.setattr(api_client, "get_monitoring_requests", fake_get_monitoring_requests)
    return calls


def _assert_page_is_reset(view: app.DashboardView) -> None:
    """Après une erreur : indicateurs à « — », graphiques et tableaux vides, aucun message d'état vide."""
    assert view.kpis_html == app.kpi_cards_html(empty_kpis())
    assert view.daily.data.empty and list(view.daily.data.columns) == DAILY_COLUMNS
    assert view.services_html == app.service_rows_html(empty_service_kpis())
    assert view.daily_errors.data.empty
    assert list(view.daily_errors.data.columns) == DAILY_ERRORS_COLUMNS
    assert view.recent_errors.empty and list(view.recent_errors.columns) == RECENT_ERRORS_COLUMNS
    assert view.recent_successes.empty and list(view.recent_successes.columns) == RECENT_SUCCESSES_COLUMNS
    assert view.daily_note_html == ""
    assert view.recent_errors_note_html == ""
    assert view.recent_successes_note_html == ""


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
    pd.testing.assert_frame_equal(view.daily.data, daily_volume_frame(summary))
    assert view.services_html == app.service_rows_html(service_kpis(summary))
    pd.testing.assert_frame_equal(view.daily_errors.data, daily_errors_frame(summary))
    pd.testing.assert_frame_equal(
        view.recent_errors, recent_errors_frame(summary_calls.errors_response)
    )
    pd.testing.assert_frame_equal(
        view.recent_successes, recent_successes_frame(summary_calls.successes_response)
    )
    assert view.daily_note_html == ""
    assert view.recent_errors_note_html == ""
    assert view.recent_successes_note_html == ""


# --- Période vide ----------------------------------------------------------------


def test_load_dashboard_with_empty_period(summary_calls):
    summary_calls.response = _summary(total=0)
    view = app.load_dashboard(7)

    assert view.status_html == ""
    assert view.kpis_html == app.kpi_cards_html(summary_kpis(summary_calls.response))
    assert "—" in view.kpis_html
    assert view.daily.data["requests"].tolist() == [0] * 6
    assert view.daily_errors.data["errors"].tolist() == [0] * 3
    assert view.services_html == app.service_rows_html(service_kpis(summary_calls.response))
    assert view.daily_note_html == app.note_html("Aucune requête sur cette période.")


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
    assert not first.daily.data.empty

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


def test_dashboard_tables_are_only_last_errors_and_last_successes():
    """Ni « Par service » ni « Erreurs par type » : seulement les deux tableaux des derniers appels."""
    demo = app.build_app()
    tables = [block for block in demo.blocks.values() if isinstance(block, gr.Dataframe)]
    texts = str([getattr(block, "value", "") for block in demo.blocks.values()])

    assert [table.headers for table in tables] == [RECENT_ERRORS_COLUMNS, RECENT_SUCCESSES_COLUMNS]
    assert "Par service" not in texts
    assert "Erreurs par type" not in texts


def test_page_order_from_kpis_to_last_successes():
    """KPI globaux, Predict / Recommend, volume, erreurs par jour, dernières erreurs, derniers succès."""
    blocks = list(app.build_app().blocks.values())

    def first(predicate) -> int:
        return next(i for i, block in enumerate(blocks) if predicate(block))

    def showing(text: str):
        return lambda block: text in str(getattr(block, "value", ""))

    plots = [i for i, block in enumerate(blocks) if isinstance(block, gr.Plot)]
    tables = [i for i, block in enumerate(blocks) if isinstance(block, gr.Dataframe)]
    order = [
        first(showing('class="mon-kpis"')),
        first(showing('class="mon-services"')),
        first(showing(">Volume quotidien<")),
        plots[0],
        first(showing(">Erreurs par jour<")),
        plots[1],
        first(showing(">Dernières erreurs<")),
        tables[0],
        first(showing(">Dernières requêtes réussies<")),
        tables[1],
    ]

    assert len(plots) == 2
    assert order == sorted(order)


# --- Erreurs dans le temps, dernières erreurs et derniers succès -----------------


def test_load_dashboard_fetches_last_ten_errors_and_successes(summary_calls):
    app.load_dashboard(30)

    assert summary_calls.requests_calls == [
        {"limit": 10, "service": None, "success": False},
        {"limit": 10, "service": None, "success": True},
    ]


def test_recent_tables_show_empty_messages(summary_calls):
    summary_calls.errors_response = _requests()
    summary_calls.successes_response = _requests()

    view = app.load_dashboard(30)

    assert view.recent_errors.empty and view.recent_successes.empty
    assert view.recent_errors_note_html == app.note_html("Aucune erreur récente.")
    assert view.recent_successes_note_html == app.note_html("Aucune requête réussie récente.")


@pytest.mark.parametrize("failing_success_flag", [False, True])
def test_failure_of_a_requests_call_resets_the_whole_page(
    summary_calls, monkeypatch: pytest.MonkeyPatch, failing_success_flag: bool
):
    """Si les dernières erreurs ou les derniers succès échouent, rien d'ancien ne reste affiché."""

    def failing_requests(*, limit, service=None, success=None):
        if success is failing_success_flag:
            raise ApiHttpError(503, "monitoring_unavailable", "Monitoring is unavailable.")
        return _requests(_call())

    monkeypatch.setattr(api_client, "get_monitoring_requests", failing_requests)
    view = app.load_dashboard(30)

    assert view.status_html == app.alert_html(
        "Le monitoring est momentanément indisponible côté API."
    )
    _assert_page_is_reset(view)


# --- Graphiques quotidiens --------------------------------------------------------


def _volume_frame(day_count: int) -> pd.DataFrame:
    """Volume de `day_count` jours (une ligne par jour et par service), comme `daily_volume_frame`."""
    first = date(2026, 7, 8)
    rows = [
        {"date": (first + timedelta(days=offset)).isoformat(), "service": service, "requests": 1}
        for offset in range(day_count)
        for service in ("predict", "recommend")
    ]
    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def _errors_frame(day_count: int, errors: int = 0) -> pd.DataFrame:
    volume = _volume_frame(day_count)
    days = volume["date"].drop_duplicates().tolist()
    return pd.DataFrame({"date": days, "errors": [errors] * len(days)})


@pytest.mark.parametrize("day_count", [7, 30, 90])
def test_daily_charts_keep_every_day_but_space_out_date_labels(day_count: int):
    """Une barre par jour ; seuls les jours de `date_axis_labels` sont étiquetés."""
    volume = _volume_frame(day_count)
    for chart, frame in (
        (app.daily_volume_chart(volume), volume),
        (app.daily_errors_chart(_errors_frame(day_count)), _errors_frame(day_count)),
    ):
        spec = chart.to_dict()
        days = frame["date"].drop_duplicates().tolist()

        assert spec["encoding"]["x"]["field"] == "date"
        assert spec["encoding"]["x"]["axis"]["values"] == date_axis_labels(days)
        assert len(set(chart.data["date"])) == day_count


def test_daily_charts_label_first_and_last_day_over_90_days():
    labels = app.daily_volume_chart(_volume_frame(90)).to_dict()["encoding"]["x"]["axis"]["values"]

    assert labels[0] == "2026-07-08"
    assert labels[-1] == "2026-10-05"
    assert len(labels) == 14  # une par semaine, plus le dernier jour


def test_volume_chart_uses_service_colors():
    color = app.daily_volume_chart(_volume_frame(7)).to_dict()["encoding"]["color"]

    assert color["field"] == "service"
    assert color["scale"] == {
        "domain": ["predict", "recommend"],
        "range": [COLORS["slate"], COLORS["teal"]],
    }


def test_errors_chart_uses_the_alert_color_without_legend():
    """Erreurs en orange brique (`alert`), sans légende : une seule série, rien à distinguer."""
    spec = app.daily_errors_chart(_errors_frame(7)).to_dict()

    assert spec["mark"]["color"] == COLORS["alert"] == app.ERRORS_COLOR
    assert "color" not in spec["encoding"]
    assert COLORS["alert"] not in app.SERVICE_COLORS.values()


@pytest.mark.parametrize("errors, expected", [(0, 1), (1, 1), (3, 3), (40, 5)])
def test_count_axis_has_integer_ticks_only(errors: int, expected: int):
    """Pas de graduation 0,5 sur un axe de 0 à 1 : au plus `max` graduations."""
    y_axis = app.daily_errors_chart(_errors_frame(7, errors)).to_dict()["encoding"]["y"]["axis"]

    assert y_axis == {"format": "d", "tickCount": expected}


@pytest.mark.parametrize("errors, top", [(0, 1.2), (1, 1.2), (40, 48.0)])
def test_count_axis_starts_at_zero_with_headroom_above_the_maximum(errors: int, top: float):
    """La plus haute barre ne touche pas le haut : 20 % de marge au-dessus du maximum."""
    scale = app.daily_errors_chart(_errors_frame(7, errors)).to_dict()["encoding"]["y"]["scale"]

    assert scale == {"domain": [0, pytest.approx(top)], "nice": False}


def test_volume_headroom_is_computed_on_the_stacked_daily_total():
    """Deux services à 1 requête par jour : barre empilée de 2, axe jusqu'à 2,4."""
    scale = app.daily_volume_chart(_volume_frame(7)).to_dict()["encoding"]["y"]["scale"]

    assert scale["domain"] == [0, pytest.approx(2.4)]


@pytest.mark.parametrize("day_count", [7, 30, 90])
def test_date_axis_has_no_title_and_outer_padding_without_extra_days(day_count: int):
    """Pas de titre « Jour (UTC) » ; de l'air avant le premier et après le dernier jour."""
    chart = app.daily_volume_chart(_volume_frame(day_count))
    x = chart.to_dict()["encoding"]["x"]

    assert x["title"] is None
    assert x["scale"] == {"range": [{"expr": "width * 0.04"}, {"expr": "width * 0.96"}]}
    assert chart.data["date"].nunique() == day_count


def test_daily_charts_without_days_still_build():
    """Après une erreur d'API, les graphiques vides se construisent sans étiquette."""
    for chart in (
        app.daily_volume_chart(pd.DataFrame(columns=DAILY_COLUMNS)),
        app.daily_errors_chart(pd.DataFrame(columns=DAILY_ERRORS_COLUMNS)),
    ):
        assert chart.to_dict()["encoding"]["x"]["axis"]["values"] == []


def test_daily_charts_target_gradio_vega_lite_5_without_vega_menu():
    """Gradio embarque Vega-Lite 5 ; pas de menu « … » (export, éditeur Vega en ligne)."""
    spec = app.daily_volume_chart(_volume_frame(7)).to_dict()

    assert "/vega-lite/v5." in spec["$schema"]
    assert spec["usermeta"] == {"embedOptions": {"renderer": "svg", "actions": False}}
    assert spec["width"] == "container"


def test_daily_charts_keep_space_inside_their_frame():
    """Axes et légende ne touchent pas le bord du cadre blanc."""
    for chart in (app.daily_volume_chart(_volume_frame(7)), app.daily_errors_chart(_errors_frame(7))):
        padding = chart.to_dict()["padding"]

        assert min(padding.values()) >= 12


def test_recent_sections_state_that_they_ignore_the_period():
    demo = app.build_app()
    texts = [str(getattr(block, "value", "")) for block in demo.blocks.values()]

    assert sum("10 derniers appels, toutes périodes confondues." in text for text in texts) == 2


def test_dashboard_never_shows_private_fields_or_token(summary_calls, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MONITORING_API_TOKEN", "secret-token-value")

    view = app.load_dashboard(30)
    def text(value) -> str:
        if isinstance(value, alt.Chart):
            return value.to_json()
        return value.to_csv() if isinstance(value, pd.DataFrame) else str(value)

    page = "".join(text(value) for value in view)

    for private in ("secret-token-value", "response_payload", "logfire_trace_id", "Authorization"):
        assert private not in page
