"""Tests de ``agritech.ui.theme``."""

from __future__ import annotations

from agritech.ui.theme import COLORS, LOGO_WHITE_PATH, logo_data_uri, page_css, predict_form_css


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOR_TYPE_RGBA = 6


def test_white_logo_is_a_png_with_alpha_channel() -> None:
    data = LOGO_WHITE_PATH.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    # Octet 25 = type de couleur de l'en-tête IHDR ; 6 = RGBA (fond transparent possible).
    assert data[25] == PNG_COLOR_TYPE_RGBA


def test_logo_data_uri_embeds_the_png() -> None:
    assert logo_data_uri().startswith("data:image/png;base64,iVBORw0KGgo")


def test_page_css_declares_every_color_variable() -> None:
    css = page_css()
    assert css.startswith("<style>") and css.endswith("</style>")
    for name, value in COLORS.items():
        assert f"--{name}:{value};" in css


def test_conditions_card_rhythm_is_shared_by_both_pages() -> None:
    css = page_css()
    assert "padding:32px 28px 36px}" in css
    assert '.st-key-cond_card .stElementContainer[class*="_slider"]{margin-top:-6px;margin-bottom:-6px}' in css


def test_narrow_screens_stack_ranking_shrink_panel_title_and_keep_ranges_whole() -> None:
    css = page_css()
    assert "@media (max-width:1200px){\n  .ag-rank-row{grid-template-columns:30px minmax(0,1fr) auto;" in css
    assert "@media (max-width:1100px){\n  .ag-panel-title{font-size:38px}\n}" in css
    assert ".ag-range{white-space:nowrap}" in css


def test_predict_form_css_keeps_only_predict_specifics() -> None:
    css = predict_form_css()
    assert css.startswith("<style>") and css.endswith("</style>")
    assert ".st-key-cond_card .ag-field-label{font-size:18px}" in css
    assert "border-top:1px solid var(--card-line)" in css
    assert "padding:32px 28px 36px" not in css
