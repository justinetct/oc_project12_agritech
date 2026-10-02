"""Tests de ``agritech.ui.icons``."""

from __future__ import annotations

import pytest

from agritech.ui.icons import icon_svg


def test_icon_svg_returns_inline_svg() -> None:
    svg = icon_svg("rendement")
    assert svg.startswith('<svg viewBox="0 0 24 24"')
    assert "currentColor" in svg


def test_icon_svg_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        icon_svg("inconnue")
