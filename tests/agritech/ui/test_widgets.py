"""Tests de ``agritech.ui.widgets`` (plage du curseur)."""

from __future__ import annotations

from agritech.ui.widgets import _snap_inward


def test_snap_inward_keeps_bounds_already_on_the_step() -> None:
    assert _snap_inward(15.0, 40.0, 0.1) == (15.0, 40.0)
    assert _snap_inward(100.0, 1000.0, 1.0) == (100.0, 1000.0)


def test_snap_inward_rounds_towards_the_inside() -> None:
    # Domaine réel de la température /recommend : 2,5667 → 30,2467 °C.
    assert _snap_inward(2.566666666666667, 30.24666666666667, 0.1) == (2.6, 30.2)
    assert _snap_inward(-10.04, 39.96, 0.1) == (-10.0, 39.9)


def test_snap_inward_keeps_a_range_narrower_than_one_step() -> None:
    assert _snap_inward(2.51, 2.59, 0.1) == (2.51, 2.59)
