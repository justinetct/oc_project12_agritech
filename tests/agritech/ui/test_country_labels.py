"""Tests de ``agritech.ui.country_labels`` (noms français, tri et recherche sans accents)."""

from __future__ import annotations

import unicodedata

from agritech.ui.country_labels import country_label, country_sort_key, find_country, searchable_label

_LABELS = {"EGY": "Égypte", "USA": "États-Unis", "FRA": "France"}


def test_country_label_falls_back_to_the_api_name() -> None:
    assert country_label("EGY", "Egypt") == "Égypte"
    assert country_label("XXX", "Nowhere") == "Nowhere"


def test_sort_key_ignores_accents_and_case() -> None:
    assert country_sort_key("Égypte") == "egypte"
    assert sorted(["France", "États-Unis", "Égypte"], key=country_sort_key) == ["Égypte", "États-Unis", "France"]


def test_find_country_ignores_accents_case_and_spaces() -> None:
    for typed in ("Egypte", "Égypte", "égypte", " EGYPTE "):
        assert find_country(typed, _LABELS) == "EGY"
    assert find_country("etats-unis", _LABELS) == "USA"


def test_find_country_returns_none_when_nothing_matches() -> None:
    assert find_country("Atlantide", _LABELS) is None
    assert find_country("Egy", _LABELS) is None  # nom complet attendu, pas un début de nom


def test_searchable_label_keeps_the_displayed_name() -> None:
    label = searchable_label("Égypte")
    assert unicodedata.normalize("NFC", label) == "Égypte"  # même nom une fois recomposé
    assert label.startswith("É")  # « E » + accent combinant : « Egypte » le retrouve
