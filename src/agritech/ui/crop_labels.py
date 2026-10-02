"""Libellés français des cultures de /recommend, pour l'affichage uniquement.

L'API reste la source de vérité : les noms envoyés et reçus, ainsi que l'ordre
du classement, sont ceux de l'API. Une culture absente de la table garde son
nom API.
"""

from __future__ import annotations

CROP_LABELS_FR: dict[str, str] = {
    "Cassava": "Manioc",
    "Maize": "Maïs",
    "Plantains and others": "Banane plantain et autres",
    "Potatoes": "Pomme de terre",
    "Rice, paddy": "Riz paddy",
    "Sorghum": "Sorgho",
    "Soybeans": "Soja",
    "Sweet potatoes": "Patate douce",
    "Wheat": "Blé",
    "Yams": "Igname",
}


def crop_label(crop: str) -> str:
    """Libellé français de ``crop`` ; le nom API tel quel si la culture est inconnue."""
    return CROP_LABELS_FR.get(crop, crop)
