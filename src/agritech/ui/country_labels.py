"""Noms français des pays de /recommend, pour l'affichage uniquement.

L'API reste la source de vérité : la page travaille avec les codes ISO3 et
les envoie tels quels. Un code absent de la table garde le nom fourni par
l'API.
"""

from __future__ import annotations

import unicodedata

COUNTRY_LABELS_FR: dict[str, str] = {
    "ALB": "Albanie",
    "DZA": "Algérie",
    "AGO": "Angola",
    "ARG": "Argentine",
    "ARM": "Arménie",
    "AUS": "Australie",
    "AUT": "Autriche",
    "AZE": "Azerbaïdjan",
    "BHS": "Bahamas",
    "BGD": "Bangladesh",
    "BLR": "Biélorussie",
    "BEL": "Belgique",
    "BOL": "Bolivie",
    "BWA": "Botswana",
    "BRA": "Brésil",
    "BGR": "Bulgarie",
    "BFA": "Burkina Faso",
    "BDI": "Burundi",
    "CMR": "Cameroun",
    "CAN": "Canada",
    "CAF": "République centrafricaine",
    "CHL": "Chili",
    "CHN": "Chine continentale",
    "COL": "Colombie",
    "COG": "Congo",
    "HRV": "Croatie",
    "CZE": "Tchéquie",
    "CIV": "Côte d'Ivoire",
    "DNK": "Danemark",
    "DOM": "République dominicaine",
    "ECU": "Équateur",
    "EGY": "Égypte",
    "SLV": "Salvador",
    "ERI": "Érythrée",
    "EST": "Estonie",
    "FIN": "Finlande",
    "FRA": "France",
    "DEU": "Allemagne",
    "GHA": "Ghana",
    "GRC": "Grèce",
    "GTM": "Guatemala",
    "GIN": "Guinée",
    "GNB": "Guinée-Bissau",
    "GUY": "Guyana",
    "HTI": "Haïti",
    "HND": "Honduras",
    "HUN": "Hongrie",
    "IND": "Inde",
    "IDN": "Indonésie",
    "IRN": "Iran",
    "IRQ": "Irak",
    "IRL": "Irlande",
    "ITA": "Italie",
    "JAM": "Jamaïque",
    "JPN": "Japon",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "LAO": "Laos",
    "LVA": "Lettonie",
    "LBN": "Liban",
    "LSO": "Lesotho",
    "LBY": "Libye",
    "LTU": "Lituanie",
    "MDG": "Madagascar",
    "MWI": "Malawi",
    "MYS": "Malaisie",
    "MLI": "Mali",
    "MRT": "Mauritanie",
    "MEX": "Mexique",
    "MAR": "Maroc",
    "MOZ": "Mozambique",
    "NAM": "Namibie",
    "NPL": "Népal",
    "NLD": "Pays-Bas",
    "NZL": "Nouvelle-Zélande",
    "NIC": "Nicaragua",
    "NER": "Niger",
    "NOR": "Norvège",
    "PAK": "Pakistan",
    "PNG": "Papouasie-Nouvelle-Guinée",
    "PER": "Pérou",
    "POL": "Pologne",
    "PRT": "Portugal",
    "QAT": "Qatar",
    "KOR": "Corée du Sud",
    "MDA": "Moldavie",
    "ROU": "Roumanie",
    "RUS": "Russie",
    "RWA": "Rwanda",
    "SAU": "Arabie saoudite",
    "SEN": "Sénégal",
    "SVK": "Slovaquie",
    "SVN": "Slovénie",
    "ZAF": "Afrique du Sud",
    "ESP": "Espagne",
    "LKA": "Sri Lanka",
    "SUR": "Suriname",
    "SWE": "Suède",
    "CHE": "Suisse",
    "SYR": "Syrie",
    "TJK": "Tadjikistan",
    "THA": "Thaïlande",
    "MKD": "Macédoine du Nord",
    "TUN": "Tunisie",
    "TUR": "Turquie",
    "UGA": "Ouganda",
    "UKR": "Ukraine",
    "GBR": "Royaume-Uni",
    "TZA": "Tanzanie",
    "USA": "États-Unis",
    "URY": "Uruguay",
    "VEN": "Venezuela",
    "VNM": "Viêt Nam",
    "ZMB": "Zambie",
    "ZWE": "Zimbabwe",
}


def country_label(iso3: str, api_name: str) -> str:
    """Nom français du pays ``iso3`` ; le nom fourni par l'API si le code est inconnu."""
    return COUNTRY_LABELS_FR.get(iso3, api_name)


def country_sort_key(label: str) -> str:
    """Clé de tri et de recherche qui ignore accents et majuscules (« Égypte » → « egypte »)."""
    return unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode().lower()


def searchable_label(label: str) -> str:
    """Nom affiché dans le sélecteur, accents décomposés (« É » = « E » + accent combinant).

    Le rendu à l'écran est identique, mais le filtre du sélecteur, qui compare les
    caractères tels quels, retrouve alors « Égypte » quand on tape « Egypte ».
    """
    return unicodedata.normalize("NFD", label)


def find_country(text: str, labels: dict[str, str]) -> str | None:
    """Code ISO3 dont le nom correspond à ``text``, sans tenir compte des accents ni des majuscules."""
    wanted = country_sort_key(text.strip())
    return next((code for code, label in labels.items() if country_sort_key(label) == wanted), None)
