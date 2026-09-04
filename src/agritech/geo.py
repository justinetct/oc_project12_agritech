"""Carte du monde pour les notebooks : contours des pays et coloration par valeur.

Les contours viennent de Natural Earth, échelle 1:110m (177 pays), au format
GeoJSON. Le fichier n'est pas versionné : voir `data/README.md` pour le
récupérer.

Aucune librairie cartographique n'est nécessaire : le GeoJSON est lu avec
`json` et les pays sont dessinés comme des polygones matplotlib. Les
coordonnées sont utilisées telles quelles (longitude, latitude), ce qui revient
à une projection équirectangulaire : les surfaces près des pôles paraissent
plus grandes qu'elles ne sont.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path as CheminFichier
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path

from agritech.config import PATHS

GEOJSON_PAR_DEFAUT = PATHS.data / "geo" / "ne_110m_admin_0_countries.geojson"

# Noms qui ne s'apparient pas automatiquement. Les fichiers FAO emploient la
# forme longue des Nations unies, Natural Earth la forme courte usuelle.
CORRESPONDANCES = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Iran (Islamic Republic of)": "Iran",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
    "Congo (Democratic Republic Of The)": "Dem. Rep. Congo",
    "Macedonia": "North Macedonia",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "China, mainland": "China",
    "Occupied Palestinian Territory": "Palestine",
    "Czechoslovakia": None,  # entités sans frontière actuelle : volontairement
    "USSR": None,  # laissées de côté plutôt que rattachées
    "Yugoslav SFR": None,  # à un pays successeur arbitraire
    "Serbia and Montenegro": None,
    "Belgium-Luxembourg": None,
    "Ethiopia PDR": None,
    "Sudan (former)": None,
    "Pacific Islands Trust Territory": None,
}


def _code_iso3(proprietes: dict) -> str | None:
    """Code ISO3 d'une entité Natural Earth.

    `ISO_A3` vaut `-99` pour quelques entités, dont la France et la Norvège.
    `ISO_A3_EH` les renseigne correctement ; `ADM0_A3` sert de dernier recours
    pour les territoires sans code ISO officiel (Kosovo, Somaliland).
    """
    for champ in ("ISO_A3", "ISO_A3_EH", "ADM0_A3"):
        valeur = proprietes.get(champ)
        if valeur and valeur != "-99":
            return valeur
    return None


def _cle(nom: str) -> str:
    """Normalise un nom de pays : sans accent, sans ponctuation, en minuscules."""
    sans_accent = unicodedata.normalize("NFKD", str(nom)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z ]", " ", sans_accent.lower()).split())


def _chemin(geometry: dict) -> Path:
    """Convertit une géométrie GeoJSON en tracé matplotlib.

    Un pays peut être fait de plusieurs polygones (îles), et chaque polygone
    d'un contour suivi d'éventuels trous — le Lesotho dans l'Afrique du Sud,
    par exemple. Tous les anneaux sont réunis dans un seul tracé.
    """
    if geometry["type"] == "MultiPolygon":
        polygones = geometry["coordinates"]
    else:
        polygones = [geometry["coordinates"]]

    sommets, codes = [], []
    for polygone in polygones:
        for anneau in polygone:
            sommets.extend(anneau)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(anneau) - 1))

    return Path(sommets, codes)


def charger_contours(chemin: CheminFichier | None = None) -> dict[str, Path]:
    """Charge le GeoJSON et retourne un tracé matplotlib par code ISO3."""
    chemin = chemin or GEOJSON_PAR_DEFAUT
    if not chemin.is_file():
        raise FileNotFoundError(
            f"Contours introuvables : {chemin}\nVoir data/README.md pour les récupérer."
        )

    geojson = json.loads(chemin.read_text(encoding="utf-8"))
    contours = {}
    for f in geojson["features"]:
        code = _code_iso3(f["properties"])
        if code:
            contours[code] = _chemin(f["geometry"])
    return contours


def _index_des_noms(chemin: CheminFichier | None = None) -> dict[str, str]:
    """Toutes les variantes de nom connues de Natural Earth, vers le code ISO3."""
    chemin = chemin or GEOJSON_PAR_DEFAUT
    geojson = json.loads(chemin.read_text(encoding="utf-8"))

    index = {}
    for f in geojson["features"]:
        p = f["properties"]
        code = _code_iso3(p)
        if code is None:
            continue
        for champ in ("NAME", "ADMIN", "NAME_LONG", "FORMAL_EN", "NAME_SORT"):
            valeur = p.get(champ)
            if valeur and valeur != "-99":
                index.setdefault(_cle(valeur), code)
    return index


def vers_iso3(noms: Iterable[str], chemin: CheminFichier | None = None) -> dict[str, str]:
    """Associe chaque nom de pays à son code ISO3.

    Les noms sans contour connu sont simplement absents du résultat : micro-États
    et territoires que Natural Earth ne dessine pas à l'échelle 110m, et entités
    historiques listées dans `CORRESPONDANCES`. Comparer les clés du résultat à
    la liste de départ donne les pays qui n'apparaîtront pas sur la carte.
    """
    index = _index_des_noms(chemin)

    codes = {}
    for nom in noms:
        cible = CORRESPONDANCES.get(nom, nom)
        if cible is None:
            continue
        code = index.get(_cle(cible))
        if code:
            codes[nom] = code
    return codes


def _dessiner(
    contours: dict[str, Path],
    couleur_par_code: dict[str, object],
    titre: str,
    ax,
    couleur_absente: str,
) -> None:
    """Trace les pays, chacun rempli avec la couleur qu'on lui a associée."""
    for code, chemin in contours.items():
        couleur = couleur_par_code.get(code, couleur_absente)
        ax.add_patch(PathPatch(chemin, facecolor=couleur, edgecolor="white", linewidth=0.3))

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)  # l'Antarctique est coupé, aucun pays du jeu n'y figure
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(titre)


def plot_world_map(
    valeurs: dict[str, float],
    titre: str = "",
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    couleur_absente: str = "#e8e8e8",
    contours: dict[str, Path] | None = None,
) -> None:
    """Colore les pays du monde selon une valeur numérique, indexée par code ISO3.

    Les pays absents de `valeurs` sont grisés. Sans `ax`, une figure est créée
    et affichée ; avec `ax`, la carte est dessinée dedans, ce qui permet de
    composer une grille de plusieurs cartes.
    """
    contours = contours if contours is not None else charger_contours()

    montrer = ax is None
    if montrer:
        _, ax = plt.subplots(figsize=(12, 6))

    presentes = [v for v in valeurs.values() if v is not None]
    vmin = vmin if vmin is not None else (min(presentes) if presentes else 0)
    vmax = vmax if vmax is not None else (max(presentes) if presentes else 1)
    palette = plt.get_cmap(cmap)
    etendue = (vmax - vmin) or 1

    couleurs = {
        code: palette((valeur - vmin) / etendue)
        for code, valeur in valeurs.items()
        if valeur is not None
    }
    _dessiner(contours, couleurs, titre, ax, couleur_absente)

    if montrer:
        plt.tight_layout()
        plt.show()


def plot_carte_categories(
    categories: dict[str, str],
    couleurs: dict[str, str],
    titre: str = "",
    ax=None,
    couleur_absente: str = "#e8e8e8",
    libelle_absent: str = "hors jeu de données",
    contours: dict[str, Path] | None = None,
) -> None:
    """Colore les pays selon une catégorie plutôt qu'une valeur numérique.

    `categories` associe un code ISO3 à un libellé, `couleurs` associe chaque
    libellé à sa couleur. L'ordre de `couleurs` est celui de la légende.
    """
    contours = contours if contours is not None else charger_contours()

    montrer = ax is None
    if montrer:
        _, ax = plt.subplots(figsize=(14, 7))

    par_code = {
        code: couleurs[libelle]
        for code, libelle in categories.items()
        if libelle in couleurs
    }
    _dessiner(contours, par_code, titre, ax, couleur_absente)

    legende = [Patch(facecolor=c, label=libelle) for libelle, c in couleurs.items()]
    legende.append(Patch(facecolor=couleur_absente, label=libelle_absent))
    ax.legend(handles=legende, loc="lower left", fontsize=9, frameon=True)

    if montrer:
        plt.tight_layout()
        plt.show()
