"""Génère la version HTML du rapport technique.

Source unique : `docs/rapport_technique.md`. Le Markdown reste lisible et
diffable dans Git ; le HTML n'ajoute que la mise en page — barre supérieure,
navigation latérale, bandeau d'en-tête et encarts. Rien n'est rédigé deux fois.

    poetry run python scripts/build_report.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

from agritech.config import PATHS

SOURCE = PATHS.docs / "rapport_technique.md"
SORTIE = PATHS.docs / "rapport_technique.html"
TEMPLATE = PATHS.src.parent / "scripts" / "report_template.html"

TITRE = "Rapport technique — Agritech Answers"
TITRE_COURT = "Rapport technique"
SOUS_TITRE = "Système de prédiction de rendement et de recommandation de cultures"
PILL = "Étape données · v1"
RESUME = (
    "Contexte métier, exploration des deux datasets, ACP, nettoyage des données "
    "historiques et construction des datasets d'entraînement de /predict et /recommend."
)
PIED = (
    "Agritech Answers — rapport technique · "
    'Source Markdown\u00a0: <a href="rapport_technique.md">rapport_technique.md</a> · '
    "Page régénérée par <code>scripts/build_report.py</code>."
)

# Encarts GitHub (> [!NOTE]) : libellé affiché et pictogramme de chaque type.
CALLOUTS = {
    "NOTE": ("Note", "◆"),
    "TIP": ("Conseil", "✦"),
    "IMPORTANT": ("Important", "▲"),
    "WARNING": ("Attention", "▲"),
    "CAUTION": ("Prudence", "■"),
}


def convertit_markdown(texte: str) -> str:
    """Rend le Markdown en HTML, avec des ancres accentuées comme sur GitHub."""
    convertisseur = markdown.Markdown(
        extensions=["tables", "fenced_code", "attr_list", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": slugify_unicode, "separator": "-"}},
    )
    return convertisseur.convert(texte)


def transforme_encarts(contenu: str) -> str:
    """Convertit les blockquotes d'alerte GitHub en encarts stylés.

    python-markdown rend `> [!NOTE]` comme un blockquote dont le premier
    paragraphe commence par le marqueur ; celui-ci devient un en-tête.
    """

    def remplace(correspondance: re.Match[str]) -> str:
        corps = correspondance.group("corps")
        marqueur = re.search(r"<p>\s*\[!(?P<type>[A-Z]+)\]\s*(<br\s*/?>)?\s*", corps)
        if marqueur is None or marqueur.group("type") not in CALLOUTS:
            return correspondance.group(0)

        libelle, icone = CALLOUTS[marqueur.group("type")]
        classe = marqueur.group("type").lower()
        corps = corps[: marqueur.start()] + "<p>" + corps[marqueur.end() :]
        corps = corps.replace("<p></p>", "")

        return (
            f'<div class="callout callout-{classe}">\n'
            f'  <div class="callout-header"><span aria-hidden="true">{icone}</span>{libelle}</div>\n'
            f'  <div class="callout-body">{corps}</div>\n'
            f"</div>"
        )

    return re.sub(
        r"<blockquote>(?P<corps>.*?)</blockquote>", remplace, contenu, flags=re.DOTALL
    )


def dimensions_image(source: str) -> tuple[int, int] | None:
    """Dimensions en pixels d'une image du dossier `docs/`, sans dépendance externe."""
    chemin = PATHS.docs / source
    if not chemin.is_file():
        return None

    if chemin.suffix == ".png":
        # En-tête PNG : 8 octets de signature, puis le chunk IHDR (largeur, hauteur).
        entete = chemin.read_bytes()[16:24]
        return int.from_bytes(entete[:4], "big"), int.from_bytes(entete[4:], "big")

    if chemin.suffix == ".svg":
        debut = chemin.read_text(encoding="utf-8")[:600]
        boite = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', debut)
        if boite:
            return round(float(boite.group(1))), round(float(boite.group(2)))
    return None


def transforme_figures(contenu: str) -> str:
    """Transforme les images isolées en `<figure>` avec légende.

    Les dimensions réelles sont ajoutées à la balise : sans elles, la hauteur
    de la page change pendant le chargement et un lien d'ancre suivi trop tôt
    tombe à côté de sa section.
    """

    def remplace(correspondance: re.Match[str]) -> str:
        balise, alt, source = (
            correspondance.group(1),
            correspondance.group("alt"),
            correspondance.group("src"),
        )
        taille = dimensions_image(source)
        if taille:
            balise = balise.replace("<img ", f'<img width="{taille[0]}" height="{taille[1]}" ', 1)
        legende = f"<figcaption>{alt}</figcaption>" if alt else ""
        return f"<figure>{balise}{legende}</figure>"

    return re.sub(
        r'<p>(<img\s[^>]*alt="(?P<alt>[^"]*)"\s+src="(?P<src>[^"]+)"[^>]*/?>)</p>',
        remplace,
        contenu,
    )


def espaces_insecables(contenu: str) -> str:
    """Colle la ponctuation double au mot qui la précède.

    Sans cela, un « ; » ou un « : » peut se retrouver seul en début de ligne.
    Le remplacement ne touche ni les balises ni le code, où l'espace compte.
    """
    morceaux = re.split(r"(<[^>]+>)", contenu)
    profondeur_code = 0
    for i, morceau in enumerate(morceaux):
        if morceau.startswith("<"):
            balise = morceau.lower()
            if balise.startswith(("<code", "<pre")):
                profondeur_code += 1
            elif balise.startswith(("</code", "</pre")):
                profondeur_code = max(0, profondeur_code - 1)
            continue
        if profondeur_code == 0:
            morceaux[i] = re.sub(r" ([;:!?»])", " \\1", morceau).replace("« ", "« ")
    return "".join(morceaux)


def rend_tableaux_scrollables(contenu: str) -> str:
    """Enveloppe les tableaux : un tableau large défile sans étirer la page."""
    return re.sub(
        r"<table>(.*?)</table>",
        lambda m: f'<div class="table-scroll"><table>{m.group(1)}</table></div>',
        contenu,
        flags=re.DOTALL,
    )


def extrait_sommaire(contenu: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Retire la section « Sommaire » et liste les sections de premier niveau.

    Le sommaire du Markdown sert la lecture sur GitHub ; dans la page HTML, la
    navigation latérale le remplace.
    """
    contenu = re.sub(
        r'<h2 id="sommaire">.*?</h2>\s*<ol>.*?</ol>', "", contenu, flags=re.DOTALL
    )

    sections = []
    for identifiant, titre in re.findall(
        r'<h2 id="([^"]+)">(.*?)</h2>', contenu, flags=re.DOTALL
    ):
        titre = re.sub(r"<[^>]+>", "", titre).strip()
        numero, _, libelle = titre.partition(". ")
        if numero.isdigit():
            sections.append((identifiant, numero.zfill(2), libelle))
        else:
            sections.append((identifiant, "", titre))
    return contenu, sections


def construit_nav(sections: list[tuple[str, str, str]]) -> str:
    """Construit les entrées de la navigation latérale."""
    lignes = []
    for identifiant, numero, libelle in sections:
        lignes.append(
            f'        <li><a href="#{identifiant}">'
            f'<span class="num">{numero}</span>'
            f'<span class="lbl">{html.escape(libelle)}</span></a></li>'
        )
    return "\n".join(lignes)


def indente(contenu: str, espaces: int = 6) -> str:
    """Aligne le corps du rapport sur l'indentation du gabarit."""
    marge = " " * espaces
    return "\n".join(marge + ligne if ligne.strip() else ligne for ligne in contenu.splitlines())


def main() -> None:
    texte = SOURCE.read_text(encoding="utf-8")

    contenu = convertit_markdown(texte)
    # Le titre de niveau 1 est remplacé par le bandeau d'en-tête du gabarit.
    contenu = re.sub(r"<h1[^>]*>.*?</h1>", "", contenu, count=1, flags=re.DOTALL)
    contenu = re.sub(r"^\s*<p><em>.*?</em></p>", "", contenu, count=1, flags=re.DOTALL)
    contenu, sections = extrait_sommaire(contenu)
    contenu = transforme_encarts(contenu)
    contenu = transforme_figures(contenu)
    contenu = rend_tableaux_scrollables(contenu)
    contenu = espaces_insecables(contenu)

    page = TEMPLATE.read_text(encoding="utf-8")
    remplacements = {
        "__TITRE__": TITRE,
        "__TITRE_COURT__": TITRE_COURT,
        "__SOUS_TITRE__": SOUS_TITRE,
        "__RESUME__": RESUME,
        "__PILL__": PILL,
        "__PIED__": PIED,
        "__NAV__": construit_nav(sections),
        "__CONTENU__": indente(contenu.strip()),
    }
    for cle, valeur in remplacements.items():
        page = page.replace(cle, valeur)

    SORTIE.write_text(page, encoding="utf-8")

    # Chemins relatifs : aucun chemin absolu de la machine dans les sorties.
    print(f"écrit : {SORTIE.relative_to(PATHS.root)}")
    print(f"        {len(sections)} sections, {SORTIE.stat().st_size / 1024:.0f} Ko")


if __name__ == "__main__":
    main()
