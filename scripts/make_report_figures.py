"""Figures du rapport technique.

Régénère les images de `docs/assets/figures/` à partir des données locales de
`data/`. Les analyses viennent des notebooks 01 à 06 ; ce script ne fait que
remettre en forme les résultats déjà établis, aux couleurs du rapport.

Le schéma `06_pipelines.svg` ne dépend d'aucune donnée : il est écrit
directement en SVG et versionné tel quel.

    poetry run python scripts/make_report_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from agritech import geo
from agritech.config import AGRICULTURE_CROP_YIELD_FILENAME, PATHS, SEED

# ---------------------------------------------------------------------------
# Identité visuelle — reprise des slides du projet
# ---------------------------------------------------------------------------

BRAND = "#14452f"
BRAND_2 = "#2c7a53"
BRAND_3 = "#5aa47c"
WASH = "#e9f3ed"
INK = "#1e2a24"
MUTED = "#63706a"
LINE = "#dde8e2"
ACCENT = "#b35c00"
ACCENT_SOFT = "#e8c98a"

# Palette de séries : verts de la charte, puis ocres et bruns pour rester
# distinguable au-delà de trois modalités.
SERIES = [BRAND, BRAND_3, ACCENT, "#7b8f86", "#8a5a00", BRAND_2, ACCENT_SOFT]

# Six modalités à distinguer d'un coup d'œil (cultures, variables de l'ACP) :
# les verts de la charte ne suffisent plus, des teintes froides sont ajoutées.
SERIES_6 = [BRAND, BRAND_3, ACCENT, "#4a6b8a", "#c9a227", "#8c5a7a"]

# Deux familles de cultures historiques : verts pour les tubercules et le
# plantain, ocres pour les céréales et les légumineuses, du plus foncé au plus clair.
TUBERCULES = ["Potatoes", "Cassava", "Yams", "Sweet potatoes", "Plantains and others"]
VERTS = [BRAND, BRAND_2, BRAND_3, "#8cc2a2", "#b9dcc7"]
OCRES = ["#8a5a00", ACCENT, "#d4832f", "#e2a864", "#eccb96"]
NOMS_CULTURES = {
    "Potatoes": "pommes de terre",
    "Sweet potatoes": "patate douce",
    "Cassava": "manioc",
    "Yams": "igname",
    "Plantains and others": "plantain",
    "Maize": "maïs",
    "Rice, paddy": "riz",
    "Wheat": "blé",
    "Sorghum": "sorgho",
    "Soybeans": "soja",
}

# Secteurs des camemberts : verts et ocres alternés. Sur les teintes foncées,
# le pourcentage est écrit en blanc.
SECTEURS = [BRAND, "#e2a864", BRAND_3, ACCENT, "#b9dcc7", "#8a5a00"]
SECTEURS_FONCES = {BRAND, ACCENT, "#8a5a00"}

# Pays hors jeu de données sur la carte.
GRIS_CARTE = "#e3e7e5"

FIGURES = PATHS.docs / "assets" / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": LINE,
        "axes.titlecolor": BRAND,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# Figures reprises des slides : textes plus grands et cadre allégé, pour rester
# lisibles une fois l'image réduite à la largeur du rapport. Ce style ne
# s'applique qu'à ces figures ; les autres images ne changent pas.
STYLE_SLIDES = {
    "font.size": 12,
    "axes.titlesize": 14.5,
    "axes.titlepad": 12,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid.axis": "y",
}
DPI_SLIDES = 200


def enregistre(fig, nom: str, dpi: int | None = None) -> None:
    """Écrit la figure dans `docs/assets/figures/` et referme la figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    chemin = FIGURES / nom
    options = {"dpi": dpi} if dpi else {}
    fig.savefig(chemin, bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)
    # Chemin relatif : aucun chemin absolu de la machine dans les sorties.
    print(f"  {chemin.relative_to(PATHS.root)}  ({chemin.stat().st_size / 1024:.0f} Ko)")


def milliers(valeur: float, _position=None) -> str:
    """Nombre entier avec une espace pour séparer les milliers (12 500)."""
    return f"{valeur:,.0f}".replace(",", " ")


def lit_agriculture(colonnes: list[str]) -> pd.DataFrame:
    """Lit les colonnes utiles du jeu parcellaire (93 Mo sur disque)."""
    return pd.read_csv(
        PATHS.data_agriculture_crop_yield / AGRICULTURE_CROP_YIELD_FILENAME,
        usecols=colonnes,
    )


def lit_historique() -> pd.DataFrame:
    """Lit le dataset historique nettoyé produit par le notebook 04."""
    return pd.read_csv(PATHS.data_processed / "crop_yield_clean.csv")


def etat_apres_jointures() -> pd.DataFrame:
    """Reconstruit l'état après jointures du notebook 04, qui n'est pas sauvegardé.

    Mêmes étapes que le notebook : période 1990-2013, noms de pays convertis en
    code ISO3, température moyennée par pays et par année, puis jointures sur le
    code pays et l'année à partir du fichier de rendement.
    """
    debut, fin = 1990, 2013
    dossier = PATHS.data_crop_yield_prediction

    def ajoute_iso3(df: pd.DataFrame, colonne: str) -> pd.DataFrame:
        correspondances = geo.vers_iso3(df[colonne].astype(str).unique())
        return df.assign(iso3=df[colonne].map(correspondances))

    rendement = pd.read_csv(dossier / "yield.csv")
    rendement = (
        rendement[rendement["Area"] != "China"]
        .loc[lambda d: d["Year"].between(debut, fin)]
        .pipe(ajoute_iso3, "Area")
        .rename(columns={"Area": "area", "Year": "year", "Item": "crop"})
        .assign(yield_t_ha=lambda d: d["Value"] / 10_000)  # hg/ha -> t/ha
        [["iso3", "area", "year", "crop", "yield_t_ha"]]
    )
    temperature = (
        pd.read_csv(dossier / "temp.csv")
        .drop_duplicates()
        .loc[lambda d: d["year"].between(debut, fin)]
        .pipe(ajoute_iso3, "country")
        .groupby(["iso3", "year"], as_index=False)["avg_temp"]
        .mean()
    )
    pluie = pd.read_csv(dossier / "rainfall.csv")
    pluie.columns = pluie.columns.str.strip()  # espace initiale dans un en-tête
    pluie = (
        pluie.assign(
            rain_mm=lambda d: pd.to_numeric(d["average_rain_fall_mm_per_year"], errors="coerce")
        )
        .loc[lambda d: d["Year"].between(debut, fin)]
        .pipe(ajoute_iso3, "Area")
        .rename(columns={"Year": "year"})
        [["iso3", "year", "rain_mm"]]
    )
    pesticides = (
        pd.read_csv(dossier / "pesticides.csv")
        .loc[lambda d: d["Year"].between(debut, fin)]
        .pipe(ajoute_iso3, "Area")
        .rename(columns={"Year": "year", "Value": "pesticides_t"})
        [["iso3", "year", "pesticides_t"]]
    )

    etat = rendement.dropna(subset=["iso3"])
    for source in (temperature, pluie, pesticides):
        etat = etat.merge(source.dropna(subset=["iso3"]), on=["iso3", "year"], how="left")

    # Chiffres vérifiés par assert dans le notebook 04.
    assert len(etat) == 22_679 and etat["iso3"].nunique() == 168
    return etat


# ---------------------------------------------------------------------------
# 1. Ce qui explique le rendement dans le jeu parcellaire
# ---------------------------------------------------------------------------


def figure_pluie_rendement() -> None:
    df = lit_agriculture(
        [
            "Rainfall_mm",
            "Yield_tons_per_hectare",
            "Fertilizer_Used",
            "Irrigation_Used",
        ]
    )

    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(11, 4.2))

    echantillon = df.sample(12_000, random_state=SEED)
    gauche.scatter(
        echantillon["Rainfall_mm"],
        echantillon["Yield_tons_per_hectare"],
        s=4,
        alpha=0.25,
        color=BRAND_2,
        edgecolors="none",
    )
    tranches = pd.cut(df["Rainfall_mm"], bins=25)
    moyennes = df.groupby(tranches, observed=True)["Yield_tons_per_hectare"].mean()
    centres = [interv.mid for interv in moyennes.index]
    gauche.plot(centres, moyennes.to_numpy(), color=ACCENT, lw=2.2)
    gauche.set_xlabel("Pluie sur la saison (mm)")
    gauche.set_ylabel("Rendement (t/ha)")
    gauche.set_title("Relation pluie / rendement  ·  r = 0,76")
    gauche.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=BRAND_2, label="12 000 parcelles"),
            Line2D([], [], color=ACCENT, lw=2.2, label="rendement moyen par tranche"),
        ],
        loc="upper left",
        fontsize=9,
    )

    tranches_larges = pd.cut(df["Rainfall_mm"], bins=12)
    combinaisons = [
        (True, True, BRAND, "engrais + irrigation"),
        (True, False, BRAND_3, "engrais seul"),
        (False, True, ACCENT, "irrigation seule"),
        (False, False, MUTED, "aucun des deux"),
    ]
    for engrais, irrigation, couleur, libelle in combinaisons:
        masque = (df["Fertilizer_Used"] == engrais) & (df["Irrigation_Used"] == irrigation)
        courbe = df[masque].groupby(tranches_larges, observed=True)[
            "Yield_tons_per_hectare"
        ].mean()
        droite.plot(
            [interv.mid for interv in courbe.index],
            courbe.to_numpy(),
            marker="o",
            ms=3.5,
            lw=1.8,
            color=couleur,
            label=libelle,
        )
    droite.set_xlabel("Pluie sur la saison (mm)")
    droite.set_ylabel("Rendement moyen (t/ha)")
    droite.set_title("Effet des pratiques, à pluie comparable")
    droite.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    enregistre(fig, "01_pluie_rendement.png")


# ---------------------------------------------------------------------------
# 2. ACP : structure des variables et absence de séparation des cultures
# ---------------------------------------------------------------------------


def figure_acp() -> None:
    colonnes = [
        "Rainfall_mm",
        "Temperature_Celsius",
        "Fertilizer_Used",
        "Irrigation_Used",
        "Days_to_Harvest",
        "Yield_tons_per_hectare",
    ]
    df = lit_agriculture(colonnes + ["Crop"])

    X = StandardScaler().fit_transform(df[colonnes].astype(float))
    acp = PCA(n_components=len(colonnes)).fit(X)
    inertie = acp.explained_variance_ratio_ * 100
    correlations = pd.DataFrame(
        acp.components_.T * np.sqrt(acp.explained_variance_), index=colonnes
    )

    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # Cercle des corrélations F1-F2
    angles = np.linspace(0, 2 * np.pi, 200)
    gauche.plot(np.cos(angles), np.sin(angles), color=LINE, lw=1.2)
    gauche.axhline(0, color=LINE, lw=1)
    gauche.axvline(0, color=LINE, lw=1)
    etiquettes = {
        "Yield_tons_per_hectare": "Yield",
        "Rainfall_mm": "Rainfall",
        "Temperature_Celsius": "Temperature",
        "Fertilizer_Used": "Fertilizer",
        "Irrigation_Used": "Irrigation",
        "Days_to_Harvest": "Days_to_Harvest",
    }
    for (nom, (cx, cy)), couleur in zip(correlations.iloc[:, [0, 1]].iterrows(), SERIES_6):
        gauche.annotate(
            "",
            xy=(cx, cy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=couleur, lw=2, shrinkA=0, shrinkB=0),
        )
        norme = float(np.hypot(cx, cy))
        k = (norme + 0.13) / norme if norme > 0.15 else 0.24 / max(norme, 1e-9)
        # Une flèche plutôt verticale reçoit son libellé au-dessus ou en
        # dessous : posé à côté, un libellé long recouvrirait ses voisins.
        verticale = abs(cy) > abs(cx)
        gauche.text(
            cx * k,
            cy * k,
            etiquettes[nom],
            color=couleur,
            fontsize=8.5,
            fontweight="bold",
            ha="center" if verticale else ("left" if cx >= 0 else "right"),
            va=("bottom" if cy >= 0 else "top") if verticale else "center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )
    gauche.set_xlim(-1.45, 1.45)
    gauche.set_ylim(-1.2, 1.2)
    gauche.set_aspect("equal")
    gauche.grid(False)
    gauche.set_xlabel(f"F1 ({inertie[0]:.1f} %)")
    gauche.set_ylabel(f"F2 ({inertie[1]:.1f} %)")
    gauche.set_title("Cercle des corrélations")

    # Plan factoriel F1-F2, coloré par culture
    idx = df.sample(5_000, random_state=SEED).index
    projection = pd.DataFrame(acp.transform(X)[:, :2], index=df.index, columns=["F1", "F2"])
    echantillon = projection.loc[idx]
    cultures = sorted(df["Crop"].unique())
    for culture, couleur in zip(cultures, SERIES_6):
        masque = df.loc[idx, "Crop"] == culture
        droite.scatter(
            echantillon.loc[masque, "F1"],
            echantillon.loc[masque, "F2"],
            s=6,
            alpha=0.55,
            color=couleur,
            edgecolors="none",
            label=culture,
        )
    droite.axhline(0, color=LINE, lw=1)
    droite.axvline(0, color=LINE, lw=1)
    droite.set_xlabel(f"F1 ({inertie[0]:.1f} %)")
    droite.set_ylabel(f"F2 ({inertie[1]:.1f} %)")
    droite.set_title("5 000 parcelles projetées, couleur = culture")
    droite.legend(fontsize=8, markerscale=2, ncol=2, loc="upper right")

    fig.tight_layout()
    enregistre(fig, "02_acp.png")


# ---------------------------------------------------------------------------
# 3. Évolution des rendements historiques
# ---------------------------------------------------------------------------


def figure_evolution_rendements() -> None:
    df = lit_historique()
    moyennes = df.groupby(["year", "crop"])["yield_t_ha"].mean().unstack()
    debut, fin = moyennes.index.min(), moyennes.index.max()

    familles = []
    for libelle, tubercule, teintes in [
        ("Tubercules et plantain", True, VERTS),
        ("Céréales et légumineuses", False, OCRES),
    ]:
        cultures = [c for c in moyennes.columns if (c in TUBERCULES) == tubercule]
        # Teinte de plus en plus claire, du rendement le plus haut au plus bas en fin de période
        cultures = list(moyennes.loc[fin, cultures].sort_values(ascending=False).index)
        familles.append((libelle, cultures, teintes))

    with plt.rc_context(STYLE_SLIDES):
        fig, ax = plt.subplots(figsize=(10, 4.9))
        for _, cultures, teintes in familles:
            for culture, couleur in zip(cultures, teintes):
                pomme_de_terre = culture == "Potatoes"
                ax.plot(
                    moyennes.index,
                    moyennes[culture].to_numpy(),
                    color=couleur,
                    lw=3.4 if pomme_de_terre else 2.1,
                    zorder=3 if pomme_de_terre else 2,
                )

        ax.annotate(
            "Pommes de terre",
            xy=(fin, moyennes.loc[fin, "Potatoes"]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=BRAND,
            fontsize=13,
            fontweight="bold",
        )

        # Nom de chaque famille, dans l'espace vide au-dessus de ses courbes
        # sur les premières années.
        premieres = moyennes.loc[debut : debut + 6]
        autres_tubercules = [c for c in familles[0][1] if c != "Potatoes"]
        positions = [
            (premieres[autres_tubercules].max().max() + premieres["Potatoes"].min()) / 2,
            (premieres[familles[1][1]].max().max() + premieres[familles[0][1]].min().min()) / 2,
        ]
        for (libelle, cultures, teintes), y in zip(familles, positions):
            ax.annotate(
                libelle,
                xy=(debut + 0.3, y),
                xytext=(0, 2),
                textcoords="offset points",
                va="bottom",
                color=teintes[1],
                fontsize=13,
                fontweight="bold",
            )
            ax.annotate(
                ", ".join(NOMS_CULTURES[c] for c in cultures),
                xy=(debut + 0.3, y),
                xytext=(0, -2),
                textcoords="offset points",
                va="top",
                color=MUTED,
                fontsize=11.5,
            )

        ax.set_xlim(debut - 0.4, fin + 0.4)
        ax.set_ylim(0, moyennes.max().max() * 1.12)
        ax.set_yticks(range(0, int(moyennes.max().max()) + 1, 5))
        ax.tick_params(length=0)
        ax.set_xlabel("Année")
        ax.set_ylabel("Rendement moyen (t/ha)")

        fig.tight_layout()
        enregistre(fig, "03_evolution_rendements.png", dpi=DPI_SLIDES)


# ---------------------------------------------------------------------------
# 4. Comparaison des deux jeux sur les cultures communes
# ---------------------------------------------------------------------------


def figure_comparaison_datasets() -> None:
    communes = ["Maize", "Rice", "Soybean", "Wheat"]

    historique = lit_historique()
    historique["crop"] = historique["crop"].replace(
        {"Rice, paddy": "Rice", "Soybeans": "Soybean"}
    )
    historique = historique[historique["crop"].isin(communes)]

    parcelle = lit_agriculture(["Crop", "Yield_tons_per_hectare"])
    parcelle = parcelle[parcelle["Crop"].isin(communes)]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    largeur = 0.34
    for decalage, (donnees, colonne_culture, colonne_valeur, couleur, libelle) in enumerate(
        [
            (historique, "crop", "yield_t_ha", BRAND, "Historique nettoyé (pays × année)"),
            (
                parcelle,
                "Crop",
                "Yield_tons_per_hectare",
                ACCENT_SOFT,
                "Agriculture CropYield (parcelle)",
            ),
        ]
    ):
        positions = np.arange(len(communes)) + (decalage - 0.5) * largeur
        valeurs = [
            donnees.loc[donnees[colonne_culture] == c, colonne_valeur].to_numpy()
            for c in communes
        ]
        boites = ax.boxplot(
            valeurs,
            positions=positions,
            widths=largeur * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=INK, lw=1.4),
            whiskerprops=dict(color=MUTED),
            capprops=dict(color=MUTED),
        )
        for boite in boites["boxes"]:
            boite.set_facecolor(couleur)
            boite.set_edgecolor(MUTED)
            boite.set_alpha(0.9)
        ax.plot([], [], color=couleur, lw=8, label=libelle)

    ax.set_xticks(np.arange(len(communes)))
    ax.set_xticklabels(communes)
    ax.set_xlabel("Culture")
    ax.set_ylabel("Rendement (t/ha)")
    ax.set_title("Rendement par culture, quatre cultures communes")
    ax.set_ylim(top=12)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    enregistre(fig, "04_comparaison_datasets.png")


# ---------------------------------------------------------------------------
# 5. Pesticides : distribution brute contre logarithme
# ---------------------------------------------------------------------------


def figure_pesticides() -> None:
    pesticides = lit_historique()["pesticides_t"]

    with plt.rc_context(STYLE_SLIDES):
        fig, (gauche, droite) = plt.subplots(1, 2, figsize=(10, 3.9))

        # Barres de 30 000 t : la première regroupe presque toutes les lignes.
        largeur = 30
        effectifs, _, _ = gauche.hist(
            pesticides / 1_000,
            bins=np.arange(0, pesticides.max() / 1_000 + largeur, largeur),
            color=ACCENT,
            edgecolor="white",
            linewidth=0.6,
        )
        gauche.annotate(
            f"{effectifs[0] / effectifs.sum() * 100:.0f} % des lignes sous 30 000 t",
            xy=(largeur, effectifs[0] * 0.92),
            xytext=(14, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=12,
        )
        gauche.annotate(
            f"jusqu'à {milliers(pesticides.max())} t",
            xy=(pesticides.max() / 1_000, 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=11.5,
        )
        gauche.set_title("Brut")
        gauche.set_xlabel("pesticides_t (milliers de tonnes)")
        gauche.xaxis.set_major_formatter(FuncFormatter(milliers))

        droite.hist(np.log1p(pesticides), bins=60, color=BRAND, edgecolor="white", linewidth=0.6)
        droite.set_title("Après log1p")
        droite.set_xlabel("log1p(pesticides_t)")
        droite.xaxis.set_major_locator(MultipleLocator(2))

        for ax in (gauche, droite):
            ax.set_ylabel("Nombre de lignes")
            ax.yaxis.set_major_formatter(FuncFormatter(milliers))
            ax.tick_params(length=0)

        fig.tight_layout(w_pad=3)
        enregistre(fig, "05_pesticides_log.png", dpi=DPI_SLIDES)


# ---------------------------------------------------------------------------
# 7. Variables catégorielles du jeu parcellaire : répartition et effet
# ---------------------------------------------------------------------------


def figure_repartitions_categorielles() -> None:
    colonnes = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
    df = lit_agriculture(colonnes + ["Yield_tons_per_hectare"])

    with plt.rc_context(STYLE_SLIDES):
        fig, axes = plt.subplots(1, 4, figsize=(10.7, 3.7))
        for ax, colonne in zip(axes, colonnes):
            effectifs = df[colonne].value_counts().sort_index()
            moyennes = df.groupby(colonne)["Yield_tons_per_hectare"].mean()
            ecart = f"{moyennes.max() - moyennes.min():.3f}".replace(".", ",")

            # Sur une seule ligne, la place manque autour des camemberts : chaque
            # modalité est nommée dans son secteur, au-dessus de son pourcentage.
            couleurs = SECTEURS[: len(effectifs)]
            _, _, textes = ax.pie(
                effectifs,
                colors=couleurs,
                autopct=lambda part: f"{part:.1f} %".replace(".", ","),
                startangle=90,
                counterclock=False,
                radius=1.2,
                pctdistance=0.6,
                wedgeprops=dict(edgecolor="white", linewidth=1.5),
                textprops=dict(fontsize=11.5, linespacing=1.25),
            )
            for modalite, couleur, texte in zip(effectifs.index, couleurs, textes):
                texte.set_text(f"{modalite}\n{texte.get_text()}")
                texte.set_color("white" if couleur in SECTEURS_FONCES else INK)

            ax.set_title(colonne, pad=40)
            ax.text(
                0.5,
                1.02,
                f"{len(effectifs)} modalités\nécart de rendement {ecart} t/ha",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color=MUTED,
                fontsize=11.5,
                linespacing=1.3,
            )

        fig.tight_layout(w_pad=0.6)
        enregistre(fig, "07_repartitions_categorielles.png", dpi=DPI_SLIDES)


# ---------------------------------------------------------------------------
# 8. Couverture des pays après jointures, avant nettoyage
# ---------------------------------------------------------------------------


def figure_couverture_pays() -> None:
    etat = etat_apres_jointures()

    complete = etat[["avg_temp", "rain_mm", "pesticides_t"]].notna().all(axis=1)
    par_pays = etat.groupby("iso3")[["avg_temp", "pesticides_t"]].count()
    incomplets = set(par_pays.index[(par_pays == 0).any(axis=1)])
    complets = set(par_pays.index) - incomplets
    # Mêmes groupes que le nettoyage du notebook 04.
    assert (len(complets), len(incomplets)) == (117, 51)

    part_complete = complete.mean() * 100
    part_pays_entiers = etat.loc[~complete, "iso3"].isin(incomplets).mean() * 100

    libelle_complet = f"contexte complet · {len(complets)} pays"
    libelle_incomplet = f"contexte incomplet · {len(incomplets)} pays"
    couleurs = {libelle_complet: BRAND, libelle_incomplet: ACCENT}
    categories = {code: libelle_complet for code in complets}
    categories.update({code: libelle_incomplet for code in incomplets})

    with plt.rc_context(STYLE_SLIDES):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        geo.plot_carte_categories(
            categories, couleurs, ax=ax, couleur_absente=GRIS_CARTE, contours=geo.charger_contours()
        )
        # Légende sous la carte, en plus grand que celle de la fonction utilitaire.
        ax.legend(
            handles=[Patch(facecolor=c, label=libelle) for libelle, c in couleurs.items()]
            + [Patch(facecolor=GRIS_CARTE, label="hors jeu de données")],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            fontsize=12.5,
            frameon=False,
            handlelength=1.3,
            handleheight=1.1,
            columnspacing=2.2,
        )
        ax.set_title(
            f"{part_complete:.0f} % des lignes complètes après jointures", loc="left", pad=30
        )
        ax.text(
            0,
            1.025,
            f"{part_pays_entiers:.0f} % des lignes incomplètes viennent des "
            f"{len(incomplets)} pays sans température ou sans pesticides",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=12,
        )

        fig.tight_layout()
        enregistre(fig, "08_couverture_pays.png", dpi=DPI_SLIDES)


def main() -> None:
    print("Figures du rapport technique :")
    figure_pluie_rendement()
    figure_acp()
    figure_evolution_rendements()
    figure_comparaison_datasets()
    figure_pesticides()
    figure_repartitions_categorielles()
    figure_couverture_pays()


if __name__ == "__main__":
    main()
