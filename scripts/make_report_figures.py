"""Figures du rapport technique.

Régénère les images de `docs/assets/figures/` à partir des données locales de
`data/`. Les analyses viennent des notebooks 01 à 06 ; ce script ne fait que
remettre en forme les résultats déjà établis, aux couleurs du rapport.

    poetry run python scripts/make_report_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

FIGURES = PATHS.docs / "assets" / "figures"
DEBUT, FIN = 1990, 2013

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


def enregistre(fig, nom: str) -> None:
    """Écrit la figure dans `docs/assets/figures/` et referme la figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    chemin = FIGURES / nom
    fig.savefig(chemin, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Chemin relatif : aucun chemin absolu de la machine dans les sorties.
    print(f"  {chemin.relative_to(PATHS.root)}  ({chemin.stat().st_size / 1024:.0f} Ko)")


def lit_agriculture(colonnes: list[str]) -> pd.DataFrame:
    """Lit les colonnes utiles du jeu parcellaire (93 Mo sur disque)."""
    return pd.read_csv(
        PATHS.data_agriculture_crop_yield / AGRICULTURE_CROP_YIELD_FILENAME,
        usecols=colonnes,
    )


def lit_historique() -> pd.DataFrame:
    """Lit le dataset historique consolidé produit par le notebook 04."""
    return pd.read_csv(PATHS.data_processed / f"crop_yield_prediction_{DEBUT}_{FIN}.csv")


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

    tubercules = ["Potatoes", "Cassava", "Yams", "Sweet potatoes", "Plantains and others"]
    familles = [
        ("Tubercules et plantain", tubercules),
        (
            "Céréales et légumineuses",
            [c for c in sorted(df["crop"].unique()) if c not in tubercules],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (titre, cultures) in zip(axes, familles):
        for culture, couleur in zip(cultures, SERIES_6):
            serie = (
                df[df["crop"] == culture]
                .groupby("year")["yield_t_ha"]
                .mean()
                .sort_index()
            )
            ax.plot(serie.index, serie.to_numpy(), lw=1.9, color=couleur, label=culture)
        ax.set_title(titre)
        ax.set_xlabel("Année")
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[1].legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Rendement moyen (t/ha)")

    fig.tight_layout()
    enregistre(fig, "03_evolution_rendements.png")


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
            (historique, "crop", "yield_t_ha", BRAND, "Historique consolidé (pays × année)"),
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
# 5. Pesticides : tonnage brut contre logarithme
# ---------------------------------------------------------------------------


def figure_pesticides() -> None:
    df = lit_historique().dropna(subset=["pesticides_t"]).copy()
    df["log_pesticides"] = np.log1p(df["pesticides_t"])

    tubercules = ["Potatoes", "Cassava", "Sweet potatoes", "Yams", "Plantains and others"]
    df["famille"] = np.where(
        df["crop"].isin(tubercules), "tubercules et plantain", "céréales et légumineuses"
    )
    echantillon = df.sample(4_000, random_state=SEED)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, colonne, titre, xlabel in [
        (axes[0], "pesticides_t", "Tonnage brut", "pesticides_t (tonnes)"),
        (axes[1], "log_pesticides", "Logarithme", "log1p(pesticides_t)"),
    ]:
        for (famille, groupe), couleur in zip(
            echantillon.groupby("famille"), [BRAND_3, ACCENT]
        ):
            ax.scatter(
                groupe[colonne],
                groupe["yield_t_ha"],
                s=6,
                alpha=0.45,
                color=couleur,
                edgecolors="none",
                label=famille,
            )
        ax.set_title(titre)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rendement (t/ha)")
    axes[0].legend(fontsize=8.5, markerscale=2, loc="upper right")

    fig.tight_layout()
    enregistre(fig, "05_pesticides_log.png")


def main() -> None:
    print("Figures du rapport technique :")
    figure_pluie_rendement()
    figure_acp()
    figure_evolution_rendements()
    figure_comparaison_datasets()
    figure_pesticides()


if __name__ == "__main__":
    main()
