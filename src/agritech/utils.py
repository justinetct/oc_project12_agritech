"""Fonctions de visualisation partagées par les notebooks du projet.

Reprises du Projet 4 (`utils.plot_distrib`) et simplifiées : seules les
fonctions utiles à l'EDA du Projet 12 sont conservées.
"""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def use_inline_backend() -> None:
    """Réactive l'affichage des figures dans le notebook.

    `ProfileReport` bascule matplotlib sur le backend non interactif `Agg`
    pour produire son rapport HTML. Toute figure créée ensuite ne s'affiche
    plus. On rebascule donc sur le backend `inline`, sans rien changer
    lorsque le code tourne hors notebook.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return
        plt.switch_backend("module://matplotlib_inline.backend_inline")
    except Exception:
        pass


def _zoom_y_on_bars(ax) -> None:
    """Fait démarrer l'axe des ordonnées à la hauteur de la plus petite barre.

    Sur une distribution quasi uniforme, un axe partant de zéro écrase les
    écarts entre barres : toutes paraissent identiques. Une marge est laissée
    de part et d'autre, sans quoi la plus petite barre aurait une hauteur nulle
    et laisserait un trou dans l'histogramme.
    """
    heights = [patch.get_height() for patch in ax.patches]
    if not heights:
        return

    basse, haute = min(heights), max(heights)
    marge = 0.05 * (haute - basse) or 1
    ax.set_ylim(basse - marge, haute + marge)


def _draw_distrib(
    values: pd.Series,
    ax_hist,
    ax_box,
    title: str,
    hist_color: str = "steelblue",
    box_color: str = "skyblue",
    xlabel: str = "Valeurs",
) -> None:
    """Trace un histogramme et son boxplot sur deux axes déjà créés."""
    sns.histplot(values, kde=True, ax=ax_hist, color=hist_color, bins=30)
    ax_hist.set_ylabel("Fréquence")
    ax_hist.set_title(title)
    _zoom_y_on_bars(ax_hist)

    sns.boxplot(x=values, ax=ax_box, color=box_color)
    ax_box.set_xlabel(xlabel)


def plot_distrib(df: pd.DataFrame, col: str, do_log: bool = False) -> None:
    """Affiche la distribution d'une variable numérique.

    Histogramme avec courbe de densité en haut, boxplot en dessous.
    Avec `do_log=True`, la transformation log1p est affichée en regard.
    """
    use_inline_backend()
    values = df[col].dropna()

    if do_log:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), height_ratios=[3, 1])
        _draw_distrib(values, axes[0, 0], axes[1, 0], "Distribution des valeurs")
        _draw_distrib(
            np.log1p(values),
            axes[0, 1],
            axes[1, 1],
            "Distribution log",
            hist_color="orange",
            box_color="gold",
            xlabel="Valeurs (log1p)",
        )
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1])
        _draw_distrib(values, axes[0], axes[1], "Distribution des valeurs")

    fig.suptitle(col, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()


def plot_distribs(df: pd.DataFrame, cols: list[str], ncols: int = 2) -> None:
    """Affiche les distributions de plusieurs variables dans une seule figure.

    Chaque variable occupe deux axes superposés : histogramme puis boxplot.
    """
    use_inline_backend()
    nrows = -(-len(cols) // ncols)  # arrondi supérieur

    fig, axes = plt.subplots(
        2 * nrows,
        ncols,
        figsize=(7 * ncols, 5 * nrows),
        height_ratios=[3, 1] * nrows,
    )
    axes = np.asarray(axes).reshape(2 * nrows, ncols)

    for position, col in enumerate(cols):
        ligne, colonne = divmod(position, ncols)
        _draw_distrib(
            df[col].dropna(), axes[2 * ligne, colonne], axes[2 * ligne + 1, colonne], col
        )

    for position in range(len(cols), nrows * ncols):  # cases d'une grille incomplète
        ligne, colonne = divmod(position, ncols)
        axes[2 * ligne, colonne].axis("off")
        axes[2 * ligne + 1, colonne].axis("off")

    fig.tight_layout()
    plt.show()


def plot_pearson_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    target: str | None = None,
    title: str = "Matrice de corrélation (Pearson)",
) -> None:
    """Affiche la matrice de corrélation de Pearson sous forme de heatmap.

    Sans `cols`, toutes les variables numériques et booléennes sont retenues.
    Avec `target`, les variables sont triées par corrélation absolue
    décroissante avec la cible, ce qui amène les plus liées en haut à gauche.
    """
    use_inline_backend()
    if cols is None:
        cols = df.select_dtypes(["number", "bool"]).columns.tolist()

    corr = df[cols].corr()
    if target is not None:
        ordre = corr[target].abs().sort_values(ascending=False).index
        corr = corr.loc[ordre, ordre]

    fig, ax = plt.subplots(figsize=(9, 7))
    # vmin/vmax figés : sinon l'échelle de couleur se cale sur les données
    # et une corrélation faible paraît aussi forte qu'une corrélation à 1.
    sns.heatmap(
        corr, cmap="coolwarm", annot=True, fmt=".2f", square=True, vmin=-1, vmax=1, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_pairs(
    df: pd.DataFrame,
    target: str,
    cols: list[str],
    ncols: int = 3,
    sample: int = 20_000,
    random_state: int = 42,
) -> None:
    """Croise la cible avec chaque variable indiquée, dans une seule figure.

    Nuage de points pour une variable continue, boxplot pour une variable
    booléenne : un nuage sur deux modalités ne donnerait que deux traits.
    Les nuages sont tracés sur un échantillon, un million de points saturant
    la figure sans rien ajouter de lisible.
    """
    use_inline_backend()
    nrows = -(-len(cols) // ncols)  # arrondi supérieur
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    flat = np.ravel(axes)

    for ax, col in zip(flat, cols):
        continue_ = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(
            df[col]
        )
        if continue_:
            extrait = df.sample(min(sample, len(df)), random_state=random_state)
            sns.scatterplot(
                data=extrait, x=col, y=target, ax=ax, s=8, alpha=0.3, edgecolor=None
            )
        else:
            sns.boxplot(data=df, x=col, y=target, ax=ax)
        ax.set_title(f"{target} vs {col}")

    for ax in flat[len(cols):]:  # cases restantes d'une grille incomplète
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def plot_cat_heatmaps(
    df: pd.DataFrame,
    target: str,
    cols: list[str],
    ncols: int = 3,
    aggfunc: str = "mean",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Une heatmap par couple de variables catégorielles.

    La valeur d'une case est l'agrégat de `target` pour ce croisement.

    Sans `vmin`/`vmax`, l'échelle de couleur se recale sur chaque table :
    de simples fluctuations d'échantillonnage prennent alors l'apparence
    d'une structure forte. Fixer les bornes rend les heatmaps comparables
    entre elles et montre si les écarts sont réels.
    """
    use_inline_backend()
    paires = list(combinations(cols, 2))
    nrows = -(-len(paires) // ncols)  # arrondi supérieur

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    flat = np.ravel(axes)

    for ax, (ligne, colonne) in zip(flat, paires):
        table = df.pivot_table(index=ligne, columns=colonne, values=target, aggfunc=aggfunc)
        sns.heatmap(
            table, cmap="coolwarm", annot=True, fmt=".2f", ax=ax, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{ligne} x {colonne}")

    for ax in flat[len(paires):]:  # cases restantes d'une grille incomplète
        ax.axis("off")

    fig.suptitle(f"{aggfunc} de {target}")
    fig.tight_layout()
    plt.show()


def _pct_and_count(pct: float, total: int) -> str:
    """Étiquette d'une part de camembert : pourcentage puis effectif."""
    n = round(pct / 100 * total)
    return f"{pct:.1f}%\n{n:,}".replace(",", " ")


def plot_pies(
    df: pd.DataFrame,
    cols: list[str],
    title: str = "Répartition des variables catégorielles",
) -> None:
    """Affiche un camembert par variable catégorielle, dans une seule figure.

    Chaque part porte son pourcentage et l'effectif correspondant.
    """
    use_inline_backend()
    nrows = -(-len(cols) // 2)  # arrondi supérieur
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))
    flat = np.ravel(axes)

    for ax, col in zip(flat, cols):
        counts = df[col].value_counts()
        ax.pie(
            counts,
            labels=counts.index,
            autopct=lambda pct, total=counts.sum(): _pct_and_count(pct, total),
            startangle=90,
        )
        ax.set_title(col)

    for ax in flat[len(cols):]:  # cases restantes d'une grille incomplète
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
