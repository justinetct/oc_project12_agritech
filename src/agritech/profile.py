"""Rapport de profiling partagé par les notebooks d'analyse exploratoire."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pandas as pd
from data_profiling import ProfileReport


def profile_report(
    df: pd.DataFrame,
    profile_path: Path | None = None,
    verbose: bool = True,
) -> ProfileReport:
    """Calcule le rapport de profiling, l'enregistre et affiche un court résumé.

    `progress_bar=False` coupe les barres de progression de la librairie. Une
    dernière barre est écrite sur `stderr` par `data_profiling` sans tenir
    compte de cette option : `redirect_stderr` la met de côté.

    Le rapport complet est retourné. Dans un notebook, il faut l'affecter à une
    variable ou terminer la ligne par `;` : laissé comme dernière expression
    d'une cellule, il incruste plus d'un Mo de HTML dans le fichier.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        profile = ProfileReport(
            df,
            title="Profiling Report",
            minimal=True,
            progress_bar=False,
            correlations={"auto": {"calculate": True, "threshold": 0.5}},
            duplicates={"head": 5},
        )
        if profile_path:
            profile.to_file(profile_path)

        desc = profile.get_description()

    if verbose:
        print(f"Observations       : {desc.table['n']:,}")
        print(f"Variables          : {desc.table['n_var']}")
        print(
            f"Cellules manquantes: {desc.table['n_cells_missing']} "
            f"({desc.table['p_cells_missing']:.2%})"
        )
        print(
            f"Lignes dupliquees  : {desc.table['n_duplicates']} "
            f"({desc.table['p_duplicates']:.2%})"
        )
        print(f"Taille en memoire  : {desc.table['memory_size'] / 1024**2:.1f} Mo")
        print(f"Types              : {desc.table['types']}")
        print("\nAlertes :")
        for a in desc.alerts:
            print(" -", a)

    return profile
