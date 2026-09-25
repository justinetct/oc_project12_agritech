"""Compromis performance / taille d'une forêt : mesurer un modèle, puis chercher les meilleurs compromis.

Outils du notebook 15, sortis du notebook pour qu'il ne garde que le raisonnement. Deux raccourcis, sur
lesquels repose toute la recherche :

- les `k` premiers arbres d'une forêt **sont** une forêt de `k` arbres, puisqu'une forêt est une moyenne
  d'arbres indépendants : `scores_par_prefixe` et `tronquer` évitent de réapprendre un modèle par valeur
  de `n_estimators` ;
- le fichier d'une forêt est proportionnel à son nombre de nœuds (72 octets par nœud avant compression) :
  une seule écriture donne la taille de toutes ses sous-forêts.

`mesurer` applique les deux à une configuration, `lancer` les parallélise, `pareto` et `voisins` guident la
recherche. Le notebook garde la grille, les tirages, les tableaux et les graphiques.
"""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

from agritech.config import SEED
from agritech.preprocessing import make_preprocessing
from agritech.recommend_service import recommander

# Hyperparamètres qui décrivent la forme des arbres, donc la taille du modèle.
STRUCTURELS = ["max_features", "min_samples_split", "min_samples_leaf", "max_depth", "max_leaf_nodes"]

# Compressions comparées : `joblib` accepte le niveau zlib directement, ou un couple (format, niveau).
COMPRESSIONS = {"brut": 0, "zlib": 3, "lzma": ("lzma", 3)}


def foret(n_estimators: int, **params) -> ExtraTreesRegressor:
    """ExtraTrees des essais : graine fixe et un seul fil, comme le modèle qui sera déployé."""
    return ExtraTreesRegressor(n_estimators=n_estimators, random_state=SEED, n_jobs=1, **params)


def preparer_matrices(X: pd.DataFrame, y, cv, categorielles: list[str], numeriques: list[str]):
    """Encode les données une fois pour toutes : un couple par fold, plus le jeu complet.

    Le preprocessing ne dépend ni du modèle ni de `y` : l'apprendre une fois par fold suffit, et les
    milliers de forêts de la recherche travaillent ensuite sur des matrices déjà encodées. Chaque fold
    n'est encodé que sur ses lignes d'apprentissage : pas de fuite de données.

    Renvoie `(plis, X_complet, y_complet, encodeur)` où `plis` contient, par fold,
    `(X_apprentissage, y_apprentissage, X_validation, y_validation)`.
    """
    colonnes = categorielles + numeriques
    valeurs = np.asarray(y, dtype=float)
    plis = []
    for apprentissage, validation in cv:
        encodeur = make_preprocessing(categorielles, numeriques).fit(X[colonnes].iloc[apprentissage])
        plis.append((encodeur.transform(X[colonnes].iloc[apprentissage]), valeurs[apprentissage],
                     encodeur.transform(X[colonnes].iloc[validation]), valeurs[validation]))
    encodeur = make_preprocessing(categorielles, numeriques).fit(X[colonnes])
    return plis, encodeur.transform(X[colonnes]), valeurs, encodeur


def deployable(encodeur, modele) -> Pipeline:
    """Le pipeline tel qu'il serait sauvegardé pour l'API : preprocessing appris sur tout, puis le modèle."""
    return Pipeline([("preprocessing", encodeur), ("model", modele)])


def scores_par_prefixe(modele, X_val, y_val, points) -> dict[int, tuple[float, float, float]]:
    """RMSE, MAE et R² des `k` premiers arbres, pour chaque `k` de `points`.

    Une forêt étant la moyenne de ses arbres, la moyenne cumulée des prédictions arbre par arbre donne
    exactement les prédictions d'une forêt de `k` arbres.
    """
    cumul, sortie, variance = np.zeros(len(y_val)), {}, float(np.var(y_val))
    for indice, arbre in enumerate(modele.estimators_, start=1):
        cumul += arbre.predict(X_val)
        if indice in points:
            erreur = cumul / indice - y_val
            sortie[indice] = (float(np.sqrt(np.mean(erreur ** 2))), float(np.mean(np.abs(erreur))),
                              float(1 - np.mean(erreur ** 2) / variance))
    return sortie


def tronquer(modele, k: int):
    """Forêt réduite à ses `k` premiers arbres, identique à un apprentissage direct à `k` arbres."""
    reduit = clone(modele)
    reduit.__dict__.update({cle: valeur for cle, valeur in modele.__dict__.items() if cle.endswith("_")})
    reduit.estimators_, reduit.n_estimators = modele.estimators_[:k], k
    return reduit


def structure(modele) -> dict:
    """Anatomie d'une forêt : arbres, nœuds, feuilles et profondeurs."""
    noeuds = np.array([arbre.tree_.node_count for arbre in modele.estimators_])
    profondeurs = np.array([arbre.tree_.max_depth for arbre in modele.estimators_])
    return {"arbres": len(noeuds), "noeuds": int(noeuds.sum()), "noeuds par arbre": round(noeuds.mean()),
            "feuilles par arbre": round(float(np.mean([(a.tree_.children_left == -1).sum() for a in modele.estimators_]))),
            "profondeur moyenne": round(profondeurs.mean(), 1), "profondeur max": int(profondeurs.max())}


def taille(objet, dossier: str, nom: str, compress=3) -> float:
    """Taille en Mo du fichier joblib réellement écrit, puis supprimé."""
    chemin = Path(dossier) / nom
    joblib.dump(objet, chemin, compress=compress)
    octets = chemin.stat().st_size
    chemin.unlink()
    return octets / 1e6


def mesurer_artefact(pipeline, demande: pd.DataFrame, dossier: str, formats=("zlib",), repetitions: int = 20) -> dict:
    """Ce que coûte un modèle une fois déployé : taille du fichier, chargement et temps d'une demande.

    Taille mesurée sans compression, en zlib et en lzma ; temps de chargement pour chaque format de
    `formats` ; enfin temps moyen d'une recommandation de 10 cultures, sur le modèle rechargé.
    """
    mesures = {f"{nom} (Mo)": taille(pipeline, dossier, "artefact.joblib", compress=compression)
               for nom, compression in COMPRESSIONS.items()}
    chemin = Path(dossier) / "artefact.joblib"
    for nom in formats:
        joblib.dump(pipeline, chemin, compress=COMPRESSIONS[nom])
        debut = time.perf_counter()
        recharge = joblib.load(chemin)
        mesures[f"chargement {nom} (s)"] = time.perf_counter() - debut
    debut = time.perf_counter()
    for _ in range(repetitions):
        recommander(recharge, demande)
    mesures["10 cultures (ms)"] = (time.perf_counter() - debut) / repetitions * 1000
    chemin.unlink()
    return mesures


def normaliser(config: dict) -> tuple:
    """Clé d'une configuration, en types Python purs.

    scikit-learn utilise `max(min_samples_split, 2 x min_samples_leaf)` : deux configurations qui ne
    diffèrent que par un `min_samples_split` plus petit donnent exactement la même forêt, et n'en font
    donc qu'une seule ici.
    """
    entier = lambda valeur: None if valeur is None or pd.isna(valeur) else int(valeur)
    propre = {"max_features": float(config["max_features"]),
              "min_samples_split": max(int(config["min_samples_split"]), 2 * int(config["min_samples_leaf"])),
              "min_samples_leaf": int(config["min_samples_leaf"]),
              "max_depth": entier(config["max_depth"]), "max_leaf_nodes": entier(config["max_leaf_nodes"])}
    return tuple(propre[nom] for nom in STRUCTURELS)


def mesurer(cle: tuple, matrices, nombres_arbres: list[int], budget_noeuds: int, annees: list, dossier: str) -> list[dict]:
    """Une configuration : validation croisée et taille du fichier, pour chaque nombre d'arbres.

    Un seul apprentissage par fold, au plus grand nombre d'arbres utile, puis les préfixes de la forêt.
    La taille est mesurée sur le modèle complet et ramenée à chaque préfixe par son nombre de nœuds.
    `budget_noeuds` arrête la forêt quand le fichier deviendrait trop gros pour un compromis de taille ;
    une sonde de 10 arbres sert à estimer le nombre de nœuds par arbre, puis la même forêt est prolongée.
    """
    plis_encodes, X_complet, y_complet, encodeur = matrices
    config = dict(zip(STRUCTURELS, cle))
    sonde = foret(10, warm_start=True, **config).fit(X_complet, y_complet)
    par_arbre = np.mean([arbre.tree_.node_count for arbre in sonde.estimators_])
    points = [k for k in nombres_arbres if k * par_arbre <= budget_noeuds] or nombres_arbres[:1]
    modele = sonde.set_params(n_estimators=max(points)).fit(X_complet, y_complet)  # prolonge la même forêt
    noeuds = np.cumsum([arbre.tree_.node_count for arbre in modele.estimators_])
    taille_max = taille(deployable(encodeur, modele), dossier, f"{os.getpid()}.joblib")
    controle = taille(deployable(encodeur, tronquer(modele, points[0])), dossier, f"{os.getpid()}_c.joblib")
    plis = [scores_par_prefixe(foret(max(points), **config).fit(X_app, y_app), X_val, y_val, set(points))
            for X_app, y_app, X_val, y_val in plis_encodes]

    lignes = []
    for k in points:
        rmse = np.array([pli[k][0] for pli in plis])
        lignes.append({**{nom: (np.nan if valeur is None else valeur) for nom, valeur in config.items()},
                       "n_estimators": k, "rmse": rmse.mean(), "rmse_std": rmse.std(),
                       "mae": np.mean([pli[k][1] for pli in plis]), "r2": np.mean([pli[k][2] for pli in plis]),
                       **{f"rmse_{annee}": valeur for annee, valeur in zip(annees, rmse)},
                       "noeuds": int(noeuds[k - 1]), "taille": taille_max * noeuds[k - 1] / noeuds[-1],
                       "taille_mesuree": controle if k == points[0] else (taille_max if k == max(points) else np.nan)})
    return lignes


def lancer(cles, matrices, nombres_arbres: list[int], budget_noeuds: int, annees: list, n_jobs: int = 12) -> pd.DataFrame:
    """Évalue les configurations en parallèle : un processus par configuration, un fil par forêt."""
    # loky relance les processus dont la mémoire grossit et refait leurs essais : l'avertissement est sans effet
    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")
    with tempfile.TemporaryDirectory() as dossier:
        paquets = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(mesurer)(cle, matrices, nombres_arbres, budget_noeuds, annees, dossier) for cle in cles)
    return pd.DataFrame([ligne for paquet in paquets for ligne in paquet])


def pareto(table: pd.DataFrame, x: str = "taille", y: str = "rmse") -> pd.DataFrame:
    """Modèles non dominés : aucun autre n'est à la fois plus petit et meilleur."""
    tri = table.sort_values([x, y]).reset_index(drop=True)
    garde, meilleur = [], np.inf
    for position, valeur in enumerate(tri[y].to_numpy()):
        if valeur < meilleur - 1e-12:
            garde.append(position)
            meilleur = valeur
    return tri.loc[garde].reset_index(drop=True)


def voisins(table: pd.DataFrame, grille: dict[str, list]) -> set[tuple]:
    """Configurations voisines de celles du front : une valeur déplacée d'un cran dans la grille."""
    sortie = set()
    for _, ligne in table.iterrows():
        config = {nom: (None if pd.isna(ligne[nom]) else ligne[nom]) for nom in STRUCTURELS}
        for nom, valeurs in grille.items():
            if config[nom] in valeurs:
                position = valeurs.index(config[nom])
                choix = valeurs[max(position - 1, 0):position + 2]
            else:  # min_samples_split relevé par la normalisation : on prend les deux valeurs qui l'encadrent
                finies = [valeur for valeur in valeurs if valeur is not None]
                choix = [max([valeur for valeur in finies if valeur < config[nom]], default=finies[0]),
                         min([valeur for valeur in finies if valeur > config[nom]], default=finies[-1])]
            for voisine in choix:
                sortie.add(normaliser({**config, nom: voisine}))
    return sortie


def reglages_depuis(ligne) -> tuple[int, dict, float, float]:
    """Relit une ligne de résultats : nombre d'arbres, réglages en types Python purs, RMSE et MAE."""
    params = {nom: (None if pd.isna(ligne[nom]) else ligne[nom]) for nom in STRUCTURELS}
    params = {nom: (float(valeur) if nom == "max_features" else None if valeur is None else int(valeur))
              for nom, valeur in params.items()}
    return int(ligne["n_estimators"]), params, float(ligne["rmse"]), float(ligne["mae"])
