# Projet 12 : Concevez un système de recommandations pour une agriculture optimisée par les données

## Objectif métier

Deux services sont visés :

### `/predict` — estimation de rendement

- l'utilisateur choisit une culture ;
- il renseigne les conditions de sa parcelle ;
- le système retourne une estimation du rendement.

### `/recommend` — classement des cultures

- l'utilisateur renseigne les conditions de sa parcelle ;
- le système teste les cultures possibles ;
- il retourne un classement par rendement prédit décroissant.

## Données

Deux jeux de données sont disponibles :

- **Agriculture CropYield Dataset** — observations au niveau parcelle.
- **CropYield Prediction Dataset** — issu de plusieurs fichiers annuels par pays.

La stratégie d'exploitation et de consolidation de ces deux sources **n'est pas
encore arrêtée** : elle sera décidée après l'analyse exploratoire.

## Prérequis

- Python 3.12
- [Poetry](https://python-poetry.org/) 2.x

## Installation

```bash
poetry install
```

## Structure du dépôt

```
.
├── data/          # jeux de données locaux (non versionnés)
├── notebooks/     # analyse exploratoire et expérimentations
├── src/agritech/  # code réutilisable
├── pyproject.toml
└── README.md
```

Le dépôt évoluera au fur et à mesure du projet (tests, service d'inférence,
industrialisation).

## État actuel

Initialisation du projet : environnement Poetry, structure de base et
documentation. Aucune analyse ni modélisation n'a encore été réalisée.
