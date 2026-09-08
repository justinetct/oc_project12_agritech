# Projet 12 : Concevez un système de recommandations pour une agriculture optimisée par les données

## Objectif métier

Deux services sont visés :

### `/predict` — estimation de rendement

- l'utilisateur choisit une culture ;
- il renseigne les conditions de sa parcelle ;
- le système retourne une estimation du rendement.

### `/recommend` — classement des cultures

- l'utilisateur indique son pays ;
- le système récupère le contexte historique disponible ;
- il prédit le rendement des cultures candidates ;
- il retourne un classement par rendement prédit décroissant.

## Données

Deux jeux de données sont utilisés :

- **Agriculture CropYield Dataset** — observations au niveau parcelle.
- **CropYield Prediction Dataset** — données historiques par pays et par année.

Les deux sources ne sont pas fusionnées ligne à ligne : elles répondent à deux usages différents.

- **Agriculture CropYield Dataset** alimente `/predict`.
- **CropYield Prediction Dataset** alimente `/recommend`.

**Observations :**
- le premier dataset différencie peu les cultures mais relie fortement le rendement aux conditions de parcelle ;
- le second différencie davantage les cultures et permet de comparer leurs rendements dans un même contexte national ;
- pour `/recommend`, les variables historiques sont construites uniquement à partir des années précédentes, avec un référentiel d'inférence en 2014 et des données disponibles jusqu'en 2013.

## Notebooks

| Notebook | Rôle | Fichier produit                                                                                  |
|---|---|--------------------------------------------------------------------------------------------------|
| `01_eda_agriculture_crop_yield.ipynb` | Analyse exploratoire du dataset Agriculture CropYield | rapport HTML de profiling, non versionné                                                         |
| `02_pca_agriculture_crop_yield.ipynb` | ACP et étude de la structure des variables | —                                                                                                |
| `03_eda_crop_yield_prediction.ipynb` | Analyse des sources historiques et pré-fusion exploratoire | rapports HTML de profiling, non versionnés                                                       |
| `04_build_crop_yield_dataset.ipynb` | Consolidation des sources historiques | `data/processed/crop_yield_prediction_1990_2013.csv`                                             |
| `05_dataset_consolidation_strategy.ipynb` | Comparaison des deux datasets et définition de la stratégie `/predict` / `/recommend` | —                                                                                                |
| `06_prepare_training_dataset.ipynb` | Préparation et contrôle des datasets d'entraînement | `data/processed/predict_training_dataset.csv` et `data/processed/recommend_training_dataset.csv` |

Les fichiers de données générés sont reproductibles depuis les notebooks et ne sont pas versionnés.

## Installation

**Prérequis**
- Python 3.12
- [Poetry](https://python-poetry.org/) 2.x

```bash
poetry install
```

## Structure du dépôt

```
.
├── data/
│   ├── agriculture-crop-yield/      # dataset parcelle
│   ├── crop-yield-prediction/       # sources historiques
│   ├── geo/                         # données géographiques
│   └── processed/                   # datasets générés, non versionnés
├── notebooks/                       # analyses et préparation des données
├── src/agritech/                    # code réutilisable
├── pyproject.toml
└── README.md
```


## État actuel

L’exploration, l’ACP, la consolidation des données et la préparation des jeux d’entraînement sont terminées.

Jeux préparés :

* /predict : 999 769 lignes, 6 variables explicatives ;
* /recommend : 15 664 lignes, 117 pays, 10 cultures, période 1991-2013.

La prochaine étape est la modélisation et la validation des deux services.