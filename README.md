# Projet 12 : Concevez un système de recommandations pour une agriculture optimisée par les données

📄 **[Rapport technique HTML](https://justinetct.github.io/oc_project12_agritech/rapport_technique.html)** — source Markdown : [`docs/rapport_technique.md`](docs/rapport_technique.md)

## Objectif métier

Deux services sont visés :

### `/predict` — estimation de rendement

- l'utilisateur choisit une culture ;
- il renseigne les conditions de sa parcelle ;
- le système retourne une estimation du rendement.

### `/recommend` — classement des cultures

- l'utilisateur choisit son pays ;
- l'application préremplit la température, la pluie et les pesticides avec les valeurs historiques connues du pays, affichées comme valeurs du pays ;
- l'utilisateur peut modifier ces valeurs pour décrire son contexte local ;
- le modèle prédit le rendement des 10 cultures ;
- l'application retourne un classement par rendement prédit décroissant.

Le pays n'est pas une variable du modèle : il sert uniquement à préremplir les valeurs.

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
- pour `/recommend`, les features historiques n'utilisent que les années précédentes : l'application est pensée pour 2014, avec les données connues jusqu'en 2013.

## Notebooks

| Notebook | Rôle | Fichier produit |
|---|---|---|
| `01_eda_agriculture_crop_yield.ipynb` | Analyse exploratoire du dataset Agriculture CropYield | rapport HTML de profiling, non versionné |
| `02_pca_agriculture_crop_yield.ipynb` | ACP exploratoire et étude de la structure des variables | — |
| `03_eda_crop_yield_prediction.ipynb` | Analyse des sources historiques et assemblage exploratoire | rapports HTML de profiling, non versionnés |
| `04_build_crop_yield_dataset.ipynb` | Construction du dataset historique : jointures (22 679 lignes, 168 pays), puis nettoyage | `data/processed/crop_yield_clean.csv` |
| `05_dataset_consolidation_strategy.ipynb` | Comparaison des deux datasets et définition de la stratégie `/predict` / `/recommend` | — |
| `06_prepare_training_dataset.ipynb` | Construction des datasets utilisés pour la modélisation : sélection des lignes et variables, création de l’historique pour /recommend et contrôles contre les fuites de données | `data/processed/predict_training_dataset.csv` et `data/processed/recommend_training_dataset.csv` |

Les fichiers de données générés sont reproductibles depuis les notebooks et ne sont pas versionnés.

## Rapport technique

Le rapport de synthèse est rédigé en Markdown dans [`docs/rapport_technique.md`](docs/rapport_technique.md)
et publié en HTML sur [GitHub Pages](https://justinetct.github.io/oc_project12_agritech/rapport_technique.html).
Le HTML est **généré** depuis le Markdown : le contenu n'est écrit qu'une fois.

```bash
# Figures du rapport, à partir des données locales
poetry run python scripts/make_report_figures.py

# Page HTML, à partir de docs/rapport_technique.md
poetry run python scripts/build_report.py
```

## Installation

**Prérequis :**
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
├── docs/                            # rapport technique (Markdown, HTML, assets)
├── notebooks/                       # analyses et préparation des données
├── scripts/                         # génération des figures et du rapport HTML
├── src/agritech/                    # code réutilisable
├── pyproject.toml
└── README.md
```


## État actuel

L’exploration, l’ACP, le nettoyage des données historiques et la construction des datasets utilisés pour la modélisation sont terminés…

Datasets préparés :

* dataset historique nettoyé (`crop_yield_clean.csv`) : 16 357 lignes, 117 pays, 10 cultures, période 1990-2013, aucune valeur manquante ;
* /predict : 999 769 lignes, 9 variables candidates, dont les 6 de la configuration métier envisagée (`Region`, `Weather_Condition` et `Days_to_Harvest` restent à évaluer) ; les 231 lignes au rendement négatif sont exclues de l’entraînement et conservées pour un contrôle après modélisation ;
* /recommend : 15 664 lignes, 117 pays, 10 cultures, période 1991-2013, 5 variables candidates (`crop`, `temp_hist`, `rain_mm`, `pest_hist`, `log_pest_hist`), les pesticides étant gardés en brut et en logarithme pour comparer les deux versions.

La prochaine étape est la modélisation et la validation des deux services ; la sélection finale des variables sera décidée à cette étape.
