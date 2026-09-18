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

Le pays sert à préremplir les valeurs. Son code (`iso3`) et l'année (`year`) doivent encore être testés comme variables du modèle.

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
| `05_dataset_comparison.ipynb` | Comparaison des deux datasets et définition de la stratégie `/predict` / `/recommend` | — |
| `06_prepare_training_dataset.ipynb` | Construction des datasets utilisés pour la modélisation : sélection des lignes et variables, création de l’historique pour /recommend et contrôles contre les fuites de données | `data/processed/predict_training_dataset.csv` et `data/processed/recommend_training_dataset.csv` |
| `07_predict_training_baseline.ipynb` | Baseline `/predict` : `DummyRegressor` et `LinearRegression` sur toutes les variables puis sur les variables réduites, comparés par validation croisée à 5 folds sur le train ; jeu de test réservé à l'évaluation finale | runs MLflow, expérience `oc_p12_agritech_predict` |
| `08_predict_feature_engineering.ipynb` | Sélection des variables et feature engineering sur la baseline linéaire : interactions testées et choix des 4 variables sélectionnées de `/predict` | runs MLflow, expérience `oc_p12_agritech_predict` |
| `09_predict_nonlinear_models.ipynb` | Comparaison `/predict` de six modèles non linéaires à la régression linéaire, sur toutes les variables et sur les variables réduites ; feature engineering du notebook 08 vérifié sur les cinq modèles d'ensemble ; importance des variables | runs MLflow, expérience `oc_p12_agritech_predict` |
| `10_predict_model_tuning.ipynb` | Tuning des modèles non linéaires sur les variables sélectionnées, comparaison fold par fold avec la régression linéaire et choix du modèle `/predict` | runs MLflow, expérience `oc_p12_agritech_predict` |
| `11_predict_final_evaluation.ipynb` | Évaluation finale `/predict` sur le jeu de test réservé, analyse des erreurs et sauvegarde du modèle | run MLflow, `models/predict_model.joblib` et `models/predict_model_metadata.json` |
| `12_recommend_training_baseline.ipynb` | Baseline `/recommend` : validation temporelle 2008-2012, `DummyRegressor`, culture seule, puis culture et conditions avec pesticides bruts ou en logarithme ; test 2013 réservé | runs MLflow, expérience `oc_p12_agritech_recommend` |

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

Les expériences sont journalisées avec MLflow ; la configuration est gérée par l'environnement. Le projet reste
exécutable sans configuration distante.

```bash
cp .env.example .env   # facultatif
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
├── models/                          # modèles finaux versionnés et leurs métadonnées
├── notebooks/                       # analyses, préparation des données et modélisation
├── scripts/                         # génération des figures et du rapport HTML
├── src/agritech/                    # code réutilisable
├── pyproject.toml
└── README.md
```


## État actuel

L’exploration, l’ACP, le nettoyage des données historiques, la construction des datasets et la modélisation `/predict`, évaluation finale et sauvegarde du modèle comprises, sont terminés. `/recommend` est le chantier en cours.

Datasets préparés :

* dataset historique nettoyé (`crop_yield_clean.csv`) : 16 319 lignes, 115 pays, 10 cultures, période 1990-2013, aucune valeur manquante ; le Monténégro et le Soudan en sont exclus, car leur valeur de pluie est recopiée depuis un autre pays dans la source (détail dans [`data/README.md`](data/README.md)) ;
* /predict : 999 769 lignes, 9 variables candidates ; les 231 lignes au rendement négatif sont exclues de l’entraînement ; le contrôle du notebook 11 montre que le modèle final leur prédit à toutes un rendement positif ;
* /recommend : 15 636 lignes, 115 pays, 10 cultures, période 1991-2013, 7 variables candidates : `iso3`, `year`, `crop`, `temp_hist`, `rain_mm`, `pest_hist` et `log_pest_hist`, les pesticides étant gardés en brut et en logarithme pour comparer les deux versions ; l’apport du pays (`iso3`) et de l’année (`year`) reste à évaluer dans le notebook 12.

### `/predict` : modèle final évalué

Tous les modèles sont comparés par validation croisée à 5 folds sur le jeu d’entraînement. **Le jeu de test n’a servi à aucun choix de modèle, de variable ou d’hyperparamètre** : il sert uniquement à l’évaluation finale du modèle retenu.

* notebook 07 : la régression linéaire réduit la RMSE d’environ 70 % par rapport au `DummyRegressor` ;
* notebook 08 : les interactions testées n’apportent rien, et le modèle garde les 4 variables sélectionnées : `Rainfall_mm`, `Temperature_Celsius`, `Fertilizer_Used` et `Irrigation_Used`. `Crop` reste une information de l’application, mais pas une variable du modèle actuel ;
* notebook 09 : arbre de décision, forêt aléatoire, HistGradientBoosting, XGBoost, LightGBM et CatBoost ne font pas mieux que la régression linéaire, sur toutes les variables comme sur les variables réduites ; le feature engineering du notebook 08 ne les améliore pas non plus ;
* notebook 10 : CatBoost et HistGradientBoosting optimisés s’approchent de la régression linéaire sans la dépasser, ni en moyenne ni sur un seul fold ;
* notebook 11 : évaluation finale du modèle retenu sur le jeu de test, analyse des erreurs, contrôle des 231 rendements négatifs et sauvegarde du modèle.

| Modèle, variables sélectionnées | RMSE CV (t/ha) | MAE CV (t/ha) | R² CV |
|---|---:|---:|---:|
| `LinearRegression` | 0,500332 | 0,399292 | 0,91289 |
| `CatBoost` optimisé | 0,500411 | 0,399369 | 0,91286 |
| `HistGradientBoosting` optimisé | 0,500640 | 0,399543 | 0,91278 |

**Modèle final : `LinearRegression` avec les 4 variables sélectionnées.** Sur le jeu de test (20 % des lignes), il obtient une RMSE de 0,499268 t/ha, une MAE de 0,398338 t/ha et un R² de 0,91323, très proches de la validation croisée. Le pipeline complet (preprocessing et régression) est sauvegardé dans `models/predict_model.joblib`, et ses métadonnées dans `models/predict_model_metadata.json`.

### `/recommend` : baseline en cours

Le notebook 12 pose une première baseline. Les modèles sont comparés par validation temporelle sur 2008-2012 : chaque année est prédite par un modèle appris sur les années précédentes. **Le test 2013 reste réservé.**

* la culture seule réduit déjà l’erreur de 32 % par rapport au `DummyRegressor` ;
* les conditions du pays la réduisent encore de 6 % ;
* la régression additive donne cependant le même classement des cultures pour les 115 pays.

**Meilleure baseline actuelle : `LinearRegression` avec `crop`, `temp_hist`, `rain_mm` et `log_pest_hist`** (RMSE CV 5,4034 t/ha, R² CV 0,5899). Avant de figer la baseline, il reste à la comparer aux versions avec `year`, avec `iso3`, et avec `year` et `iso3`. Viendront ensuite le feature engineering temporel et des modèles plus riches.
