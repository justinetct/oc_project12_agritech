# Projet 12 : Concevez un système de recommandations pour une agriculture optimisée par les données

📄 **[Rapport HTML](https://justinetct.github.io/oc_project12_agritech/rapport_technique.html)** — source Markdown : [`docs/rapport_technique.md`](docs/rapport_technique.md)

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

Le pays sert à préremplir les valeurs et à le situer sur le globe : le modèle final utilise sa position géographique et l'année (`year`), mais pas son code (`iso3`). Ce parcours décrit l'application visée : l'API qui le mettra en œuvre, y compris la modification des valeurs par l'utilisateur, reste à construire.

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
| `06_prepare_training_dataset.ipynb` | Construction des datasets utilisés pour la modélisation : sélection des lignes et des variables, sans fuite de données ; `/recommend` garde tout le dataset historique nettoyé, avec ses variables d’origine | `data/processed/predict_training_dataset.csv` et `data/processed/recommend_training_dataset.csv` |
| `07_predict_training_baseline.ipynb` | Baseline `/predict` : `DummyRegressor` et `LinearRegression` sur toutes les variables puis sur les variables réduites, comparés par validation croisée à 5 folds sur le train ; jeu de test réservé à l'évaluation finale | runs MLflow, expérience `oc_p12_agritech_predict` |
| `08_predict_feature_engineering.ipynb` | Sélection des variables et feature engineering sur la baseline linéaire : interactions testées et choix des 4 variables sélectionnées de `/predict` | runs MLflow, expérience `oc_p12_agritech_predict` |
| `09_predict_nonlinear_models.ipynb` | Comparaison `/predict` de six modèles non linéaires à la régression linéaire, sur toutes les variables et sur les variables réduites ; feature engineering du notebook 08 vérifié sur les cinq modèles d'ensemble ; importance des variables | runs MLflow, expérience `oc_p12_agritech_predict` |
| `10_predict_model_tuning.ipynb` | Tuning des modèles non linéaires sur les variables sélectionnées, comparaison fold par fold avec la régression linéaire et choix du modèle `/predict` | runs MLflow, expérience `oc_p12_agritech_predict` |
| `11_predict_final_evaluation.ipynb` | Évaluation finale `/predict` sur le jeu de test réservé, analyse des erreurs et sauvegarde du modèle | run MLflow, `models/predict_model.joblib` et `models/predict_model_metadata.json` |
| `12_recommend_training_baseline.ipynb` | Baseline `/recommend` : découpage temporel (test 2013 réservé, validation par année sur 2008-2012), `DummyRegressor`, puis `LinearRegression` standardisée sur quatre jeux de variables, avec ou sans `year` et `iso3` ; choix de la baseline | runs MLflow, expérience `oc_p12_agritech_recommend` |
| `13_recommend_feature_engineering.ipynb` | Feature engineering `/recommend` avec la régression linéaire : logarithme des pesticides, interactions culture × conditions, géographie des pays, comparaison avec `iso3`, conditions des 3 années précédentes et rôle de `year` | runs MLflow, expérience `oc_p12_agritech_recommend` |
| `14_recommend_nonlinear_models.ipynb` | Modèles non linéaires `/recommend` : six familles comparées à la meilleure régression linéaire, apport de la géographie, de `iso3` et des conditions historiques, tuning léger puis approfondi, choix de trois finalistes | runs MLflow, expérience `oc_p12_agritech_recommend` |
| `15_recommend_final_model.ipynb` | Modèle final `/recommend` : recommandations réelles des finalistes dans cinq pays, carte mondiale des cultures classées n°1, comparaison avec les rendements observés en 2012, limite des cultures jamais observées dans le pays, stress test, compromis performance / taille d’ExtraTrees, choix du modèle, importance des variables du modèle final (permutation par famille sur la validation 2008-2012), évaluation finale sur 2013 et sauvegarde | run MLflow, `models/recommend_model.joblib` et `models/recommend_model_metadata.json` |

Les fichiers de données générés sont reproductibles depuis les notebooks et ne sont pas versionnés.

## Rapport

Le rapport final du projet est rédigé en Markdown dans [`docs/rapport_technique.md`](docs/rapport_technique.md)
et publié en HTML sur [GitHub Pages](https://justinetct.github.io/oc_project12_agritech/rapport_technique.html).
Le HTML est **généré** depuis le Markdown : le contenu n'est écrit qu'une fois.

```bash
# Page HTML, à partir de docs/rapport_technique.md
poetry run python scripts/build_report.py
```

Les figures sont des fichiers de `docs/assets/figures/` : graphiques de l'exploration, schéma des deux pipelines,
cartes et importance des variables reprises des sorties du notebook 15, captures des expériences MLflow.

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
├── docs/                            # rapport (Markdown, HTML publié, figures)
├── models/                          # modèles finaux versionnés et leurs métadonnées
├── notebooks/                       # analyses, préparation des données et modélisation
├── scripts/                         # génération du rapport HTML
├── src/agritech/                    # code réutilisable
├── pyproject.toml
└── README.md
```


## État actuel

L’exploration, l’ACP, le nettoyage des données historiques, la construction des datasets et la modélisation des deux services, `/predict` et `/recommend`, évaluations finales et sauvegarde des modèles comprises, sont terminés. Reste à construire l’API qui servira les deux modèles.

Datasets préparés :

* dataset historique nettoyé (`crop_yield_clean.csv`) : 16 319 lignes, 115 pays, 10 cultures, période 1990-2013, aucune valeur manquante ; le Monténégro et le Soudan en sont exclus, car leur valeur de pluie est recopiée depuis un autre pays dans la source (détail dans [`data/README.md`](data/README.md)) ;
* /predict : 999 769 lignes, 9 variables candidates ; les 231 lignes au rendement négatif sont exclues de l’entraînement ; le contrôle du notebook 11 montre que le modèle final leur prédit à toutes un rendement positif ;
* /recommend : tout le dataset historique nettoyé (16 319 lignes), avec les variables d’origine `avg_temp`, `rain_mm` et `pesticides_t` ; les variables historiques et géographiques sont calculées dans les notebooks de modélisation. La première année de chaque pays n’a pas d’historique : elle sert seulement à calculer celui de l’année suivante (1990 pour 1991). Il reste 14 941 lignes pour l’entraînement (1991-2012) et 695 pour le test (2013).

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

### `/recommend` : modèle final évalué

Les modèles sont comparés par validation temporelle sur 2008-2012 : chaque année est prédite par un modèle appris sur les années précédentes. **2013 est réservée au test final** : elle n’a servi à aucun choix de modèle, de variable ou d’hyperparamètre, et seul le modèle retenu y est évalué.

* notebook 12 : la baseline `LinearRegression` (culture, conditions de l’année et `year`) obtient une RMSE de 5,51 t/ha, contre 8,50 pour le `DummyRegressor` ; `iso3` l’améliore (4,53 t/ha), mais la baseline est choisie sans identifiant du pays ;
* notebook 13 : le logarithme des pesticides, un effet des conditions propre à chaque culture et la géographie des pays ramènent la régression linéaire à 4,67 t/ha ; les conditions des 3 années précédentes, connues au moment de recommander, font à peu près aussi bien que celles de l’année ;
* notebook 14 : les modèles à arbres font beaucoup mieux que la régression linéaire (RMSE de 1,5 à 1,9 t/ha avec la géographie et leurs réglages de départ), et les conditions historiques les améliorent encore. Après tuning, trois finalistes très proches : ExtraTrees 1,436, CatBoost 1,450 et LightGBM 1,465 t/ha ;
* notebook 15 : choix du modèle avant 2013, sur la validation 2008-2012, la taille et le déploiement ; puis évaluation finale sur 2013 et sauvegarde.

**Modèle final : `ExtraTreesRegressor` à 150 arbres** (`max_features=0.9`, `min_samples_split=3`, `random_state=42`, `n_jobs=1`), appris sur les 14 941 lignes 1991-2012. Variables : `crop`, `year`, `temp_hist` et `log_pest_hist` (moyennes des 3 années précédentes du pays, pesticides en logarithme), `rain_mm` (pluie fixe du pays), `lat_abs`, `geo_x`, `geo_y` et `geo_z` (position du pays sur le globe).

| ExtraTrees, 150 arbres | RMSE (t/ha) | MAE (t/ha) | R² |
|---|---:|---:|---:|
| Validation temporelle 2008-2012 | 1,4349 | 0,7114 | 0,9710 |
| Test final 2013 (695 lignes) | 1,6584 | 0,7615 | 0,9638 |

150 arbres font aussi bien que 300 pour un fichier deux fois plus petit : le pipeline complet (preprocessing et modèle) pèse 44,7 Mo compressés (lzma) dans `models/recommend_model.joblib`, sous la recommandation de 50 Mo par fichier de GitHub. Ses métadonnées sont dans `models/recommend_model_metadata.json`.

**Limite :** les 695 lignes de 2013 portent toutes sur des couples pays × culture déjà observés, comme 3 469 des 3 472 lignes de la validation. Le test ne mesure donc pas les recommandations de cultures jamais observées dans le pays, alors que, dans les demandes 2012, 25 des 115 cultures classées n°1 n’avaient jamais été observées dans le pays.
