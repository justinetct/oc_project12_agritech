# Reconstruction des modèles servis

`make rebuild-models` reconstruit les 5 fichiers de `models/` chargés par l'API à partir des seules données
brutes : sans `data/processed/`, sans notebook et sans MLflow.

| Script | Rôle |
|---|---|
| `prepare_data.py` | prépare en mémoire les datasets d'entraînement `/predict` et `/recommend`, avec les mêmes étapes que les notebooks 04 et 06 ; n'écrit aucun fichier |
| `rebuild_models.py` | appelle `prepare_data.py`, réentraîne les deux modèles servis, les compare aux fichiers de `models/`, puis écrit les 5 fichiers |

## Commandes

```bash
make rebuild-models
```

La commande écrit dans `models/`. Pour une reconstruction de contrôle dans un autre dossier :

```bash
poetry run python scripts/rebuild_models.py --output-dir /tmp/agritech-models
```

## Entrées et sorties

| Entrée, non versionnée (voir [`data/README.md`](../data/README.md)) | Usage |
|---|---|
| `data/agriculture-crop-yield/crop_yield.csv` | dataset `/predict` |
| `data/crop-yield-prediction/` : `yield.csv`, `temp.csv`, `rainfall.csv`, `pesticides.csv` | dataset `/recommend` (`yield_df.csv` n'est pas utilisé) |
| `data/geo/ne_110m_admin_0_countries.geojson` | codes ISO3 des pays, puis coordonnées des variables géographiques |

Sorties, dans `models/` ou dans `--output-dir` : `predict_model.joblib`, `predict_model_metadata.json`,
`recommend_model.joblib`, `recommend_model_metadata.json` et `recommend_context.json`.

La reconstruction reproduit les protocoles finaux, sans refaire la sélection de modèle, le tuning, la validation
croisée ni le suivi MLflow : `/predict` est réentraîné sur les 999 769 lignes et `/recommend` sur 1991-2013. Les
métriques d'évaluation finale sont recopiées dans les métadonnées, pas recalculées.

## Vérifications de reproductibilité

1. **Datasets** : `prepare_data.py` fait passer les données par le format CSV, en mémoire, comme les notebooks par
   les fichiers de `data/processed/`. Ce passage compte : une moyenne de températures écrite `18.240000000000002`
   est relue `18.24`, et les modèles servis ont été appris sur les valeurs relues. Les deux datasets passent ensuite
   les contrôles de leur lecture dans les notebooks : 999 769 et 16 319 lignes, colonnes attendues, aucune valeur
   manquante, cible positive ou nulle.
2. **Modèles** : effectifs, variables, modalités, hyperparamètres, domaines d'apprentissage et contexte (115 pays).
3. **Avant l'écriture** : comparaison avec les fichiers déjà présents dans `models/` (prédictions sur 1 000 lignes
   du test `/predict` et sur les 15 636 lignes `/recommend`, métadonnées hors `created_on` et `versions`,
   contexte) ; le script s'arrête s'ils diffèrent.
4. **Après l'écriture** : rechargement des 5 fichiers par le code de serving de l'API, prédictions de référence et
   classement de la France.

## Après une réexécution des notebooks 11 et 15

Ces notebooks écrivent leur modèle évalué dans `models/`. Vérifier d'abord `git status models/` pour ne pas
écraser une modification volontaire, puis `git restore models/` remet les artefacts servis ; sinon, supprimer les
fichiers réécrits puis lancer `make rebuild-models`.
