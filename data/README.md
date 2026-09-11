# Données

Les fichiers de données ne sont **pas versionnés** dans Git (≈ 96 Mo, dont un CSV
de 89 Mo). Ils sont fournis dans le cadre du Projet 12 OpenClassrooms et doivent
être placés localement dans ce dossier.

## Arborescence attendue

```
data/
├── agriculture-crop-yield/
│   └── crop_yield.csv                 # ~89 Mo, 1 000 000 lignes
├── crop-yield-prediction/
│   ├── yield.csv                      # ~3.9 Mo
│   ├── yield_df.csv                   # ~1.5 Mo
│   ├── temp.csv                       # ~1.4 Mo
│   ├── pesticides.csv                 # ~380 Ko
│   └── rainfall.csv                   # ~136 Ko
├── geo/
│   └── ne_110m_admin_0_countries.geojson   # ~820 Ko, contours des pays
└── processed/                              # générés par les notebooks
    ├── crop_yield_clean.csv                # ~0,9 Mo, dataset historique nettoyé
    ├── predict_negative_yield_rows.csv     # ~22 Ko, 231 rendements négatifs, hors entraînement
    ├── predict_training_dataset.csv        # ~89 Mo, dataset d'entraînement /predict
    └── recommend_training_dataset.csv      # ~1,3 Mo, dataset d'entraînement /recommend
```

## Provenance

Les deux jeux de données sont fournis dans le cadre du Projet 12 OpenClassrooms.
Ils ne sont pas redistribués dans ce repository.

- `agriculture-crop-yield` — observations au niveau parcelle
- `crop-yield-prediction` — fichiers annuels agrégés par pays

## Contours des pays

`data/geo/ne_110m_admin_0_countries.geojson` est utilisé par `agritech.geo` pour
les cartes du monde. Il vient de [Natural Earth](https://www.naturalearthdata.com/)
(échelle 1:110m, 177 pays, domaine public) et n'est pas versionné non plus :

```bash
mkdir -p data/geo
curl -sSL -o data/geo/ne_110m_admin_0_countries.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
```

À cette échelle, les micro-États et la plupart des territoires insulaires n'ont
pas de contour : ils n'apparaissent pas sur les cartes, même présents dans les
données.

Une fois les fichiers en place, l'arborescence ci-dessus doit être respectée :
le code et les notebooks s'appuient sur ces chemins.

## Dataset historique nettoyé

`data/processed/crop_yield_clean.csv` rassemble les quatre fichiers de
`crop-yield-prediction/`, avec une ligne par pays, année et culture (clé
`iso3 + year + crop`), sur 1990-2013. Il est construit par
`notebooks/04_build_crop_yield_dataset.ipynb` en deux étapes :

1. **État après jointures** : 22 679 lignes, 168 pays, avec des valeurs manquantes.
   Non sauvegardé.
2. **Nettoyage** : une année sans pluie reprend la valeur connue du pays (2003, absente
   de `rainfall.csv`, et Bahamas 1990-1991) ; les 51 pays sans aucune température ou
   sans aucun pesticide sont retirés (36 sans température, 24 sans pesticides, 9 dans les
   deux groupes).

Résultat : **16 357 lignes, 117 pays, 10 cultures, 1990-2013, aucune valeur manquante**.
Les notebooks 05 et 06 partent de ce fichier. Il n'est pas versionné et se reconstruit en
réexécutant le notebook 04.

| Colonne | Unité | Source |
|---|---|---|
| `iso3` | code ISO3 | conversion des noms de pays via `agritech.geo` |
| `area` | nom du pays | `yield.csv` |
| `year` | année | — |
| `crop` | culture | `yield.csv` |
| `yield_t_ha` | tonnes/hectare | `yield.csv`, converti depuis hg/ha |
| `avg_temp` | °C | `temp.csv`, moyenne des relevés du pays et de l'année |
| `rain_mm` | mm/an | `rainfall.csv`, pluie moyenne habituelle du pays, reprise pour les années sans valeur |
| `pesticides_t` | tonnes de matière active | `pesticides.csv` |

## Datasets d'entraînement

Produits par `notebooks/06_prepare_training_dataset.ipynb`, non versionnés.

| Fichier | Lignes | Colonnes |
|---|---|---|
| `predict_training_dataset.csv` | 999 769 | 9 variables candidates `Crop`, `Soil_Type`, `Rainfall_mm`, `Temperature_Celsius`, `Fertilizer_Used`, `Irrigation_Used`, `Region`, `Weather_Condition`, `Days_to_Harvest` ; cible `Yield_tons_per_hectare` |
| `recommend_training_dataset.csv` | 15 664 (117 pays, 1991-2013) | hors modèle `iso3`, `area`, `year` ; 5 variables candidates `crop`, `temp_hist`, `rain_mm`, `pest_hist`, `log_pest_hist` ; cible `yield_t_ha` |
| `predict_negative_yield_rows.csv` | 231 | rendements négatifs, toutes les colonnes d'origine ; hors entraînement et évaluation |
