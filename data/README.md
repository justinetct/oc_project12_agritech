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
   deux groupes), puis les 2 pays dont la valeur de pluie est erronée (Monténégro et
   Soudan, voir plus bas).

Résultat : **16 319 lignes, 115 pays, 10 cultures, 1990-2013, aucune valeur manquante**.
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
| `rain_mm` | mm/an | `rainfall.csv`, valeur de pluie fixe par pays sur toute la période, reprise pour les années sans valeur |
| `pesticides_t` | tonnes de matière active | `pesticides.csv` |

### Valeurs de pluie erronées dans `rainfall.csv`

Dans `rainfall.csv`, 14 entités portent exactement la valeur de pluie d'une autre entité, proche
dans l'ordre alphabétique, répétée sur toutes les années. Pour les repérer, les 20 valeurs
partagées par deux entités du fichier ont été comparées le 2026-09-16 à l'indicateur de la Banque
mondiale `AG.LND.PRCP.MM` (« Average precipitation in depth (mm per year) »), dont le nom et les
valeurs vérifiées correspondent pour l'essentiel au fichier :

- 6 sont de vraies coïncidences, identiques à la Banque mondiale pour les deux entités (par
  exemple Burundi et Corée du Sud, 1 274 mm) : elles restent en place ;
- 14 sont des copies : la valeur appartient à l'autre entité, et la Banque mondiale n'a aucune
  valeur, ou une valeur différente, pour l'entité qui la reprend.

| Entité | Pluie du fichier (mm) | Valeur recopiée depuis | Banque mondiale (mm) | Dans le pipeline |
|---|---|---|---|---|
| Andorra | 1 010 | Angola | aucune valeur | absent : pas de code ISO3 |
| Kiribati | 630 | Kenya | aucune valeur | absent : pas de code ISO3 |
| Liechtenstein | 56 | Libya | aucune valeur | absent : pas de code ISO3 |
| Marshall Islands | 560 | Malta | aucune valeur | absent : pas de code ISO3 |
| Micronesia | 758 | Mexico | aucune valeur | absent : pas de code ISO3 |
| Montenegro | 241 | Mongolia | aucune valeur | **retiré : pluie erronée** |
| Nauru | 285 | Namibia | aucune valeur | absent : pas de code ISO3 |
| Palau | 494 | Pakistan | aucune valeur | absent : pas de code ISO3 |
| Samoa | 1 583 | St. Vincent and the Grenadines | 2 880 | absent : pas de code ISO3 |
| Serbia | 686 | Senegal | aucune valeur | retiré avant : sans pesticides |
| South Sudan | 495 | South Africa | 900 | retiré avant : ni température ni pesticides |
| Sudan | 1 712 | Sri Lanka | 250 | **retiré : pluie erronée** |
| Tuvalu | 161 | Turkmenistan | aucune valeur | absent : pas de code ISO3 |
| Vanuatu | 206 | Uzbekistan | 2 000 | retiré avant : sans température |

Seuls le Monténégro et le Soudan arrivaient encore jusqu'au dataset nettoyé. Leurs valeurs ne sont
pas remplacées par celles de la Banque mondiale : la version utilisée pour construire le fichier
n'est pas connue, et certaines valeurs ont été révisées depuis (Égypte : 51 mm dans le fichier,
18,1 aujourd'hui). Plutôt que d'inventer ou de mélanger des données, ces deux pays sont retirés
(38 lignes) ; aucune autre valeur de pluie n'est modifiée.

## Datasets d'entraînement

Produits par `notebooks/06_prepare_training_dataset.ipynb`, non versionnés.

| Fichier | Lignes | Colonnes |
|---|---|---|
| `predict_training_dataset.csv` | 999 769 | 9 variables candidates `Crop`, `Soil_Type`, `Rainfall_mm`, `Temperature_Celsius`, `Fertilizer_Used`, `Irrigation_Used`, `Region`, `Weather_Condition`, `Days_to_Harvest` ; cible `Yield_tons_per_hectare` |
| `recommend_training_dataset.csv` | 15 636 (115 pays, 1991-2013) | `area` hors modèle ; variables candidates `iso3`, `year`, `crop`, `temp_hist`, `rain_mm`, `pest_hist`, `log_pest_hist` ; cible `yield_t_ha` |
| `predict_negative_yield_rows.csv` | 231 | rendements négatifs, toutes les colonnes d'origine ; hors entraînement et évaluation |
