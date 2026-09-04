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
└── processed/
    └── crop_yield_prediction_1990_2013.csv # ~1,2 Mo, dataset consolidé
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

## Dataset consolidé

`data/processed/crop_yield_prediction_1990_2013.csv` assemble les quatre fichiers de
`crop-yield-prediction/` à la granularité **pays × année × culture**, clé
`iso3 + year + crop`, sur 1990-2013.

Il n'est pas versionné et se reconstruit en réexécutant
`notebooks/04_build_crop_yield_dataset.ipynb`.

| Colonne | Unité | Source |
|---|---|---|
| `iso3` | code ISO3 | harmonisation via `agritech.geo` |
| `area` | libellé pays | `yield.csv` |
| `year` | année | — |
| `crop` | culture | `yield.csv` |
| `yield_t_ha` | tonnes/hectare | `yield.csv`, converti depuis hg/ha |
| `avg_temp` | °C | `temp.csv`, moyenne des relevés du pays et de l'année |
| `rain_mm` | mm/an | `rainfall.csv`, normale climatique constante par pays |
| `pesticides_t` | tonnes de matière active | `pesticides.csv` |

Les valeurs manquantes sont conservées : elles traduisent une absence de couverture du
pays dans la source concernée, pas des trous dans une série existante.
