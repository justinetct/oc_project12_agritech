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
└── geo/
    └── ne_110m_admin_0_countries.geojson   # ~820 Ko, contours des pays
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
