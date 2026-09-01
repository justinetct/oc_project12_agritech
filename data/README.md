# Données

Les fichiers de données ne sont **pas versionnés** dans Git (≈ 96 Mo, dont un CSV
de 89 Mo). Ils sont fournis dans le cadre du Projet 12 OpenClassrooms et doivent
être placés localement dans ce dossier.

## Arborescence attendue

```
data/
├── agriculture-crop-yield-dataset/
│   └── crop_yield.csv                 # ~89 Mo, 1 000 000 lignes
└── crop-yield-prediction-dataset/
    ├── yield.csv                      # ~3.9 Mo
    ├── yield_df.csv                   # ~1.5 Mo
    ├── temp.csv                       # ~1.4 Mo
    ├── pesticides.csv                 # ~380 Ko
    └── rainfall.csv                   # ~136 Ko
```

## Provenance

Les deux jeux de données sont fournis dans le cadre du Projet 12 OpenClassrooms.
Ils ne sont pas redistribués dans ce repository.

- `agriculture-crop-yield-dataset` — observations au niveau parcelle
- `crop-yield-prediction-dataset` — fichiers annuels agrégés par pays

Une fois les fichiers en place, l'arborescence ci-dessus doit être respectée :
le code et les notebooks s'appuient sur ces chemins.
