# Image de service pour l'API Agritech Answers.
#
# Approche minimale inspirée du Projet 8 : Poetry gère l'environnement local,
# l'image ne contient que Python, pip et les paquets listés dans
# requirements.txt. Le requirements.txt est généré séparément à partir de
# pyproject.toml (sous-étape 3 de la tâche 20) et attendu dans le contexte
# de build.
#
# Variables d'environnement (DATABASE_URL, ENVIRONMENT, LOGFIRE_*) fournies
# au runtime par docker-compose, jamais figées dans l'image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dépendances Python — couche stable, rebuild seulement si requirements.txt change.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Code applicatif — copié directement à la racine de /app, donc importable
#    en tant que `agritech` sans PYTHONPATH ni installation du paquet.
COPY src/agritech ./agritech

# 3. Artefacts modèles — couche dédiée (~45 Mo), indépendante du code.
COPY models ./models

# 4. Répertoire de persistance du monitoring SQLite.
#    Le volume nommé sera monté ici par docker-compose ; le mkdir garantit
#    que le chemin existe même sans volume, pour un `docker run` isolé.
RUN mkdir -p /app/data/monitoring

EXPOSE 8000

CMD ["uvicorn", "agritech.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
