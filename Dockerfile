# Images de service d'Agritech Answers : API et interface de monitoring.
#
# Approche minimale inspirée du Projet 8 : Poetry gère l'environnement local,
# les images ne contiennent que Python, pip et les paquets listés dans les
# requirements, générés à partir de pyproject.toml et attendus dans le
# contexte de build :
#
#   poetry export --with api --without-hashes -f requirements.txt -o requirements.txt
#   poetry export --only gradio --without-hashes -f requirements.txt -o requirements-gradio.txt
#
# Un seul Dockerfile, trois étapes :
#   - base   : Python, dépendances communes (requirements.txt) et code `agritech` ;
#   - gradio : interface de monitoring, qui n'appelle l'API qu'en HTTP ;
#   - api    : service FastAPI avec les modèles. Dernière étape, donc cible par
#              défaut d'un `docker build .` sans `--target`.
#
# Variables d'environnement (DATABASE_URL, ENVIRONMENT, LOGFIRE_*,
# MONITORING_API_TOKEN, AGRITECH_API_URL) fournies au runtime par
# docker-compose, jamais figées dans l'image.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 1. Dépendances Python — couche stable, rebuild seulement si requirements.txt change.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Code applicatif — copié sous /app/src pour préserver la structure du dépôt.
#    `config.py` fait `Path(__file__).resolve().parents[2]` et attend donc /app
#    comme PATHS.root. PYTHONPATH=/app/src (défini plus haut) rend `agritech.*`
#    importable sans installation du paquet.
COPY src ./src


# --- Interface de monitoring (Gradio) ----------------------------------------
# Aucun modèle, aucune base SQLite : l'interface lit uniquement l'API HTTP
# (`AGRITECH_API_URL`, token `MONITORING_API_TOKEN`).
FROM base AS gradio

COPY requirements-gradio.txt ./
RUN pip install --no-cache-dir -r requirements-gradio.txt

COPY gradio_app ./gradio_app

# Écoute sur toutes les interfaces du conteneur (127.0.0.1 par défaut chez
# Gradio, injoignable depuis l'hôte) ; pas de statistiques d'usage envoyées.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

EXPOSE 7860

CMD ["python", "gradio_app/app.py"]


# --- API (FastAPI) -------------------------------------------------------------
FROM base AS api

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
