# Images de service d'Agritech Answers : API, interface métier et interface de monitoring.
#
# Approche minimale inspirée du Projet 8 : Poetry gère l'environnement local,
# les images ne contiennent que Python, pip et les paquets listés dans les
# requirements, générés à partir de pyproject.toml et attendus dans le
# contexte de build :
#
#   poetry export --only main,api --without-hashes -f requirements.txt -o requirements.txt
#   poetry export --only streamlit --without-hashes -f requirements.txt -o requirements-streamlit.txt
#   poetry export --only gradio --without-hashes -f requirements.txt -o requirements-gradio.txt
#
# Un seul Dockerfile, quatre étapes :
#   - base      : Python et variables communes, sans aucune dépendance installée ;
#   - gradio    : interface de monitoring (requirements-gradio.txt), qui n'appelle l'API qu'en HTTP ;
#   - streamlit : interface métier (requirements-streamlit.txt), qui n'appelle l'API qu'en HTTP ;
#   - api       : service FastAPI avec les modèles (requirements.txt). Dernière étape, donc cible
#                 par défaut d'un `docker build .` sans `--target`.
#
# Dans chaque étape, les dépendances sont installées avant de copier le code :
# modifier le code ne réinstalle pas les paquets.
#
# Variables d'environnement (DATABASE_URL, ENVIRONMENT, LOGFIRE_*,
# MONITORING_API_TOKEN, AGRITECH_API_URL) fournies au runtime par
# docker-compose, jamais figées dans l'image.

FROM python:3.12-slim AS base

# Le code `agritech` est copié sous /app/src pour préserver la structure du dépôt :
# `config.py` fait `Path(__file__).resolve().parents[2]` et attend donc /app comme
# PATHS.root. PYTHONPATH=/app/src rend `agritech.*` importable sans installer le paquet.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app


# --- Interface de monitoring (Gradio) ----------------------------------------
# Aucun modèle, aucune base SQLite : l'interface lit uniquement l'API HTTP
# (`AGRITECH_API_URL`, token `MONITORING_API_TOKEN`).
FROM base AS gradio

COPY requirements-gradio.txt ./
RUN pip install --no-cache-dir -r requirements-gradio.txt

COPY src ./src
COPY gradio_app ./gradio_app

# Écoute sur toutes les interfaces du conteneur (127.0.0.1 par défaut chez
# Gradio, injoignable depuis l'hôte) ; pas de statistiques d'usage envoyées.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

EXPOSE 7860

CMD ["python", "gradio_app/app.py"]


# --- Interface métier (Streamlit) --------------------------------------------
# Aucun modèle, aucune base SQLite : l'interface appelle l'API en HTTP
# (`AGRITECH_API_URL`). `.streamlit/config.toml` porte le thème.
FROM base AS streamlit

COPY requirements-streamlit.txt ./
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY src ./src
COPY .streamlit ./.streamlit
COPY streamlit_app ./streamlit_app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]


# --- API (FastAPI) -------------------------------------------------------------
FROM base AS api

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Artefacts modèles — couche dédiée (~45 Mo), indépendante du code.
COPY models ./models
COPY src ./src

# Répertoire de persistance du monitoring SQLite. Le volume nommé sera monté ici
# par docker-compose ; le mkdir garantit que le chemin existe même sans volume,
# pour un `docker run` isolé.
RUN mkdir -p /app/data/monitoring

EXPOSE 8000

CMD ["uvicorn", "agritech.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
