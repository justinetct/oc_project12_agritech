"""Version publique de l'API Agritech Answers.

Module à part pour pouvoir lire la version sans importer l'application
FastAPI : `agritech.api.main` l'expose (`/health`, OpenAPI, monitoring) et
l'historique de démonstration du monitoring l'enregistre dans ses lignes.
Elle doit rester alignée avec `[project] version` de `pyproject.toml`.
"""

API_VERSION = "1.0.0"
