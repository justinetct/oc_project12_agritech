"""Middlewares HTTP branchés sur l'application FastAPI Agritech Answers.

À ce jour, un seul middleware : `RequestLoggerMiddleware`, qui persiste
chaque appel métier (`POST /predict` et `POST /recommend`) dans la table
`api_requests` de la base de monitoring.
"""
