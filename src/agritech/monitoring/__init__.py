"""Persistance de l'observabilité API : configuration, modèle SQLAlchemy, session, repository.

Ce package concentre tout le code lié à la table `api_requests` (persistance
des appels `/predict` et `/recommend`) et au script CLI de rejeu. Il ne
connaît pas HTTP : le middleware `src/agritech/api/middleware/` fait le pont
au moment où il sera introduit.
"""
