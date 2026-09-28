"""État de l'application, initialisé au démarrage par le lifespan.

Chaque attribut vaut `None` avant le démarrage, puis contient l'objet chargé
une fois pour toutes. Les endpoints les lisent sans les modifier.
"""

from __future__ import annotations

from agritech.serving import Bundle, RecommendContext


bundle_predict: Bundle | None = None
bundle_recommend: Bundle | None = None
recommend_context: RecommendContext | None = None
