"""Exceptions applicatives de l'API Agritech Answers.

Les exceptions métier vivent ici pour que le code des routers reste indépendant
des codes HTTP : les handlers de `error_handlers.py` les traduisent en
`ErrorResponse` unifiées.
"""

from __future__ import annotations


class ModelUnavailableError(Exception):
    """Un modèle attendu par un endpoint n'est pas chargé.

    Est levée quand `runtime.bundle_predict` (ou l'équivalent `/recommend`) vaut
    `None` alors qu'un endpoint en a besoin. Le handler HTTP la traduit en 503
    avec le code `model_unavailable`.
    """
