"""Repository de la table `api_requests` : une fonction d'insertion.

Le repository ne connaît ni HTTP, ni la sémantique métier d'une ligne : il
prend un dict de valeurs, construit un `ApiRequest` et l'écrit. Toute la
politique « une panne de persistance ne casse jamais une requête HTTP » vit
dans le middleware (lot 3) qui appelle cette fonction dans un `try/except`.

Volontairement peu de code : rien à factoriser à ce stade.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from agritech.monitoring.models import ApiRequest


def insert_api_request(session: Session, values: dict[str, Any]) -> int:
    """Insère une ligne `api_requests` et renvoie son `id` auto-incrémenté.

    `values` reprend les noms des colonnes du modèle `ApiRequest` — le
    middleware construit ce dict à partir de la requête HTTP, de la réponse
    et du contexte (api_version, model_version, environment, trace id).
    Toute clé inconnue lèvera un `TypeError` immédiat de SQLAlchemy, ce qui
    fait remonter l'erreur au plus tôt.
    """
    row = ApiRequest(**values)
    session.add(row)
    session.commit()
    return row.id
