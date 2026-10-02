"""Repository de la table `api_requests` : insertion et lectures dédiées.

Le repository ne connaît ni HTTP, ni la sémantique métier d'une ligne :

- `insert_api_request` prend un dict de valeurs, construit un `ApiRequest`
  et l'écrit. Toute la politique « une panne de persistance ne casse jamais
  une requête HTTP » vit dans le middleware, qui appelle cette fonction dans
  un `try/except` ;
- `list_recent_requests` lit les dernières lignes archivées, avec des
  filtres simples. Elle ne fait que des SELECT : aucune écriture, aucun
  commit.

Les lectures restent des fonctions dédiées aux besoins du monitoring, pas un
accès générique à la table. La validation des paramètres (bornes de `limit`,
valeurs de `service`) appartient à la couche HTTP.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
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


def list_recent_requests(
    session: Session,
    *,
    limit: int,
    service: str | None = None,
    success: bool | None = None,
) -> list[ApiRequest]:
    """Renvoie les `limit` lignes les plus récentes, éventuellement filtrées.

    Tri du plus récent au plus ancien sur `timestamp`, puis sur `id` pour
    départager deux lignes au même timestamp : l'ordre reste déterministe.

    Filtres facultatifs et combinables :
        service : `"predict"` ou `"recommend"` ; `None` garde les deux.
        success : `True` (succès) ou `False` (erreurs) ; `None` garde tout.

    Aucune validation de `limit` ici : les bornes sont imposées par
    l'endpoint HTTP qui appelle cette fonction.
    """
    query = select(ApiRequest)
    if service is not None:
        query = query.where(ApiRequest.service == service)
    if success is not None:
        query = query.where(ApiRequest.success.is_(success))
    query = query.order_by(ApiRequest.timestamp.desc(), ApiRequest.id.desc()).limit(limit)
    return list(session.execute(query).scalars())
