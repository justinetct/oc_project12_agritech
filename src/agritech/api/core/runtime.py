"""État de l'application, initialisé au démarrage par le lifespan.

`bundle_predict` vaut `None` avant le démarrage, puis contient le bundle chargé
une fois pour toutes. Les endpoints le lisent sans le modifier.
"""

from __future__ import annotations

from agritech.serving import Bundle


bundle_predict: Bundle | None = None
