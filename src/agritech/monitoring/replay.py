"""CLI de rejeu d'appels métier archivés dans `api_requests`.

Deux modes d'utilisation :

- **Rejouer une requête précise** — passe l'id de la ligne :

  ```
  poetry run python -m agritech.monitoring.replay 42
  ```

- **Lister/rejouer les erreurs récentes** — mode batch :

  ```
  poetry run python -m agritech.monitoring.replay --failed --since 2026-09-29
  ```

Le module ne passe **pas** par HTTP/FastAPI : il rejoue le calcul métier en
appelant directement la couche `agritech.serving`, avec le modèle
actuellement disponible sur disque. Il n'y a pas de registre de modèles :
si `model_version` archivée diffère de la version actuelle, un WARNING est
affiché et le replay bit-identique n'est pas garanti.

Le module reste volontairement mince : aucune abstraction supplémentaire,
aucun framework CLI, aucune tolérance numérique fabriquée. Une égalité
JSON stricte suffit pour ce lot ; un diff récursif expose les écarts
éventuels sans dépendre d'une librairie externe.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable

from dotenv import load_dotenv
from sqlalchemy import select

from agritech.config import PATHS
from agritech.monitoring.config import load_config
from agritech.monitoring.models import ApiRequest
from agritech.monitoring.session import (
    create_monitoring_engine,
    create_session_factory,
)
from agritech.serving import (
    Bundle,
    RecommendContext,
    load_bundle,
    load_recommend_context,
    predict as serving_predict,
    recommend as serving_recommend,
)


_SUPPORTED_SERVICES = ("predict", "recommend")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI : renvoie l'exit code du programme.

    Chargé par `python -m agritech.monitoring.replay` via le bloc `__main__`
    en bas de fichier. Retour `0` en cas de succès, `2` pour un id inconnu,
    et l'exit code d'argparse (`2`) sur une erreur d'argument.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.failed:
        if args.since is None:
            parser.error("--failed requires --since YYYY-MM-DD")
    else:
        if args.request_id is None:
            parser.error("request_id is required (or use --failed --since)")

    load_dotenv(override=False)
    config = load_config()
    engine = create_monitoring_engine(config)
    factory = create_session_factory(engine)

    try:
        if args.failed:
            return _replay_failed_since(factory, args.since)
        return _replay_single(factory, args.request_id)
    finally:
        engine.dispose()


def _build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI. Argparse gère seul les erreurs de syntaxe."""
    parser = argparse.ArgumentParser(
        prog="python -m agritech.monitoring.replay",
        description=(
            "Rejoue un appel métier archivé dans api_requests. "
            "Utilise le modèle actuellement disponible ; un mismatch de "
            "model_version est signalé mais ne bloque pas."
        ),
    )
    parser.add_argument(
        "request_id",
        type=int,
        nargs="?",
        help="Identifiant de la ligne api_requests à rejouer.",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Mode batch : parcourt les requêtes en erreur depuis --since.",
    )
    parser.add_argument(
        "--since",
        type=_parse_iso_date,
        default=None,
        help="Date ISO YYYY-MM-DD (inclusive) utilisée avec --failed.",
    )
    return parser


def _parse_iso_date(value: str) -> datetime:
    """Parse une date `YYYY-MM-DD` en `datetime` UTC. Lève `ArgumentTypeError` sinon."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date '{value}', expected YYYY-MM-DD"
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _replay_single(factory: Callable, request_id: int) -> int:
    """Charge et rejoue la ligne d'id `request_id`, ou renvoie `2` si introuvable."""
    with factory() as session:
        row = session.get(ApiRequest, request_id)
        if row is None:
            print(
                f"No api_request found with id={request_id}", file=sys.stderr
            )
            return 2
        _print_replay(row)
    return 0


def _replay_failed_since(factory: Callable, since: datetime) -> int:
    """Liste et rejoue les erreurs archivées depuis `since`. Exit `0` même si vide."""
    with factory() as session:
        rows = (
            session.execute(
                select(ApiRequest)
                .where(ApiRequest.success.is_(False))
                .where(ApiRequest.timestamp >= since)
                .order_by(ApiRequest.id)
            )
            .scalars()
            .all()
        )

    if not rows:
        print(f"No failed api_request since {since.date().isoformat()}.")
        return 0

    for index, row in enumerate(rows):
        if index > 0:
            print("-" * 60)
        _print_replay(row)
    return 0


def _print_replay(row: ApiRequest) -> None:
    """Affiche l'en-tête, le calcul et le diff pour une ligne archivée."""
    print(f"REPLAY api_request #{row.id}")
    print(f"Service: {row.service}")
    print(f"Endpoint: {row.endpoint}")
    print(f"Timestamp: {row.timestamp.isoformat()}")
    print(f"API version archived: {row.api_version}")
    print(f"Model version archived: {row.model_version}")

    if row.service not in _SUPPORTED_SERVICES:
        print(
            f"Replay not applicable: service '{row.service}' is not supported "
            f"by agritech.serving."
        )
        return

    if not row.success:
        print(
            f"Archived response was an error "
            f"(status={row.status_code}, error_type={row.error_type})."
        )
        print()
        print("ARCHIVED RESPONSE")
        print(json.dumps(row.response_payload, indent=2, ensure_ascii=False))
        print(
            "Replay not applicable: request did not reach the service layer "
            "(rejected upstream by FastAPI/Pydantic or by the service itself)."
        )
        return

    try:
        bundle, context = _load_service(row.service)
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot load service '{row.service}': {type(exc).__name__}: {exc}")
        return

    current_version = bundle.metadata.get("model_version")
    print(f"Model version current:  {current_version}")
    if row.model_version and current_version != row.model_version:
        print(
            "WARNING: archived model_version differs from the currently "
            "available model. Replay uses the current model — this is NOT a "
            "strict reproduction of the historical execution."
        )

    print()
    print("REQUEST")
    print(json.dumps(row.request_payload, indent=2, ensure_ascii=False))
    print()
    print("ARCHIVED RESPONSE")
    print(json.dumps(row.response_payload, indent=2, ensure_ascii=False))
    print()

    try:
        current = _run_service(row.service, bundle, context, row.request_payload)
    except Exception as exc:  # noqa: BLE001
        print(f"Replay error: {type(exc).__name__}: {exc}")
        return

    print("CURRENT RESPONSE")
    print(json.dumps(current, indent=2, ensure_ascii=False))
    print()
    print("DIFF")
    diff_lines = _diff_json(row.response_payload, current)
    if not diff_lines:
        print("No difference.")
    else:
        for line in diff_lines:
            print(line)


def _load_service(service: str) -> tuple[Bundle, RecommendContext | None]:
    """Charge le bundle du service, et le contexte pour /recommend."""
    bundle = load_bundle(service)
    if service == "recommend":
        context = load_recommend_context(
            PATHS.root / "models" / "recommend_context.json"
        )
        return bundle, context
    return bundle, None


def _run_service(
    service: str,
    bundle: Bundle,
    context: RecommendContext | None,
    request_payload: Any,
) -> dict:
    """Appelle le bon service métier avec le payload archivé."""
    if not isinstance(request_payload, dict):
        raise ValueError("archived request_payload is not a JSON object")

    if service == "predict":
        return serving_predict(bundle, request_payload)
    if service == "recommend":
        iso3 = request_payload.get("iso3")
        if not isinstance(iso3, str):
            raise ValueError("archived request_payload has no iso3 string")
        conditions = request_payload.get("conditions") or {}
        return serving_recommend(bundle, context, iso3, conditions)
    raise ValueError(f"unsupported service: {service}")


def _diff_json(archived: Any, current: Any, path: str = "") -> list[str]:
    """Diff JSON récursif simple. Renvoie une liste de lignes prêtes à imprimer.

    Les valeurs sont comparées par égalité stricte : deux floats qui diffèrent
    d'un epsilon sortiront comme différence. C'est volontaire — le lot 5
    n'a pas de règle de tolérance numérique.
    """
    if archived == current:
        return []

    if type(archived) is not type(current):
        return [
            f"- {path or 'root'}:",
            f"    archived: {json.dumps(archived, ensure_ascii=False)}",
            f"    current:  {json.dumps(current, ensure_ascii=False)}",
        ]

    if isinstance(archived, dict):
        lines: list[str] = []
        keys = sorted(set(archived) | set(current))
        for key in keys:
            if archived.get(key) != current.get(key):
                subpath = f"{path}.{key}" if path else key
                lines.extend(_diff_json(archived.get(key), current.get(key), subpath))
        return lines

    if isinstance(archived, list):
        if len(archived) != len(current):
            return [
                f"- {path or 'root'} (list length differs):",
                f"    archived: {len(archived)}",
                f"    current:  {len(current)}",
            ]
        lines = []
        for index, (a, c) in enumerate(zip(archived, current)):
            if a != c:
                lines.extend(_diff_json(a, c, f"{path}[{index}]"))
        return lines

    # scalaires ou types égaux mais valeurs différentes
    return [
        f"- {path or 'root'}:",
        f"    archived: {json.dumps(archived, ensure_ascii=False)}",
        f"    current:  {json.dumps(current, ensure_ascii=False)}",
    ]


if __name__ == "__main__":  # pragma: no cover — testé via `main()`
    sys.exit(main())
