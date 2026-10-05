"""Règles d'architecture de `docker-compose.yml` et du `Dockerfile`.

On ne teste que ce qui protège l'architecture : FastAPI seul propriétaire de
la base de monitoring, dashboard Gradio qui ne la lit que par l'API HTTP, et
secrets fournis par l'environnement, jamais écrits dans les fichiers.
"""

from __future__ import annotations

import re

import pytest
import yaml

from agritech.config import PATHS


@pytest.fixture(scope="module")
def services() -> dict:
    return yaml.safe_load((PATHS.root / "docker-compose.yml").read_text())["services"]


def test_only_the_api_mounts_the_monitoring_volume(services):
    assert services["api"]["volumes"] == ["agritech_monitoring:/app/data/monitoring"]
    assert "volumes" not in services["gradio"]
    assert "DATABASE_URL" not in services["gradio"]["environment"]


def test_gradio_calls_the_api_through_the_compose_network(services):
    gradio = services["gradio"]

    assert gradio["environment"]["AGRITECH_API_URL"] == "http://api:8000"
    assert gradio["depends_on"] == {"api": {"condition": "service_healthy"}}
    assert "healthcheck" in services["api"]


def test_ports_are_published_on_localhost_only(services):
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    assert services["gradio"]["ports"] == ["127.0.0.1:7860:7860"]


def test_monitoring_token_comes_from_the_environment(services):
    """Même token pour l'API et le dashboard, interpolé, jamais écrit en clair."""
    for name in ("api", "gradio"):
        assert services[name]["environment"]["MONITORING_API_TOKEN"] == "${MONITORING_API_TOKEN:-}"


def test_each_service_builds_its_own_dockerfile_target(services):
    dockerfile = (PATHS.root / "Dockerfile").read_text()
    stages = re.findall(r"^FROM \S+ AS (\w+)$", dockerfile, flags=re.MULTILINE)

    assert services["api"]["build"]["target"] == "api"
    assert services["gradio"]["build"]["target"] == "gradio"
    # `api` en dernier : un `docker build .` sans cible construit toujours l'API.
    assert stages == ["base", "gradio", "api"]
    assert "GRADIO_SERVER_NAME=0.0.0.0" in dockerfile
