"""Règles d'architecture de `docker-compose.yml`, du `Dockerfile` et de `make docker-demo`.

On ne teste que ce qui protège l'architecture : FastAPI seul propriétaire de
la base de monitoring, interfaces Streamlit et Gradio qui ne passent que par
l'API HTTP, secrets fournis par l'environnement (jamais écrits dans les
fichiers) et démo locale étiquetée `local`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
import yaml

from agritech.config import PATHS


INTERFACES = ["streamlit", "gradio"]
REQUIREMENTS = {
    "api": "requirements.txt",
    "streamlit": "requirements-streamlit.txt",
    "gradio": "requirements-gradio.txt",
}


@pytest.fixture(scope="module")
def services() -> dict:
    return yaml.safe_load((PATHS.root / "docker-compose.yml").read_text())["services"]


@pytest.fixture(scope="module")
def stages() -> dict[str, str]:
    """Contenu de chaque étape du `Dockerfile`, indexé par son nom."""
    parts = re.split(r"^FROM \S+ AS (\w+)$", (PATHS.root / "Dockerfile").read_text(), flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2]))


def _packages(requirements: str) -> set[str]:
    lines = (PATHS.root / requirements).read_text().splitlines()
    return {line.split("==")[0] for line in lines if "==" in line}


@pytest.fixture(scope="module")
def demo_commands() -> str:
    """Commandes de `make docker-demo`, affichées sans être exécutées (`make -n`)."""
    if shutil.which("make") is None:
        pytest.skip("make absent")
    result = subprocess.run(
        ["make", "-n", "docker-demo"], cwd=PATHS.root, capture_output=True, text=True, check=True
    )
    return result.stdout


def test_the_demo_has_three_services(services):
    assert set(services) == {"api", "streamlit", "gradio"}


@pytest.mark.parametrize("name", INTERFACES)
def test_only_the_api_mounts_the_monitoring_volume(services, name):
    assert services["api"]["volumes"] == ["agritech_monitoring:/app/data/monitoring"]
    assert "volumes" not in services[name]
    assert "DATABASE_URL" not in services[name]["environment"]


@pytest.mark.parametrize("name", INTERFACES)
def test_interfaces_call_the_api_through_the_compose_network(services, name):
    interface = services[name]

    assert interface["environment"]["AGRITECH_API_URL"] == "http://api:8000"
    assert interface["depends_on"] == {"api": {"condition": "service_healthy"}}
    assert "healthcheck" in services["api"]


def test_ports_are_published_on_localhost_only(services):
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    assert services["streamlit"]["ports"] == ["127.0.0.1:8501:8501"]
    assert services["gradio"]["ports"] == ["127.0.0.1:7860:7860"]


def test_tokens_come_from_the_environment(services):
    """Tokens interpolés, jamais écrits en clair ; Streamlit ne reçoit que l'URL de l'API."""
    for name in ("api", "gradio"):
        assert services[name]["environment"]["MONITORING_API_TOKEN"] == "${MONITORING_API_TOKEN:-}"
    assert services["api"]["environment"]["LOGFIRE_TOKEN"] == "${LOGFIRE_TOKEN:-}"
    assert services["streamlit"]["environment"] == {"AGRITECH_API_URL": "http://api:8000"}


def test_the_local_demo_is_labelled_local(services):
    environment = services["api"]["environment"]

    assert environment["ENVIRONMENT"] == "local"
    assert environment["LOGFIRE_ENVIRONMENT"] == "local"


def test_each_service_builds_its_own_dockerfile_target(services):
    dockerfile = (PATHS.root / "Dockerfile").read_text()
    stages = re.findall(r"^FROM \S+ AS (\w+)$", dockerfile, flags=re.MULTILINE)

    for name in ("api", "streamlit", "gradio"):
        assert services[name]["build"]["target"] == name
    assert stages == ["base", "gradio", "streamlit", "api"]
    assert "GRADIO_SERVER_NAME=0.0.0.0" in dockerfile
    assert "--server.address 0.0.0.0" in dockerfile


def test_without_target_the_service_argument_chooses_the_image():
    """Render ne choisit pas de cible : la dernière étape reprend celle nommée par SERVICE, l'API par défaut."""
    dockerfile = (PATHS.root / "Dockerfile").read_text()
    instructions = [line for line in dockerfile.splitlines() if line and not line.startswith(("#", " "))]

    assert instructions[0] == "ARG SERVICE=api"
    assert instructions[1].startswith("FROM ")
    assert instructions[-1] == "FROM ${SERVICE}"


@pytest.mark.parametrize(
    ("name", "port_option"),
    [
        ("api", "--port ${PORT:-8000}"),
        ("streamlit", "--server.port ${PORT:-8501}"),
        ("gradio", "export GRADIO_SERVER_PORT=${PORT:-7860}"),
    ],
)
def test_each_service_listens_on_port_or_its_local_port(stages, name, port_option):
    """$PORT fourni par Render, sinon le port local ; `sh -c` lit la variable, `exec` garde les signaux."""
    command = re.search(r"^CMD (.+)$", stages[name], flags=re.MULTILINE).group(1)

    assert command.startswith('["sh", "-c", ')
    assert port_option in command
    assert "exec " in command


def test_base_stage_installs_no_dependency(stages):
    assert "pip install" not in stages["base"]
    assert "requirements" not in stages["base"]


@pytest.mark.parametrize("name", ["api", "streamlit", "gradio"])
def test_each_image_installs_only_its_own_requirements(stages, name):
    installed = re.findall(r"pip install --no-cache-dir -r (\S+)", stages[name])

    assert installed == [REQUIREMENTS[name]]


@pytest.mark.parametrize("name", INTERFACES)
def test_interface_images_carry_their_imports_but_not_the_api_stack(name):
    """Les paquets importés par `agritech.ui`, sans scikit-learn, SQLAlchemy ni Logfire."""
    packages = _packages(REQUIREMENTS[name])

    assert {name, "requests", "pydantic"} <= packages
    assert not packages & {"scikit-learn", "scipy", "joblib", "sqlalchemy", "logfire"}


def test_gradio_image_can_read_a_local_env_file():
    assert "python-dotenv" in _packages(REQUIREMENTS["gradio"])


def test_docker_demo_keeps_the_logfire_token_from_the_environment(demo_commands):
    assert "docker compose up -d api streamlit gradio" in demo_commands
    assert "LOGFIRE_TOKEN=" not in demo_commands


def test_docker_demo_waits_for_the_three_services_before_opening_them(demo_commands):
    waits = [
        demo_commands.index(url)
        for url in (
            "http://127.0.0.1:8000/health",
            "http://127.0.0.1:8501/_stcore/health",
            "http://127.0.0.1:7860/",
        )
    ]
    first_open = demo_commands.index("open http://")

    assert waits == sorted(waits)
    assert max(waits) < first_open
    for url in ("http://127.0.0.1:8000/docs", "http://127.0.0.1:8501", "http://127.0.0.1:7860"):
        assert f"open {url}" in demo_commands
