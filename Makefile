.PHONY: lint test test-durations api streamlit gradio seed-monitoring rebuild-models health predict recommend \
        requirements docker-build docker-up docker-down docker-logs docker-demo

lint:
	poetry run ruff check .

test:
	poetry run pytest

test-durations:
	poetry run pytest --durations=20

api:
	poetry run uvicorn agritech.api.main:app --reload

streamlit:
	poetry run streamlit run streamlit_app/app.py

# Monitoring : lit AGRITECH_API_URL et MONITORING_API_TOKEN dans l'environnement.
gradio:
	poetry run python gradio_app/app.py

# Historique de démonstration du monitoring : aperçu seulement par défaut.
# Écriture : make seed-monitoring ARGS=--write (ajout seul, refusé si ENVIRONMENT=prod).
seed-monitoring:
	poetry run python -m agritech.monitoring.seed_history $(ARGS)

# Reconstruit les 5 artefacts servis de models/ à partir des datasets préparés (voir README).
rebuild-models:
	poetry run python scripts/rebuild_models.py

health:
	curl http://127.0.0.1:8000/health

predict:
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"rainfall_mm":500,"temperature_celsius":25,"fertilizer_used":true,"irrigation_used":false}'

recommend:
	curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d '{"iso3":"FRA","conditions":{"average_temperature_celsius":25,"annual_rainfall_mm":500,"average_annual_pesticides_tons":60000}}'

# -- Docker ----------------------------------------------------------------
# Trois services : `api` (FastAPI, seul à monter la base de monitoring),
# `streamlit` (interface métier) et `gradio` (dashboard de monitoring). Les deux
# interfaces appellent l'API par le réseau Compose.
# LOGFIRE_TOKEN et MONITORING_API_TOKEN viennent du shell, sinon du `.env` local
# (interpolation Compose). Sans LOGFIRE_TOKEN, l'API tourne sans Logfire.

DOCKER_SERVICES = api streamlit gradio

# Requirements des trois images, exportés depuis poetry.lock (plugin poetry-plugin-export).
# À relancer après chaque changement de dépendances, avant de reconstruire les images.
requirements:
	poetry export --only main,api --without-hashes -f requirements.txt -o requirements.txt
	poetry export --only streamlit --without-hashes -f requirements.txt -o requirements-streamlit.txt
	poetry export --only gradio --without-hashes -f requirements.txt -o requirements-gradio.txt

docker-build:
	docker compose build $(DOCKER_SERVICES)

docker-up:
	docker compose up -d $(DOCKER_SERVICES)

docker-down:
	docker compose down

docker-logs:
	docker compose logs --tail=200 $(DOCKER_SERVICES)

# Attend qu'une URL réponde, 30 s au plus : $(call wait_for,nom,url)
define wait_for
	@echo "Waiting for $(1)..."
	@for i in $$(seq 1 60); do \
	     curl -sf $(2) >/dev/null && exit 0; \
	     sleep 0.5; \
	 done; \
	 echo "FAIL: $(1) did not respond within 30s."; \
	 echo "See 'make docker-logs' to inspect, then 'make docker-down' to clean up."; \
	 exit 1
endef

docker-demo: docker-build docker-up
	$(call wait_for,the API,http://127.0.0.1:8000/health)
	$(call wait_for,Streamlit,http://127.0.0.1:8501/_stcore/health)
	$(call wait_for,the Gradio dashboard,http://127.0.0.1:7860/)
	@echo "Swagger:   http://127.0.0.1:8000/docs"
	@echo "Streamlit: http://127.0.0.1:8501"
	@echo "Dashboard: http://127.0.0.1:7860"
	@echo "Opening the three interfaces and streaming logs (Ctrl+C to stop)."
	@open http://127.0.0.1:8000/docs
	@open http://127.0.0.1:8501
	@open http://127.0.0.1:7860
	@docker compose logs -f --tail=100 $(DOCKER_SERVICES) || true
