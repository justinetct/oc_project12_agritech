.PHONY: test test-durations api streamlit gradio seed-monitoring health predict recommend \
        docker-build docker-up docker-down docker-logs docker-demo

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

health:
	curl http://127.0.0.1:8000/health

predict:
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"rainfall_mm":500,"temperature_celsius":25,"fertilizer_used":true,"irrigation_used":false}'

recommend:
	curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d '{"iso3":"FRA","conditions":{"average_temperature_celsius":25,"annual_rainfall_mm":500,"average_annual_pesticides_tons":60000}}'

# -- Docker ----------------------------------------------------------------
# Deux services : `api` (FastAPI, seul à monter la base de monitoring) et
# `gradio` (dashboard de monitoring, qui appelle l'API par le réseau Compose).
# LOGFIRE_TOKEN= devant chaque commande compose vide l'éventuel token présent
# dans le shell : la démo locale reste en mode silencieux, aucun envoi réseau
# vers Logfire. Retirer le préfixe (ou exporter LOGFIRE_TOKEN) pour activer.
# MONITORING_API_TOKEN vient du shell, sinon du `.env` local (interpolation
# Compose) : le même token pour l'API et pour le dashboard.

docker-build:
	LOGFIRE_TOKEN= docker compose build api gradio

docker-up:
	LOGFIRE_TOKEN= docker compose up -d api gradio

docker-down:
	LOGFIRE_TOKEN= docker compose down

docker-logs:
	LOGFIRE_TOKEN= docker compose logs --tail=200 api gradio

docker-demo: docker-build docker-up
	@echo "Waiting for /health..."
	@READY=false; \
	 for i in $$(seq 1 60); do \
	     if curl -sf http://127.0.0.1:8000/health >/dev/null; then \
	         READY=true; break; \
	     fi; \
	     sleep 0.5; \
	 done; \
	 if [ "$$READY" = "false" ]; then \
	     echo "FAIL: /health did not respond within 30s."; \
	     echo "See 'make docker-logs' to inspect, then 'make docker-down' to clean up."; \
	     exit 1; \
	 fi; \
	 echo "API ready. Waiting for the Gradio dashboard..."; \
	 READY=false; \
	 for i in $$(seq 1 60); do \
	     if curl -sf http://127.0.0.1:7860/ >/dev/null; then \
	         READY=true; break; \
	     fi; \
	     sleep 0.5; \
	 done; \
	 if [ "$$READY" = "false" ]; then \
	     echo "FAIL: Gradio did not respond within 30s."; \
	     echo "See 'make docker-logs' to inspect, then 'make docker-down' to clean up."; \
	     exit 1; \
	 fi; \
	 echo "Swagger:   http://127.0.0.1:8000/docs"; \
	 echo "Dashboard: http://127.0.0.1:7860"; \
	 echo "Opening both and streaming logs (Ctrl+C to stop)."; \
	 open http://127.0.0.1:8000/docs; \
	 open http://127.0.0.1:7860; \
	 LOGFIRE_TOKEN= docker compose logs -f --tail=100 api gradio || true
