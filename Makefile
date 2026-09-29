.PHONY: test test-durations api health predict recommend

test:
	poetry run pytest

test-durations:
	poetry run pytest --durations=20

api:
	poetry run uvicorn agritech.api.main:app --reload

health:
	curl http://127.0.0.1:8000/health

predict:
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"rainfall_mm":500,"temperature_celsius":25,"fertilizer_used":true,"irrigation_used":false}'

recommend:
	curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d '{"iso3":"FRA"}'
