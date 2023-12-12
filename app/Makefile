.PHONY: start
start:
	uvicorn main:app --reload --host 0.0.0.0 --port 8123

.PHONY: format
format:
	black .
	isort .
