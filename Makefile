PY ?= .venv/bin/python

.PHONY: help install data train contract-demo pipeline dashboard mlflow-ui lint test all clean

help:
	@echo "make install        - cria .venv e instala o projeto (pip) — ou use 'uv sync'"
	@echo "make data           - baixa o dataset, aplica o contrato e separa referencia x producao"
	@echo "make train          - treina o modelo baseline na referencia"
	@echo "make contract-demo  - demonstra o contrato aceitando um lote bom e bloqueando um ruim"
	@echo "make pipeline       - roda tudo: ingestao, treino, 6 lotes monitorados, relatorios"
	@echo "make dashboard      - regenera reports/dashboard.html a partir do ultimo resumo"
	@echo "make mlflow-ui      - abre a UI do MLflow em http://127.0.0.1:5000"
	@echo "make lint | test    - ruff e pytest"

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e . pytest pytest-cov ruff

data:
	$(PY) -m credit_monitor.data.prepare

train:
	$(PY) -m credit_monitor.models.train

contract-demo:
	$(PY) -m credit_monitor.contracts.demo

pipeline:
	$(PY) -m credit_monitor.pipeline

dashboard:
	$(PY) -m credit_monitor.monitoring.dashboard

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

test:
	$(PY) -m pytest --cov=credit_monitor --cov-report=term-missing

all: lint test pipeline

clean:
	rm -rf data/reference data/production data/quarantine models logs mlruns mlflow.db
