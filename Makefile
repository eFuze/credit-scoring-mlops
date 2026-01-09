.PHONY: install data train api dashboard docker-build docker-up docker-down clean test help

# Variables
PYTHON = python
PIP = pip

help:
	@echo "Credit Scoring MLOps - Commandes disponibles:"
	@echo ""
	@echo "  make install      - Installer les dépendances"
	@echo "  make data         - Télécharger/créer les données"
	@echo "  make train        - Entraîner le modèle"
	@echo "  make api          - Lancer l'API (port 8000)"
	@echo "  make dashboard    - Lancer le dashboard (port 8501)"
	@echo "  make docker-build - Construire les images Docker"
	@echo "  make docker-up    - Lancer avec Docker Compose"
	@echo "  make docker-down  - Arrêter Docker Compose"
	@echo "  make clean        - Nettoyer les fichiers générés"
	@echo "  make test         - Lancer les tests"
	@echo ""

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py --sample

prepare:
	$(PYTHON) -m src.data_processing

train: data prepare
	$(PYTHON) -m src.train

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.parquet
	rm -rf models/*.joblib models/*.json
	rm -rf mlflow.db mlruns/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache

test:
	pytest tests/ -v

# Raccourcis
run: train api

all: install data train
