# Credit Scoring MLOps

Modele de scoring credit avec optimisation du cout metier et explicabilite SHAP.

## Objectif

Predire le risque de defaut de credit avec une fonction de cout asymetrique :
- **Faux Negatif** (defaut non detecte) : cout = 10
- **Faux Positif** (bon client refuse) : cout = 1

## Structure du Projet

```
credit-scoring-mlops/
├── src/                    # Code source
│   ├── config.py          # Configuration
│   ├── cost_functions.py  # Fonctions de cout metier
│   ├── data_processing.py # Preparation des donnees
│   └── train.py           # Entrainement du modele
├── api/                    # API FastAPI
│   └── main.py            # Endpoints de prediction
├── dashboard/              # Dashboard Streamlit
│   └── app.py             # Interface utilisateur
├── scripts/                # Scripts utilitaires
│   └── download_data.py   # Telechargement donnees Kaggle
├── notebooks/              # Notebooks d'analyse
├── models/                 # Modeles sauvegardes
├── data/                   # Donnees (raw/processed)
├── tests/                  # Tests unitaires
├── Dockerfile             # Image Docker API
├── docker-compose.yml     # Orchestration Docker
├── Makefile               # Commandes Make
└── requirements.txt       # Dependances Python
```

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Preparer les donnees et entrainer

```bash
# Creer des donnees d'exemple (sans compte Kaggle)
python scripts/download_data.py --sample

# Preparer les features
python -m src.data_processing

# Entrainer le modele
python -m src.train

# Ou tout en une commande :
make train
```

### 3. Lancer l'API

```bash
uvicorn api.main:app --reload
```

Documentation interactive : http://localhost:8000/docs

### 4. Lancer le Dashboard

```bash
streamlit run dashboard/app.py
```

## API Endpoints

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/` | GET | Info API |
| `/health` | GET | Health check |
| `/model/info` | GET | Info modele |
| `/model/features` | GET | Liste des features |
| `/predict` | POST | Prediction simple |
| `/predict/batch` | POST | Predictions en batch |
| `/predict/explain` | POST | Prediction avec SHAP |

### Exemple de requete

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6}}'
```

Reponse :
```json
{
  "probability": 0.0823,
  "prediction": 0,
  "threshold": 0.083,
  "decision": "ACCEPTE",
  "risk_level": "FAIBLE"
}
```

## Docker

```bash
docker-compose up -d

# Services disponibles :
# - API : http://localhost:8000
# - Dashboard : http://localhost:8501
# - MLflow : http://localhost:5000
```

## Modele

- **Algorithme** : LightGBM
- **Optimisation** : Cout metier (10xFN + 1xFP)
- **AUC** : ~0.78
- **Seuil optimal** : ~0.08

## Dataset

Home Credit Default Risk (Kaggle)
