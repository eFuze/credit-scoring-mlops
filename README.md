# Projet 7 - Implémentez un modèle de scoring

## Scoring Crédit avec MLOps

Application de scoring crédit permettant de prédire la probabilité de défaut de paiement d'un client, avec déploiement MLOps complet.

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)

---

## Structure du projet

```
credit-scoring-mlops/
│
├── api/                          # API FastAPI
│   └── main.py                   # Endpoints de prédiction
│
├── streamlit_app/                # Dashboard interactif
│   └── app.py                    # Application Streamlit
│
├── notebooks/                    # Notebooks d'analyse
│   └── analysis.ipynb            # Modélisation + MLFlow tracking
│
├── scripts/                      # Scripts utilitaires
│   └── generate_drift_report.py  # Génération rapport Evidently
│
├── tests/                        # Tests unitaires
│   └── test_api.py               # Tests de l'API
│
├── reports/                      # Rapports générés
│   └── data_drift_report.html    # Rapport de data drift
│
├── models/                       # Modèles sauvegardés
│   └── model.pkl                 # Modèle LightGBM
│
├── mlruns/                       # Expériences MLFlow
│
├── .github/workflows/            # CI/CD
│   └── ci-cd.yml                 # Pipeline GitHub Actions
│
├── Dockerfile                    # Conteneurisation
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

---

## Installation

### Prérequis
- Python 3.10+
- pip ou conda

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/votre-username/credit-scoring-mlops.git
cd credit-scoring-mlops

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### 1. Notebook de modélisation

Le notebook `notebooks/analysis.ipynb` contient :
- Chargement et exploration des données
- Entraînement des modèles (LogReg, LightGBM)
- Optimisation du seuil avec fonction de coût métier
- **Tracking MLFlow** des expériences
- Interprétabilité SHAP

```bash
# Lancer Jupyter
jupyter notebook notebooks/analysis.ipynb
```

### 2. Interface MLFlow UI

```bash
# Lancer le serveur MLFlow
mlflow ui --port 5000

# Ouvrir http://localhost:5000
```

### 3. API de prédiction

```bash
# Lancer l'API
uvicorn api.main:app --reload --port 8000

# Documentation: http://localhost:8000/docs
```

**Endpoints disponibles:**
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | État de l'API |
| `/model/info` | GET | Informations du modèle |
| `/predict` | POST | Prédiction simple |
| `/predict/explain` | POST | Prédiction + SHAP |

**Exemple de requête:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.7}}'
```

### 4. Dashboard Streamlit

```bash
# Lancer le dashboard
streamlit run streamlit_app/app.py

# Ouvrir http://localhost:8501
```

### 5. Rapport de Data Drift

```bash
# Générer le rapport Evidently
python scripts/generate_drift_report.py

# Le rapport est sauvegardé dans reports/data_drift_report.html
```

---

## Déploiement

### Docker

```bash
# Construire l'image
docker build -t credit-scoring-api .

# Lancer le conteneur
docker run -p 8000:8000 credit-scoring-api
```

### CI/CD (GitHub Actions)

Le pipeline `.github/workflows/ci-cd.yml` :
1. Exécute les tests unitaires
2. Vérifie la qualité du code (flake8)
3. Build l'image Docker
4. Déploie sur le cloud (Heroku/Render/AWS)

---

## Modèle

### Algorithme
**LightGBM** (Gradient Boosting) avec optimisation de seuil

### Fonction de coût métier
```
Coût = 10 × FN + 1 × FP
```
- **FN (Faux Négatif):** Client qui fait défaut mais accepté -> Coût 10
- **FP (Faux Positif):** Bon client refusé -> Coût 1

### Performances
| Métrique | Valeur |
|----------|--------|
| AUC (Hold-out) | 0.787 |
| Seuil optimal | 0.083 |
| Taux de détection défauts | 68.4% |
| Taux de refus | 29.5% |

### Top Features (SHAP)
1. EXT_SOURCE_2
2. EXT_SOURCE_3
3. EXT_SOURCE_1
4. DAYS_BIRTH
5. DAYS_EMPLOYED

---

## Technologies

| Catégorie | Technologies |
|-----------|--------------|
| ML | LightGBM, scikit-learn |
| Interprétabilité | SHAP |
| MLOps | MLflow |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Data Drift | Evidently |
| CI/CD | GitHub Actions |
| Conteneurisation | Docker |

---

## Tests

```bash
# Exécuter les tests
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=api --cov-report=html
```

---

## Auteur

Projet réalisé dans le cadre du parcours **Data Scientist** d'OpenClassrooms.

---

## Licence

Ce projet est sous licence MIT.
