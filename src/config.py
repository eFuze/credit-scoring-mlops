"""Configuration du projet."""
from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Fichiers de données
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Modèle
MODEL_PATH = MODELS_DIR / "model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Coûts métier
FN_COST = 10.0  # Coût d'un faux négatif (défaut non détecté)
FP_COST = 1.0   # Coût d'un faux positif (bon client refusé)

# Paramètres modèle
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 20,
    "random_state": 7,
    "n_jobs": -1,
    "verbosity": -1
}

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "credit-scoring"

# Créer les dossiers
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
