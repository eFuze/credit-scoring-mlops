"""API FastAPI pour le scoring crédit."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH, FN_COST, FP_COST
from src.cost_functions import get_probas


# Global model storage
model_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage."""
    import joblib

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modèle non trouvé: {MODEL_PATH}\n"
            "Exécutez d'abord: python -m src.train"
        )

    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model_data["model"] = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as f:
        model_data["threshold_data"] = json.load(f)

    with open(FEATURE_NAMES_PATH, "r") as f:
        model_data["feature_names"] = json.load(f)

    print(f"Modèle chargé. Seuil: {model_data['threshold_data']['threshold']:.3f}")
    yield
    model_data.clear()


app = FastAPI(
    title="Credit Scoring API",
    description="API de prédiction du risque de défaut crédit",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClientFeatures(BaseModel):
    """Features d'un client pour la prédiction."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire des features du client"
    )


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    client_id: Optional[str] = None
    probability: float = Field(..., description="Probabilité de défaut")
    prediction: int = Field(..., description="0=accepté, 1=refusé")
    threshold: float = Field(..., description="Seuil de décision utilisé")
    decision: str = Field(..., description="ACCEPTÉ ou REFUSÉ")
    risk_level: str = Field(..., description="Niveau de risque")


class BatchPredictionRequest(BaseModel):
    """Requête pour prédictions en batch."""
    clients: List[Dict[str, float]]


class BatchPredictionResponse(BaseModel):
    """Réponse pour prédictions en batch."""
    predictions: List[PredictionResponse]


class ModelInfo(BaseModel):
    """Informations sur le modèle."""
    model_type: str
    threshold: float
    fn_cost: float
    fp_cost: float
    n_features: int


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/predict/batch", "/model/info", "/health"]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "model_loaded": "model" in model_data}


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Retourne les informations sur le modèle."""
    return ModelInfo(
        model_type="LightGBM",
        threshold=model_data["threshold_data"]["threshold"],
        fn_cost=model_data["threshold_data"].get("fn_cost", FN_COST),
        fp_cost=model_data["threshold_data"].get("fp_cost", FP_COST),
        n_features=len(model_data["feature_names"])
    )


@app.get("/model/features")
async def get_features():
    """Retourne la liste des features attendues."""
    return {"features": model_data["feature_names"]}


def get_risk_level(proba: float) -> str:
    """Détermine le niveau de risque."""
    if proba < 0.05:
        return "TRÈS FAIBLE"
    elif proba < 0.10:
        return "FAIBLE"
    elif proba < 0.20:
        return "MODÉRÉ"
    elif proba < 0.40:
        return "ÉLEVÉ"
    else:
        return "TRÈS ÉLEVÉ"


@app.post("/predict", response_model=PredictionResponse)
async def predict(client: ClientFeatures, client_id: Optional[str] = None):
    """
    Prédit le risque de défaut pour un client.

    - **features**: Dictionnaire des features du client
    - **client_id**: Identifiant optionnel du client
    """
    try:
        feature_names = model_data["feature_names"]
        threshold = model_data["threshold_data"]["threshold"]

        # Créer le DataFrame avec les features
        df = pd.DataFrame([client.features])

        # Ajouter les features manquantes avec NaN
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = np.nan

        # Ordonner les colonnes
        df = df[feature_names]

        # Renommer pour LightGBM
        df.columns = [f"feat_{i}" for i in range(len(feature_names))]

        # Prédiction
        proba = get_probas(model_data["model"], df)[0]
        prediction = int(proba >= threshold)

        return PredictionResponse(
            client_id=client_id,
            probability=round(float(proba), 4),
            prediction=prediction,
            threshold=threshold,
            decision="REFUSÉ" if prediction == 1 else "ACCEPTÉ",
            risk_level=get_risk_level(proba)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Prédictions en batch pour plusieurs clients."""
    predictions = []

    for i, client_features in enumerate(request.clients):
        client = ClientFeatures(features=client_features)
        pred = await predict(client, client_id=str(i))
        predictions.append(pred)

    return BatchPredictionResponse(predictions=predictions)


@app.post("/predict/explain")
async def predict_with_explanation(client: ClientFeatures, client_id: Optional[str] = None):
    """
    Prédit avec explication SHAP des principales features.
    """
    try:
        import shap

        feature_names = model_data["feature_names"]
        threshold = model_data["threshold_data"]["threshold"]

        # Créer le DataFrame
        df = pd.DataFrame([client.features])
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = np.nan
        df = df[feature_names]

        # DataFrame pour le modèle
        df_model = df.copy()
        df_model.columns = [f"feat_{i}" for i in range(len(feature_names))]

        # Prédiction
        proba = get_probas(model_data["model"], df_model)[0]
        prediction = int(proba >= threshold)

        # SHAP
        clf = model_data["model"].named_steps["clf"]
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(df_model)

        # Top features
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Classe positive
        else:
            shap_vals = shap_values[0]

        # Trier par importance absolue
        indices = np.argsort(np.abs(shap_vals))[::-1][:10]

        top_features = []
        for idx in indices:
            top_features.append({
                "feature": feature_names[idx],
                "value": float(df.iloc[0, idx]) if not pd.isna(df.iloc[0, idx]) else None,
                "shap_value": round(float(shap_vals[idx]), 4),
                "impact": "augmente risque" if shap_vals[idx] > 0 else "diminue risque"
            })

        return {
            "client_id": client_id,
            "probability": round(float(proba), 4),
            "prediction": prediction,
            "threshold": threshold,
            "decision": "REFUSÉ" if prediction == 1 else "ACCEPTÉ",
            "risk_level": get_risk_level(proba),
            "explanation": {
                "base_value": round(float(explainer.expected_value), 4) if not isinstance(explainer.expected_value, list) else round(float(explainer.expected_value[1]), 4),
                "top_features": top_features
            }
        }

    except ImportError:
        # Si SHAP n'est pas installé, retourner juste la prédiction
        pred = await predict(client, client_id)
        return {**pred.dict(), "explanation": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
