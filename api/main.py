"""
API de Scoring Crédit - Projet 7 OpenClassrooms
================================================
API FastAPI pour prédire la probabilité de défaut de paiement d'un client.

Endpoints:
- GET /health : Vérification de l'état de l'API
- GET /model/info : Informations sur le modèle
- POST /predict : Prédiction pour un client
- POST /predict/explain : Prédiction avec explications SHAP
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shap

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.083"))
FN_COST = float(os.getenv("FN_COST", "10.0"))
FP_COST = float(os.getenv("FP_COST", "1.0"))

# Initialisation de l'API
app = FastAPI(
    title="API Scoring Crédit",
    description="API de prédiction du risque de défaut de paiement - Projet 7 OC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS pour permettre les appels depuis le dashboard Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
model = None
feature_names = None
explainer = None


class ClientData(BaseModel):
    """Données d'un client pour la prédiction."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire des features du client",
        example={"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.7}
    )


class ClientDataList(BaseModel):
    """Liste de features sous forme de liste (ordre des colonnes du modèle)."""
    features: List[float] = Field(
        ...,
        description="Liste des valeurs des features dans l'ordre du modèle"
    )


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    client_id: Optional[str] = None
    probability: float = Field(..., description="Probabilité de défaut (0-1)")
    prediction: int = Field(..., description="Prédiction binaire (0=accepté, 1=refusé)")
    decision: str = Field(..., description="Décision en texte")
    threshold: float = Field(..., description="Seuil de décision utilisé")
    risk_level: str = Field(..., description="Niveau de risque (low/medium/high)")


class PredictionWithExplanation(PredictionResponse):
    """Réponse de prédiction avec explications SHAP."""
    shap_values: Dict[str, float] = Field(
        ...,
        description="Contributions SHAP des features principales"
    )
    base_value: float = Field(..., description="Valeur de base SHAP")
    top_positive_features: List[Dict[str, Any]] = Field(
        ...,
        description="Features augmentant le risque"
    )
    top_negative_features: List[Dict[str, Any]] = Field(
        ...,
        description="Features diminuant le risque"
    )


class ModelInfo(BaseModel):
    """Informations sur le modèle."""
    model_type: str
    n_features: int
    threshold: float
    fn_cost: float
    fp_cost: float
    feature_names: List[str]


def load_model():
    """Charge le modèle et initialise l'explainer SHAP."""
    global model, feature_names, explainer

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {MODEL_PATH}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # Support pour différents formats de sauvegarde
    if isinstance(model_data, dict):
        model = model_data.get("model")
        feature_names = model_data.get("feature_names", [])
        # Utiliser le seuil sauvegardé si disponible
        if "threshold" in model_data:
            global THRESHOLD
            THRESHOLD = model_data["threshold"]
    else:
        model = model_data
        feature_names = [f"feat_{i}" for i in range(model.n_features_in_)]

    # Initialiser l'explainer SHAP
    try:
        if hasattr(model, "named_steps"):
            # Pipeline sklearn
            clf = model.named_steps.get("clf", model)
        else:
            clf = model
        explainer = shap.TreeExplainer(clf)
    except Exception as e:
        print(f"Warning: Impossible d'initialiser SHAP explainer: {e}")
        explainer = None

    return model


def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque basé sur la probabilité."""
    if probability < 0.05:
        return "low"
    elif probability < 0.15:
        return "medium"
    else:
        return "high"


@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API."""
    try:
        load_model()
        print(f"Modèle chargé avec succès depuis {MODEL_PATH}")
        print(f"Seuil de décision: {THRESHOLD}")
    except FileNotFoundError:
        print(f"ATTENTION: Modèle non trouvé à {MODEL_PATH}")
        print("L'API démarre mais les prédictions échoueront.")


@app.get("/", tags=["General"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "API Scoring Crédit - Projet 7 OpenClassrooms",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Vérifie l'état de l'API."""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "threshold": THRESHOLD
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Retourne les informations sur le modèle."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    model_type = type(model).__name__
    if hasattr(model, "named_steps"):
        model_type = f"Pipeline({type(model.named_steps['clf']).__name__})"

    return ModelInfo(
        model_type=model_type,
        n_features=len(feature_names),
        threshold=THRESHOLD,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
        feature_names=feature_names[:50]  # Limiter pour la réponse
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(client: ClientData, client_id: Optional[str] = None):
    """
    Prédit la probabilité de défaut pour un client.

    - **features**: Dictionnaire des features du client
    - **client_id**: Identifiant optionnel du client
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Convertir en DataFrame
        df = pd.DataFrame([client.features])

        # Réordonner les colonnes selon le modèle si nécessaire
        if feature_names:
            missing = set(feature_names) - set(df.columns)
            for col in missing:
                df[col] = 0  # Valeur par défaut
            df = df[feature_names]

        # Prédiction
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0, 1])
        else:
            probability = float(model.predict(df)[0])

        prediction = 1 if probability >= THRESHOLD else 0
        decision = "Crédit refusé" if prediction == 1 else "Crédit accordé"
        risk_level = get_risk_level(probability)

        return PredictionResponse(
            client_id=client_id,
            probability=round(probability, 4),
            prediction=prediction,
            decision=decision,
            threshold=THRESHOLD,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/explain", response_model=PredictionWithExplanation, tags=["Prediction"])
async def predict_with_explanation(client: ClientData, client_id: Optional[str] = None):
    """
    Prédit avec explications SHAP détaillées.

    Retourne la prédiction ainsi que les contributions de chaque feature.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer SHAP non disponible")

    try:
        # Convertir en DataFrame
        df = pd.DataFrame([client.features])

        if feature_names:
            missing = set(feature_names) - set(df.columns)
            for col in missing:
                df[col] = 0
            df = df[feature_names]

        # Prédiction
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0, 1])
        else:
            probability = float(model.predict(df)[0])

        prediction = 1 if probability >= THRESHOLD else 0
        decision = "Crédit refusé" if prediction == 1 else "Crédit accordé"
        risk_level = get_risk_level(probability)

        # Calcul SHAP
        if hasattr(model, "named_steps"):
            shap_input = df
        else:
            shap_input = df

        shap_values = explainer.shap_values(shap_input)

        # Gérer les différents formats de sortie SHAP
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Classe positive
        else:
            shap_vals = shap_values[0]

        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, np.ndarray):
            base_value = float(explainer.expected_value[1])

        # Top features
        feature_importance = list(zip(feature_names, shap_vals))
        sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)

        top_positive = [
            {"feature": f, "shap_value": round(float(v), 4), "value": float(df[f].iloc[0])}
            for f, v in sorted_features[:5] if v > 0
        ]

        top_negative = [
            {"feature": f, "shap_value": round(float(v), 4), "value": float(df[f].iloc[0])}
            for f, v in sorted_features[-5:] if v < 0
        ]

        # Dictionnaire SHAP (top 20)
        shap_dict = {f: round(float(v), 4) for f, v in sorted_features[:20]}

        return PredictionWithExplanation(
            client_id=client_id,
            probability=round(probability, 4),
            prediction=prediction,
            decision=decision,
            threshold=THRESHOLD,
            risk_level=risk_level,
            shap_values=shap_dict,
            base_value=round(base_value, 4),
            top_positive_features=top_positive,
            top_negative_features=top_negative
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")


@app.get("/features", tags=["Model"])
async def get_feature_names():
    """Retourne la liste des features attendues par le modèle."""
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"features": feature_names}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
