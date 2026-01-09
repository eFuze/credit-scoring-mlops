#!/usr/bin/env python3
"""Module d'entraînement du modèle de scoring crédit."""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from src.config import (
    LGBM_PARAMS, MODEL_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH,
    MODELS_DIR, FN_COST, FP_COST, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from src.cost_functions import (
    get_probas, best_cost_and_threshold, biz_scorer, thr_scorer
)
from src.data_processing import load_processed_data, prepare_data


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any] = None,
    use_mlflow: bool = True
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """
    Entraîne le modèle LightGBM avec validation croisée.

    Args:
        X: Features
        y: Target
        params: Paramètres LightGBM (utilise LGBM_PARAMS par défaut)
        use_mlflow: Activer le tracking MLflow

    Returns:
        model: Modèle entraîné
        optimal_threshold: Seuil optimal pour la classification
        metrics: Métriques de performance
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE SCORING CRÉDIT")
    print("=" * 60)

    # Nettoyer les noms de colonnes pour LightGBM
    feature_names = X.columns.tolist()
    X_clean = X.copy()
    X_clean.columns = [f"feat_{i}" for i in range(X.shape[1])]

    # Créer le pipeline
    model = Pipeline([
        ("clf", LGBMClassifier(**params))
    ])

    # Cross-validation
    print("\n[1/3] Cross-validation (3 folds)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

    cv_results = cross_validate(
        model, X_clean, y,
        cv=cv,
        scoring={"biz": biz_scorer, "auc": "roc_auc", "thr": thr_scorer},
        return_train_score=False,
        n_jobs=1
    )

    cv_biz = cv_results["test_biz"].mean()
    cv_auc = cv_results["test_auc"].mean()
    cv_thr = cv_results["test_thr"].mean()

    print(f"  Score métier CV: {cv_biz:.0f}")
    print(f"  AUC CV: {cv_auc:.4f}")
    print(f"  Seuil optimal moyen: {cv_thr:.3f}")

    # Hold-out validation
    print("\n[2/3] Validation hold-out (20%)...")
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_clean, y, test_size=0.2, random_state=7, stratify=y
    )

    model.fit(X_train, y_train)
    proba_holdout = get_probas(model, X_holdout)

    holdout_cost, holdout_thr = best_cost_and_threshold(
        y_holdout.values, proba_holdout, fn_cost=FN_COST, fp_cost=FP_COST
    )
    holdout_auc = roc_auc_score(y_holdout, proba_holdout)

    print(f"  AUC hold-out: {holdout_auc:.4f}")
    print(f"  Coût hold-out: {holdout_cost:.0f}")
    print(f"  Seuil optimal hold-out: {holdout_thr:.3f}")

    # Entraînement final sur toutes les données
    print("\n[3/3] Entraînement final sur 100% des données...")
    model.fit(X_clean, y)

    # Recalculer le seuil optimal sur les données complètes
    proba_full = get_probas(model, X_clean)
    _, optimal_threshold = best_cost_and_threshold(
        y.values, proba_full, fn_cost=FN_COST, fp_cost=FP_COST
    )

    print(f"  Seuil final: {optimal_threshold:.3f}")

    metrics = {
        "cv_business_score": float(cv_biz),
        "cv_auc": float(cv_auc),
        "cv_threshold": float(cv_thr),
        "holdout_auc": float(holdout_auc),
        "holdout_cost": float(holdout_cost),
        "holdout_threshold": float(holdout_thr),
        "final_threshold": float(optimal_threshold),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "default_rate": float(y.mean()),
    }

    # MLflow tracking
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")

            print("\n  Métriques enregistrées dans MLflow")
        except Exception as e:
            print(f"\n  MLflow non disponible: {e}")

    return model, optimal_threshold, metrics, feature_names


def save_model(
    model: Pipeline,
    threshold: float,
    feature_names: list,
    metrics: Dict[str, Any]
) -> None:
    """Sauvegarde le modèle et ses métadonnées."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle
    joblib.dump(model, MODEL_PATH)
    print(f"\nModèle sauvegardé: {MODEL_PATH}")

    # Sauvegarder le seuil
    threshold_data = {
        "threshold": threshold,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Seuil sauvegardé: {THRESHOLD_PATH}")

    # Sauvegarder les noms de features
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    print(f"Features sauvegardées: {FEATURE_NAMES_PATH}")

    # Sauvegarder les métriques
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées: {metrics_path}")


def load_model() -> Tuple[Pipeline, float, list]:
    """Charge le modèle sauvegardé."""
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as f:
        threshold_data = json.load(f)

    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    return model, threshold_data["threshold"], feature_names


def main():
    """Pipeline d'entraînement complet."""
    # Vérifier/préparer les données
    try:
        X_train, y_train, X_test = load_processed_data()
        print("Données chargées depuis le cache.")
    except FileNotFoundError:
        print("Préparation des données...")
        X_train, y_train, X_test, _ = prepare_data(save=True)

    # Entraîner le modèle
    model, threshold, metrics, feature_names = train_model(
        X_train, y_train, use_mlflow=True
    )

    # Sauvegarder
    save_model(model, threshold, feature_names, metrics)

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    print(f"\nPour lancer l'API: uvicorn api.main:app --reload")
    print(f"Pour lancer le dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
