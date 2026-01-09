"""Module pour l'analyse du Data Drift avec Evidently."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, DATA_DIR


def load_train_test_data():
    """Charge les données train et test pour l'analyse du drift."""
    # Données d'entraînement (référence)
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")

    # Données de test (production simulée)
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")

    return X_train, X_test


def get_top_features(n_features: int = 20) -> list:
    """Récupère les top features par importance."""
    try:
        import joblib
        model = joblib.load(MODELS_DIR / "model.joblib")

        with open(MODELS_DIR / "feature_names.json", "r") as f:
            feature_names = json.load(f)

        clf = model.named_steps["clf"]
        importances = clf.feature_importances_

        indices = np.argsort(importances)[::-1][:n_features]
        return [feature_names[i] for i in indices]
    except Exception:
        return None


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path = None,
    top_features: list = None
) -> dict:
    """
    Génère un rapport de Data Drift avec Evidently.

    Args:
        reference_data: Données de référence (train)
        current_data: Données actuelles (test/production)
        output_path: Chemin pour sauvegarder le rapport HTML
        top_features: Liste des features à analyser (optionnel)

    Returns:
        Dictionnaire avec les résultats du drift
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric, DataDriftTable
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        raise ImportError(
            "Evidently n'est pas installé correctement. "
            "Installez-le avec: pip install evidently"
        )

    # Filtrer sur les top features si spécifié
    if top_features:
        common_cols = [c for c in top_features if c in reference_data.columns and c in current_data.columns]
        reference_data = reference_data[common_cols]
        current_data = current_data[common_cols]

    # Aligner les colonnes
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    print(f"Analyse du drift sur {len(common_cols)} features...")

    # Créer le rapport
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Sauvegarder le rapport HTML
    if output_path is None:
        output_path = DATA_DIR / "drift_report.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"Rapport HTML sauvegardé: {output_path}")

    # Extraire les résultats
    results = report.as_dict()

    # Résumé
    drift_summary = {
        "timestamp": datetime.now().isoformat(),
        "n_features_analyzed": len(common_cols),
        "dataset_drift": results["metrics"][0]["result"]["dataset_drift"],
        "drift_share": results["metrics"][0]["result"]["drift_share"],
        "n_drifted_features": results["metrics"][0]["result"]["number_of_drifted_columns"],
    }

    # Sauvegarder le résumé JSON
    summary_path = output_path.parent / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(drift_summary, f, indent=2)
    print(f"Résumé JSON sauvegardé: {summary_path}")

    return drift_summary


def generate_full_report(output_dir: Path = None):
    """
    Génère un rapport complet incluant Data Drift et Data Quality.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError:
        raise ImportError("Installez evidently: pip install evidently")

    if output_dir is None:
        output_dir = DATA_DIR

    # Charger les données
    X_train, X_test = load_train_test_data()

    # Top features
    top_features = get_top_features(20)
    if top_features:
        print(f"Analyse des {len(top_features)} features les plus importantes")
        common = [c for c in top_features if c in X_train.columns and c in X_test.columns]
        X_train_subset = X_train[common]
        X_test_subset = X_test[common]
    else:
        # Prendre un échantillon de colonnes
        cols = list(set(X_train.columns) & set(X_test.columns))[:50]
        X_train_subset = X_train[cols]
        X_test_subset = X_test[cols]

    # Rapport Data Drift
    print("\n[1/2] Génération du rapport Data Drift...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=X_train_subset, current_data=X_test_subset)
    drift_report.save_html(str(output_dir / "data_drift_report.html"))

    # Rapport Data Quality
    print("[2/2] Génération du rapport Data Quality...")
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=X_train_subset, current_data=X_test_subset)
    quality_report.save_html(str(output_dir / "data_quality_report.html"))

    print(f"\nRapports générés dans {output_dir}:")
    print(f"  - data_drift_report.html")
    print(f"  - data_quality_report.html")

    return {
        "drift_report": output_dir / "data_drift_report.html",
        "quality_report": output_dir / "data_quality_report.html"
    }


def check_drift_threshold(drift_share: float, threshold: float = 0.3) -> bool:
    """
    Vérifie si le drift dépasse le seuil acceptable.

    Args:
        drift_share: Proportion de features avec drift
        threshold: Seuil d'alerte (défaut 30%)

    Returns:
        True si drift acceptable, False si alerte
    """
    if drift_share > threshold:
        print(f"⚠️ ALERTE: Drift détecté sur {drift_share:.1%} des features (seuil: {threshold:.1%})")
        return False
    else:
        print(f"✅ Drift acceptable: {drift_share:.1%} des features (seuil: {threshold:.1%})")
        return True


def main():
    """Exécute l'analyse complète du Data Drift."""
    print("=" * 60)
    print("ANALYSE DU DATA DRIFT AVEC EVIDENTLY")
    print("=" * 60)

    # Test import evidently
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        print("Evidently importé avec succès")
    except ImportError as e:
        print(f"\nErreur d'import Evidently: {e}")
        print("Installez evidently: pip install evidently")
        return

    try:
        # Charger les données
        print("\nChargement des données...")
        X_train, X_test = load_train_test_data()
        print(f"  Référence (train): {X_train.shape}")
        print(f"  Production (test): {X_test.shape}")

        # Top features
        top_features = get_top_features(20)

        if top_features:
            print(f"  Top features: {len(top_features)}")
            common = [c for c in top_features if c in X_train.columns and c in X_test.columns]
            X_train_subset = X_train[common].copy()
            X_test_subset = X_test[common].copy()
        else:
            cols = list(set(X_train.columns) & set(X_test.columns))[:30]
            X_train_subset = X_train[cols].copy()
            X_test_subset = X_test[cols].copy()

        print(f"\nGénération du rapport de drift sur {X_train_subset.shape[1]} features...")

        # Créer le rapport
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_train_subset, current_data=X_test_subset)

        # Sauvegarder
        output_path = DATA_DIR / "drift_report.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))

        print(f"\n" + "=" * 60)
        print("RAPPORT GÉNÉRÉ AVEC SUCCÈS")
        print("=" * 60)
        print(f"\nOuvrez le rapport HTML:")
        print(f"  {output_path}")

    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        print("Exécutez d'abord: python -m src.train")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
