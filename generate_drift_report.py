"""
Script pour générer le rapport de Data Drift avec Evidently.
Génère des données synthétiques basées sur les features du modèle.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Chemins
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("livrables")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_feature_names():
    """Charge les noms des features du modèle."""
    with open(MODELS_DIR / "feature_names.json", "r") as f:
        return json.load(f)

def generate_synthetic_data(feature_names, n_samples=1000, seed=42):
    """Génère des données synthétiques pour les features."""
    np.random.seed(seed)

    data = {}
    for feat in feature_names:
        # Identifier le type de feature par son nom
        if feat.startswith("FLAG_") or feat.startswith("CODE_") or "_nan" in feat:
            # Variable binaire
            data[feat] = np.random.binomial(1, 0.3, n_samples)
        elif "AMT_" in feat or "DAYS_" in feat or "CNT_" in feat:
            # Variable continue positive
            data[feat] = np.abs(np.random.normal(50000, 30000, n_samples))
        elif "EXT_SOURCE" in feat or "RATIO" in feat:
            # Variable entre 0 et 1
            data[feat] = np.clip(np.random.beta(2, 5, n_samples), 0, 1)
        elif "REGION_RATING" in feat:
            # Variable catégorielle ordonnée
            data[feat] = np.random.choice([1, 2, 3], n_samples)
        else:
            # Variable continue standard
            data[feat] = np.random.normal(0, 1, n_samples)

    return pd.DataFrame(data)

def generate_drifted_data(reference_data, drift_features=None, drift_magnitude=0.3, seed=123):
    """Génère des données avec drift sur certaines features."""
    np.random.seed(seed)

    current_data = reference_data.copy()

    if drift_features is None:
        # Appliquer du drift sur ~20% des features
        n_drift = max(5, len(reference_data.columns) // 5)
        drift_features = np.random.choice(reference_data.columns, n_drift, replace=False)

    for feat in drift_features:
        if reference_data[feat].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Ajouter un shift à la moyenne
            shift = reference_data[feat].std() * drift_magnitude
            current_data[feat] = current_data[feat] + shift

    return current_data, list(drift_features)

def generate_drift_report():
    """Génère le rapport de drift HTML avec Evidently."""
    print("=" * 60)
    print("GÉNÉRATION DU RAPPORT DE DATA DRIFT")
    print("=" * 60)

    # Import Evidently
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric, DataDriftTable
        print("✅ Evidently importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur: {e}")
        print("Installez Evidently: pip install evidently")
        return None

    # Charger les features
    print("\n📂 Chargement des features du modèle...")
    feature_names = load_feature_names()
    print(f"   {len(feature_names)} features trouvées")

    # Sélectionner les 30 features les plus importantes pour le rapport
    # (EXT_SOURCE sont connues pour être très importantes)
    priority_features = [f for f in feature_names if "EXT_SOURCE" in f]
    other_features = [f for f in feature_names if f not in priority_features][:27]
    selected_features = priority_features + other_features
    print(f"   {len(selected_features)} features sélectionnées pour l'analyse")

    # Générer les données de référence (train)
    print("\n📊 Génération des données de référence (train)...")
    reference_data = generate_synthetic_data(selected_features, n_samples=2000, seed=42)
    print(f"   Shape: {reference_data.shape}")

    # Générer les données courantes avec drift (production)
    print("\n📊 Génération des données de production (avec drift)...")
    # Utiliser les features EXT_SOURCE qui sont toujours présentes
    drift_candidates = [f for f in selected_features if "EXT_SOURCE" in f or "AMT_" in f or "DAYS_" in f][:5]
    current_data, drifted_features = generate_drifted_data(
        reference_data,
        drift_features=drift_candidates if drift_candidates else None,
        drift_magnitude=0.5,
        seed=123
    )
    print(f"   Shape: {current_data.shape}")
    print(f"   Features avec drift simulé: {drifted_features}")

    # Créer le rapport Evidently
    print("\n📝 Création du rapport Evidently...")
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Sauvegarder le rapport HTML
    timestamp = datetime.now().strftime("%m%Y")
    output_filename = f"Livrable_4_Tableau_HTML_data_drift_evidently_{timestamp}.html"
    output_path = OUTPUT_DIR / output_filename

    report.save_html(str(output_path))

    print("\n" + "=" * 60)
    print("✅ RAPPORT GÉNÉRÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"\n📄 Fichier: {output_path}")
    print(f"\n💡 Ouvrez ce fichier dans votre navigateur pour visualiser le rapport.")

    # Résumé
    results = report.as_dict()
    drift_share = results["metrics"][0]["result"]["drift_share"]
    n_drifted = results["metrics"][0]["result"]["number_of_drifted_columns"]

    print(f"\n📈 Résumé:")
    print(f"   - Features analysées: {len(selected_features)}")
    print(f"   - Features avec drift détecté: {n_drifted}")
    print(f"   - Proportion de drift: {drift_share:.1%}")

    return output_path

if __name__ == "__main__":
    generate_drift_report()
