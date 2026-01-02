"""
Script de génération du rapport de Data Drift avec Evidently
=============================================================
Projet 7 OpenClassrooms - Data Science

Ce script génère un rapport HTML analysant le data drift entre :
- Les données d'entraînement (référence)
- Les données de production/test (courantes)

Le rapport est sauvegardé dans reports/data_drift_report.html
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Evidently pour le data drift
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric
)

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Features les plus importantes (top 20 du modèle)
TOP_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "HOUR_APPR_PROCESS_START",
    "DAYS_LAST_PHONE_CHANGE",
    "OWN_CAR_AGE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS"
]


def load_data(train_path: Path = None, test_path: Path = None):
    """
    Charge les données d'entraînement et de test.

    Si les fichiers ne sont pas fournis, génère des données simulées
    pour démonstration.
    """

    # Essayer de charger les vraies données
    if train_path and train_path.exists():
        print(f"Chargement des données train: {train_path}")
        df_train = pd.read_parquet(train_path)
    else:
        print("Génération de données d'entraînement simulées...")
        df_train = generate_sample_data(n_samples=10000, seed=42)

    if test_path and test_path.exists():
        print(f"Chargement des données test: {test_path}")
        df_test = pd.read_parquet(test_path)
    else:
        print("Génération de données de test simulées (avec drift)...")
        df_test = generate_sample_data(n_samples=2000, seed=123, add_drift=True)

    # Ne garder que les features disponibles
    available_features = [f for f in TOP_FEATURES if f in df_train.columns and f in df_test.columns]

    if len(available_features) == 0:
        # Utiliser toutes les colonnes numériques
        available_features = df_train.select_dtypes(include=[np.number]).columns[:20].tolist()

    return df_train[available_features], df_test[available_features], available_features


def generate_sample_data(n_samples: int, seed: int, add_drift: bool = False) -> pd.DataFrame:
    """
    Génère des données simulées pour la démonstration.

    Si add_drift=True, introduit un décalage pour simuler le data drift.
    """
    np.random.seed(seed)

    # Générer les features principales
    data = {
        "EXT_SOURCE_1": np.random.beta(2, 5, n_samples),
        "EXT_SOURCE_2": np.random.beta(3, 3, n_samples),
        "EXT_SOURCE_3": np.random.beta(2, 4, n_samples),
        "DAYS_BIRTH": np.random.randint(-25000, -7000, n_samples),
        "DAYS_EMPLOYED": np.random.randint(-15000, 0, n_samples),
        "DAYS_REGISTRATION": np.random.randint(-20000, 0, n_samples),
        "DAYS_ID_PUBLISH": np.random.randint(-6000, 0, n_samples),
        "AMT_CREDIT": np.random.lognormal(12, 1, n_samples),
        "AMT_ANNUITY": np.random.lognormal(9, 0.8, n_samples),
        "AMT_INCOME_TOTAL": np.random.lognormal(11, 0.7, n_samples),
        "AMT_GOODS_PRICE": np.random.lognormal(12, 1, n_samples),
        "REGION_POPULATION_RELATIVE": np.random.uniform(0, 0.1, n_samples),
        "HOUR_APPR_PROCESS_START": np.random.randint(0, 24, n_samples),
        "DAYS_LAST_PHONE_CHANGE": np.random.randint(-4000, 0, n_samples),
        "OWN_CAR_AGE": np.random.exponential(10, n_samples),
        "CODE_GENDER": np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        "FLAG_OWN_CAR": np.random.choice([0, 1], n_samples, p=[0.66, 0.34]),
        "FLAG_OWN_REALTY": np.random.choice([0, 1], n_samples, p=[0.31, 0.69]),
        "CNT_CHILDREN": np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.2, 0.07, 0.02, 0.01]),
        "CNT_FAM_MEMBERS": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.25, 0.1, 0.05])
    }

    if add_drift:
        # Simuler un drift sur certaines features
        # EXT_SOURCE décalés (clients avec moins de données externes)
        data["EXT_SOURCE_1"] = data["EXT_SOURCE_1"] * 0.8 + 0.1
        data["EXT_SOURCE_2"] = data["EXT_SOURCE_2"] * 0.9

        # Population plus jeune
        data["DAYS_BIRTH"] = data["DAYS_BIRTH"] * 0.85

        # Montants de crédit plus élevés (inflation)
        data["AMT_CREDIT"] = data["AMT_CREDIT"] * 1.15
        data["AMT_ANNUITY"] = data["AMT_ANNUITY"] * 1.12

        # Plus de demandes le soir (changement comportemental)
        data["HOUR_APPR_PROCESS_START"] = np.clip(
            data["HOUR_APPR_PROCESS_START"] + np.random.normal(2, 1, n_samples),
            0, 23
        ).astype(int)

    return pd.DataFrame(data)


def generate_drift_report(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_names: list,
    output_path: Path = None
) -> str:
    """
    Génère le rapport de data drift avec Evidently.

    Args:
        df_reference: DataFrame de référence (données d'entraînement)
        df_current: DataFrame courant (données de production)
        feature_names: Liste des features à analyser
        output_path: Chemin du fichier HTML de sortie

    Returns:
        Chemin du fichier HTML généré
    """

    if output_path is None:
        output_path = OUTPUT_DIR / "data_drift_report.html"

    print(f"\nGénération du rapport de data drift...")
    print(f"  - Données de référence: {len(df_reference)} échantillons")
    print(f"  - Données courantes: {len(df_current)} échantillons")
    print(f"  - Features analysées: {len(feature_names)}")

    # Configuration des colonnes
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = [
        f for f in feature_names
        if df_reference[f].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    # Créer le rapport avec plusieurs métriques
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    # Exécuter le rapport
    report.run(
        reference_data=df_reference,
        current_data=df_current,
        column_mapping=column_mapping
    )

    # Sauvegarder en HTML
    report.save_html(str(output_path))
    print(f"\n✅ Rapport sauvegardé: {output_path}")

    # Afficher un résumé
    try:
        result = report.as_dict()
        metrics = result.get("metrics", [])

        for metric in metrics:
            metric_result = metric.get("result", {})
            if "drift_share" in metric_result:
                drift_share = metric_result["drift_share"]
                print(f"\n📊 Résumé du drift:")
                print(f"   - Proportion de features avec drift: {drift_share:.1%}")

            if "dataset_drift" in metric_result:
                dataset_drift = metric_result["dataset_drift"]
                if dataset_drift:
                    print("   ⚠️  DRIFT DÉTECTÉ au niveau du dataset")
                else:
                    print("   ✅ Pas de drift significatif détecté")
    except Exception as e:
        print(f"Note: Impossible d'extraire le résumé: {e}")

    return str(output_path)


def main():
    """Fonction principale."""

    print("=" * 60)
    print("GÉNÉRATION DU RAPPORT DE DATA DRIFT - EVIDENTLY")
    print("Projet 7 OpenClassrooms")
    print("=" * 60)

    # Chemins des données (à adapter selon votre configuration)
    train_path = DATA_DIR / "X_train_fe.parquet"
    test_path = DATA_DIR / "X_test_fe.parquet"

    # Charger les données
    df_train, df_test, features = load_data(train_path, test_path)

    print(f"\nFeatures utilisées ({len(features)}):")
    for f in features[:10]:
        print(f"  - {f}")
    if len(features) > 10:
        print(f"  ... et {len(features) - 10} autres")

    # Générer le rapport
    output_path = generate_drift_report(df_train, df_test, features)

    print("\n" + "=" * 60)
    print("RAPPORT GÉNÉRÉ AVEC SUCCÈS")
    print(f"Ouvrez le fichier: {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    main()
