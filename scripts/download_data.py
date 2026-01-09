#!/usr/bin/env python3
"""
Script pour télécharger les données Home Credit Default Risk depuis Kaggle.

Prérequis:
1. Installer kaggle: pip install kaggle
2. Configurer l'API Kaggle:
   - Créer un compte sur kaggle.com
   - Aller dans Account -> API -> Create New Token
   - Placer le fichier kaggle.json dans ~/.kaggle/

Usage:
    python scripts/download_data.py
"""
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR


def download_kaggle_data():
    """Télécharge les données depuis Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Erreur: kaggle n'est pas installé.")
        print("Installez-le avec: pip install kaggle")
        print("Puis configurez ~/.kaggle/kaggle.json")
        sys.exit(1)

    print("Connexion à l'API Kaggle...")
    api = KaggleApi()
    api.authenticate()

    competition = "home-credit-default-risk"

    print(f"Téléchargement des données {competition}...")
    print(f"Destination: {RAW_DATA_DIR}")

    # Télécharger les fichiers principaux
    files_to_download = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv",
    ]

    for filename in files_to_download:
        print(f"  Téléchargement de {filename}...")
        try:
            api.competition_download_file(
                competition,
                filename,
                path=str(RAW_DATA_DIR),
                quiet=False
            )
        except Exception as e:
            print(f"  Erreur pour {filename}: {e}")

    # Décompresser les fichiers .zip
    import zipfile
    for zip_file in RAW_DATA_DIR.glob("*.zip"):
        print(f"  Extraction de {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(RAW_DATA_DIR)
        zip_file.unlink()  # Supprimer le zip après extraction

    print("\nTéléchargement terminé!")
    print(f"Fichiers disponibles dans {RAW_DATA_DIR}:")
    for f in sorted(RAW_DATA_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")


def create_sample_data():
    """Crée des données d'exemple si Kaggle n'est pas configuré."""
    import numpy as np
    import pandas as pd

    print("Création de données d'exemple (sans Kaggle)...")

    np.random.seed(42)
    n_samples = 10000
    n_features = 100

    # Générer des features synthétiques
    X = np.random.randn(n_samples, n_features)

    # Variable cible déséquilibrée (~8% de défauts)
    y = (np.random.rand(n_samples) < 0.08).astype(int)

    # Ajouter une corrélation entre certaines features et la cible
    X[:, 0] = X[:, 0] - 2 * y  # EXT_SOURCE_1
    X[:, 1] = X[:, 1] - 1.5 * y  # EXT_SOURCE_2
    X[:, 2] = X[:, 2] - 1.8 * y  # EXT_SOURCE_3

    # Créer les DataFrames
    feature_names = [f"feature_{i}" for i in range(n_features)]
    feature_names[0] = "EXT_SOURCE_1"
    feature_names[1] = "EXT_SOURCE_2"
    feature_names[2] = "EXT_SOURCE_3"

    df_train = pd.DataFrame(X, columns=feature_names)
    df_train["TARGET"] = y
    df_train["SK_ID_CURR"] = range(1, n_samples + 1)

    # Données de test (sans TARGET)
    X_test = np.random.randn(2000, n_features)
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test["SK_ID_CURR"] = range(n_samples + 1, n_samples + 2001)

    # Sauvegarder
    train_path = RAW_DATA_DIR / "application_train.csv"
    test_path = RAW_DATA_DIR / "application_test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Données d'entraînement: {train_path}")
    print(f"  - {len(df_train)} lignes, {len(df_train.columns)} colonnes")
    print(f"  - Taux de défaut: {y.mean():.2%}")
    print(f"Données de test: {test_path}")
    print(f"  - {len(df_test)} lignes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Télécharger les données")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Créer des données d'exemple (sans Kaggle)"
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.sample:
        create_sample_data()
    else:
        # Vérifier si kaggle est configuré
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print("Fichier ~/.kaggle/kaggle.json non trouvé.")
            print("Création de données d'exemple à la place...")
            create_sample_data()
        else:
            download_kaggle_data()
