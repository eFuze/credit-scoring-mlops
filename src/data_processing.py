"""Module de préparation et feature engineering des données."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données brutes."""
    train_path = RAW_DATA_DIR / "application_train.csv"
    test_path = RAW_DATA_DIR / "application_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {train_path}\n"
            "Exécutez d'abord: python scripts/download_data.py --sample"
        )

    print(f"Chargement des données depuis {RAW_DATA_DIR}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"  Train: {df_train.shape[0]:,} lignes, {df_train.shape[1]} colonnes")
    print(f"  Test: {df_test.shape[0]:,} lignes, {df_test.shape[1]} colonnes")

    return df_train, df_test


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering basique."""
    df = df.copy()

    # Ratios financiers si les colonnes existent
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)

    # Interactions EXT_SOURCE
    ext_sources = [c for c in df.columns if c.startswith("EXT_SOURCE")]
    if len(ext_sources) >= 2:
        df["EXT_SOURCE_MEAN"] = df[ext_sources].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_sources].std(axis=1)
        df["EXT_SOURCE_PROD"] = df[ext_sources].prod(axis=1)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles."""
    df = df.copy()

    # Identifier les colonnes catégorielles
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        # One-hot encoding pour les colonnes avec peu de modalités
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
        else:
            # Label encoding pour les autres
            df[col] = df[col].astype('category').cat.codes

    return df


def prepare_data(save: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, list]:
    """
    Pipeline complet de préparation des données.

    Returns:
        X_train: Features d'entraînement
        y_train: Target
        X_test: Features de test
        feature_names: Liste des noms de features
    """
    # Charger les données brutes
    df_train, df_test = load_raw_data()

    # Séparer target et ID
    y_train = df_train["TARGET"].astype(int)
    train_ids = df_train["SK_ID_CURR"]
    test_ids = df_test["SK_ID_CURR"]

    df_train = df_train.drop(["TARGET", "SK_ID_CURR"], axis=1)
    df_test = df_test.drop(["SK_ID_CURR"], axis=1)

    print("\nFeature engineering...")
    df_train = basic_feature_engineering(df_train)
    df_test = basic_feature_engineering(df_test)

    print("Encodage des variables catégorielles...")
    df_train = encode_categoricals(df_train)
    df_test = encode_categoricals(df_test)

    # Aligner les colonnes
    common_cols = list(set(df_train.columns) & set(df_test.columns))
    df_train = df_train[common_cols]
    df_test = df_test[common_cols]

    # Remplacer les infinis
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    feature_names = df_train.columns.tolist()

    print(f"\nDonnées préparées:")
    print(f"  X_train: {df_train.shape}")
    print(f"  X_test: {df_test.shape}")
    print(f"  Taux de défaut: {y_train.mean():.2%}")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        df_train.to_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
        y_train.to_frame().to_parquet(PROCESSED_DATA_DIR / "y_train.parquet")
        df_test.to_parquet(PROCESSED_DATA_DIR / "X_test.parquet")

        # Sauvegarder les IDs
        train_ids.to_frame().to_parquet(PROCESSED_DATA_DIR / "train_ids.parquet")
        test_ids.to_frame().to_parquet(PROCESSED_DATA_DIR / "test_ids.parquet")

        print(f"\nDonnées sauvegardées dans {PROCESSED_DATA_DIR}")

    return df_train, y_train, df_test, feature_names


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Charge les données déjà préparées."""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")["TARGET"]
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")

    return X_train, y_train, X_test


if __name__ == "__main__":
    prepare_data(save=True)
