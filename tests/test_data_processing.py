"""Tests pour le module de traitement des données."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import basic_feature_engineering, encode_categoricals


class TestBasicFeatureEngineering:
    """Tests pour le feature engineering basique."""

    def test_credit_income_ratio(self):
        """Test du calcul du ratio crédit/revenu."""
        df = pd.DataFrame({
            "AMT_CREDIT": [100000, 200000],
            "AMT_INCOME_TOTAL": [50000, 100000]
        })
        result = basic_feature_engineering(df)

        assert "CREDIT_INCOME_RATIO" in result.columns
        assert result["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(2.0, rel=0.01)

    def test_ext_source_aggregations(self):
        """Test des agrégations EXT_SOURCE."""
        df = pd.DataFrame({
            "EXT_SOURCE_1": [0.5, 0.6],
            "EXT_SOURCE_2": [0.7, 0.8],
            "EXT_SOURCE_3": [0.3, 0.4]
        })
        result = basic_feature_engineering(df)

        assert "EXT_SOURCE_MEAN" in result.columns
        assert "EXT_SOURCE_STD" in result.columns
        assert "EXT_SOURCE_PROD" in result.columns

    def test_handles_missing_columns(self):
        """Test que les colonnes manquantes ne causent pas d'erreur."""
        df = pd.DataFrame({
            "OTHER_COLUMN": [1, 2, 3]
        })
        result = basic_feature_engineering(df)

        # Ne devrait pas planter
        assert "OTHER_COLUMN" in result.columns


class TestEncodeCategoricals:
    """Tests pour l'encodage des variables catégorielles."""

    def test_one_hot_encoding_few_categories(self):
        """Test du one-hot encoding pour peu de catégories."""
        df = pd.DataFrame({
            "CATEGORY": ["A", "B", "A", "C"]
        })
        result = encode_categoricals(df)

        # La colonne originale devrait être remplacée par des dummies
        assert "CATEGORY" not in result.columns
        assert any("CATEGORY_" in col for col in result.columns)

    def test_label_encoding_many_categories(self):
        """Test du label encoding pour beaucoup de catégories."""
        df = pd.DataFrame({
            "MANY_CATS": [f"cat_{i}" for i in range(20)]
        })
        result = encode_categoricals(df)

        # Devrait être encodé en numérique
        assert result["MANY_CATS"].dtype in [np.int8, np.int16, np.int32, np.int64]

    def test_preserves_numeric_columns(self):
        """Test que les colonnes numériques sont préservées."""
        df = pd.DataFrame({
            "NUMERIC": [1.0, 2.0, 3.0],
            "CATEGORY": ["A", "B", "A"]
        })
        result = encode_categoricals(df)

        assert "NUMERIC" in result.columns
        assert result["NUMERIC"].tolist() == [1.0, 2.0, 3.0]


class TestDataIntegrity:
    """Tests d'intégrité des données."""

    def test_no_duplicate_columns(self):
        """Test qu'il n'y a pas de colonnes dupliquées."""
        df = pd.DataFrame({
            "COL_A": [1, 2],
            "COL_B": [3, 4]
        })
        result = basic_feature_engineering(df)

        assert len(result.columns) == len(set(result.columns))

    def test_handles_nan_values(self):
        """Test que les NaN sont gérés correctement."""
        df = pd.DataFrame({
            "AMT_CREDIT": [100000, np.nan],
            "AMT_INCOME_TOTAL": [50000, 60000]
        })
        result = basic_feature_engineering(df)

        # Ne devrait pas planter
        assert len(result) == 2
