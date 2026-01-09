"""Configuration pytest et fixtures partagées."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Génère des données d'exemple pour les tests."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        "feature_0": np.random.randn(n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "EXT_SOURCE_1": np.random.rand(n_samples),
        "EXT_SOURCE_2": np.random.rand(n_samples),
        "AMT_CREDIT": np.random.uniform(10000, 500000, n_samples),
        "AMT_INCOME_TOTAL": np.random.uniform(20000, 200000, n_samples),
    })

    y = pd.Series((np.random.rand(n_samples) < 0.1).astype(int), name="TARGET")

    return X, y


@pytest.fixture
def sample_probas():
    """Génère des probabilités d'exemple."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def binary_labels():
    """Génère des labels binaires d'exemple."""
    np.random.seed(42)
    return (np.random.rand(100) < 0.1).astype(int)
