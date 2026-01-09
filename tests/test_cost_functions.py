"""Tests pour les fonctions de coût métier."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cost_functions import (
    cost_from_threshold,
    best_cost_and_threshold,
    business_score
)


def test_cost_from_threshold_all_positive():
    """Test avec prédiction tout positif."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.9, 0.9, 0.9])
    threshold = 0.5

    cost = cost_from_threshold(y_true, y_proba, threshold)

    # 2 FP (y=0, pred=1), 0 FN
    assert cost == 2.0  # 0*10 + 2*1


def test_cost_from_threshold_all_negative():
    """Test avec prédiction tout négatif."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.1, 0.1])
    threshold = 0.5

    cost = cost_from_threshold(y_true, y_proba, threshold)

    # 0 FP, 2 FN (y=1, pred=0)
    assert cost == 20.0  # 2*10 + 0*1


def test_best_threshold_finds_minimum():
    """Test que best_cost_and_threshold trouve le minimum."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])

    cost, threshold = best_cost_and_threshold(y_true, y_proba)

    # Le seuil optimal devrait être entre 0.4 et 0.7
    assert 0.4 <= threshold <= 0.7


def test_business_score_is_negative_cost():
    """Test que business_score retourne -cost."""
    y_true = np.array([0, 1])
    y_proba = np.array([0.5, 0.5])

    score = business_score(y_true, y_proba)

    assert score <= 0  # Score est toujours négatif ou nul
