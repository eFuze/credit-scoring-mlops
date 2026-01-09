"""Tests pour le modèle et le module d'entraînement."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cost_functions import get_probas, cost_from_threshold, best_cost_and_threshold


class TestGetProbas:
    """Tests pour la fonction get_probas."""

    def test_with_predict_proba(self):
        """Test avec un modèle ayant predict_proba."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])

        X = pd.DataFrame({"a": [1, 2]})
        result = get_probas(mock_model, X)

        assert len(result) == 2
        assert result[0] == 0.3
        assert result[1] == 0.6

    def test_with_decision_function(self):
        """Test avec un modèle ayant decision_function."""
        mock_model = MagicMock(spec=["decision_function"])
        mock_model.decision_function.return_value = np.array([0, 2])

        X = pd.DataFrame({"a": [1, 2]})
        result = get_probas(mock_model, X)

        assert len(result) == 2
        assert 0 < result[0] < 1  # Sigmoid de 0 = 0.5
        assert result[1] > 0.5  # Sigmoid de 2 > 0.5


class TestCostFunction:
    """Tests pour la fonction de coût métier."""

    def test_perfect_predictions(self):
        """Test avec des prédictions parfaites."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = 0.5

        cost = cost_from_threshold(y_true, y_proba, threshold)

        # Pas de FP, pas de FN
        assert cost == 0.0

    def test_cost_asymmetry(self):
        """Test que FN coûte plus cher que FP."""
        y_true = np.array([0, 1])

        # Un FP
        cost_fp = cost_from_threshold(y_true, np.array([0.9, 0.1]), 0.5)
        # Un FN
        cost_fn = cost_from_threshold(y_true, np.array([0.1, 0.1]), 0.5)

        assert cost_fn > cost_fp  # FN (10) > FP (1)

    def test_custom_costs(self):
        """Test avec des coûts personnalisés."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.9, 0.1])  # 1 FP, 1 FN

        cost = cost_from_threshold(y_true, y_proba, 0.5, fn_cost=5.0, fp_cost=2.0)

        assert cost == 7.0  # 5*1 + 2*1


class TestBestThreshold:
    """Tests pour la recherche du seuil optimal."""

    def test_finds_optimal_threshold(self):
        """Test que le seuil optimal minimise le coût."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7])

        best_cost, best_thr = best_cost_and_threshold(y_true, y_proba)

        # Vérifier que c'est bien le minimum
        for thr in np.linspace(0, 1, 11):
            cost = cost_from_threshold(y_true, y_proba, thr)
            assert cost >= best_cost - 0.001  # Tolérance numérique

    def test_threshold_in_valid_range(self):
        """Test que le seuil est entre 0 et 1."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.3, 0.7])

        _, threshold = best_cost_and_threshold(y_true, y_proba)

        assert 0 <= threshold <= 1

    def test_with_custom_thresholds(self):
        """Test avec une liste de seuils personnalisée."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.3, 0.7])
        custom_thresholds = np.array([0.2, 0.4, 0.6, 0.8])

        _, threshold = best_cost_and_threshold(
            y_true, y_proba, thresholds=custom_thresholds
        )

        assert threshold in custom_thresholds


class TestModelPredictions:
    """Tests de prédiction du modèle."""

    def test_predictions_are_binary(self):
        """Test que les prédictions finales sont binaires."""
        y_proba = np.array([0.1, 0.5, 0.9])
        threshold = 0.5

        predictions = (y_proba >= threshold).astype(int)

        assert set(predictions).issubset({0, 1})

    def test_threshold_effect(self):
        """Test de l'effet du seuil sur les prédictions."""
        y_proba = np.array([0.3, 0.5, 0.7])

        # Seuil bas = plus de positifs
        pred_low = (y_proba >= 0.2).astype(int)
        # Seuil haut = moins de positifs
        pred_high = (y_proba >= 0.8).astype(int)

        assert pred_low.sum() > pred_high.sum()
