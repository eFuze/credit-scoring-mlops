"""Tests pour l'API FastAPI."""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock du modèle pour les tests sans fichier modèle
@pytest.fixture
def mock_model_data():
    """Fixture pour mocker les données du modèle."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    return {
        "model": mock_model,
        "threshold_data": {"threshold": 0.083, "fn_cost": 10, "fp_cost": 1},
        "feature_names": ["feature_0", "feature_1", "feature_2"]
    }


@pytest.fixture
def client(mock_model_data):
    """Fixture pour créer un client de test."""
    from fastapi.testclient import TestClient

    # Patcher le chargement du modèle
    with patch.dict("api.main.model_data", mock_model_data):
        from api.main import app
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""

    def test_health_check(self, client):
        """Test que /health retourne un status healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests pour l'endpoint racine."""

    def test_root_returns_info(self, client):
        """Test que / retourne les informations de l'API."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict."""

    def test_predict_returns_valid_response(self, client):
        """Test qu'une prédiction retourne une réponse valide."""
        payload = {
            "features": {
                "feature_0": 0.5,
                "feature_1": 0.3,
                "feature_2": 0.7
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "probability" in data
        assert "prediction" in data
        assert "threshold" in data
        assert "decision" in data
        assert "risk_level" in data

    def test_predict_probability_in_range(self, client):
        """Test que la probabilité est entre 0 et 1."""
        payload = {"features": {"feature_0": 0.5}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert 0 <= data["probability"] <= 1

    def test_predict_decision_is_valid(self, client):
        """Test que la décision est ACCEPTÉ ou REFUSÉ."""
        payload = {"features": {"feature_0": 0.5}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["decision"] in ["ACCEPTÉ", "REFUSÉ"]

    def test_predict_with_empty_features(self, client):
        """Test avec features vides."""
        payload = {"features": {}}
        response = client.post("/predict", json=payload)
        # Devrait fonctionner (features manquantes = NaN)
        assert response.status_code == 200


class TestModelInfoEndpoint:
    """Tests pour l'endpoint /model/info."""

    def test_model_info_returns_valid_data(self, client):
        """Test que /model/info retourne les infos du modèle."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_type" in data
        assert "threshold" in data
        assert "fn_cost" in data
        assert "fp_cost" in data
        assert "n_features" in data


class TestBatchPredictEndpoint:
    """Tests pour l'endpoint /predict/batch."""

    def test_batch_predict_multiple_clients(self, client):
        """Test des prédictions en batch."""
        payload = {
            "clients": [
                {"feature_0": 0.5, "feature_1": 0.3},
                {"feature_0": 0.8, "feature_1": 0.1},
                {"feature_0": 0.2, "feature_1": 0.9}
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    def test_batch_predict_empty_list(self, client):
        """Test avec liste vide."""
        payload = {"clients": []}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["predictions"] == []
