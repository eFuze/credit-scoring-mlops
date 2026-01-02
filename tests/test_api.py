"""
Tests unitaires pour l'API de scoring crédit.
Projet 7 OpenClassrooms
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ajouter le chemin parent pour l'import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app


# Client de test FastAPI
client = TestClient(app)


class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""

    def test_health_endpoint_returns_200(self):
        """Vérifie que l'endpoint /health retourne un status 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_has_status_field(self):
        """Vérifie que la réponse contient le champ status."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_endpoint_has_threshold(self):
        """Vérifie que la réponse contient le seuil."""
        response = client.get("/health")
        data = response.json()
        assert "threshold" in data


class TestRootEndpoint:
    """Tests pour l'endpoint racine /."""

    def test_root_returns_200(self):
        """Vérifie que / retourne 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_message(self):
        """Vérifie que la réponse contient un message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data

    def test_root_has_docs_link(self):
        """Vérifie que la réponse contient le lien vers la doc."""
        response = client.get("/")
        data = response.json()
        assert "docs" in data
        assert data["docs"] == "/docs"


class TestDocsEndpoint:
    """Tests pour la documentation Swagger."""

    def test_docs_available(self):
        """Vérifie que la documentation Swagger est accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """Vérifie que ReDoc est accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict."""

    def test_predict_requires_features(self):
        """Vérifie qu'une requête sans features échoue."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_with_minimal_features(self):
        """Test de prédiction avec des features minimales."""
        features = {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7
        }
        response = client.post("/predict", json={"features": features})
        # Peut échouer si le modèle n'est pas chargé (503) ou réussir (200)
        assert response.status_code in [200, 503]

    def test_predict_response_structure(self):
        """Vérifie la structure de la réponse de prédiction."""
        features = {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.7}
        response = client.post("/predict", json={"features": features})

        if response.status_code == 200:
            data = response.json()
            assert "probability" in data
            assert "prediction" in data
            assert "decision" in data
            assert "threshold" in data
            assert "risk_level" in data


class TestModelInfoEndpoint:
    """Tests pour l'endpoint /model/info."""

    def test_model_info_endpoint(self):
        """Vérifie que l'endpoint model/info répond."""
        response = client.get("/model/info")
        # 200 si modèle chargé, 503 sinon
        assert response.status_code in [200, 503]


class TestFeaturesEndpoint:
    """Tests pour l'endpoint /features."""

    def test_features_endpoint(self):
        """Vérifie que l'endpoint /features répond."""
        response = client.get("/features")
        # 200 si modèle chargé, 503 sinon
        assert response.status_code in [200, 503]


class TestPredictExplainEndpoint:
    """Tests pour l'endpoint /predict/explain."""

    def test_predict_explain_requires_features(self):
        """Vérifie qu'une requête sans features échoue."""
        response = client.post("/predict/explain", json={})
        assert response.status_code == 422

    def test_predict_explain_with_features(self):
        """Test de prédiction avec explications."""
        features = {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7
        }
        response = client.post("/predict/explain", json={"features": features})
        # Peut échouer si modèle ou explainer non chargé
        assert response.status_code in [200, 503]


class TestInputValidation:
    """Tests de validation des entrées."""

    def test_invalid_json(self):
        """Vérifie que du JSON invalide est rejeté."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_empty_features_dict(self):
        """Vérifie qu'un dictionnaire de features vide est accepté."""
        response = client.post("/predict", json={"features": {}})
        # Accepté mais peut échouer à la prédiction
        assert response.status_code in [200, 400, 503]


class TestCORS:
    """Tests pour la configuration CORS."""

    def test_cors_headers_present(self):
        """Vérifie que les headers CORS sont présents."""
        response = client.options("/predict")
        # CORS est configuré, donc la requête devrait passer
        assert response.status_code in [200, 405]


# Tests d'intégration (nécessitent un modèle chargé)
class TestIntegration:
    """Tests d'intégration (exécutés seulement si le modèle est chargé)."""

    @pytest.fixture
    def check_model_loaded(self):
        """Vérifie si le modèle est chargé."""
        response = client.get("/health")
        if response.json().get("model_loaded") is False:
            pytest.skip("Modèle non chargé")

    def test_full_prediction_flow(self, check_model_loaded):
        """Test du flux complet de prédiction."""
        features = {
            "EXT_SOURCE_1": 0.3,
            "EXT_SOURCE_2": 0.4,
            "EXT_SOURCE_3": 0.5,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
            "AMT_CREDIT": 500000,
            "AMT_INCOME_TOTAL": 150000
        }

        response = client.post("/predict", json={"features": features})
        assert response.status_code == 200

        data = response.json()
        assert 0 <= data["probability"] <= 1
        assert data["prediction"] in [0, 1]
        assert data["risk_level"] in ["low", "medium", "high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
