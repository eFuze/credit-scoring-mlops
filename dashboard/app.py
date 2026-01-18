"""Dashboard Streamlit allégé pour le scoring crédit.

Version simplifiée: prédiction accepté/refusé uniquement.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH, PROCESSED_DATA_DIR


# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring",
    page_icon="💳",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Charge le modèle et les métadonnées."""
    import joblib

    if not MODEL_PATH.exists():
        return None, None, None

    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as f:
        threshold_data = json.load(f)

    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    return model, threshold_data, feature_names


@st.cache_data
def load_data():
    """Charge les données clients."""
    try:
        X = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
        return X
    except FileNotFoundError:
        return None


def get_proba(model, X):
    """Calcule la probabilité de défaut."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1][0]
    else:
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))[0]


def main():
    st.title("💳 Scoring Crédit")
    st.markdown("---")

    # Charger le modèle
    model, threshold_data, feature_names = load_model()

    if model is None:
        st.error("Modèle non trouvé. Exécutez: `python -m src.train`")
        st.stop()

    threshold = threshold_data["threshold"]
    X = load_data()

    # Sélection du client
    st.subheader("Sélection du client")

    if X is not None:
        client_idx = st.number_input(
            "Numéro du client",
            min_value=0,
            max_value=len(X) - 1,
            value=0
        )

        client_data = X.iloc[[client_idx]].copy()
        client_data.columns = [f"feat_{i}" for i in range(len(feature_names))]
    else:
        st.warning("Données non disponibles. Mode manuel.")
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5)

        client_data = pd.DataFrame(
            np.zeros((1, len(feature_names))),
            columns=[f"feat_{i}" for i in range(len(feature_names))]
        )
        for i, feat in enumerate(feature_names):
            if feat == "EXT_SOURCE_1":
                client_data.iloc[0, i] = ext1
            elif feat == "EXT_SOURCE_2":
                client_data.iloc[0, i] = ext2
            elif feat == "EXT_SOURCE_3":
                client_data.iloc[0, i] = ext3

    st.markdown("---")

    # Bouton de prédiction
    if st.button("Analyser la demande de crédit", type="primary", use_container_width=True):
        proba = get_proba(model, client_data)
        accepted = proba < threshold

        # Résultat principal
        st.markdown("### Décision")

        if accepted:
            st.success("## CRÉDIT ACCEPTÉ", icon="✅")
        else:
            st.error("## CRÉDIT REFUSÉ", icon="❌")

        # Informations complémentaires
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Probabilité de défaut", f"{proba:.1%}")

        with col2:
            st.metric("Seuil de décision", f"{threshold:.1%}")

        # Jauge visuelle
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={'suffix': "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if not accepted else "darkgreen"},
                'steps': [
                    {'range': [0, threshold * 100], 'color': "lightgreen"},
                    {'range': [threshold * 100, 100], 'color': "lightcoral"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
