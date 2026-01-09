"""Dashboard Streamlit pour le scoring crédit."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH, PROCESSED_DATA_DIR


# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
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
    """Charge les données pour l'analyse."""
    try:
        X = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
        y = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")["TARGET"]
        return X, y
    except FileNotFoundError:
        return None, None


def get_probas(model, X):
    """Calcule les probabilités."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))


def main():
    st.title("💳 Dashboard Scoring Crédit")

    # Charger le modèle
    model, threshold_data, feature_names = load_model()

    if model is None:
        st.error("⚠️ Modèle non trouvé. Exécutez d'abord: `python -m src.train`")
        st.stop()

    threshold = threshold_data["threshold"]

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Section",
        ["Prédiction Client", "Analyse Globale", "Comparaison Clients"]
    )

    # Charger les données
    X, y = load_data()

    if page == "Prédiction Client":
        prediction_page(model, threshold, feature_names, X, y)
    elif page == "Analyse Globale":
        analysis_page(model, threshold, feature_names, X, y)
    else:
        comparison_page(model, threshold, feature_names, X, y)


def prediction_page(model, threshold, feature_names, X, y):
    """Page de prédiction pour un client."""
    st.header("🎯 Prédiction pour un client")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Sélection du client")

        if X is not None:
            # Mode: sélection depuis les données
            client_idx = st.number_input(
                "Index du client",
                min_value=0,
                max_value=len(X) - 1,
                value=0
            )

            client_data = X.iloc[[client_idx]].copy()
            client_data.columns = [f"feat_{i}" for i in range(len(feature_names))]

            # Afficher quelques features importantes
            st.write("**Features principales:**")
            display_features = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            for feat in display_features:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    val = X.iloc[client_idx, idx]
                    st.write(f"- {feat}: {val:.4f}" if not pd.isna(val) else f"- {feat}: N/A")
        else:
            st.warning("Données non disponibles. Mode manuel.")
            # Mode manuel simplifié
            st.write("Entrez les valeurs des features principales:")
            ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
            ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
            ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5)

            # Créer un DataFrame avec des valeurs par défaut
            client_data = pd.DataFrame(
                np.zeros((1, len(feature_names))),
                columns=[f"feat_{i}" for i in range(len(feature_names))]
            )
            # Remplir les EXT_SOURCE si présents
            for i, feat in enumerate(feature_names):
                if feat == "EXT_SOURCE_1":
                    client_data.iloc[0, i] = ext1
                elif feat == "EXT_SOURCE_2":
                    client_data.iloc[0, i] = ext2
                elif feat == "EXT_SOURCE_3":
                    client_data.iloc[0, i] = ext3

    with col2:
        st.subheader("Résultat de la prédiction")

        if st.button("🔮 Prédire", type="primary"):
            # Prédiction
            proba = get_probas(model, client_data)[0]
            prediction = int(proba >= threshold)

            # Affichage
            if prediction == 0:
                st.success(f"### ✅ CRÉDIT ACCEPTÉ")
            else:
                st.error(f"### ❌ CRÉDIT REFUSÉ")

            # Jauge de probabilité
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilité de défaut (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if prediction else "darkgreen"},
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
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Niveau de risque
            if proba < 0.05:
                risk = "🟢 TRÈS FAIBLE"
            elif proba < 0.10:
                risk = "🟡 FAIBLE"
            elif proba < 0.20:
                risk = "🟠 MODÉRÉ"
            elif proba < 0.40:
                risk = "🔴 ÉLEVÉ"
            else:
                risk = "⛔ TRÈS ÉLEVÉ"

            st.metric("Niveau de risque", risk)
            st.metric("Seuil de décision", f"{threshold:.1%}")

            # Explication SHAP
            try:
                import shap
                st.subheader("📊 Explication de la décision")

                clf = model.named_steps["clf"]
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(client_data)

                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]
                else:
                    shap_vals = shap_values[0]

                # Top 10 features
                indices = np.argsort(np.abs(shap_vals))[::-1][:10]

                df_shap = pd.DataFrame({
                    "Feature": [feature_names[i] for i in indices],
                    "Impact": [shap_vals[i] for i in indices]
                })

                fig = px.bar(
                    df_shap,
                    x="Impact",
                    y="Feature",
                    orientation="h",
                    color="Impact",
                    color_continuous_scale=["green", "lightgray", "red"],
                    color_continuous_midpoint=0
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.info(f"Explication SHAP non disponible: {e}")


def analysis_page(model, threshold, feature_names, X, y):
    """Page d'analyse globale."""
    st.header("📈 Analyse Globale du Modèle")

    if X is None or y is None:
        st.warning("Données non disponibles pour l'analyse.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        # Distribution des probabilités
        st.subheader("Distribution des probabilités")

        X_clean = X.copy()
        X_clean.columns = [f"feat_{i}" for i in range(len(feature_names))]

        # Échantillon pour la vitesse
        sample_size = min(5000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X_clean.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]

        probas = get_probas(model, X_sample)

        df_dist = pd.DataFrame({
            "Probabilité": probas,
            "Classe": y_sample.map({0: "Remboursé", 1: "Défaut"})
        })

        fig = px.histogram(
            df_dist,
            x="Probabilité",
            color="Classe",
            nbins=50,
            opacity=0.7,
            barmode="overlay"
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Seuil ({threshold:.2f})")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Métriques
        st.subheader("Métriques du modèle")

        metrics_path = Path(__file__).parent.parent / "models" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("AUC (hold-out)", f"{metrics.get('holdout_auc', 0):.4f}")
                st.metric("Seuil optimal", f"{metrics.get('final_threshold', threshold):.3f}")
            with col_b:
                st.metric("Score métier CV", f"{metrics.get('cv_business_score', 0):,.0f}")
                st.metric("Taux de défaut", f"{metrics.get('default_rate', 0):.2%}")
        else:
            st.info("Métriques non disponibles.")

    # Feature importance
    st.subheader("📊 Importance des features")

    try:
        clf = model.named_steps["clf"]
        importances = clf.feature_importances_

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(20)

        fig = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Feature importance non disponible: {e}")


def comparison_page(model, threshold, feature_names, X, y):
    """Page de comparaison de clients."""
    st.header("🔄 Comparaison de Clients")

    if X is None:
        st.warning("Données non disponibles.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Client 1")
        idx1 = st.number_input("Index client 1", 0, len(X) - 1, 0, key="c1")

    with col2:
        st.subheader("Client 2")
        idx2 = st.number_input("Index client 2", 0, len(X) - 1, 100, key="c2")

    if st.button("🔍 Comparer", type="primary"):
        X_clean = X.copy()
        X_clean.columns = [f"feat_{i}" for i in range(len(feature_names))]

        clients = X_clean.iloc[[idx1, idx2]]
        probas = get_probas(model, clients)

        col1, col2 = st.columns(2)

        for i, (col, idx, proba) in enumerate(zip([col1, col2], [idx1, idx2], probas)):
            with col:
                pred = int(proba >= threshold)
                if pred == 0:
                    st.success(f"✅ Client {idx}: ACCEPTÉ")
                else:
                    st.error(f"❌ Client {idx}: REFUSÉ")

                st.metric("Probabilité", f"{proba:.2%}")
                st.metric("Classe réelle", "Défaut" if y.iloc[idx] == 1 else "Remboursé")

        # Comparaison des features
        st.subheader("Comparaison des features principales")

        # Top features par importance
        try:
            clf = model.named_steps["clf"]
            top_idx = np.argsort(clf.feature_importances_)[::-1][:10]

            df_comp = pd.DataFrame({
                "Feature": [feature_names[i] for i in top_idx],
                f"Client {idx1}": [X.iloc[idx1, i] for i in top_idx],
                f"Client {idx2}": [X.iloc[idx2, i] for i in top_idx]
            })

            st.dataframe(df_comp, use_container_width=True)

        except Exception as e:
            st.warning(f"Comparaison non disponible: {e}")


if __name__ == "__main__":
    main()
