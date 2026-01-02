"""
Dashboard Streamlit - Test de l'API Scoring Crédit
===================================================
Application interactive pour tester l'API de scoring crédit.
Permet de visualiser les prédictions et les explications SHAP.

Projet 7 OpenClassrooms - Data Science
"""

import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Configuration de la page
st.set_page_config(
    page_title="Scoring Crédit - Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (configurable via variable d'environnement)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> Dict[str, Any]:
    """Vérifie l'état de l'API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_model_info() -> Dict[str, Any]:
    """Récupère les informations du modèle."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return None


def predict(features: Dict[str, float], client_id: str = None) -> Dict[str, Any]:
    """Effectue une prédiction via l'API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            params={"client_id": client_id} if client_id else {},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def predict_with_explanation(features: Dict[str, float], client_id: str = None) -> Dict[str, Any]:
    """Effectue une prédiction avec explications SHAP."""
    try:
        response = requests.post(
            f"{API_URL}/predict/explain",
            json={"features": features},
            params={"client_id": client_id} if client_id else {},
            timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def create_gauge_chart(probability: float, threshold: float) -> go.Figure:
    """Crée un graphique gauge pour la probabilité de défaut."""
    color = "green" if probability < threshold else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de défaut (%)", 'font': {'size': 20}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': 'lightgreen'},
                {'range': [threshold * 100, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_shap_bar_chart(top_positive: list, top_negative: list) -> go.Figure:
    """Crée un graphique en barres pour les contributions SHAP."""
    features = []
    values = []
    colors = []

    for item in reversed(top_negative):
        features.append(item['feature'][:30])  # Tronquer les noms longs
        values.append(item['shap_value'])
        colors.append('blue')

    for item in top_positive:
        features.append(item['feature'][:30])
        values.append(item['shap_value'])
        colors.append('red')

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors
    ))

    fig.update_layout(
        title="Contributions des features (SHAP)",
        xaxis_title="Impact sur la prédiction",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def main():
    """Fonction principale de l'application Streamlit."""

    # En-tête
    st.title("🏦 Dashboard Scoring Crédit")
    st.markdown("**Projet 7 OpenClassrooms - Implémentez un modèle de scoring**")

    # Sidebar - Configuration et état de l'API
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Afficher l'URL de l'API
        st.text_input("URL de l'API", value=API_URL, disabled=True)

        # Vérifier l'état de l'API
        st.subheader("État de l'API")
        health = check_api_health()

        if health.get("status") == "healthy":
            st.success("✅ API opérationnelle")
            st.metric("Seuil de décision", f"{health.get('threshold', 0.083):.1%}")
        elif health.get("status") == "degraded":
            st.warning("⚠️ API en mode dégradé")
        else:
            st.error("❌ API non disponible")
            st.caption(health.get("message", "Erreur de connexion"))

        # Informations sur le modèle
        st.subheader("Informations modèle")
        model_info = get_model_info()
        if model_info:
            st.write(f"**Type:** {model_info.get('model_type', 'N/A')}")
            st.write(f"**Features:** {model_info.get('n_features', 'N/A')}")
            st.write(f"**Coût FN:** {model_info.get('fn_cost', 10)}")
            st.write(f"**Coût FP:** {model_info.get('fp_cost', 1)}")

    # Tabs principaux
    tab1, tab2, tab3 = st.tabs(["📊 Prédiction individuelle", "📁 Prédiction par fichier", "ℹ️ À propos"])

    # Tab 1: Prédiction individuelle
    with tab1:
        st.header("Prédiction pour un client")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Saisie des features principales")

            # Features principales (EXT_SOURCE sont les plus importantes)
            ext_source_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01,
                                     help="Score de source externe 1 (bureau de crédit)")
            ext_source_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01,
                                     help="Score de source externe 2")
            ext_source_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01,
                                     help="Score de source externe 3")

            st.subheader("Autres features")
            days_birth = st.number_input("Âge (années)", 18, 100, 35)
            days_employed = st.number_input("Ancienneté emploi (années)", 0, 50, 5)
            amt_credit = st.number_input("Montant du crédit", 0, 5000000, 500000)
            amt_income = st.number_input("Revenu annuel", 0, 2000000, 150000)

            # Construire le dictionnaire de features
            features = {
                "EXT_SOURCE_1": ext_source_1,
                "EXT_SOURCE_2": ext_source_2,
                "EXT_SOURCE_3": ext_source_3,
                "DAYS_BIRTH": -days_birth * 365,
                "DAYS_EMPLOYED": -days_employed * 365,
                "AMT_CREDIT": float(amt_credit),
                "AMT_INCOME_TOTAL": float(amt_income),
            }

            # Bouton de prédiction
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_btn = st.button("🔮 Prédire", type="primary", use_container_width=True)
            with col_btn2:
                explain_btn = st.button("🔍 Prédire + Expliquer", use_container_width=True)

        with col2:
            st.subheader("Résultat")

            if predict_btn:
                with st.spinner("Calcul en cours..."):
                    result = predict(features)

                if "error" in result:
                    st.error(f"Erreur: {result['error']}")
                else:
                    # Afficher la décision
                    if result['prediction'] == 0:
                        st.success(f"✅ {result['decision']}")
                    else:
                        st.error(f"❌ {result['decision']}")

                    # Gauge de probabilité
                    fig = create_gauge_chart(result['probability'], result['threshold'])
                    st.plotly_chart(fig, use_container_width=True)

                    # Métriques
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Probabilité", f"{result['probability']:.2%}")
                    col_m2.metric("Seuil", f"{result['threshold']:.2%}")
                    col_m3.metric("Risque", result['risk_level'].upper())

            if explain_btn:
                with st.spinner("Calcul des explications SHAP..."):
                    result = predict_with_explanation(features)

                if "error" in result:
                    st.error(f"Erreur: {result['error']}")
                elif "detail" in result:
                    st.error(f"Erreur API: {result['detail']}")
                else:
                    # Afficher la décision
                    if result['prediction'] == 0:
                        st.success(f"✅ {result['decision']}")
                    else:
                        st.error(f"❌ {result['decision']}")

                    # Gauge
                    fig = create_gauge_chart(result['probability'], result['threshold'])
                    st.plotly_chart(fig, use_container_width=True)

                    # Graphique SHAP
                    if result.get('top_positive_features') or result.get('top_negative_features'):
                        fig_shap = create_shap_bar_chart(
                            result.get('top_positive_features', []),
                            result.get('top_negative_features', [])
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                        # Explication textuelle
                        st.subheader("Explication de la décision")
                        st.write(f"**Valeur de base:** {result.get('base_value', 0):.4f}")

                        if result.get('top_positive_features'):
                            st.write("**Facteurs augmentant le risque:**")
                            for f in result['top_positive_features'][:3]:
                                st.write(f"- {f['feature']}: +{f['shap_value']:.4f}")

                        if result.get('top_negative_features'):
                            st.write("**Facteurs diminuant le risque:**")
                            for f in result['top_negative_features'][:3]:
                                st.write(f"- {f['feature']}: {f['shap_value']:.4f}")

    # Tab 2: Prédiction par fichier
    with tab2:
        st.header("Prédiction par fichier CSV")

        st.info("""
        Téléchargez un fichier CSV contenant les features des clients.
        Le fichier doit avoir les colonnes correspondant aux features du modèle.
        """)

        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"**{len(df)} clients chargés**")
                st.dataframe(df.head())

                if st.button("🚀 Prédire pour tous les clients"):
                    predictions = []
                    progress_bar = st.progress(0)

                    for i, row in df.iterrows():
                        features = row.to_dict()
                        result = predict(features, client_id=str(i))
                        predictions.append({
                            'client_id': i,
                            'probability': result.get('probability', None),
                            'prediction': result.get('prediction', None),
                            'decision': result.get('decision', 'Erreur')
                        })
                        progress_bar.progress((i + 1) / len(df))

                    results_df = pd.DataFrame(predictions)

                    st.success(f"✅ {len(results_df)} prédictions effectuées")
                    st.dataframe(results_df)

                    # Statistiques
                    col1, col2 = st.columns(2)
                    with col1:
                        accepted = len(results_df[results_df['prediction'] == 0])
                        refused = len(results_df[results_df['prediction'] == 1])
                        st.metric("Crédits accordés", accepted)
                        st.metric("Crédits refusés", refused)

                    with col2:
                        fig = px.histogram(results_df, x='probability', nbins=20,
                                          title="Distribution des probabilités de défaut")
                        st.plotly_chart(fig, use_container_width=True)

                    # Téléchargement des résultats
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger les résultats",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {e}")

    # Tab 3: À propos
    with tab3:
        st.header("À propos du projet")

        st.markdown("""
        ## Projet 7 - Implémentez un modèle de scoring

        ### Contexte
        Ce dashboard permet aux chargés de relation client d'évaluer la probabilité
        de défaut de paiement d'un demandeur de crédit.

        ### Modèle
        - **Algorithme:** LightGBM (Gradient Boosting)
        - **Métrique optimisée:** Coût métier (10×FN + 1×FP)
        - **Seuil optimal:** ~8.3%
        - **AUC:** ~0.787

        ### Fonction de coût
        Le modèle optimise une fonction de coût asymétrique :
        - **Faux Négatif (FN):** Client qui fait défaut mais accepté → **Coût 10**
        - **Faux Positif (FP):** Bon client refusé → **Coût 1**

        Cette asymétrie reflète la réalité bancaire : un défaut coûte bien plus cher
        (perte du capital) qu'un refus de bon client (manque à gagner sur les intérêts).

        ### Interprétabilité
        Les explications sont fournies via **SHAP (SHapley Additive exPlanations)**,
        permettant de comprendre l'impact de chaque feature sur la décision.

        ### Technologies
        - **API:** FastAPI
        - **Dashboard:** Streamlit
        - **ML:** LightGBM, scikit-learn
        - **Interprétabilité:** SHAP
        - **MLOps:** MLflow
        """)


if __name__ == "__main__":
    main()
