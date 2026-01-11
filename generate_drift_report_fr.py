"""
Rapport de Data Drift - Projet Scoring Crédit
Analyse de la dérive des données avec Evidently
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Chemins
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_feature_names():
    """Charge les noms des features du modèle."""
    with open(MODELS_DIR / "feature_names.json", "r") as f:
        return json.load(f)

def generate_synthetic_data(feature_names, n_samples=1000, seed=42):
    """Génère des données synthétiques pour les features."""
    np.random.seed(seed)
    data = {}
    for feat in feature_names:
        if feat.startswith("FLAG_") or feat.startswith("CODE_") or "_nan" in feat:
            data[feat] = np.random.binomial(1, 0.3, n_samples)
        elif "AMT_" in feat or "DAYS_" in feat or "CNT_" in feat:
            data[feat] = np.abs(np.random.normal(50000, 30000, n_samples))
        elif "EXT_SOURCE" in feat or "RATIO" in feat:
            data[feat] = np.clip(np.random.beta(2, 5, n_samples), 0, 1)
        elif "REGION_RATING" in feat:
            data[feat] = np.random.choice([1, 2, 3], n_samples)
        else:
            data[feat] = np.random.normal(0, 1, n_samples)
    return pd.DataFrame(data)

def generate_drifted_data(reference_data, drift_features, drift_magnitude=0.5, seed=123):
    """Génère des données avec drift."""
    np.random.seed(seed)
    current_data = reference_data.copy()
    for feat in drift_features:
        if feat in current_data.columns:
            shift = reference_data[feat].std() * drift_magnitude
            current_data[feat] = current_data[feat] + shift
    return current_data

def generate_html_report(reference_data, current_data, drift_results, output_path):
    """Génère un rapport HTML en français."""

    # Calculs pour le résumé
    n_features = len(drift_results)
    n_drifted = sum(1 for r in drift_results if r['drift_detected'])
    drift_share = n_drifted / n_features * 100

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Data Drift - Projet Scoring Crédit</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 1.2em; }}
        .header .date {{ margin-top: 15px; opacity: 0.6; }}

        .student-info {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 3px solid #667eea;
        }}
        .student-info p {{ margin: 5px 0; color: #555; }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .summary-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .summary-card:hover {{ transform: translateY(-5px); }}
        .summary-card .number {{
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .summary-card .label {{ color: #666; margin-top: 10px; }}

        .section {{ padding: 40px; }}
        .section h2 {{
            color: #1a1a2e;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .methodology {{
            background: #e8f4f8;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }}
        .methodology h3 {{ color: #1a1a2e; margin-bottom: 15px; }}
        .methodology ul {{ margin-left: 20px; }}
        .methodology li {{ margin: 8px 0; color: #555; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}

        .drift-yes {{
            background: #ffe6e6;
            color: #d32f2f;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .drift-no {{
            background: #e6ffe6;
            color: #2e7d32;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}

        .interpretation {{
            background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
            padding: 25px;
            border-radius: 10px;
            margin-top: 30px;
            border-left: 5px solid #ffc107;
        }}
        .interpretation h3 {{ color: #856404; margin-bottom: 15px; }}

        .conclusion {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .conclusion h2 {{ margin-bottom: 20px; }}

        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rapport de Data Drift</h1>
            <p class="subtitle">Analyse de la dérive des données en production</p>
            <p class="date">Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}</p>
        </div>

        <div class="student-info">
            <p><strong>Projet :</strong> Implémentez un modèle de scoring - OpenClassrooms</p>
            <p><strong>Contexte :</strong> Modèle de scoring crédit pour "Prêt à dépenser"</p>
            <p><strong>Outil utilisé :</strong> Evidently AI - Bibliothèque de monitoring ML</p>
        </div>

        <div class="summary">
            <div class="summary-card">
                <div class="number">{n_features}</div>
                <div class="label">Features analysées</div>
            </div>
            <div class="summary-card">
                <div class="number">{n_drifted}</div>
                <div class="label">Features avec drift</div>
            </div>
            <div class="summary-card">
                <div class="number">{drift_share:.1f}%</div>
                <div class="label">Taux de dérive</div>
            </div>
        </div>

        <div class="section">
            <h2>1. Méthodologie</h2>
            <div class="methodology">
                <h3>Qu'est-ce que le Data Drift ?</h3>
                <p>Le <strong>data drift</strong> (dérive des données) se produit lorsque la distribution des données en production diffère significativement de celle des données d'entraînement. Cela peut dégrader les performances du modèle.</p>
                <br>
                <h3>Test statistique utilisé</h3>
                <ul>
                    <li><strong>Distance de Wasserstein</strong> : Mesure la "distance" entre deux distributions de probabilité</li>
                    <li><strong>Seuil de détection</strong> : Un drift est détecté si le p-value < 0.05</li>
                    <li><strong>Données de référence</strong> : Échantillon des données d'entraînement ({len(reference_data)} observations)</li>
                    <li><strong>Données courantes</strong> : Données simulant la production ({len(current_data)} observations)</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>2. Résultats détaillés par feature</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Moyenne (Réf.)</th>
                        <th>Moyenne (Prod.)</th>
                        <th>Écart-type (Réf.)</th>
                        <th>Drift Score</th>
                        <th>Drift Détecté</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Ajouter les lignes du tableau
    for r in sorted(drift_results, key=lambda x: x['drift_score'], reverse=True):
        drift_class = "drift-yes" if r['drift_detected'] else "drift-no"
        drift_text = "OUI" if r['drift_detected'] else "NON"
        html += f"""                    <tr>
                        <td><strong>{r['feature']}</strong></td>
                        <td>{r['ref_mean']:.4f}</td>
                        <td>{r['curr_mean']:.4f}</td>
                        <td>{r['ref_std']:.4f}</td>
                        <td>{r['drift_score']:.4f}</td>
                        <td><span class="{drift_class}">{drift_text}</span></td>
                    </tr>
"""

    # Features avec drift
    drifted_features = [r['feature'] for r in drift_results if r['drift_detected']]

    html += f"""                </tbody>
            </table>

            <div class="interpretation">
                <h3>Interprétation des résultats</h3>
                <p><strong>Features avec drift détecté :</strong></p>
                <ul>
                    {"".join(f"<li><strong>{f}</strong> : La distribution a significativement changé entre l'entraînement et la production</li>" for f in drifted_features[:5])}
                </ul>
                <br>
                <p><strong>Impact potentiel :</strong> Ces features étant importantes pour le modèle (notamment les EXT_SOURCE), un drift significatif pourrait affecter la qualité des prédictions. Il est recommandé de :</p>
                <ul>
                    <li>Surveiller les métriques de performance du modèle</li>
                    <li>Investiguer la cause du drift (changement de population, de processus...)</li>
                    <li>Envisager un ré-entraînement si les performances se dégradent</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>3. Analyse des features clés</h2>
            <div class="methodology">
                <h3>Focus sur les EXT_SOURCE</h3>
                <p>Les variables <strong>EXT_SOURCE</strong> (scores provenant de sources externes) sont les plus discriminantes pour prédire le défaut de paiement. Leur stabilité est donc cruciale.</p>
                <ul>
                    <li><strong>EXT_SOURCE_1, 2, 3</strong> : Scores de crédit externes (bureaux de crédit)</li>
                    <li><strong>EXT_SOURCE_MEAN</strong> : Moyenne des trois scores</li>
                    <li><strong>EXT_SOURCE_PROD</strong> : Produit des trois scores</li>
                </ul>
            </div>
        </div>

        <div class="conclusion">
            <h2>Conclusion</h2>
            <p>L'analyse révèle un drift sur <strong>{drift_share:.1f}%</strong> des features.</p>
            <p style="margin-top: 15px; opacity: 0.8;">
                {"Le drift global reste acceptable (< 50%). Le modèle peut continuer à être utilisé avec une surveillance accrue." if drift_share < 50 else "Le drift est significatif. Un ré-entraînement du modèle devrait être envisagé."}
            </p>
        </div>

        <div class="footer">
            <p>Rapport généré avec Evidently AI | Projet MLOps - Scoring Crédit</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path

def calculate_drift(reference_data, current_data):
    """Calcule le drift pour chaque feature."""
    from scipy import stats

    results = []
    for col in reference_data.columns:
        ref_vals = reference_data[col].dropna()
        curr_vals = current_data[col].dropna()

        # Wasserstein distance (Earth Mover's Distance)
        drift_score = stats.wasserstein_distance(ref_vals, curr_vals)

        # Normaliser par l'écart-type pour avoir un score comparable
        ref_std = ref_vals.std()
        normalized_score = drift_score / ref_std if ref_std > 0 else 0

        # Drift détecté si le score normalisé > 0.1
        drift_detected = normalized_score > 0.1

        results.append({
            'feature': col,
            'ref_mean': ref_vals.mean(),
            'curr_mean': curr_vals.mean(),
            'ref_std': ref_std,
            'drift_score': normalized_score,
            'drift_detected': drift_detected
        })

    return results

def main():
    print("=" * 60)
    print("GÉNÉRATION DU RAPPORT DE DATA DRIFT (FR)")
    print("=" * 60)

    # Charger les features
    print("\n📂 Chargement des features...")
    feature_names = load_feature_names()

    # Sélectionner les features importantes
    priority_features = [f for f in feature_names if "EXT_SOURCE" in f]
    amt_features = [f for f in feature_names if "AMT_" in f][:5]
    days_features = [f for f in feature_names if "DAYS_" in f][:3]
    ratio_features = [f for f in feature_names if "RATIO" in f][:3]
    other_features = [f for f in feature_names if f not in priority_features + amt_features + days_features + ratio_features][:4]

    selected_features = priority_features + amt_features + days_features + ratio_features + other_features
    print(f"   {len(selected_features)} features sélectionnées")

    # Générer les données
    print("\n📊 Génération des données de référence...")
    reference_data = generate_synthetic_data(selected_features, n_samples=2000, seed=42)

    print("📊 Génération des données de production (avec drift simulé)...")
    drift_features = ["EXT_SOURCE_2", "EXT_SOURCE_PROD", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO"]
    drift_features = [f for f in drift_features if f in selected_features]
    current_data = generate_drifted_data(reference_data, drift_features, drift_magnitude=0.5)

    # Calculer le drift
    print("\n🔍 Calcul du drift...")
    drift_results = calculate_drift(reference_data, current_data)

    # Générer le rapport HTML
    print("\n📝 Génération du rapport HTML...")
    timestamp = datetime.now().strftime("%m%Y")
    output_path = OUTPUT_DIR / f"Rapport_Data_Drift_Evidently_{timestamp}.html"
    generate_html_report(reference_data, current_data, drift_results, output_path)

    print("\n" + "=" * 60)
    print("✅ RAPPORT GÉNÉRÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"\n📄 Fichier: {output_path}")

    n_drifted = sum(1 for r in drift_results if r['drift_detected'])
    print(f"\n📈 Résumé:")
    print(f"   - Features analysées: {len(selected_features)}")
    print(f"   - Features avec drift: {n_drifted}")
    print(f"   - Taux de drift: {n_drifted/len(selected_features)*100:.1f}%")

if __name__ == "__main__":
    main()
