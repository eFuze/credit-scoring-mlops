# Guide de Deploiement

Ce guide explique comment deployer l'API et le dashboard sur differentes plateformes cloud gratuites.

## Option 1: Render (Recommande)

### Deploiement automatique

1. Connectez-vous sur [Render](https://render.com)
2. Cliquez sur "New" > "Blueprint"
3. Connectez votre repo GitHub
4. Render detectera automatiquement le fichier `render.yaml`
5. Cliquez sur "Apply"

### Deploiement manuel

1. Connectez-vous sur [Render](https://render.com)
2. Cliquez sur "New" > "Web Service"
3. Connectez votre repo GitHub
4. Configurez:
   - **Name**: credit-scoring-api
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Cliquez sur "Create Web Service"

URL de l'API: `https://credit-scoring-api.onrender.com`

## Option 2: Railway

1. Connectez-vous sur [Railway](https://railway.app)
2. Cliquez sur "New Project" > "Deploy from GitHub repo"
3. Selectionnez votre repo
4. Railway detectera automatiquement le Procfile
5. Ajoutez les variables d'environnement si necessaire

## Option 3: Heroku

```bash
# Installer Heroku CLI
heroku login
heroku create credit-scoring-api
git push heroku main
heroku open
```

## Option 4: Google Cloud Run

```bash
# Build et push l'image Docker
gcloud builds submit --tag gcr.io/PROJECT_ID/credit-scoring-api

# Deployer
gcloud run deploy credit-scoring-api \
  --image gcr.io/PROJECT_ID/credit-scoring-api \
  --platform managed \
  --allow-unauthenticated
```

## Verification du deploiement

Une fois deploye, testez l'API:

```bash
# Health check
curl https://YOUR-APP-URL/health

# Prediction
curl -X POST https://YOUR-APP-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6}}'
```

## Variables d'environnement

| Variable | Description | Defaut |
|----------|-------------|--------|
| PORT | Port de l'application | 8000 |
| PYTHON_VERSION | Version Python | 3.10 |

## Notes importantes

1. **Modele**: Le fichier `models/model.joblib` doit etre present dans le repo
2. **Donnees**: Ne pas commiter les fichiers de donnees (`.csv`, `.parquet`)
3. **Free tier**: Les services gratuits peuvent etre mis en veille apres inactivite
