# Prévision des nuitées touristiques au Maroc

Projet du cours « Déploiement d'applications intelligentes » — Rania0497.
But : prédire le nombre de nuitées par ville et par année (régression, métrique MAE).
Données : data.gov.ma (ministère du Tourisme), 2012-2020.

## Installer (une seule fois, Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancer

Terminal 1 : le serveur MLflow (le laisser ouvert)
```powershell
.venv\Scripts\Activate.ps1
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Terminal 2 : le travail
```powershell
.venv\Scripts\Activate.ps1
dvc pull                   # récupère les données depuis le stockage DVC
dvc repro                  # preparer -> explorer -> entrainer -> evaluer (seulement ce qui a changé)
dvc metrics show           # affiche la MAE
python src/journaliser.py  # relie le run MLflow à l'empreinte des données
```
`pipeline.py` (TP2) reste pour enregistrer un modèle validé dans le registre MLflow.
Puis ouvrir http://127.0.0.1:5000

## Résultat (année test 2019)
| Méthode | MAE (nuitées) |
|---|---|
| Naïve : « cette année = l'an dernier » | 84 381 |
| Ridge sur logarithmes | 60 955 |

## Dossiers
- `data/raw/` : les fichiers Excel d'origine
- `data/processed/tourisme.csv` : le tableau propre
- `src/` : preparer, explorer, entrainer, evaluer
- `dvc.yaml` / `dvc.lock` : le pipeline DVC et sa « photo »
- `data/raw.dvc` : le ticket DVC des données brutes
- `pipeline.py` : version TP2 (MLflow + registre)
- `params.yaml` : les réglages

## API (étape 8)
```powershell
uvicorn api.main:app --reload          # puis http://127.0.0.1:8000/docs
docker build -t nuitees-api:1.0 .
docker run -d --name nuitees -p 8000:8000 nuitees-api:1.0
```

## Interface (étape 8c)
```powershell
streamlit run app/streamlit_app.py     # l'API doit tourner sur le port 8000
```

## Tout lancer avec Docker Compose
```powershell
docker compose up --build      # API sur :8000, interface sur http://localhost:8501
docker compose down            # tout arrêter
```
