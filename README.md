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
python src/preparer.py     # tableau propre + découpage train/test
python src/explorer.py     # graphiques dans reports/figures/
python pipeline.py         # preparer -> entrainer -> evaluer, suivi dans MLflow
```
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
- `pipeline.py` : enchaîne les étapes et enregistre tout dans MLflow
- `params.yaml` : les réglages
