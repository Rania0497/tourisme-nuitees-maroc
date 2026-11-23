# Prévision des nuitées touristiques au Maroc

Projet du cours « Déploiement d'applications intelligentes ».
But : prédire le nombre de nuitées par ville et par année (régression, métrique MAE).
Données : data.gov.ma (ministère du Tourisme), 2012-2020.

## Lancer le projet (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/preparer.py
```

## Dossiers
- `data/raw/` : les fichiers Excel d'origine
- `data/processed/tourisme.csv` : le tableau propre (créé par `preparer.py`)
- `src/` : le code du pipeline
- `params.yaml` : les réglages
