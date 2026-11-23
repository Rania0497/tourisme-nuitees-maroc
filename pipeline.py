"""Chef d'orchestre (comme pipeline.py du TP2).

Lance preparer -> entrainer -> evaluer, enregistre tout dans MLflow,
et n'enregistre le modèle que s'il bat la méthode naïve.

Avant : lancer le serveur MLflow dans un autre terminal (voir README).
"""
import json
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

sys.path.insert(0, "src")
from entrainer import COLONNES  # noqa: E402

P = yaml.safe_load(open("params.yaml", encoding="utf-8"))
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nuitees-pipeline")


def lancer(etape):
    # sys.executable = le python du .venv (celui qui lance ce script)
    subprocess.run([sys.executable, f"src/{etape}.py"], check=True)


refuse = False
with mlflow.start_run(run_name="pipeline"):                        # exécution PARENTE
    with mlflow.start_run(run_name="preparer", nested=True):        # enfant 1
        lancer("preparer")
        mlflow.log_params(P["preparation"])
    with mlflow.start_run(run_name="entrainer", nested=True):       # enfant 2
        lancer("entrainer")
        mlflow.log_params(P["entrainement"])
    with mlflow.start_run(run_name="evaluer", nested=True):         # enfant 3
        lancer("evaluer")
        mets = json.load(open("reports/metrics.json"))
        mlflow.log_metrics(mets)
        mlflow.log_artifact("reports/predictions.png")

    mlflow.log_metrics(mets)                       # visible aussi au niveau parent

    # RÈGLE DE VALIDATION : faire mieux que "cette année = l'an dernier"
    if mets["mae"] < mets["mae_naif"]:
        exemple = pd.read_csv("data/prepare/test.csv")[COLONNES].head(3)
        mlflow.sklearn.log_model(
            joblib.load("models/model.pkl"),
            name="model",
            input_example=exemple,
            registered_model_name="nuitees-maroc",
        )
        mlflow.set_tag("statut", "accepte")
        print(f"modele accepte (MAE {mets['mae']} < naif {mets['mae_naif']}) et enregistre")
    else:
        mlflow.set_tag("statut", "refuse")
        refuse = True

if refuse:
    raise SystemExit(f"modele refuse : MAE {mets['mae']} >= naif {mets['mae_naif']}")
