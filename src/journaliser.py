"""Relier MLflow à la version EXACTE des données (comme au TP3).

Lit l'empreinte (md5) des données dans le ticket DVC data/raw.dvc
et l'enregistre dans MLflow avec le commit Git et les métriques.

Avant : serveur MLflow lancé (terminal 1) et `dvc repro` fait.
"""
import json
import subprocess

import mlflow
import yaml

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nuitees-dvc")

P = yaml.safe_load(open("params.yaml", encoding="utf-8"))
ticket = yaml.safe_load(open("data/raw.dvc", encoding="utf-8"))["outs"][0]
mets = json.load(open("reports/metrics.json"))
commit = subprocess.getoutput("git rev-parse --short HEAD")
empreinte = ticket["md5"].replace(".dir", "")   # data/raw est un dossier

with mlflow.start_run(run_name=f"donnees-{empreinte[:8]}"):
    mlflow.log_params(P["entrainement"])
    mlflow.log_metrics(mets)
    mlflow.set_tags({
        "commit": commit,              # quel code
        "donnees_md5": ticket["md5"],  # quelles données, exactement
        "donnees_fichiers": ticket.get("nfiles", ""),
    })
    mlflow.log_artifact("dvc.lock")    # la photo complète du pipeline

print("run MLflow cree : donnees", empreinte[:8], "- MAE", mets["mae"])
