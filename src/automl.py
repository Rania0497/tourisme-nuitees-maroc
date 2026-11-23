"""Étape 6 : AutoML avec FLAML (comme au TP4).

FLAML essaie tout seul plusieurs familles de modèles et réglages
pendant un temps donné, puis garde le meilleur.
On le compare ensuite à notre modèle Ridge, dans MLflow.

Avant : serveur MLflow lancé (terminal 1) et `dvc repro` fait.
"""
import json
import os
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from flaml import AutoML
from sklearn.metrics import mean_absolute_error

P = yaml.safe_load(open("params.yaml", encoding="utf-8"))["automl"]
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nuitees-automl")

train = pd.read_csv("data/prepare/train.csv")
test = pd.read_csv("data/prepare/test.csv")
VILLES = sorted(train["ville"].unique())


def entrees(d):
    """Mêmes informations que pour Ridge, mais déjà en chiffres pour FLAML :
    log des grands nombres + une colonne 0/1 par ville."""
    x = pd.DataFrame({"annee": d["annee"]})
    for c in ["lits", "nuitees_lag1", "nuitees_lag2"]:
        x[f"log_{c}"] = np.log(d[c])
    for v in VILLES:
        x[f"ville_{v}"] = (d["ville"] == v).astype(int)
    return x


# 1) La recherche automatique
automl = AutoML()
t0 = time.time()
automl.fit(
    entrees(train), np.log(train["nuitees"]),   # on apprend en log, comme Ridge
    task="regression", metric="mae",
    time_budget=P["budget_s"],                  # secondes de recherche
    estimator_list=P["familles"],               # familles de modèles à essayer
    eval_method="cv", n_splits=5,               # peu de lignes -> validation croisée
    seed=42, verbose=0,
    log_file_name="reports/flaml.log",
)
duree = round(time.time() - t0, 1)

# 2) La note sur 2019 (jamais vue pendant la recherche)
pred = np.exp(automl.predict(entrees(test)))
mae_flaml = round(mean_absolute_error(test["nuitees"], pred))
ridge = json.load(open("reports/metrics.json"))   # la note de notre Ridge (dvc repro)

os.makedirs("models", exist_ok=True)
joblib.dump(automl.model.estimator, "models/model_automl.pkl")

# 3) Deux runs dans MLflow pour comparer côte à côte
with mlflow.start_run(run_name="ridge-manuel"):
    mlflow.log_params({"outil": "manuel", "famille": "ridge"})
    mlflow.log_metrics({"mae": ridge["mae"], "mae_naif": ridge["mae_naif"]})

with mlflow.start_run(run_name=f"flaml-{P['budget_s']}s"):
    mlflow.log_params({"outil": "flaml", "budget_s": P["budget_s"],
                       "famille": automl.best_estimator})
    mlflow.log_params({f"cfg_{k}": v for k, v in automl.best_config.items()})
    mlflow.log_metrics({"mae": mae_flaml, "mae_naif": ridge["mae_naif"],
                        "duree_recherche_s": duree})
    mlflow.log_artifact("reports/flaml.log")

print(f"FLAML : famille {automl.best_estimator}, MAE {mae_flaml} ({duree} s)")
print(f"Ridge : MAE {ridge['mae']}   |   naif : MAE {ridge['mae_naif']}")
