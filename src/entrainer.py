"""Étape 2 du pipeline : entraîner le modèle.

Lit seulement data/prepare/train.csv (fait par preparer.py)
et enregistre le modèle dans models/model.pkl.

Modèle : régression Ridge sur les LOGARITHMES.
Le log fait raisonner le modèle en pourcentages ("+8 % par rapport à l'an
dernier") au lieu de chiffres bruts. C'est utile car Marrakech (8 millions)
et Meknès (250 000) n'ont pas du tout la même taille.
"""
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Les colonnes que le modèle regarde (les "entrées")
COLONNES = ["ville", "annee", "lits", "nuitees_lag1", "nuitees_lag2"]
CIBLE = "nuitees"   # ce qu'on veut deviner


def construire_modele(p):
    """3 machines collées l'une après l'autre :
    1) préparer les entrées : log des grands nombres, ville -> colonnes 0/1
    2) une régression Ridge (une droite, avec un frein contre l'excès)
    3) TransformedTargetRegressor : la cible passe en log pour apprendre,
       puis revient en vraies nuitées (exp) au moment de prédire
    """
    entrees = ColumnTransformer([
        ("log", FunctionTransformer(np.log), ["lits", "nuitees_lag1", "nuitees_lag2"]),
        ("annee", "passthrough", ["annee"]),
        ("ville", OneHotEncoder(handle_unknown="ignore"), ["ville"]),
    ])
    pipe = Pipeline([("entrees", entrees), ("ridge", Ridge(alpha=p["alpha"]))])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


def main():
    p = yaml.safe_load(open("params.yaml", encoding="utf-8"))["entrainement"]
    train = pd.read_csv("data/prepare/train.csv")
    modele = construire_modele(p).fit(train[COLONNES], train[CIBLE])

    os.makedirs("models", exist_ok=True)
    joblib.dump(modele, "models/model.pkl")
    print("entrainement : modele ecrit dans models/model.pkl")


if __name__ == "__main__":
    main()
