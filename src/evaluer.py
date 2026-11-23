"""Étape 3 du pipeline : noter le modèle sur l'année test.

Compare le modèle à une méthode "naïve" (cette année = l'an dernier).
Écrit reports/metrics.json et reports/predictions.png.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from entrainer import CIBLE, COLONNES


def main():
    test = pd.read_csv("data/prepare/test.csv")
    modele = joblib.load("models/model.pkl")

    vrai = test[CIBLE]
    pred = modele.predict(test[COLONNES])
    naif = test["nuitees_lag1"]   # méthode naïve : on recopie l'année d'avant

    mets = {
        "mae": round(mean_absolute_error(vrai, pred)),          # erreur moyenne en nuitées
        "mape": round(mean_absolute_percentage_error(vrai, pred), 4),  # erreur en %
        "mae_naif": round(mean_absolute_error(vrai, naif)),
    }

    os.makedirs("reports", exist_ok=True)
    json.dump(mets, open("reports/metrics.json", "w"), indent=2)

    # Image : vraies valeurs contre prédictions, ville par ville
    d = test.assign(pred=pred).sort_values(CIBLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d[CIBLE] / 1e6, width=0.4, label="vrai")
    ax.bar([i + 0.2 for i in x], d["pred"] / 1e6, width=0.4, label="prédit")
    ax.set_xticks(list(x))
    ax.set_xticklabels(d["ville"], rotation=45, ha="right")
    ax.set_ylabel("Nuitées (millions)")
    ax.set_title(f"Année test {int(d['annee'].iloc[0])} : vrai contre prédit")
    ax.legend()
    fig.savefig("reports/predictions.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("evaluation :", mets)


if __name__ == "__main__":
    main()
