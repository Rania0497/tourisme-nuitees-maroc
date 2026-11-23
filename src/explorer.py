"""Étape d'exploration : regarder les données avec des graphiques.

Lancer depuis le dossier du projet :  python src/explorer.py
Les images sont enregistrées dans reports/figures/
"""
import os

import matplotlib
matplotlib.use("Agg")                 # pas de fenêtre : on enregistre des images
import matplotlib.pyplot as plt
import pandas as pd
import yaml

DOSSIER = "reports/figures"


def courbes_par_ville(df):
    """Graphique 1 : l'évolution des nuitées de chaque ville dans le temps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ville, d in df.groupby("ville"):
        ax.plot(d["annee"], d["nuitees"] / 1e6, marker="o", label=ville)
    ax.set_title("Nuitées par ville et par année")
    ax.set_xlabel("Année")
    ax.set_ylabel("Nuitées (millions)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(f"{DOSSIER}/1_nuitees_par_ville.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def lits_contre_nuitees(df):
    """Graphique 2 : plus de lits = plus de nuitées ? (sans 2020)"""
    d = df[df["annee"] != 2020]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["lits"], d["nuitees"] / 1e6)
    ax.set_title("Lits contre nuitées (2014-2019)")
    ax.set_xlabel("Nombre de lits")
    ax.set_ylabel("Nuitées (millions)")
    fig.savefig(f"{DOSSIER}/2_lits_vs_nuitees.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def lag1_contre_nuitees(df):
    """Graphique 3 : l'année d'avant prédit-elle bien l'année d'après ?"""
    d = df[df["annee"] != 2020]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["nuitees_lag1"] / 1e6, d["nuitees"] / 1e6)
    maxi = d["nuitees"].max() / 1e6
    ax.plot([0, maxi], [0, maxi], "--", color="grey", label="cette année = l'an dernier")
    ax.set_title("Nuitées de l'an dernier contre cette année")
    ax.set_xlabel("Nuitées l'année d'avant (millions)")
    ax.set_ylabel("Nuitées cette année (millions)")
    ax.legend()
    fig.savefig(f"{DOSSIER}/3_lag1_vs_nuitees.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["donnees"]
    df = pd.read_csv(params["table_propre"])
    os.makedirs(DOSSIER, exist_ok=True)

    courbes_par_ville(df)
    lits_contre_nuitees(df)
    lag1_contre_nuitees(df)

    # Quelques chiffres à lire dans le terminal
    sans_covid = df[df["annee"] != 2020]
    print("Résumé des nuitées (2014-2019) :")
    print(sans_covid["nuitees"].describe().round(0).to_string())
    print("\nLien avec les nuitées (corrélation, 1 = lien parfait) :")
    print(sans_covid[["lits", "nuitees_lag1", "nuitees_lag2", "nuitees"]]
          .corr()["nuitees"].round(3).to_string())
    print(f"\n3 images enregistrées dans {DOSSIER}/")


if __name__ == "__main__":
    main()
