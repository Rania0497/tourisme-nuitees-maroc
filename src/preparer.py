"""Étape 1 du pipeline : préparer les données.

Lit les 2 fichiers Excel de data.gov.ma (nuitées et lits),
les met au même format, les colle ensemble, puis ajoute
les nuitées des années d'avant (lag1, lag2).
Enfin, coupe le tableau en 2 : train (pour apprendre) et test (pour noter).

Lancer depuis le dossier du projet :  python src/preparer.py
"""
import os

import pandas as pd
import yaml


def lire_excel_large(chemin, nom_valeur):
    """Transforme un tableau 'large' (1 colonne par année) en tableau 'long'.

    Avant : ville | 2012 | 2013 | ...
    Après : ville | annee | valeur
    """
    df = pd.read_excel(chemin)
    df = df.rename(columns={"provinces": "ville"})
    df = df.melt(id_vars="ville", var_name="annee", value_name=nom_valeur)
    df["annee"] = df["annee"].astype(int)
    return df


def nettoyer_noms(df):
    """Le même lieu est écrit de 2 façons selon le fichier : on unifie."""
    df["ville"] = df["ville"].str.strip()
    df["ville"] = df["ville"].replace({"Agadir+Taghazout": "Agadir-Taghazout"})
    return df


def ajouter_lags(df):
    """Ajoute les nuitées de l'année d'avant (lag1) et d'il y a 2 ans (lag2)."""
    df = df.sort_values(["ville", "annee"]).reset_index(drop=True)
    groupe = df.groupby("ville")["nuitees"]
    df["nuitees_lag1"] = groupe.shift(1)
    df["nuitees_lag2"] = groupe.shift(2)
    # 2012 et 2013 n'ont pas assez d'historique : on les enlève
    return df.dropna(subset=["nuitees_lag1", "nuitees_lag2"])


def main():
    tout = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    params = tout["donnees"]
    p_prep = tout["preparation"]

    nuitees = nettoyer_noms(lire_excel_large(params["nuitees_brut"], "nuitees"))
    lits = nettoyer_noms(lire_excel_large(params["lits_brut"], "lits"))

    # inner = on garde seulement les villes présentes dans LES DEUX fichiers
    # (Ifrane n'a que des lits, pas de nuitées -> elle part)
    table = nuitees.merge(lits, on=["ville", "annee"], how="inner")
    table = ajouter_lags(table)

    for col in ["nuitees", "lits", "nuitees_lag1", "nuitees_lag2"]:
        table[col] = table[col].astype(int)

    os.makedirs(os.path.dirname(params["table_propre"]), exist_ok=True)
    table.to_csv(params["table_propre"], index=False)
    print(f"OK : {len(table)} lignes, {table['ville'].nunique()} villes, "
          f"années {table['annee'].min()}-{table['annee'].max()}")

    # Découpage dans le TEMPS (pas au hasard) : on apprend sur le passé,
    # on teste sur l'année suivante, comme dans la vraie vie.
    table = table[~table["annee"].isin(p_prep["annees_exclues"])]
    train = table[table["annee"] < p_prep["annee_test"]]
    test = table[table["annee"] == p_prep["annee_test"]]
    os.makedirs("data/prepare", exist_ok=True)
    train.to_csv("data/prepare/train.csv", index=False)
    test.to_csv("data/prepare/test.csv", index=False)
    print(f"preparation : {len(train)} lignes train / {len(test)} lignes test")


if __name__ == "__main__":
    main()
