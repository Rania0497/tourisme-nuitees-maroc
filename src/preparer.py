"""Étape 1 du pipeline : préparer les données.

Lit les 2 fichiers Excel de data.gov.ma (nuitées et lits),
les met au même format, les colle ensemble, puis ajoute
les nuitées des années d'avant (lag1, lag2).

Lancer depuis le dossier du projet :  python src/preparer.py
"""
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
    params = yaml.safe_load(open("params.yaml"))["donnees"]

    nuitees = nettoyer_noms(lire_excel_large(params["nuitees_brut"], "nuitees"))
    lits = nettoyer_noms(lire_excel_large(params["lits_brut"], "lits"))

    # inner = on garde seulement les villes présentes dans LES DEUX fichiers
    # (Ifrane n'a que des lits, pas de nuitées -> elle part)
    table = nuitees.merge(lits, on=["ville", "annee"], how="inner")
    table = ajouter_lags(table)

    # Repère COVID : 2020 est une année anormale
    table["covid"] = (table["annee"] == 2020).astype(int)

    for col in ["nuitees", "lits", "nuitees_lag1", "nuitees_lag2"]:
        table[col] = table[col].astype(int)

    table.to_csv(params["table_propre"], index=False)
    print(f"OK : {len(table)} lignes, {table['ville'].nunique()} villes, "
          f"années {table['annee'].min()}-{table['annee'].max()}")
    print(table.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
