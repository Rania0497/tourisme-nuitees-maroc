"""Étape 8a : l'API qui prédit les nuitées (comme au TP8).

Lancer depuis le dossier du projet :  uvicorn api.main:app --reload
Puis ouvrir http://127.0.0.1:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

VERSION = "1.0"
modele = joblib.load("models/model.pkl")   # chargé UNE fois, au démarrage

app = FastAPI(title="API nuitées touristiques Maroc")


class Demande(BaseModel):
    """Les 5 informations que l'utilisateur envoie."""
    ville: str = Field(examples=["Marrakech"])
    annee: int = Field(examples=[2019])
    lits: int = Field(gt=0, examples=[73048])
    nuitees_lag1: int = Field(gt=0, examples=[7873933])   # nuitées de l'année d'avant
    nuitees_lag2: int = Field(gt=0, examples=[7111152])   # nuitées d'il y a 2 ans


@app.get("/health")
def health():
    """Le service est-il prêt ?"""
    return {"statut": "ok", "version": VERSION}


@app.post("/predict")
def predict(d: Demande):
    """Renvoie le nombre de nuitées prédit."""
    X = pd.DataFrame([d.model_dump()])
    nuitees = float(modele.predict(X)[0])
    return {"ville": d.ville, "annee": d.annee,
            "nuitees_predites": round(nuitees), "version": VERSION}
