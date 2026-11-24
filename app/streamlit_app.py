"""Interface web (Streamlit) qui appelle l'API de prédiction des nuitées.

Avant : l'API tourne (dans Docker ou avec uvicorn) sur le port 8000.
Lancer depuis le dossier du projet :  streamlit run app/streamlit_app.py

NOUVEAU (design) :
- une grande bannière aux couleurs du Maroc
- les résultats dans des cartes colorées
- un graphique : historique de la ville + prédiction
- photo de bannière facultative : mettre une image dans app/banniere.jpg
"""
import base64
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

# CHANGÉ : l'adresse de l'API peut venir de docker-compose (variable URL_API),
# sinon localhost par défaut
URL_API = os.environ.get("URL_API", "http://localhost:8000/predict")
DERNIERE_ANNEE = 2020   # dernière année connue dans les données
DEPART_FUTUR = 2019     # le futur part de 2019 (2020 = COVID, année anormale)

# NOUVEAU : la palette de couleurs (tourisme au Maroc)
TERRACOTTA = "#C4532D"
OCRE = "#D9A441"
MAJORELLE = "#3047A0"
VERT = "#2E7D5B"


def appeler_api(ville, annee, lits, lag1, lag2):
    """Envoie une demande à l'API et renvoie le nombre de nuitées prédit."""
    demande = {"ville": ville, "annee": int(annee), "lits": int(lits),
               "nuitees_lag1": int(lag1), "nuitees_lag2": int(lag2)}
    rep = requests.post(URL_API, json=demande, timeout=10)
    rep.raise_for_status()
    return rep.json()["nuitees_predites"]


def joli(n):
    """6168237 -> '6 168 237'."""
    return f"{int(n):,}".replace(",", " ")


def fond_banniere():
    """NOUVEAU : la photo app/banniere.jpg si elle existe, sinon un dégradé
    terracotta -> ocre avec un motif géométrique (style zellige)."""
    photo = Path(__file__).parent / "banniere.jpg"
    voile = "linear-gradient(rgba(30,20,15,.45), rgba(30,20,15,.55))"
    if photo.exists():
        code = base64.b64encode(photo.read_bytes()).decode()
        return f"{voile}, url(data:image/jpeg;base64,{code}) center/cover"
    motif = ("repeating-linear-gradient(45deg, rgba(255,255,255,.06) 0 2px, transparent 2px 28px),"
             "repeating-linear-gradient(-45deg, rgba(255,255,255,.06) 0 2px, transparent 2px 28px)")
    return f"{motif}, linear-gradient(135deg, {TERRACOTTA} 0%, #A8432A 45%, {OCRE} 100%)"


def carte(titre, valeur, couleur, note=""):
    """NOUVEAU : une carte de résultat (HTML) avec une bande de couleur."""
    return f"""
    <div class="carte" style="border-top: 6px solid {couleur}">
      <div class="carte-titre">{titre}</div>
      <div class="carte-valeur" style="color:{couleur}">{valeur}</div>
      <div class="carte-note">{note}</div>
    </div>"""


def graphique(histo, preds):
    """NOUVEAU : courbe de l'historique (plein) et de la prédiction (pointillés)."""
    h = pd.DataFrame({"Année": list(histo.keys()), "Nuitées": list(histo.values()),
                      "Série": "Historique"})
    p = pd.DataFrame({"Année": list(preds.keys()), "Nuitées": list(preds.values()),
                      "Série": "Prédiction"})
    df = pd.concat([h, p])
    couleurs = alt.Scale(domain=["Historique", "Prédiction"], range=[MAJORELLE, TERRACOTTA])
    base = alt.Chart(df).encode(
        x=alt.X("Année:O", axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("Nuitées:Q", axis=alt.Axis(format="~s", title=None)),
        color=alt.Color("Série:N", scale=couleurs, legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Série:N", legend=None,
                                  scale=alt.Scale(domain=["Historique", "Prédiction"],
                                                  range=[[1, 0], [6, 4]])),
        tooltip=["Année", alt.Tooltip("Nuitées:Q", format=",")],
    )
    return (base.mark_line(strokeWidth=3) + base.mark_circle(size=70)).properties(height=320)


# ------------------------------------------------------------------
# NOUVEAU : le style (CSS) de toute la page
# ------------------------------------------------------------------
st.set_page_config(page_title="Nuitées Maroc", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem; max-width: 1150px; }}

.hero {{
  background: {fond_banniere()};
  border-radius: 22px; padding: 70px 40px 50px; text-align: center; color: white;
  box-shadow: 0 18px 40px rgba(120,50,20,.25);
}}
.hero .sur-titre {{ letter-spacing: .35em; text-transform: uppercase; font-size: .85rem; opacity: .9; }}
.hero h1 {{
  font-family: 'Playfair Display', serif; font-size: 4.2rem; line-height: 1.05;
  margin: .3rem 0 .6rem; color: white;
}}
.hero p {{ font-size: 1.1rem; opacity: .95; margin-bottom: 1.8rem; }}
/* CHANGÉ : avertissement COVID bien visible */
.alerte {{
  background: #FFF4E0; border-left: 6px solid {OCRE}; border-radius: 12px;
  padding: 14px 20px; margin-top: 18px; color: #7A4B10; font-weight: 500;
}}

.section-titre {{
  font-family: 'Playfair Display', serif; font-size: 1.8rem; color: #2B2420;
  margin: 2rem 0 .8rem; border-left: 6px solid {OCRE}; padding-left: 14px;
}}
.carte {{
  background: white; border-radius: 16px; padding: 22px 24px; height: 100%;
  box-shadow: 0 6px 20px rgba(0,0,0,.06);
}}
.carte-titre {{ text-transform: uppercase; letter-spacing: .1em; font-size: .75rem; color: #7A6A5E; }}
.carte-valeur {{ font-family: 'Playfair Display', serif; font-size: 2.3rem; font-weight: 800; margin-top: 6px; }}
.carte-note {{ font-size: .85rem; color: #7A6A5E; margin-top: 4px; }}

div[data-testid="stForm"] {{
  background: white; border: none; border-radius: 18px; padding: 24px 28px;
  box-shadow: 0 6px 20px rgba(0,0,0,.06); margin-top: 8px;
}}
div[data-testid="stFormSubmitButton"] button {{
  background: {TERRACOTTA}; color: white; border: none; border-radius: 30px;
  padding: 10px 36px; font-weight: 600; letter-spacing: .05em; width: 100%;
}}
div[data-testid="stFormSubmitButton"] button:hover {{ background: #A8432A; color: white; }}
/* NOUVEAU : les onglets et le texte de "À propos" */
button[data-baseweb="tab"] p {{ font-size: 1.05rem; font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: {TERRACOTTA}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {TERRACOTTA}; }}
.texte {{
  background: white; border-radius: 16px; padding: 20px 24px; line-height: 1.8;
  box-shadow: 0 6px 20px rgba(0,0,0,.06); color: #3D342E;
}}
.pied {{ text-align: center; color: #9A8A7E; font-size: .8rem; margin: 3rem 0 1rem; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Données
# ------------------------------------------------------------------
table = pd.read_csv("data/processed/tourisme.csv")
villes = sorted(table["ville"].unique())

# ------------------------------------------------------------------
# NOUVEAU : la bannière
# ------------------------------------------------------------------
# CHANGÉ : plus de boutons de villes (la ville se choisit dans "Destination")
st.markdown(f"""
<div class="hero">
  <div class="sur-titre">Observatoire du tourisme</div>
  <h1>Maroc</h1>
  <p>Prévision des nuitées dans les établissements d'hébergement touristique</p>
</div>""", unsafe_allow_html=True)

# CHANGÉ : une seule page, sans onglets
onglet_prev = st.container()   # CHANGÉ : plus d'onglets, l'onglet "À propos" est retiré

with onglet_prev:
    # ------------------------------------------------------------------
    # Le formulaire : ville + année choisies dans des listes
    # ------------------------------------------------------------------
    with st.form("demande"):
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        ville = c1.selectbox("Destination", villes, index=villes.index("Marrakech"))
        # CHANGÉ : l'année se choisit dans une liste (2014 à 2030), 2025 par défaut
        annees = list(range(2014, 2031))
        annee = c2.selectbox("Année", annees, index=annees.index(2025))
        lancer = c3.form_submit_button("Prédire")

    d = table[table["ville"] == ville].set_index("annee")
    # historique complet 2012-2020 (2012 et 2013 sont dans les colonnes lag)
    histo = {2012: d.loc[2014, "nuitees_lag2"], 2013: d.loc[2014, "nuitees_lag1"]}
    histo.update(d["nuitees"].to_dict())

    if lancer:
        try:
            # ----- Vérifier le passé -----
            if annee <= DERNIERE_ANNEE:
                ligne = d.loc[annee]
                pred = appeler_api(ville, annee, ligne["lits"],
                                   ligne["nuitees_lag1"], ligne["nuitees_lag2"])
                vrai = int(ligne["nuitees"])
                ecart = abs(pred - vrai) / vrai

                st.markdown(f'<div class="section-titre">{ville}, {annee}</div>',
                            unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                k1.markdown(carte("Nuitées prédites", joli(pred), TERRACOTTA, "par le modèle"),
                            unsafe_allow_html=True)
                k2.markdown(carte("Nuitées réelles", joli(vrai), MAJORELLE, "source : data.gov.ma"),
                            unsafe_allow_html=True)
                k3.markdown(carte("Écart", f"{ecart:.1%}", VERT if ecart < .1 else OCRE,
                                  "erreur relative"), unsafe_allow_html=True)
                # CHANGÉ : avertissement COVID pour 2020
                if annee == 2020:
                    st.markdown('<div class="alerte">Attention : en 2020, les nuitées ont chuté '
                                'à cause du COVID. Le modèle ne pouvait pas prévoir cette chute, '
                                'd\'où le grand écart.</div>', unsafe_allow_html=True)
                entrees = (ligne["lits"], ligne["nuitees_lag1"], ligne["nuitees_lag2"],
                           annee - 1, annee - 2, "")
                preds = {annee: pred}

            # ----- Prédire le futur (en chaîne) -----
            else:
                lits = d.loc[DERNIERE_ANNEE, "lits"]
                lag1 = d.loc[DEPART_FUTUR, "nuitees"]
                lag2 = d.loc[DEPART_FUTUR - 1, "nuitees"]
                entrees = (lits, lag1, lag2, DEPART_FUTUR, DEPART_FUTUR - 1,
                           f" ({DERNIERE_ANNEE})")
                preds = {DEPART_FUTUR: lag1}     # la courbe part du dernier point réel
                for a in range(DEPART_FUTUR + 1, annee + 1):
                    pred = appeler_api(ville, a, lits, lag1, lag2)
                    preds[a] = pred
                    lag2, lag1 = lag1, pred
                hausse = pred / d.loc[DEPART_FUTUR, "nuitees"] - 1

                st.markdown(f'<div class="section-titre">{ville}, {annee}</div>',
                            unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                # CHANGÉ : les 3 petites phrases sous les cartes sont retirées
                k1.markdown(carte(f"Nuitées prédites en {annee}", joli(pred), TERRACOTTA),
                            unsafe_allow_html=True)
                k2.markdown(carte(f"Évolution depuis {DEPART_FUTUR}", f"{hausse:+.0%}", VERT),
                            unsafe_allow_html=True)
                # CHANGÉ : « Horizon » remplacé par « Fiabilité »
                ecart_ans = annee - DEPART_FUTUR      # nombre d'années après 2019
                fiab = "Élevée" if ecart_ans <= 2 else "Moyenne" if ecart_ans <= 5 else "Faible"
                k3.markdown(carte("Fiabilité", fiab, OCRE), unsafe_allow_html=True)

            # ----- Données utilisées (après le clic) -----
            lits_u, l1, l2, a1, a2, info_lits = entrees
            st.markdown('<div class="section-titre">Données utilisées</div>',
                        unsafe_allow_html=True)
            u1, u2, u3 = st.columns(3)
            u1.markdown(carte(f"Lits{info_lits}", joli(lits_u), OCRE), unsafe_allow_html=True)
            u2.markdown(carte(f"Nuitées {a1}", joli(l1), MAJORELLE), unsafe_allow_html=True)
            u3.markdown(carte(f"Nuitées {a2}", joli(l2), MAJORELLE), unsafe_allow_html=True)

            # ----- Graphique -----
            st.markdown('<div class="section-titre">Évolution des nuitées</div>',
                        unsafe_allow_html=True)
            st.altair_chart(graphique(histo, preds), width="stretch")

        except requests.RequestException:
            st.error("L'API ne répond pas. Vérifiez qu'elle tourne sur le port 8000.")

# CHANGÉ : pied de page raccourci
st.markdown('<div class="pied">Données : Ministère du Tourisme, data.gov.ma (2012-2020)</div>',
            unsafe_allow_html=True)
