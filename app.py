import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')   # masque le UserWarning bénin de LightGBM
import streamlit as st
st.write("✅ App démarrée - chargement en cours...")   # ligne de test temporaire
# ---------- Chargement (mis en cache pour la rapidité) ----------
@st.cache_resource
def charger():
    bundle = joblib.load('model_pipeline.joblib')
    options = joblib.load('app_options.joblib')
    return bundle, options

bundle, options = charger()
model  = bundle['model']
lower  = bundle['lower']
upper  = bundle['upper']
cat_options  = options['cat_options']
brand_models = options['brand_models']

# ---------- En-tête ----------
st.set_page_config(page_title="Estimation prix voiture", page_icon="🚗", layout="centered")
st.title("🚗 Estimation du prix d'une voiture d'occasion")
st.write("Renseignez les caractéristiques du véhicule pour obtenir une estimation de prix.")

# ---------- Saisie utilisateur ----------
st.header("Caractéristiques du véhicule")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Marque", cat_options['brand'])
    # Modèle dépendant de la marque (Option A)
    modeles_dispo = brand_models.get(brand, ['unknown'])
    model_choisi = st.selectbox("Modèle", modeles_dispo)
    vehicleType = st.selectbox("Type de véhicule", cat_options['vehicleType'])
    gearbox = st.selectbox("Boîte de vitesses", cat_options['gearbox'])

with col2:
    fuelType = st.selectbox("Carburant", cat_options['fuelType'])
    notRepairedDamage = st.selectbox("Dégât non réparé", cat_options['notRepairedDamage'])
    powerPS = st.slider("Puissance (ch)", 1, 1000, 116)
    kilometer = st.slider("Kilométrage (km)", 5000, 150000, 100000, step=5000)
    age = st.slider("Âge du véhicule (années)", 0, 66, 13)

# ---------- Construire la ligne d'entrée (mêmes colonnes que X_train) ----------
entree = pd.DataFrame([{
    'vehicleType': vehicleType,
    'gearbox': gearbox,
    'powerPS': powerPS,
    'model': model_choisi,
    'kilometer': kilometer,
    'fuelType': fuelType,
    'brand': brand,
    'notRepairedDamage': notRepairedDamage,
    'age': age
}])

# ---------- Affichage des données saisies (demandé par le sujet) ----------
st.header("Données saisies")
st.dataframe(entree)

# ---------- Prédiction ----------
if st.button("Estimer le prix", type="primary"):
    centre = np.expm1(model.predict(entree))[0]
    bas    = np.expm1(lower.predict(entree))[0]
    haut   = np.expm1(upper.predict(entree))[0]

    st.header("Résultat")
    st.metric("Prix estimé", f"{centre:,.0f} €")
    st.write(f"**Intervalle de confiance (~80%)** : {bas:,.0f} € — {haut:,.0f} €")
    st.progress(min(1.0, centre / 50000))   # barre visuelle simple
    st.caption("Estimation basée sur un modèle LightGBM entraîné sur ~340 000 annonces.")