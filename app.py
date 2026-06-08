import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------- Chargement (mis en cache) ----------
@st.cache_resource
def charger():
    bundle = joblib.load('model_pipeline.joblib')
    options = joblib.load('app_options.joblib')
    return bundle, options

bundle, options = charger()
model = bundle['model']
lower = bundle['lower']
upper = bundle['upper']
cat_options = options['cat_options']
brand_models = options['brand_models']

# ---------- Traductions (valeur allemande -> libellé français) ----------
TRAD = {
    'vehicleType': {
        'andere': 'Autre', 'bus': 'Monospace', 'cabrio': 'Cabriolet',
        'coupe': 'Coupé', 'kleinwagen': 'Citadine', 'kombi': 'Break',
        'limousine': 'Berline', 'suv': 'SUV'
    },
    'gearbox': {'automatik': 'Automatique', 'manuell': 'Manuelle'},
    'fuelType': {
        'andere': 'Autre', 'benzin': 'Essence', 'cng': 'GNV',
        'diesel': 'Diesel', 'elektro': 'Électrique',
        'hybrid': 'Hybride', 'lpg': 'GPL'
    },
    'notRepairedDamage': {'ja': 'Oui', 'nein': 'Non'}
}

def selectbox_traduit(label, col):
    valeurs_all = cat_options[col]
    mapping = TRAD[col]
    libelles_fr = [mapping[v] for v in valeurs_all]
    choix_fr = st.selectbox(label, libelles_fr)
    inverse = {fr: all_ for all_, fr in mapping.items()}
    return inverse[choix_fr]

# ---------- En-tête ----------
st.set_page_config(page_title="Estimation prix voiture", page_icon="🚗", layout="centered")
st.title("🚗 Estimation du prix d'une voiture d'occasion")
st.write("Renseignez les caractéristiques du véhicule pour obtenir une estimation de prix.")
st.caption("Modèle entraîné sur des annonces du marché allemand (2016). "
           "Estimation indicative : certaines combinaisons rares peuvent être moins fiables.")

# ---------- Saisie utilisateur ----------
st.header("Caractéristiques du véhicule")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Marque", cat_options['brand'])
    modeles_dispo = brand_models.get(brand, ['unknown'])
    model_choisi = st.selectbox("Modèle", modeles_dispo)
    vehicleType = selectbox_traduit("Type de véhicule", 'vehicleType')
    gearbox = selectbox_traduit("Boîte de vitesses", 'gearbox')

with col2:
    fuelType = selectbox_traduit("Carburant", 'fuelType')
    notRepairedDamage = selectbox_traduit("Dégât non réparé", 'notRepairedDamage')
    powerPS = st.slider("Puissance (ch)", 1, 600, 116)
    kilometer = st.slider("Kilométrage (km)", 5000, 150000, 100000, step=5000)
    age = st.slider("Âge du véhicule (années)", 0, 50, 13)

# ---------- Construction de la ligne d'entrée ----------
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

# ---------- Affichage des données saisies (en français) ----------
st.header("Données saisies")
affichage = pd.DataFrame([{
    'Marque': brand,
    'Modèle': model_choisi,
    'Type': TRAD['vehicleType'][vehicleType],
    'Boîte': TRAD['gearbox'][gearbox],
    'Carburant': TRAD['fuelType'][fuelType],
    'Dégât non réparé': TRAD['notRepairedDamage'][notRepairedDamage],
    'Puissance (ch)': powerPS,
    'Kilométrage': kilometer,
    'Âge': age
}])
st.dataframe(affichage, width='stretch')
# ---------- Prédiction ----------
if st.button("Estimer le prix", type="primary"):
    centre = np.expm1(model.predict(entree))[0]
    bas = np.expm1(lower.predict(entree))[0]
    haut = np.expm1(upper.predict(entree))[0]

    st.header("Résultat")
    st.metric("Prix estimé", f"{centre:,.0f} €")
    st.write(f"**Intervalle de confiance (~80%)** : {bas:,.0f} € — {haut:,.0f} €")
    st.progress(min(1.0, centre / 50000))
    st.caption("Estimation basée sur un modèle LightGBM entraîné sur ~340 000 annonces.")