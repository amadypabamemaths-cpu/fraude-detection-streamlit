import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="Détection de Fraude", page_icon="💳", layout="wide")

st.title("💳 Système de Détection de Fraude Bancaire")
st.write("Analysez une transaction ou un lot de transactions pour détecter un risque de fraude.")

# Chargement du modèle, du scaler et de la structure des colonnes
@st.cache_resource
def load_resources():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    model_columns = joblib.load("model/model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_resources()

# Interface de saisie manuelle
st.subheader("Saisie manuelle d'une transaction")

col1, col2 = st.columns(2)

with col1:
    montant = st.number_input("Montant de la transaction", min_value=0.0, value=50000.0, step=1000.0)
    heure = st.slider("Heure de la transaction (0-23)", 0, 23, 12)

with col2:
    localisation = st.selectbox("Localisation habituelle du client ?", ["Oui", "Non"])

if st.button("Analyser la transaction", type="primary"):
    # 1. Création du DataFrame de saisie
    input_df = pd.DataFrame([{
        'Montant': montant,
        'Heure': heure,
        'Localisation': localisation
    }])

    # 2. Encodage similaire à l'entraînement
    input_encoded = pd.get_dummies(input_df, columns=['Localisation'])

    # 3. Réalignement des colonnes pour correspondre exactement au modèle
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # 4. Normalisation et prédiction
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # 5. Affichage du résultat
    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **Alerte : Transaction Suspecte / Fraude détectée !** (Probabilité : {proba:.2%})")
    else:
        st.success(f"✅ **Transaction Légitime.** (Probabilité de fraude : {proba:.2%})")