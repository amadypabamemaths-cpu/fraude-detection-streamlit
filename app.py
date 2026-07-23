import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Détection de Fraude", page_icon="💳", layout="wide")

st.title("💳 Système de Détection de Fraude Bancaire")
st.write("Analysez une transaction ou un lot de transactions pour détecter un risque de fraude.")

# Chargement dynamique des catégories depuis le dataset
@st.cache_data
def load_options():
    df = pd.read_csv("data/transactions.csv", sep=";", on_bad_lines="skip")
    villes = sorted(df['Localisation'].dropna().unique().tolist())
    types_trans = sorted(df['Type de transaction'].dropna().unique().tolist())
    statuts_op = sorted(df['Status operation'].dropna().unique().tolist())
    return villes, types_trans, statuts_op

@st.cache_resource
def load_resources():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    model_columns = joblib.load("model/model_columns.pkl")
    return model, scaler, model_columns

villes_disponibles, types_disponibles, statuts_disponibles = load_options()
model, scaler, model_columns = load_resources()

st.subheader("Saisie manuelle d'une transaction")

col1, col2 = st.columns(2)

with col1:
    montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
    heure = st.slider("Heure de la transaction (0-23)", 0, 23, 12)
    type_trans = st.selectbox("Type de transaction", types_disponibles)

with col2:
    localisation = st.selectbox("Ville / Localisation", villes_disponibles)
    statut_op = st.selectbox("Statut de l'opération", statuts_disponibles)

if st.button("Analyser la transaction", type="primary"):
    input_df = pd.DataFrame([{
        'Montant': montant,
        'Heure': heure,
        'Localisation': localisation,
        'Type de transaction': type_trans,
        'Status operation': statut_op
    }])

    # Encodage One-Hot des colonnes
    input_encoded = pd.get_dummies(input_df, columns=['Localisation', 'Type de transaction', 'Status operation'])

    # Alignement avec la structure exacte attendue par le modèle
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Normalisation & prédiction
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **Alerte : Transaction Suspecte / Fraude détectée !** (Probabilité : {proba:.2%})")
    else:
        st.success(f"✅ **Transaction Légitime.** (Probabilité de fraude : {proba:.2%})")