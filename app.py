import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="Détection de Fraude", page_icon="💳", layout="wide")

st.title("💳 Système de Détection de Fraude Bancaire")
st.write("Analysez une transaction ou un lot de transactions pour détecter un risque de fraude.")

# Chargement dynamique des options
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

# Fonction de prétraitement commune
def preprocess_and_predict(data_df):
    # Sécurisation si la colonne Date existe
    if 'Date' in data_df.columns:
        data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
        data_df['Heure'] = data_df['Date'].dt.hour.fillna(0).astype(int)
    
    # Encodage One-Hot
    encoded_df = pd.get_dummies(data_df, columns=['Localisation', 'Type de transaction', 'Status operation'])
    
    # Alignement des colonnes avec celles du modèle
    for col in model_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[model_columns]
    
    # Prédiction
    scaled_data = scaler.transform(encoded_df)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]
    
    return predictions, probabilities

# Organisation en deux onglets
tab1, tab2 = st.tabs(["📝 Saisie manuelle", "📁 Analyse par fichier CSV"])

# --- ONGLET 1 : SAISIE MANUELLE ---
with tab1:
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
        
        pred, proba = preprocess_and_predict(input_df)
        
        st.markdown("---")
        if pred[0] == 1:
            st.error(f"🚨 **Alerte : Transaction Suspecte / Fraude détectée !** (Probabilité : {proba[0]:.2%})")
        else:
            st.success(f"✅ **Transaction Légitime.** (Probabilité de fraude : {proba[0]:.2%})")

# --- ONGLET 2 : FICHIER CSV ---
# --- ONGLET 2 : FICHIER CSV ---
with tab2:
    st.subheader("Analyse d'un fichier de transactions (CSV)")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier chargé
            file_df = pd.read_csv(uploaded_file, sep=";", on_bad_lines="skip")
            st.write("📋 **Aperçu des données chargées :**")
            st.dataframe(file_df.head())
            
            if st.button("Lancer l'analyse du fichier", type="primary"):
                preds, probas = preprocess_and_predict(file_df)
                
                # Ajout des résultats dans le DataFrame
                results_df = file_df.copy()
                results_df['Prédiction_Fraude'] = preds
                results_df['Probabilité_Fraude'] = [f"{p:.2%}" for p in probas]
                
                st.markdown("---")
                st.write("📊 **Résultats de la détection :**")
                st.dataframe(results_df)
                
                fraudes_detectees = sum(preds)
                st.warning(f"🔍 **Nombre total de transactions suspectes détectées :** {fraudes_detectees} sur {len(file_df)}")
                
                # =========================================================
                # 🔽 AJOUTEZ LE BOUTON DE TÉLÉCHARGEMENT À PARTIR D'ICI 🔽
                # =========================================================
                st.markdown("---")
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Télécharger le rapport d'analyse (CSV)",
                    data=csv_export,
                    file_name="rapport_detection_fraude.csv",
                    mime="text/csv"
                )
                # =========================================================

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")