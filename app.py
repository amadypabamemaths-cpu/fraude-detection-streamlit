import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide"
)

# --- Chargement du modèle et du scaler (mis en cache) ---
@st.cache_resource
def load_model():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --- En-tête ---
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown("Analysez une transaction ou un lot de transactions pour détecter un risque de fraude.")

# --- Menu latéral ---
mode = st.sidebar.radio("Mode d'analyse", ["Transaction unique", "Fichier CSV (lot)"])

# ============================
# MODE 1 : Transaction unique
# ============================
if mode == "Transaction unique":
    st.subheader("Saisie manuelle d'une transaction")
    col1, col2 = st.columns(2)
    with col1:
        montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0)
        heure = st.slider("Heure de la transaction (0-23)", 0, 23, 12)
    with col2:
        localisation_habituelle = st.selectbox(
            "Localisation habituelle du client ?", ["Oui", "Non"]
        )
        canal = st.selectbox("Canal", ["Mobile Money", "Carte", "Virement", "ATM"])
        
    if st.button("Analyser la transaction", type="primary"):
        # Construction du vecteur de features (à adapter selon les features de votre modèle)
        features = np.array([[montant, heure, 1 if localisation_habituelle == "Non" else 0]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]
        
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Transaction suspecte** — Probabilité de fraude : {proba:.1%}")
        else:
            st.success(f"✅ **Transaction légitime** — Probabilité de fraude : {proba:.1%}")
            
        st.progress(float(proba))

# ============================
# MODE 2 : Fichier CSV (lot)
# ============================
else:
    st.subheader("Analyse par lot (fichier CSV)")
    fichier = st.file_uploader("Déposez un fichier CSV de transactions", type=["csv"])
    
    if fichier is not None:
        df = pd.read_csv(fichier)
        st.write("Aperçu des données :", df.head())
        
        if st.button("Lancer l'analyse du lot"):
            # Note : Assurez-vous que le CSV déposé respecte les mêmes colonnes que lors de l'entraînement
            X_scaled = scaler.transform(df)
            df["prediction"] = model.predict(X_scaled)
            df["probabilite_fraude"] = model.predict_proba(X_scaled)[:, 1]
            
            nb_fraudes = df["prediction"].sum()
            st.warning(f"**{nb_fraudes}** transaction(s) suspecte(s) détectée(s) sur {len(df)}.")
            
            st.dataframe(
                df.style.apply(
                    lambda row: ["background-color: #ffcccc" if row["prediction"] == 1 else "" for _ in row],
                    axis=1
                )
            )
            
            csv_export = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les résultats", csv_export, "resultats_analyse.csv", "text/csv")

# --- Pied de page ---
st.sidebar.markdown("---")
st.sidebar.caption("Projet pédagogique — Détection de fraude bancaire par IA")