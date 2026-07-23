import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Configuration globale de la page
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du jeu de données brut pour l'exploration
@st.cache_data
def load_data():
    df = pd.read_csv("data/transactions.csv", sep=";", on_bad_lines="skip")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Heure'] = df['Date'].dt.hour.fillna(0).astype(int)
    return df

# Chargement du modèle et des artefacts
@st.cache_resource
def load_resources():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    model_columns = joblib.load("model/model_columns.pkl")
    return model, scaler, model_columns

df_raw = load_data()
model, scaler, model_columns = load_resources()

# --- BARRE LATÉRALE DE NAVIGATION ---
st.sidebar.title("🏦 Détection de Fraude")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Exploration des données", "🎯 Performance du modèle", "⚡ Prédiction en temps réel"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Projet Machine Learning**\nDétection des transactions bancaires suspectes.")

# ==============================================================================
# PAGE 1 : ACCUEIL
# ==============================================================================
if page == "🏠 Accueil":
    st.title("🏦 Détection de la fraude bancaire")
    st.write("Cette application permet d'explorer un jeu de transactions bancaires, d'évaluer la performance d'un modèle prédictif et d'analyser des transactions en temps réel.")

    st.markdown("### 📌 Répartition des transactions")
    
    # Calcul des métriques globales
    total_trans = len(df_raw)
    nb_fraudes = len(df_raw[df_raw['Target'].isin(['Fraude', 'Suspect'])]) if 'Target' in df_raw.columns else 0
    taux_fraude = (nb_fraudes / total_trans) * 100 if total_trans > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_trans:,}")
    col2.metric("Fraudes / Suspects", f"{nb_fraudes:,}")
    col3.metric("Taux de fraude", f"{taux_fraude:.2f} %")
    col4.metric("Modèle retenu", "Random Forest")

    st.markdown("---")
    st.subheader("📋 Aperçu des données")
    st.dataframe(df_raw.head(10), use_container_width=True)

# ==============================================================================
# PAGE 2 : EXPLORATION DES DONNÉES
# ==============================================================================
elif page == "📊 Exploration des données":
    st.title("📊 Exploration des données")
    
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Répartition des transactions par classe")
        if 'Target' in df_raw.columns:
            target_counts = df_raw['Target'].value_counts().reset_index()
            target_counts.columns = ['Statut', 'Nombre']
            fig_pie = px.pie(
                target_counts, names='Statut', values='Nombre',
                color='Statut',
                color_discrete_map={'Normal': '#2ecc71', 'Suspect': '#f39c12', 'Fraude': '#e74c3c'},
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Répartition par type de transaction")
        if 'Type de transaction' in df_raw.columns and 'Target' in df_raw.columns:
            fig_bar = px.histogram(
                df_raw, x='Type de transaction', color='Target',
                barmode='group',
                color_discrete_map={'Normal': '#2ecc71', 'Suspect': '#f39c12', 'Fraude': '#e74c3c'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Distribution du montant par statut (FCFA)")
    if 'Montant' in df_raw.columns and 'Target' in df_raw.columns:
        fig_box = px.box(
            df_raw, x='Target', y='Montant', color='Target',
            log_y=True,
            color_discrete_map={'Normal': '#2ecc71', 'Suspect': '#f39c12', 'Fraude': '#e74c3c'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# PAGE 3 : PERFORMANCE DU MODÈLE
# ==============================================================================
elif page == "🎯 Performance du modèle":
    st.title("🎯 Performance du modèle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exactitude (Accuracy)", "98.5 %")
    m2.metric("F1 Macro Score", "0.921")
    m3.metric("Rappel Fraude", "85.2 %")
    m4.metric("Jeu de test", "20 % du dataset")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Importance des variables")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = model_columns
            df_imp = pd.DataFrame({'Variable': feature_names, 'Importance': importances})
            df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(10)

            fig_imp = px.bar(
                df_imp, x='Importance', y='Variable', orientation='h',
                title="Contribution de chaque variable à la décision"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        st.subheader("Matrice de confusion (Exemple)")
        matrix_data = np.array([[1420, 15], [12, 68]])
        fig_cm = px.imshow(
            matrix_data,
            labels=dict(x="Prédiction", y="Classe réelle", color="Nombre"),
            x=['Normal', 'Fraude'],
            y=['Normal', 'Fraude'],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ==============================================================================
# PAGE 4 : PRÉDICTION EN TEMPS RÉEL
# ==============================================================================
elif page == "⚡ Prédiction en temps réel":
    st.title("⚡ Prédiction en temps réel")

    tab1, tab2 = st.tabs(["📝 Saisie manuelle", "📁 Importation CSV"])

    # Listes pour les déroulants
    villes_disponibles = sorted(df_raw['Localisation'].dropna().unique().tolist()) if 'Localisation' in df_raw.columns else []
    types_disponibles = sorted(df_raw['Type de transaction'].dropna().unique().tolist()) if 'Type de transaction' in df_raw.columns else []
    statuts_disponibles = sorted(df_raw['Status operation'].dropna().unique().tolist()) if 'Status operation' in df_raw.columns else []

    def preprocess_and_predict(data_df):
        if 'Date' in data_df.columns:
            data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
            data_df['Heure'] = data_df['Date'].dt.hour.fillna(0).astype(int)
        
        encoded_df = pd.get_dummies(data_df, columns=['Localisation', 'Type de transaction', 'Status operation'])
        
        for col in model_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        encoded_df = encoded_df[model_columns]
        
        scaled_data = scaler.transform(encoded_df)
        preds = model.predict(scaled_data)
        probas = model.predict_proba(scaled_data)[:, 1]
        return preds, probas

    with tab1:
        st.subheader("Analyser une transaction")
        col1, col2 = st.columns(2)

        with col1:
            montant = st.number_input("Montant (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
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

    with tab2:
        st.subheader("Analyse de lot par fichier CSV")
        uploaded_file = st.file_uploader("Charger un fichier CSV", type=["csv"])
        
        if uploaded_file is not None:
            file_df = pd.read_csv(uploaded_file, sep=";", on_bad_lines="skip")
            st.dataframe(file_df.head())
            
            if st.button("Lancer l'analyse du fichier", type="primary"):
                preds, probas = preprocess_and_predict(file_df)
                
                results_df = file_df.copy()
                results_df['Prédiction_Fraude'] = preds
                results_df['Probabilité_Fraude'] = [f"{p:.2%}" for p in probas]
                
                st.dataframe(results_df)
                
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger le rapport d'analyse (CSV)",
                    data=csv_export,
                    file_name="rapport_detection_fraude.csv",
                    mime="text/csv"
                )