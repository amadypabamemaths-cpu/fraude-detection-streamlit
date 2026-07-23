import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

os.makedirs("model", exist_ok=True)

# 1. Chargement des données
df = pd.read_csv("data/transactions.csv", sep=";", on_bad_lines="skip")

# Prétraitement de la cible
df['Class'] = df['Target'].apply(lambda x: 1 if x in ['Fraude', 'Suspect'] else 0)

# Extraction de l'heure depuis Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Heure'] = df['Date'].dt.hour.fillna(0).astype(int)

# Sélection des 5 colonnes pour l'entraînement
features_cols = ['Montant', 'Heure', 'Localisation', 'Type de transaction', 'Status operation']
df_model = df[features_cols].copy()
df_model['Class'] = df['Class']

# Encodage One-Hot des variables catégorielles
categorical_cols = ['Localisation', 'Type de transaction', 'Status operation']
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# 2. Séparation features / cible
X = df_encoded.drop("Class", axis=1)
y = df_encoded["Class"]

# Sauvegarde des colonnes
joblib.dump(list(X.columns), "model/model_columns.pkl")

# 3. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entraînement
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# 6. Sauvegarde
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Modèle entraîné et sauvegardé avec 5 caractéristiques !")