import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Création du dossier pour sauvegarder le modèle s'il n'existe pas
os.makedirs("model", exist_ok=True)

# 1. Chargement des données
# (Adapté au chemin et séparateur de votre notebook)
df = pd.read_csv("data/transactions.csv", sep=";", on_bad_lines="skip")
#df = pd.read_csv("data/transactions.csv")

# --- Prétraitement des données ---
# Création de la variable cible binaire : 1 si fraude ou suspect, 0 si normal
df['Class'] = df['Target'].apply(lambda x: 1 if x in ['Fraude', 'Suspect'] else 0)

# Suppression des colonnes non pertinentes pour l'apprentissage
df_model = df.drop(columns=['ID Clients', 'Numero de compte', 'Identifiant operation', 'Date', 'Target'])

# Encodage des variables catégorielles (One-Hot Encoding)
df_encoded = pd.get_dummies(df_model, columns=['Type de transaction', 'Status operation', 'Localisation'], drop_first=True)
# ---------------------------------

# 2. Séparation features / cible
X = df_encoded.drop("Class", axis=1)
y = df_encoded["Class"]

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

# 6. Évaluation
y_pred = model.predict(X_test)
print("--- Rapport de Classification ---")
print(classification_report(y_test, y_pred))

# 7. Sauvegarde du modèle ET du scaler
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Modèle et outils de prétraitement sauvegardés avec succès dans le dossier 'model/'.")