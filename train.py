import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

#  Chargement des données
df = pd.read_csv("data/ref_data.csv")

# Séparation des features (X) et de la target (y)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Séparer en train (80%) et validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition des modèles à tester
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Grilles d’hyperparamètres pour chaque modèle
param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10]
    },
    "Random Forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20]
    },
    "XGBoost": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.2]
    }
}

# Test des modèles avec GridSearchCV
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"Entraînement de {name}...")

    # Pipeline avec normalisation et modèle
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    # GridSearch pour trouver les meilleurs hyperparamètres
    grid = GridSearchCV(pipeline, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # Évaluer sur les données de validation
    y_pred = grid.best_estimator_.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"{name} - Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
    print(classification_report(y_val, y_pred))

    # Garder le meilleur modèle
    if acc > best_score:
        best_score = acc
        best_model = grid.best_estimator_
        best_name = name

# Sauvegarde du meilleur modèle
print(f"\n🏆 Meilleur modèle : {best_name} avec une accuracy de {best_score:.4f}")
joblib.dump(best_model, "artifacts/best_model.pkl")
print("Modèle enregistré dans 'artifacts/best_model.pkl' !")
