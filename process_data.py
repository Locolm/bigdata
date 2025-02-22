import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger les données
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Sélectionner les colonnes utiles
columns_to_keep = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[columns_to_keep]

#  Gérer les valeurs manquantes
df["Age"].fillna(df["Age"].median(), inplace=True)  # Remplace NaN par la médiane
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # Remplace NaN par la valeur la plus fréquente

# Encodage des variables catégorielles
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # 0 pour femme, 1 pour homme
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])  # Convertir en valeurs numériques

# Normalisation des variables numériques
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# Sauvegarde du dataset préparé
df.to_csv("data/ref_data.csv", index=False)

print("Prétraitement terminé, fichier 'ref_data.csv' créé !")
