import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger les données de référence
df_ref = pd.read_csv("data/ref_data.csv")

# Déterminer les distributions des données réelles
pclass_dist = df_ref["Pclass"].value_counts(normalize=True).to_dict()
sex_dist = df_ref["Sex"].value_counts(normalize=True).to_dict()
embarked_dist = df_ref["Embarked"].value_counts(normalize=True).to_dict()

# Moyenne et écart-type pour normaliser Age et Fare
age_mean, age_std = df_ref["Age"].mean(), df_ref["Age"].std()
fare_mean, fare_std = df_ref["Fare"].mean(), df_ref["Fare"].std()

# Générer un nouveau dataset de 1000 passagers fictifs
np.random.seed(42)
num_samples = 1000

data_fake = {
    "Survived": np.random.randint(0, 2, size=num_samples),
    "Pclass": np.random.choice(list(pclass_dist.keys()), size=num_samples, p=list(pclass_dist.values())),
    "Sex": np.random.choice(list(sex_dist.keys()), size=num_samples, p=list(sex_dist.values())),
    "Age": np.random.normal(age_mean, age_std, size=num_samples),  # Généré avec une distribution normale
    "SibSp": np.random.randint(0, 5, size=num_samples),  # 0 à 4 frères/sœurs/conjoints
    "Parch": np.random.randint(0, 5, size=num_samples),  # 0 à 4 parents/enfants
    "Fare": np.random.normal(fare_mean, fare_std, size=num_samples),  # Tarif normalisé
    "Embarked": np.random.choice(list(embarked_dist.keys()), size=num_samples, p=list(embarked_dist.values())),
}

df_fake = pd.DataFrame(data_fake)

# Normaliser Age et Fare comme dans ref_data.csv
scaler = StandardScaler()
df_fake[["Age", "Fare"]] = scaler.fit_transform(df_fake[["Age", "Fare"]])

# Sauvegarder les nouvelles données
df_fake.to_csv("data/test_data.csv", index=False)

print("✅ Fichier 'test_data.csv' généré avec 1000 passagers fictifs !")
