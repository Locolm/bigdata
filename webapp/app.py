import streamlit as st
import requests
import io
import pandas as pd

# URL de l'API de serving
API_URL = "http://127.0.0.1:8080/predict-file"

# Titre de l'application
st.title("Titanic Survival Prediction")

# Chargement du fichier
uploaded_file = st.file_uploader("Upload a text-based file (CSV) 📂", type=["csv"])

# Bouton pour effectuer la prédiction
if uploaded_file is not None:
    # Lire le fichier en DataFrame pandas
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    # elif uploaded_file.name.endswith(".xls") or uploaded_file.name.endswith(".xlsx"):
    #     df = pd.read_excel(uploaded_file)
    # elif uploaded_file.name.endswith(".txt"):
    #     df = pd.read_csv(uploaded_file, delimiter=";")

    # Afficher le DataFrame chargé
    st.subheader("Data Preview:")
    st.dataframe(df)
    if st.button("Did the passenger survive?"):
        # Lire le fichier sous forme de bytes
        files = {"file": uploaded_file.getvalue()}

        # Requête POST vers l'API de serving
        response = requests.post(API_URL, files=files, headers={
        })

        # Afficher le résultat
        if response.status_code == 200:
            prediction = response.json()
            prediction_df = pd.DataFrame(prediction)
            st.dataframe(prediction_df)
        else:
            st.error("Erreur lors de la requête à l'API")