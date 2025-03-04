# Projet Big data - Titanic

Double diplôme IA
*BONADA Nathan - LAURENT Clément - LE TRUNG Ethan - RANDRIANTSOA Matthieu*

Ce projet a pour objectif d’entraîner un modèle de Machine Learning sur le dataset Titanic afin de prédire si un passager a survécu ou non en fonction de ses caractéristiques. L'objectif est de mettre en production ce modèle en développant une API de serving et en l'intégrant dans une pipeline complete, le tout conteneurisé avec Docker.

## Structure du projet

```
├── data/
│   └── ref_data.csv
│   └── test_data.csv
│   └── test.csv
├── artifcats/
│   └── pipeline.pkl
├── scripts/
│   └── pipeline.py
|   └── EDA.ipynb
|   └── notebook.ipynb
├── serving/
│   └── api.py
└── webapp/
    └── app.py
```

**data/** : Contient le fichier *ref_data.csv* après prétraitement. ainsi que le fichier *test.csv* qui sont les données sur lesquelles on peut tester notre modèle.

**scripts/** : Contient les scripts Python pour le prétraitement et l'entraînement du modèle.

**artifacts/** : Contient le modèle entraîné, le scaler et la pipeline.

**serving/** : Contient l'API de serving développée avec FastAPI.

**serving/** : Contient l'interface applicative Streamlit.

## Prétraitement des Données

Le fichier *EDA.ipynb* détaille les données du dataset Titanic afin de mieux comprendre le liens entre les caractéristiques et les personnes ayant survécu. Il explique en détail la réflexion et le processus réalisé.

Le script *process_data.py* réalise :
- Chargement des données depuis un fichier CSV
- Suppression des colonnes inutiles (Name, Ticket, Cabin)
- Imputation des valeurs manquantes (Age, Embarked)
- Encodage des variables catégorielles (Sex, Embarked)
- Normalisation des variables numériques (Age, Fare)
- Sauvegarde du dataset nettoyé sous data/ref_data.csv

Format du fichier *ref_data.csv* après prétraitement:

|   colonne    |   Description    |
|---    |--:    |
|    Survived   |    0 = N’a pas survécu, 1 = A survécu    |
|    Pclass   |    Classe du passager (1 = Première, 2 = Deuxième, 3 = Troisième)   |
|   Sex    |    0 = Femme, 1 = Homme   |
|   Age    |    Âge du passager (normalisé avec StandardScaler)   |
|   SibSp    |   Nombre de frères/sœurs/conjoints à bord    |
|   Parch    |    Nombre de parents/enfants à bord   |
|   Fare    |   Prix du billet (normalisé avec StandardScaler)    |
|   Embarked    |   Port d’embarquement (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)    |

### Génération de passagers fictifs

Le script *generate_data.py* permet de simuler des données supplémentaires pour tester le modèle.
- Analyse la distribution des colonnes (Pclass, Sex, Age, etc.).
- Génère 1 000 passagers fictifs.
- Normalise  les colonnes Age et Fare comme pour *ref_data.csv*.
- Sauvegarde des passagers générés dans *test_data.csv*.

à la fin on obtient test_data.csv qui sera le fichier utilisé pour tester notre modèle.

## Entraînement du Modèle
Le choix et l'entraînement du modèle sont expliqués et effectués dans *notebook.ipynb*.

Le modèle est ensuite enregistré sous forme de pipeline dans **artifacts/**.

Les différents composants de la pipeline sont les suivants :
- drop_columns_transformer : Suppression des colonnes inutiles / Garder uniquement les colonnes utiles
- encoder : Encodage des variables catégorielles (Sex & Embarked)
- scaler : Normalisation des variables numériques (Age & Fare)
- train_model : Random Forest Classifier entrainé sur le dataset Titanic
- pipeline : Pipeline regroupant les différents composants
- fixed_pipeline : Pipeline sans le composant drop_columns_transformer (fonctionne mieux pour l'API)

<img alt="pipeline.png" src="misc/pipeline.png" width="600"/>

## Partie API

Une API de serving a été développée avec FastAPI pour permettre d'interroger le modèle.

Le code se situe dans **serving/**.

[POST]
http://127.0.0.1:8080/predict/

<img alt="pipeline.png" src="misc/postman.png" width="600"/>

Pour lancer l'API manuellement, il suffit de lancer la commande suivante depuis le repertoire ./serving :

```bash
$ uvicorn api:app --host 127.0.0.1 --port 8080
```

## Interface

Une interface est mise à disposition, pour l'utiliser manuellement il suffit de se rendre dans **webapp/** et d'utiliser la commande suivante :

```bash
$ streamlit run app.py
```

voici à quoi devrait ressembler l'interface après avoir lancé la prédiction : 

<img alt="pipeline.png" src="misc/webapp.png" width="600"/>

## Dockerisation

Les images docker peuvent être construites et lancées séparément via les commandes suivante : 

```bash
$ docker compose -f ./serving/docker-compose.yml
```

```bash
$ docker compose -f ./webapp/docker-compose.yml
```

## Déploiement automatique

Un fichier de configuration global permet de lancer toutes les briques de l'application. Pour cela, depuis la racine du dossier, entrer la commande suivante : 

```bash
docker compose up --build
```
