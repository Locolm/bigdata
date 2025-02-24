# Projet Big data - Titanic

*BONADA Nathan - LAURENT Clément - LE TRUNG Ethan - RANDRIANTSOA Matthieu*

Ce projet a pour objectif d’entraîner un modèle de Machine Learning sur le dataset Titanic afin de prédire si un passager a survécu ou non en fonction de ses caractéristiques.

## Prétraitement des Données

### Explication des Données

Le script preprocess.py réalise :
- Chargement des données depuis un fichier CSV
- Suppression des colonnes inutiles (Name, Ticket, Cabin)
- Remplacement des valeurs manquantes (Age, Embarked)
- Encodage des variables catégorielles (Sex, Embarked)
- Normalisation des variables numériques (Age, Fare)
- Sauvegarde du dataset nettoyé sous data/ref_data.csv

Le fichier ref_data.csv contient ces données après prétraitement:

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

### Génération de passagers aléatoires que l'on devra prédire

generate_data.py réalise les actions suivantes :
- Analyse la distribution des colonnes (Pclass, Sex, Age, etc.).
- Génère 1 000 passagers fictifs.
- Normalise  les colonnes Age et Fare comme pour ref_data.csv.

à la fin on obtient test_data.csv qui sera le fichier utiliser pour tester notre modèle.

## Entraînement du Modèle


## Partie API

La librairie pipreqs a été utilisée ici afin de générer le fichier requirements.txt et ne lister que les librairies nécessaires à ce projet via la commande :

```bash
$ pipreqs --encoding=utf8
```

Pour lancer l'API manuellement, il suffit de lancer la commande :

```bash
$ fastapi dev api.py
```

Pour build l'image Docker de l'api et la run:

```bash
$ Docker build -t api_titanic .
$ Docker run -p 8080:8080 api_titanic
```

Le -t permet de donner à l'image le nom api_titanic.
Le "." sert à indiquer que le Dockerfile utilisé pour construire cette image se trouve dans le répertoire actuel.

Une erreur peut survenir durant le build de l'image car l'import de certaines librairies prend énormément de temps (xgboost notamment).
A voir si on les laisse.
