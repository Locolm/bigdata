import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import pickle

# Choisi une métrique de scoring personnalisée

# def custom_scoring(y_true, y_pred):
#     """
#     Calculer la métrique : (accuracy + precision classe 1) / 2
#     """
#     report = classification_report(y_true, y_pred, output_dict=True)
#     return (report['accuracy'] + report['1.0']['precision']) / 2

def train_test_split_transform(X, y, normalize=True, use_pca=False, n_components_pca=3, test_size=0.5):
    """
    Décompose les données en données d'entrainement et données de test et applique les transformations voulues.
    """

    # Étape 1 : Séparation des données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # Étape 2 : Normalisation et/ou ACP (avant sélection des variables)
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if use_pca:
        pca = PCA(n_components=n_components_pca)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Ajouter les composantes principales aux données originales
        X_train = np.hstack((X_train, X_train_pca))
        X_test = np.hstack((X_test, X_test_pca))

    return X_train, X_test, y_train, y_test


def transform_all(X, y, normalize=True, use_pca=False, n_components_pca=3):
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if use_pca:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=n_components_pca)
        X_pca = pca.fit_transform(X)

        # Ajouter les composantes principales aux données originales
        X = np.hstack((X, X_pca))

    return X

def run_classifiers(X, y, clfs, verbose=False, scorer=None):
    """
    Retourne le meilleur modèle avec la meilleure stratégie

    Compare tous les modèles fournis avec les trois stratégies (default, normalize, normalize + pca)
    """

    stretegies = [
        "default", "normalize", "normalize + pca"
    ]

    best_model = None
    best_score = -np.inf
    best_strategy = None

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for clf_name, clf in clfs.items():
        for strategy in stretegies:
            if verbose:
                print(f"\nTest du modèle : {clf_name} avec la stratégie : {strategy}")

            if strategy == "default":
                X = transform_all(X, y, normalize=False, use_pca=False)
            elif strategy == "normalize":
                X = transform_all(X, y, normalize=True, use_pca=False)
            elif strategy == "normalize + pca":
                X = transform_all(X, y, normalize=True, use_pca=True)

            cv_acc = cross_val_score(clf, X, y, cv=kf, scoring=scorer, n_jobs=-1)
            score = np.mean(cv_acc)

            if verbose:
                print(f"Score for {clf_name} is: {score:.3f} +/- {np.std(cv_acc):.3f}")

            if score > best_score:
                best_score = score
                best_model = clf
                best_strategy = strategy

    return best_model, best_strategy

def feature_selection(X_train, X_test, y_train, y_test, best_clf, features=None, scoring=None, verbose=False):
    """
    Applique la sélection de variable à partir du meilleur modèle et retourne le nombre de variables à sélectionner.
    """

    selector = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    std = np.std([tree.feature_importances_ for tree in selector.estimators_], axis=0)

    sorted_idx = np.argsort(importances)[::-1]
    scores = np.zeros(X_train.shape[1] + 1)

    for f in np.arange(0, X_train.shape[1] + 1):
        X1_f = X_train[:, sorted_idx[:f + 1]]
        X2_f = X_test[:, sorted_idx[:f + 1]]
        best_clf.fit(X1_f, y_train)
        YMLP = best_clf.predict(X2_f)
        score = scoring(y_test, YMLP)
        scores[f] = np.round(score, 3)

    n = scores.argmax() + 1

    selected_features = sorted_idx[:n]

    if verbose:
        padding = np.arange(X_train.size / len(X_train)) + 0.5
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[0].barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center')
        ax[0].set_yticks(padding)
        ax[0].set_yticklabels(features[sorted_idx])
        ax[0].set_xlabel("Relative Importance")
        ax[0].set_title("Variable Importance")

        ax[1].plot(scores)
        ax[1].set_xlabel("Nombre de Variables")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_title("Evolution de l'accuracy en fonction des variables")

        plt.tight_layout()
        plt.show()
        print("Features selected: ", n)
        print(f"Sélection de {len(selected_features)} variables pertinentes.")

    return n, selected_features

def create_pipeline(df, test_size=0.5, clfs=None, param_grid=None, verbose=False, scoring=None, output="pipeline.pkl"):
    """
    Pipeline général automatisé pour l'entraînement et l'évaluation des modèles.

    Paramètres :
    - df : DataFrame contenant les données
    - test_size : Taille du jeu de test
    - clfs : Dictionnaire contenant les modèles à tester
    - param_grids : Dictionnaire contenant les hyperparamètres à tester pour chaque modèle
    - verbose : Afficher les informations
    """


    # Étape 1 : Séparation des données
    data = df.values

    X = data[:, :-1]
    y = data[:, -1]

    # Étape 2 : Déterminer le meilleur modèle et la meilleure stratégie
    scorer = make_scorer(scoring, greater_is_better=True)
    best_model, best_strategy = run_classifiers(X, y, clfs, verbose=verbose, scorer=scorer)

    if verbose:
        print(f"Meilleur modèle : {best_model}")
        print(f"Meilleure stratégie : {best_strategy}")

    # Étape 3 : Feature selection

    # Transform taining and testing data from the best strategy
    X_train, X_test, y_train, y_test, features_name = None, None, None, None, df.columns
    if best_strategy == "default":
        X = transform_all(X, y , normalize=False, use_pca=False)
        X_train, X_test, y_train, y_test = train_test_split_transform(X, y, normalize=False, use_pca=False, test_size=test_size)
    elif best_strategy == "normalize":
        X = transform_all(X, y, normalize=True, use_pca=False)
        X_train, X_test, y_train, y_test = train_test_split_transform(X, y, normalize=True, use_pca=False, test_size=test_size)
    elif best_strategy == "normalize + pca":
        X = transform_all(X, y, normalize=True, use_pca=True)
        X_train, X_test, y_train, y_test = train_test_split_transform(X, y, normalize=True, use_pca=True, test_size=test_size)
        features_name = np.hstack((features_name, [f"PCA_{i}" for i in range(X_train.shape[1] - len(features_name))])) # Ajoute le noms des variables PCA

    n, selected_features = feature_selection(X_train, X_test, y_train, y_test, best_model, features=features_name, scoring=scoring, verbose=verbose)

    X = X[:, selected_features]

    # Étape 4 : Entraînement du modèle final avec GridSearchCV
    if verbose:
        print(f"\nEntraînement du modèle final avec GridSearchCV pour {best_model} | cv=10")

    grid_search = GridSearchCV(best_model, param_grid[type(best_model).__name__], cv=10, scoring=scorer, n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    if verbose:
        print(f"Meilleur modèle : {best_model}")
        print(f"Meilleurs hyperparamètres : {best_params}")
        print(f"Meilleur score : {best_score:.3f}")

    # Étape 5 : Création de la pipeline finale

    rf = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)

    P = None
    if best_strategy == "normalize":
        P = Pipeline([('scaler', StandardScaler()),
                      ("fs", SelectFromModel(estimator=rf, max_features=n)),
                      ('clf', best_model)])
    elif best_strategy == "normalize + pca":
        P = Pipeline([('scaler', StandardScaler()),
                      ('fu', FeatureUnion([("scaler", StandardScaler()),("pca", PCA(n_components=0.95))])),
                      ("fs", SelectFromModel(estimator=rf, max_features=n)),
                      ('clf', best_model)])
    else:
        P=Pipeline([("fs", SelectFromModel(estimator=rf, max_features=n)),
                      ('clf', best_model)])

    # Étape 6 : Entraînement du modèle final
    data = df.values

    X = data[:, :-1]
    y = data[:, -1]

    P.fit(X, y)

    with open(output, "wb") as f:
        pickle.dump(P, f)

    print(f"Pipeline saved as '{output}'")

    return P


