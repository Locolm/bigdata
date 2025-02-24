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




