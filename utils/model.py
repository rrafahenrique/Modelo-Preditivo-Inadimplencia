import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, StratifiedKFold, RandomizedSearchCV

from imblearn.pipeline import Pipeline

def train_and_compare_models(X_train, X_test, y_train, y_test, random_state = 42):
    """
    Treina e avalia múltiplos modelos de classificação binária,
    retornando as métricas de avaliação.
    """

    # Modelos
    models: Dict[str, object] = {
        "Regressão Logística": LogisticRegression(random_state=random_state),
        "Floresta Randômica": RandomForestClassifier(random_state=random_state),
        "K-Vizinhos Mais Próximos (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=random_state),
        "LightGBM": LGBMClassifier(random_state=random_state, verbosity=-1)
    }

    results = []

    # Loop de treino e avaliação
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Modelo": name,
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        })

    # DataFrame final
    results_df = pd.DataFrame(results)

    return results_df.round(3)

#--------------------------------------------------------------------

def train_cv_repeat( X, y, n_splits = 5, n_repeats = 3, random_state = 42):
    """
    Treina e avalia múltiplos modelos de classificação binária
    usando RepeatedStratifiedKFold Cross-Validation.

    Métricas:
    - ROC-AUC
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    """

    models: Dict[str, object] = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Floresta Randômica": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "K-Vizinhos Mais Próximos (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=random_state),
        "LightGBM": LGBMClassifier(n_estimators=300,learning_rate=0.05,random_state=random_state,verbosity=-1)
    }

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    results = []

    for name, model in models.items():
        metrics = {
            "roc_auc": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))
            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_test, y_pred))
            metrics["f1"].append(f1_score(y_test, y_pred))

        results.append({
            "Modelo": name,
            "ROC-AUC": np.mean(metrics["roc_auc"]),
            "Accuracy": np.mean(metrics["accuracy"]),
            "Precision": np.mean(metrics["precision"]),
            "Recall": np.mean(metrics["recall"]),
            "F1-Score": np.mean(metrics["f1"])
        })

    return pd.DataFrame(results).round(3)

#---------------------------------------------------------------------------------------------
# Função OVERSAMPLING
def validacao_cruzada(X, y, oversampling = False, undersampling = False, n_splits = 5, random_state = 42):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models: Dict[str, object] = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Floresta Randômica": RandomForestClassifier(n_estimators=300,random_state=random_state),
        "K-Vizinhos Mais Próximos (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=random_state),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state, verbosity=-1)
    }

    results = []         #Guardar as métricas finais 
    trained_models = {}  #Guardar os modelos finais

    for name, model in models.items():
        metrics = {
            "roc_auc": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        # iterando sobre os splits
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # -------- SAMPLING APENAS NO TREINO --------
            if oversampling:
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            if undersampling:
                tomek = TomekLinks()
                X_train, y_train = tomek.fit_resample(X_train, y_train)
            # -------------------------------------------

            # Com os dados balanceados só no treino, hora de treino o modelo
            model.fit(X_train, y_train)

            # Validação SEM oversampling
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
            metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_val, y_pred))
            metrics["f1"].append(f1_score(y_val, y_pred))

        #Refit final com todos os dados
        X_final, y_final = X.copy(), y.copy()

        if oversampling:
            X_final, y_final = SMOTE(random_state=random_state).fit_resample(X_final, y_final)

        if undersampling:
            X_final, y_final = TomekLinks().fit_resample(X_final, y_final)

        model.fit(X_final, y_final)
        trained_models[name] = model  #salva o modelo treinado

        results.append({
            "Modelo": name,
            "ROC-AUC": np.mean(metrics["roc_auc"]),
            "Accuracy": np.mean(metrics["accuracy"]),
            "Precision": np.mean(metrics["precision"]),
            "Recall": np.mean(metrics["recall"]),
            "F1-Score": np.mean(metrics["f1"])
        })

    return pd.DataFrame(results).round(3), trained_models
#--------------------------------------------------------------------------------


