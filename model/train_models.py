
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a classification model
    """
  
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics, model


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 classification models and return results
    """
    results = []
    
    # 1. Logistic Regression
    lr_metrics, lr_model = evaluate_model(
        LogisticRegression(random_state=42, max_iter=1000),
        X_train, X_test, y_train, y_test,
        'Logistic Regression'
    )
    results.append(lr_metrics)
    
    # 2. Decision Tree
    dt_metrics, dt_model = evaluate_model(
        DecisionTreeClassifier(random_state=42),
        X_train, X_test, y_train, y_test,
        'Decision Tree'
    )
    results.append(dt_metrics)
    
    # 3. K-Nearest Neighbor
    knn_metrics, knn_model = evaluate_model(
        KNeighborsClassifier(),
        X_train, X_test, y_train, y_test,
        'K-Nearest Neighbor'
    )
    results.append(knn_metrics)
    
    # 4. Naive Bayes
    nb_metrics, nb_model = evaluate_model(
        GaussianNB(),
        X_train, X_test, y_train, y_test,
        'Naive Bayes'
    )
    results.append(nb_metrics)
    
    # 5. Random Forest
    rf_metrics, rf_model = evaluate_model(
        RandomForestClassifier(random_state=42, n_estimators=100),
        X_train, X_test, y_train, y_test,
        'Random Forest'
    )
    results.append(rf_metrics)
    
    # 6. XGBoost
    xgb_metrics, xgb_model = evaluate_model(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        X_train, X_test, y_train, y_test,
        'XGBoost'
    )
    results.append(xgb_metrics)
    
    return pd.DataFrame(results)


