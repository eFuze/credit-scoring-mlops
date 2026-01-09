"""Fonctions de coût métier pour le scoring crédit."""
import numpy as np
from sklearn.metrics import make_scorer


def get_probas(estimator, X):
    """Retourne P(y=1). Utilise predict_proba si dispo, sinon decision_function."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1]
    elif hasattr(estimator, "decision_function"):
        z = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    else:
        return estimator.predict(X).astype(float)


def cost_from_threshold(y_true, y_proba, threshold, fn_cost=10.0, fp_cost=1.0):
    """Calcule le coût métier pour un seuil donné."""
    pred = (y_proba >= threshold).astype(int)
    fp = np.sum((y_true == 0) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))
    return fn_cost * fn + fp_cost * fp


def best_cost_and_threshold(y_true, y_proba, thresholds=None, fn_cost=10.0, fp_cost=1.0):
    """Trouve le seuil optimal minimisant le coût métier."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    costs = np.array([
        cost_from_threshold(y_true, y_proba, t, fn_cost, fp_cost)
        for t in thresholds
    ])
    i = int(costs.argmin())
    return costs[i], float(thresholds[i])


def business_score(y_true, y_proba, fn_cost=10.0, fp_cost=1.0):
    """Score métier (négatif du coût) pour sklearn."""
    best_cost, _ = best_cost_and_threshold(y_true, y_proba, fn_cost=fn_cost, fp_cost=fp_cost)
    return -best_cost


def business_threshold(y_true, y_proba, fn_cost=10.0, fp_cost=1.0):
    """Retourne le seuil optimal."""
    _, t = best_cost_and_threshold(y_true, y_proba, fn_cost=fn_cost, fp_cost=fp_cost)
    return t


# Scorers pour GridSearchCV
biz_scorer = make_scorer(business_score, response_method='predict_proba', greater_is_better=True)
thr_scorer = make_scorer(business_threshold, response_method='predict_proba', greater_is_better=True)
