# Created by Hansi at 11/15/2021
from sklearn.metrics import f1_score, recall_score, precision_score


def token_macro_f1(y_true, y_pred):
    f1_scores = []
    for t, p in zip(y_true, y_pred):
        f1_scores.append(f1_score(t, p, average="macro"))
    return {"F1 macro score": sum(f1_scores) / len(f1_scores), "Total": len(f1_scores)}


def token_macro_recall(y_true, y_pred):
    recall_scores = []
    for t, p in zip(y_true, y_pred):
        recall_scores.append(recall_score(t, p, average="macro"))
    return {"Recall macro score": sum(recall_scores) / len(recall_scores), "Total": len(recall_scores)}


def token_macro_precision(y_true, y_pred):
    precision_scores = []
    for t, p in zip(y_true, y_pred):
        precision_scores.append(precision_score(t, p, average="macro"))
    return {"Precision macro score": sum(precision_scores) / len(precision_scores), "Total": len(precision_scores)}