"""
metrics.py
Computes all required evaluation metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, average="weighted"):
    """
    Evaluates model using all required metrics.
    """

    y_pred = model.predict(X_test)

    # For AUC, probability output is needed
    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        except:
            auc = None

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=average),
        "Recall": recall_score(y_test, y_pred, average=average),
        "F1": f1_score(y_test, y_pred, average=average),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        # "Report": classification_report(y_test, y_pred)
    }
