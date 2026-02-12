"""
train_models.py
Train models and save them.
"""

import joblib
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_preprocess
from metrics import evaluate_model

import pprint

def train_logistic_regression(X_train, X_test, y_train, y_test, dump_path="models/Logistic_Regression.pkl"):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    joblib.dump(model, dump_path)
    result = evaluate_model(model, X_test, y_test)
    return model, result


if __name__ == "__main__":
    csv_path = "data/obesity.csv"
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess(csv_path)

    model, result = train_logistic_regression(X_train, X_test, y_train, y_test)
    print("Logistic Regression:")
    pprint.pprint(result, indent=4)