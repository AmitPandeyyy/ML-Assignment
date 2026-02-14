"""
train_models.py
Train models and save them.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_and_preprocess
from metrics import evaluate_model

import pprint

def train_logistic_regression(X_train, X_test, y_train, y_test, dump_path="models/Logistic_Regression.pkl"):
    model = LogisticRegression(max_iter=2000, C=10, penalty="l1", solver="saga")
    model.fit(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    return model, result

def train_decision_tree(X_train, X_test, y_train, y_test, dump_path="models/Decision_Tree.pkl"):
    model = DecisionTreeClassifier(criterion='entropy',ccp_alpha=0.00039154001198452845)
    model.fit(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    return model, result

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=6, p=1, weights="distance")
    model.fit(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    return model, result

def train_naive_bayes(X_train, X_test, y_train, y_test):
    model = Pipeline([
        ("pca", PCA(n_components=0.95)),  # keep 95% variance
        ("nb", GaussianNB())
    ])
    model.fit(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    return model, result

def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=1)
    model.fit(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    return model, result

if __name__ == "__main__":
    csv_path = "data/obesity.csv"
    X_train, X_test, y_train, y_test, _, le = load_and_preprocess(csv_path)

    model, result = train_logistic_regression(X_train, X_test, y_train, y_test)
    joblib.dump({
        "model": model,
        "label_encoder": le
    }, "models/Logistic_Regression.pkl")
    print("Logistic Regression:")
    pprint.pprint(result, indent=4)

    model, result = train_decision_tree(X_train, X_test, y_train, y_test)
    joblib.dump({
        "model": model,
        "label_encoder": le
    }, "models/Decision_Tree.pkl")
    print("Decision Tree:")
    pprint.pprint(result, indent=4)

    model, result = train_knn(X_train, X_test, y_train, y_test)
    joblib.dump({
        "model": model,
        "label_encoder": le
    }, "models/kNN.pkl")
    print("kNN:")
    pprint.pprint(result, indent=4)

    model, result = train_naive_bayes(X_train, X_test, y_train, y_test)
    joblib.dump({
        "model": model,
        "label_encoder": le
    }, "models/Naive_Bayes.pkl")
    print("Naive Bayes:")
    pprint.pprint(result, indent=4)

    model, result = train_random_forest(X_train, X_test, y_train, y_test)
    joblib.dump({
        "model": model,
        "label_encoder": le
    }, "models/Random_Forest.pkl")
    print("Random Forest:")
    pprint.pprint(result, indent=4)
