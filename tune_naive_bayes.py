import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from preprocess import load_and_preprocess
from metrics import evaluate_model

import pprint

csv_path = "data/obesity.csv"
X_train, X_test, y_train, y_test, _, le = load_and_preprocess(csv_path)


param_grid = {
    "pca__n_components": [0.90, 0.95, 0.99, None],
    "nb__var_smoothing": np.logspace(-12, -6, 20)
}

print("Var Smoothing Values:", param_grid["nb__var_smoothing"])


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
model = Pipeline([
        ("pca", PCA(n_components=0.95)),  # keep 95% variance
        ("nb", GaussianNB())
    ])

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1_macro"
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

best_model = grid.best_estimator_
print(grid.best_params_)
result = evaluate_model(best_model, X_test, y_test)
pprint.pprint(result)