from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from preprocess import load_and_preprocess
from metrics import evaluate_model

import pandas as pd
import pprint
import numpy as np

csv_path = "data/obesity.csv"
X_train, X_test, y_train, y_test, _, le = load_and_preprocess(csv_path)

# print("Training samples:", len(X_train))
# print("Testing samples:", len(X_test))
# breakpoint()
# param_grid = {
#     "knn__n_neighbors": [7, 11, 21, 31, 41, 45],
#     "knn__weights": ["uniform", "distance"],
#     "knn__p": [1, 2]
# }
param_grid = {
    "n_neighbors": [5, 6, 7, 8, 9, 10, 11, 12],
    # "n_neighbors": [5,6, 7, 9, 11, 15, 21, 31, 41, 45],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

model = KNeighborsClassifier()
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
best_model = grid.best_estimator_
print(grid.best_params_)
result = evaluate_model(best_model, X_test, y_test)
pprint.pprint(result)