from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from preprocess import load_and_preprocess
from metrics import evaluate_model

import pandas as pd
import pprint
import numpy as np

param_grid = {
    "C": [0.1, 0.3, 0.5, 0.7, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["saga"]
}

# param_grid = {
#     "C": [10],
#     "penalty": ["l1"],
#     "solver": ["saga"]
# }

# if __name__ == "__main__":
csv_path = "data/obesity.csv"
X_train, X_test, y_train, y_test, _, le = load_and_preprocess(csv_path)

grid = GridSearchCV(
    LogisticRegression(max_iter=5000),
    param_grid,
    cv=5,
    scoring="f1_macro"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(grid.best_params_)
results_df = pd.DataFrame(grid.cv_results_)
print(results_df[["params", "mean_test_score"]].sort_values(by="mean_test_score", ascending=False))
coef = best_model.coef_
print("Number of zero coefficients:", np.sum(coef == 0))
print("Number of non-zero coefficients:", np.sum(coef != 0))
result = evaluate_model(best_model, X_test, y_test)
pprint.pprint(result)
#     return model, result