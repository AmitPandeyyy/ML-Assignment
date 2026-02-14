from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from preprocess import load_and_preprocess
from metrics import evaluate_model

import pandas as pd
import pprint
import numpy as np

csv_path = "data/obesity.csv"
X_train, X_test, y_train, y_test, _, le = load_and_preprocess(csv_path)

path = DecisionTreeClassifier().cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

alphas_sampled = np.logspace(
    np.log10(ccp_alphas[1]),
    np.log10(ccp_alphas[-2]),
    20
)

param_grid = {
    "max_depth": [None, 5, 8, 10, 12 , 15, 20],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3],
    "criterion": ["gini", "entropy"],
    "ccp_alpha": alphas_sampled
}

print("CCP Alphas:", alphas_sampled)

breakpoint()

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

# if __name__ == "__main__":

grid.fit(X_train, y_train)

best_tree = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

coef = best_tree.feature_importances_
print("Feature Importances:", coef)

result = evaluate_model(best_tree, X_test, y_test)
pprint.pprint(result)


print(grid.best_params_)
results_df = pd.DataFrame(grid.cv_results_)
print(results_df[["params", "mean_test_score"]].sort_values(by="mean_test_score", ascending=False))
