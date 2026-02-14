from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocess import load_and_preprocess
from metrics import evaluate_model

import pprint

param_grid = {
    "n_estimators": [300],
    # "n_estimators": [100, 150, 200, 250,300],
    # "max_depth": [8, 12, 15, None],
    "max_depth": [15],
    "min_samples_leaf": [1],
    # "min_samples_leaf": [1, 2, 5],
}

csv_path = "data/obesity.csv"
X_train, X_test, y_train, y_test, _, le = load_and_preprocess(csv_path)
# print("Training samples:", len(X_train))
# breakpoint()

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

best_tree = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

result = evaluate_model(best_tree, X_test, y_test)
pprint.pprint(result)
importances = best_tree.feature_importances_
print("Feature Importances:", importances)
