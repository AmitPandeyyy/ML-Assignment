"""
preprocess.py
Handles all data preprocessing logic:
- Encoding categorical features
- Scaling numerical features
- Train/test split
- Visualizations of data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import math


# load dataset and apply preprocessing(one-hot encoding for categorical, standard scaling for numerical)
def load_and_preprocess(csv_path, df=None):
    """
    Loads dataset and applies preprocessing.
    Returns processed train/test data and fitted preprocessor.
    """
    if df is None:
        df = pd.read_csv(csv_path)


    # Target column
    target_col = "NObeyesdad"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target (multiclass)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify feature types
    categorical_nominal = [
        "Gender", "family_history_with_overweight",
        "FAVC", "SMOKE", "SCC",
        "MTRANS"
    ]
    ordinal_featuers = ["CAEC","CALC"]

    numerical_features = [
        col for col in X.columns if (col not in categorical_nominal and col not in ordinal_featuers)
    ]

    # Preprocessing pipelines
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    ordinal_pipeline = Pipeline(
        steps=[("ordinal", OrdinalEncoder(categories=[["no", "Sometimes", "Frequently", "Always"], ["no", "Sometimes", "Frequently", "Always"]]))]
    )

    categorical_pipeline = Pipeline(
        # todo::check if drop="first" is needed
        steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_nominal),
            ("ord", ordinal_pipeline, ordinal_featuers),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Fit & transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        label_encoder
    )


# viaualize data distributions and relationships with target variable
def visualize_data(df):
    x_train_processed, x_test_processed, y_train, y_test, preprocessor, label_encoder = load_and_preprocess("data/obesity.csv")

    if hasattr(x_train_processed, "toarray"):
        x_train_processed = x_train_processed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(x_train_processed, columns=feature_names)
    df_processed["NObeyesdad"] = label_encoder.inverse_transform(y_train)

    # visualize numerical features
    # categorical_nominal = [
    #     "Gender", "family_history_with_overweight",
    #     "FAVC", "SMOKE", "SCC",
    #     "MTRANS"
    # ]
    # ordinal_featuers = ["CAEC","CALC"]
    # print(df_processed.columns)
    # breakpoint()

    numerical_features = [
        col for col in df_processed.columns if (col.startswith("num__"))
    ]

    # Visualize numerical features
    # df[numerical_features].hist(bins=15, figsize=(15, 10))
    # plt.suptitle("Numerical Feature Distributions")
    # plt.show()
    
    n_cols = 4  # number of plots per row
    n_rows = math.ceil(len(numerical_features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_features):
        sns.boxplot(
            x="NObeyesdad",
            y=col,
            data=df_processed,
            ax=axes[i]
        )
        axes[i].set_title(f"{col} vs Target")
        axes[i].tick_params(axis='x', rotation=45)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # for i, col in enumerate(numerical_features):
    #     plt.figure(figsize=(8, 4))
    #     sns.boxplot(x="NObeyesdad", y=col, data=df)
    #     plt.xticks(rotation=45)
    #     plt.title(f"{col} vs Target")
    #     plt.show()


if __name__ == "__main__":
    # x_train_processed, x_test_processed, y_train, y_test, preprocessor, label_encoder = load_and_preprocess("data/obesity.csv")
    # visualize data
    visualize_data(pd.read_csv("data/obesity.csv"))