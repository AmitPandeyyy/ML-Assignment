import streamlit as st
import pandas as pd
import joblib

from preprocess import load_and_preprocess
from metrics import evaluate_model

import matplotlib.pyplot as plt
import seaborn as sns

st.title("Multiclass Classification - Obesity Dataset")

st.sidebar.header("Model Selection")

MODEL_FILES = {
    "Logistic Regression": "models/Logistic_Regression.pkl",
    "Decision Tree": "models/Decision_Tree.pkl",
    "k Nearest Neighbors": "models/kNN.pkl",
    "Naive Bayes": "models/Naive_Bayes.pkl",
    "Random Forest": "models/Random_Forest.pkl",
    "XGBoost": "models/XGBoost.pkl",
}

model_name = st.sidebar.selectbox("Choose Model", list(MODEL_FILES.keys()))
model = joblib.load(MODEL_FILES[model_name])
le = model["label_encoder"]
preprocessor = model["preprocessor"]
model = model["model"]

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    # print("File uploaded:", uploaded_file.name)
    # st.write("File uploaded:", uploaded_file.name)
    # st.write("File uploaded:", uploaded_file)
    # raw_bytes = uploaded_file.getvalue()
    # st.write(raw_bytes[:200])
    # df = pd.read_csv(uploaded_file.getvalue())
    # st.write("Preview of uploaded data:")
    df = pd.read_csv(uploaded_file)
    # st.dataframe(df.head())

    # X_train, X_test, y_train, y_test, _, label_encoder = load_and_preprocess(
    #     uploaded_file, df = df
    # )

    target_col = "NObeyesdad"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_processed = preprocessor.transform(X)
    y_processed = le.transform(y)

    # results = evaluate_model(model, X_test, y_test)
    results = evaluate_model(model, X_processed, y_processed)

    st.subheader("Evaluation Metrics")
    selected_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    filtered_metrics = {k: results[k] for k in selected_keys if k in results}
    df = pd.DataFrame(filtered_metrics.items(), columns=["Metric", "Value"])
    df = df.set_index("Metric")
    st.table(df)

    st.subheader("Confusion Matrix")
    cm = results['Confusion Matrix']
    class_names = le.classes_
    fig, ax = plt.subplots()
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False,
                ax=ax)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)