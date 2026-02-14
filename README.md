# ML Assignment-2
# Problem Statement
Build and compare multiple machine learning classification models to predict obesity levels based on anthropometric measurements and lifestyle-related features. Goal is to evaluate and compare model performance using Accuracy, F1-score, Precision, Recall, MCC, and AUC metrices.
# Dataset Description
Dataset used is "Estimation of Obesity Levels Based on Eating Habits and Physical Condition dataset (UCI / Kaggle version)".
Various features like height, weight, eating habits, physical activity, gender and age, etc. are given for each sample.
The target variable is obesity level, classified into 7 different classes.
Total features = 16 (8 numerical, 6 categorical, 2 nominal)
Total Samples = 2111
## Models used
- Logistic Regression (L1 regularization, solver=saga, C=10)
- Decision Tree (criterion=entropy, ccp_alpha=0.00039)
- kNN (n_neighbours=6, p=1, distance weighting)
- Naive Bayes (GaussianNB with PCA)
- Random Forest (300 estimators,max_depth=15, min_sample_leaf=1)
- XGBoost (evaluation_metric=mlogloss, num_classes=7)
## Preprocessing Used
- Label encoding of target variable
- One-hot encoding for 6 categorical features
- Ordinal encoding for 2 ordinal features
- Standardizing for 8 numerical features
- Additionaly, added PCA for Naive Bayes classifier to improve its accuracy
- Train-test split - 80/20

# Model Comp arison Results

| Model               |   Accuracy |     F1 |   Precision |   Recall |    MCC |    AUC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9669 | 0.9669 |      0.9677 |   0.9669 | 0.9615 | 0.9982 |
| Decision Tree       |     0.9338 | 0.934  |      0.9361 |   0.9338 | 0.923  | 0.9603 |
| kNN                 |     0.8936 | 0.8901 |      0.8908 |   0.8936 | 0.8764 | 0.9904 |
| Naive Bayes         |     0.6596 | 0.6535 |      0.6582 |   0.6596 | 0.6036 | 0.9105 |
| Random Forest       |     0.9551 | 0.9559 |      0.9596 |   0.9551 | 0.948  | 0.9968 |
| XGBoost             |     0.9527 | 0.9532 |      0.9552 |   0.9527 | 0.9451 | 0.9963 |

# Comments

| Model               |   Observation about model performance |
|:--------------------|--------------------------------------|
| Logistic Regression |Best overall performer. High Accuracy, F1, MCC, and AUC. L1 regularization helped remove irrelevant/correlated features.|
| Decision Tree       |Good performance but slightly lower than ensemble methods.(Single tree can still have some variance/overfitting)|
| kNN                 |Moderate performance. Reducing n=6 and using manhattan distance helped improve accuracy.|
| Naive Bayes         |Poor performance compared to others. Feature Independence assumption violated due to correlated features.|
| Random Forest       |Strong performance with high stability. Reduced variance compared to single Decision Tree. Slightly lower than Logistic Regression.|
| XGBoost             |Very good performance, slightly lower than Logistic Regression. Boosting improves bias reduction.|
