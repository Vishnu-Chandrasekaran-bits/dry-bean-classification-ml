import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

DATA_FILE = "model/dataset.xlsx"
TARGET_COL = "Class"
MODEL_DIR = "model"


def get_metrics(y_true, y_pred, y_prob):
    """Work out all six metrics for one model."""
    is_multiclass = len(np.unique(y_true)) > 2

    if is_multiclass:
        average = "weighted"
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    else:
        average = "binary"
        auc = roc_auc_score(y_true, y_prob[:, 1])

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# load the data and drop any rows with missing values
data = pd.read_excel(DATA_FILE)
data = data.dropna()

features = data.drop(columns=[TARGET_COL])
labels = data[TARGET_COL]

# turn text columns (if there are any) into numbers
features = pd.get_dummies(features)

# encode the target labels as integers
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# save the encoder and column names so the app can use them later
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
joblib.dump(list(features.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, stratify=y, random_state=42
)

# keep a copy of the test set (with the real label names) to upload in the app
test_data = X_test.copy()
test_data[TARGET_COL] = encoder.inverse_transform(y_test)
test_data.to_csv("test_data.csv", index=False)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []

for name, clf in classifiers.items():
    # scaling is inside the pipeline so it gets saved together with the model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipeline.fit(X_train, y_train)

    file_name = name.replace(" ", "_").lower() + ".joblib"
    joblib.dump(pipeline, os.path.join(MODEL_DIR, file_name))

    scores = get_metrics(
        y_test,
        pipeline.predict(X_test),
        pipeline.predict_proba(X_test),
    )

    row = {"ML Model Name": name}
    for metric, value in scores.items():
        row[metric] = round(value, 4)
    results.append(row)

results_df = pd.DataFrame(results)

# needs the 'tabulate' package for to_markdown
print(results_df.to_markdown(index=False))
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_table.csv"), index=False)
