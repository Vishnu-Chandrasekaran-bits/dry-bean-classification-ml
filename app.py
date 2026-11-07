import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Classifier Explorer", layout="wide")
st.title("Classification Model Explorer")

# must match the names used in train_models.py
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
TARGET_COL = "Class"

encoder = joblib.load("model/label_encoder.joblib")
feature_cols = joblib.load("model/feature_columns.joblib")

is_multiclass = len(encoder.classes_) > 2
average = "weighted" if is_multiclass else "binary"


def calc_auc(y_true, y_prob):
    if is_multiclass:
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    return roc_auc_score(y_true, y_prob[:, 1])


def calc_all_metrics(y_true, y_pred, y_prob):
    """Returns the six metrics as a dict."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": calc_auc(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# sidebar
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox("Select model", list(MODEL_PATHS.keys()))
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")

if uploaded is None:
    st.info("Upload the test CSV (features + target column) from the sidebar.")
    st.stop()

# load and prepare the data
df = pd.read_csv(uploaded)

if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' is missing from the file.")
    st.stop()

X = pd.get_dummies(df.drop(columns=[TARGET_COL]))
# make sure the columns line up with what the models were trained on
X = X.reindex(columns=feature_cols, fill_value=0)
y_true = encoder.transform(df[TARGET_COL])

model = joblib.load(MODEL_PATHS[selected_model])
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

# metrics
st.subheader(f"Metrics: {selected_model}")

metrics = calc_all_metrics(y_true, y_pred, y_prob)
columns = st.columns(len(metrics))
for col, (name, value) in zip(columns, metrics.items()):
    col.metric(name, f"{value:.4f}")

# confusion matrix and report
left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[str(c) for c in encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T)

# optional comparison of every model
if st.checkbox("Compare all models on this data"):
    table_rows = []

    for name, path in MODEL_PATHS.items():
        m = joblib.load(path)
        scores = calc_all_metrics(y_true, m.predict(X), m.predict_proba(X))
        table_rows.append({"Model": name, **scores})

    comparison = pd.DataFrame(table_rows).round(4)
    st.dataframe(comparison)
