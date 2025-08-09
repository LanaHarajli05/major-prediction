import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="🎓 Major Grouping Predictor", layout="wide")
st.title("🎓 Major Grouping Predictor")
st.caption("Multiclass model to infer a student's **Major Grouping** with per-student probabilities, filters, and confusion matrix.")

# -------------------------
# Helpers
# -------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def pick_col(df: pd.DataFrame, candidates):
    normalized = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in normalized:
            return normalized[key]
    return ""

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric_cols, categorical_cols

def impute_simple(X: pd.DataFrame, numeric_cols, categorical_cols):
    X = X.copy()
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in categorical_cols:
        X[c] = X[c].fillna("Unknown")
    return X

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def topk_from_proba(proba: np.ndarray, classes: np.ndarray, k: int):
    topk_labels, topk_probs = [], []
    for row in proba:
        idx = np.argsort(row)[::-1][:k]
        topk_labels.append(", ".join([str(classes[i]) for i in idx]))
        topk_probs.append(", ".join([f"{row[i]:.2f}" for i in idx]))
    return topk_labels, topk_probs

def train_and_predict(df: pd.DataFrame, target_col: str, id_col: str,
                      test_size: float, random_state: int, model_key: str):
    y = df[target_col].astype(str)
    feature_cols = [c for c in df.columns if c != target_col]
    if id_col and id_col in feature_cols:
        feature_cols.remove(id_col)
    X = df[feature_cols].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    X = impute_simple(X, numeric_cols, categorical_cols)

    pre, _, _ = build_preprocessor(X)

    if model_key == "LR":
        model = LogisticRegression(
            multi_class="multinomial",
            solver="saga",
            max_iter=3000,
        )
    elif model_key == "RF":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError("Unknown model key.")

    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=test_size, random_state=random_state, stratify=strat
    )

    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)

    has_proba = hasattr(pipe, "predict_proba")
    if has_proba:
        proba_full = pipe.predict_proba(X)
        classes = pipe.classes_
    else:
        proba_full, classes = None, None

    base_cols = {}
    if id_col and id_col in df.columns:
        base_cols[id_col] = df[id_col].values
    else:
        base_cols["ID"] = np.arange(len(df))
    base_cols["True Major Grouping"] = y.values
    base_cols["Predicted Major Grouping"] = pipe.predict(X)
    out_df = pd.DataFrame(base_cols)

    return {
        "pipe": pipe,
        "X": X,
        "y": y,
        "idx_test": idx_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "out_df": out_df,
        "proba_full": proba_full,
        "classes": classes,
        "has_proba": has_proba,
    }

# -------------------------
# Sidebar: data + settings + filters
# -------------------------
with st.sidebar:
    st.header("1) Upload Data")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    st.header("2) Target & ID")
    target_default = "Major Grouping"
    id_default = "NAME"
    target_col_in = st.text_input("Target column:", value=target_default)
    id_col_in = st.text_input("ID column (optional):", value=id_default)

    st.header("3) Modeling")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", 0, 9999, 42)
    top_k = st.slider("Top-K predictions to show", 1, 5, 3)

if not file:
    st.info("⬅️ Upload your dataset (CSV/XLSX) in the sidebar to begin.")
    st.stop()

# Read uploaded file
if file.name.lower().endswith(".csv"):
    raw = pd.read_csv(file)
else:
    raw = pd.read_excel(file)

df0 = normalize_cols(raw)

# Pick columns robustly
target_col = pick_col(df0, [target_col_in, "Major Grouping", "Major grouping", "major grouping"])
if not target_col:
    st.error("Target column not found. Make sure your file has a 'Major Grouping' column.")
    st.stop()

id_col = pick_col(df0, [id_col_in, "NAME", "Name", "Id", "ID"])

# Clean targets
df0 = df0.dropna(subset=[target_col]).copy()
df0[target_col] = (
    df0[target_col]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
canon_map = {
    "engineering": "Engineering",
    "business": "Business",
    "cs/it/mis": "CS/IT/MIS",
    "sciences": "Sciences",
    "health": "Health",
    "arts & others": "Arts & Others",
    "arts and others": "Arts & Others",
}
df0[target_col] = df0[target_col].apply(lambda v: canon_map.get(v.lower(), v))

# Filters (dynamic if columns exist)
candidate_filters = [
    ["Cohort", "Admit Semester", "Admit semester"],
    ["Gender"],
    ["University"],
    ["COR", "Country of Residence"],
    ["Nationality"],
    ["Employment"],
    ["Final Status", "Final Status "],
]
filter_cols = [pick_col(df0, cands) for cands in candidate_filters]
filter_cols = [c for c in filter_cols if c]

with st.sidebar:
    st.header("4) Filters (apply to metrics & tables)")
    filters = {}
    for c in filter_cols:
        vals = sorted([v for v in df0[c].dropna().unique().tolist()])
        selected = st.multiselect(f"{c}", vals, default=[])
        filters[c] = set(selected)

# Apply filters to views (not to training)
view_mask = pd.Series(True, index=df0.index)
for c, chosen in filters.items():
    if chosen:
        view_mask &= df0[c].isin(chosen)
df_view = df0.loc[view_mask].copy()

# -------------------------
# EDA
# -------------------------
st.subheader("Target Distribution (Major Grouping)")
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    counts = df0[target_col].value_counts().sort_values(ascending=False)
    st.bar_chart(counts)
with c2:
    st.write("**Counts**")
    st.dataframe(counts.rename_axis("Group").reset_index(name="Count"), use_container_width=True)

st.divider()

# -------------------------
# Tabs for models (LR, RF)
# -------------------------
tab_names = ["Logistic Regression", "Random Forest"]
tabs = st.tabs(tab_names)
model_keys = {"Logistic Regression": "LR", "Random Forest": "RF"}

for model_name, tab in zip(tab_names, tabs):  # safe: don't use private attributes
    with tab:
        st.markdown(f"### {model_name}")

        results = train_and_predict(
            df=df0,
            target_col=target_col,
            id_col=id_col,
            test_size=test_size,
            random_state=random_state,
            model_key=model_keys[model_name],
        )

        out_df = results["out_df"]
        has_proba = results["has_proba"]
        classes = results["classes"]
        proba_full = results["proba_full"]

        if has_proba:
            topk_labels, topk_probs = topk_from_proba(proba_full, classes, top_k)
            out_df[f"Top-{top_k} Groups"] = topk_labels
            out_df[f"Top-{top_k} Probs"] = topk_probs

        # Add context columns (filters) to the display table
        attach_cols = [c for c in filter_cols if c]
        display_df = out_df.copy()
        if attach_cols:
            display_df = pd.concat(
                [df0[attach_cols].reset_index(drop=True), display_df.reset_index(drop=True)],
                axis=1
            )

        # ----- Metrics (respecting filters) -----
        idx_test = results["idx_test"]
        y_test = results["y_test"]
        y_pred_test = results["y_pred_test"]
        labels_all = sorted(df0[target_col].astype(str).unique().tolist())

        # Filter test indices by current view mask
        test_mask = pd.Series(False, index=df0.index)
        test_mask.iloc[idx_test] = True
        test_and_view = (test_mask & view_mask)
        if test_and_view.any():
            idx_positions = np.where(test_and_view.iloc[idx_test].values)[0]
            y_test_f = y_test.iloc[idx_positions]
            y_pred_f = y_pred_test[idx_positions]
        else:
            y_test_f = y_test
            y_pred_f = y_pred_test

        st.subheader("Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy_score(y_test_f, y_pred_f):.3f}")
        m2.metric("Macro F1", f"{f1_score(y_test_f, y_pred_f, average='macro'):.3f}")
        m3.metric("# Classes", f"{len(labels_all)}")

        st.text("Classification Report")
        st.code(classification_report(y_test_f, y_pred_f, labels=labels_all), language="text")

        st.subheader("Confusion Matrix (respecting filters)")
        plot_confusion(y_test_f, y_pred_f, labels_all)

        st.subheader("Per-Student Predictions (filtered by sidebar)")
        display_mask = view_mask.values
        display_df_f = display_df.loc[display_mask].reset_index(drop=True)
        st.dataframe(display_df_f, use_container_width=True, height=420)

        st.download_button(
            label="Download filtered predictions (CSV)",
            data=display_df_f.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_name.replace(' ', '_').lower()}_major_grouping_predictions.csv",
            mime="text/csv",
        )

st.success("Ready. Use the sidebar to filter cohorts/universities/etc., and switch model tabs for comparison.")
