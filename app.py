"""
app.py
------
Streamlit web application for Assignment 2 (Machine Learning, M.Tech AIML/DSE).

BUSINESS QUESTION THIS APP ANSWERS:
    "Given information known at order time, is this order at risk of a
    late delivery -- and how well do different ML models detect that risk?"

The app lets a user:
    1. Upload a CSV of test/order data (features + optional true label column)
    2. Pick which trained classification model to score the data with
    3. See the predictions, evaluation metrics, confusion matrix and a
       classification report so business stakeholders can compare models
       and decide which one to trust for flagging at-risk orders.

All 5 models were trained in `model/train_models.ipynb` and saved as
scikit-learn Pipelines (preprocessing + classifier bundled together), so
this app never has to re-implement scaling/encoding -- it just loads the
pipeline and calls .predict() / .predict_proba().
"""

import json
import os
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# pyrefly: ignore [missing-import]
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Late Delivery Risk Predictor",
    page_icon="📦",
    layout="wide",
)

MODEL_DIR = "model"

# Human-readable model name -> saved filename (must match train_models.ipynb)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}


# --------------------------------------------------------------------------
# Cached helpers -- loaded once per session, not on every rerun
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(model_filename: str):
    """Load a trained scikit-learn Pipeline (preprocessing + classifier) from disk."""
    path = os.path.join(MODEL_DIR, model_filename)
    return joblib.load(path)


@st.cache_resource
def load_schema():
    """Load the expected feature-column schema saved during training."""
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        return json.load(f)


def show_dataframe(df):
    """Render dataframe full-width cleanly across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.dataframe(df, use_container_width=True)


def read_csv_robust(file_obj):
    """Read a CSV file, automatically falling back to common encodings.

    Uploaded CSVs are often saved as Latin-1/Windows-1252 rather than UTF-8,
    which makes pandas' default utf-8 reader raise an encoding error. This
    tries UTF-8 first, then Windows-1252, then Latin-1 (which never fails).
    """
    first_bytes = file_obj.read(2048)
    file_obj.seek(0)
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(file_obj, encoding=encoding), encoding
        except UnicodeDecodeError:
            file_obj.seek(0)
    # latin-1 maps every byte, so this is only reached for non-encoding errors
    file_obj.seek(0)
    return pd.read_csv(file_obj, encoding="latin-1"), "latin-1"


def show_business_question():
    """Render the business question as a clear callout above the answer."""
    st.markdown(
        """
        <div style="background:#fff7e6;border:1px solid #ffd591;border-left:6px solid #fa8c16;
             border-radius:10px;padding:14px 18px;margin-bottom:8px;">
            <div style="font-size:0.85rem;font-weight:700;color:#d46b08;letter-spacing:0.5px;">
                THE BUSINESS QUESTION ❓
            </div>
            <div style="color:#873800;font-size:1.02rem;line-height:1.6;margin-top:4px;">
                <b>Can we predict, at order time,</b> whether a supply-chain order is at risk of arriving
                late - so operations teams can intervene
                <i>(expedite shipping, reallocate carrier, notify the customer)</i> <b>before it happens?</b>
            </div>
            <div style="color:#ad6800;font-size:0.88rem;margin-top:6px;">
                The answer - based on this run's results - is below 👇
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_css():
    """Inject custom CSS for a cleaner, more polished UI."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 10px 14px;
        }
        [data-testid="stMetricLabel"] { font-size: 0.85rem; }
        [data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


METRIC_HELP = {
    "Accuracy": "Share of ALL orders the model classified correctly (late + on-time). "
    "High is good, but can be misleading if the classes are imbalanced.",
    "AUC": "Chance the model ranks a random late order as riskier than a random "
    "on-time order. 1.0 = perfect, 0.5 = random guessing.",
    "Precision": "When the model flags an order as LATE, how often it is actually late. "
    "High = few false alarms.",
    "Recall": "Of all orders that were ACTUALLY late, how many the model caught. "
    "High = few missed late orders.",
    "F1 Score": "Single score balancing Precision and Recall (harmonic mean). Useful "
    "when you care about both false alarms and missed orders.",
    "MCC": "Correlation between predicted and actual labels (range -1 to 1). "
    "0 = no better than random, 1 = perfect. Robust to class imbalance.",
}


def plot_risk_tier_donut(probabilities):
    """Plot a donut chart showing proportions of Low, Medium, and High risk orders."""
    low = (probabilities < 0.4).sum()
    med = ((probabilities >= 0.4) & (probabilities <= 0.65)).sum()
    high = (probabilities > 0.65).sum()

    labels = ["Low Risk\n(<40%)", "Medium Risk\n(40-65%)", "High Risk\n(>65%)"]
    sizes = [low, med, high]
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
        pctdistance=0.78,
    )
    for t in texts:
        t.set_color("#334e68")
        t.set_fontsize(8.5)
    for a in autotexts:
        a.set_color("white")
        a.set_fontweight("bold")
        a.set_fontsize(10)
    ax.text(
        0, 0, f"Total\n{len(probabilities):,}",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#334e68",
    )
    ax.set_title("Order Risk Tier Breakdown", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_risk_distribution(probabilities):
    """Plot risk probability distribution histogram with KDE curve."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axvspan(0.0, 0.4, color="#2ecc71", alpha=0.12)
    ax.axvspan(0.4, 0.65, color="#f1c40f", alpha=0.15)
    ax.axvspan(0.65, 1.0, color="#e74c3c", alpha=0.12)
    sns.histplot(probabilities, kde=True, bins=25, color="#2b5c8f", ax=ax, edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="#d9534f", linestyle="--", linewidth=1.6, label="Decision threshold (0.50)")
    ax.set_title("Risk Probability Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Late Probability", fontsize=9)
    ax.set_ylabel("Order Count", fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=8, loc="upper center")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_shipping_mode_risk(results_df):
    """Plot late delivery risk rate breakdown by Shipping Mode."""
    if "Shipping Mode" not in results_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    group = results_df.groupby("Shipping Mode")["Predicted_Late_Delivery_Risk"].mean() * 100
    group = group.sort_values(ascending=False)
    bar_colors = ["#27ae60" if v < 40 else ("#f1c40f" if v <= 65 else "#e74c3c") for v in group.values]
    bars = ax.bar(group.index, group.values, color=bar_colors, edgecolor="none", width=0.55, alpha=0.9)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#334e68",
        )
    ax.set_title("Late-Delivery Risk by Shipping Mode", fontsize=11, fontweight="bold")
    ax.set_ylabel("% Predicted Late", fontsize=9)
    ax.set_ylim(0, max(group.values) * 1.25 if len(group) > 0 and max(group.values) > 0 else 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, model_name, auc_score):
    """Plot ROC Curve with AUC score region shaded."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def render_benchmark_table(df_comp, current_model_name):
    """Build an HTML comparison table with best-per-metric and selected-model highlights."""
    df = df_comp.sort_values("AUC", ascending=False).reset_index(drop=True)
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    best_idx = {m: df[m].idxmax() for m in metrics}
    ranks = ["🥇", "🥈", "🥉"] + [f"{i}." for i in range(4, len(df) + 1)]

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        name = row["ML Model Name"]
        selected = name == current_model_name
        tr_style = ' style="background:#fff4e6;"' if selected else ""
        badge = "🏅 Best overall" if i == best_idx["AUC"] else ranks[i]
        name_cell = ("⭐ " if selected else "") + name
        cells = [
            f"<td style='padding:9px 10px;border-bottom:1px solid #edf2f7;text-align:center;'>{badge}</td>",
            f"<td style='padding:9px 10px;border-bottom:1px solid #edf2f7;font-weight:700;text-align:left;'>{name_cell}</td>",
        ]
        for m in metrics:
            if best_idx[m] == i:
                cells.append(
                    f"<td style='padding:9px 10px;border-bottom:1px solid #edf2f7;text-align:center;"
                    f"color:#1e7e34;font-weight:800;'>{row[m]:.3f} ✓</td>"
                )
            else:
                cells.append(
                    f"<td style='padding:9px 10px;border-bottom:1px solid #edf2f7;text-align:center;'>{row[m]:.3f}</td>"
                )
        rows.append(f"<tr{tr_style}>{''.join(cells)}</tr>")

    header = (
        "<tr><th style='padding:10px;text-align:center;'>Rank</th>"
        "<th style='padding:10px;text-align:left;'>ML Model</th>"
        + "".join(f"<th style='padding:10px;text-align:center;'>{m}</th>" for m in metrics)
        + "</tr>"
    )
    return (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;width:100%;font-size:0.9rem;"
        "font-family:Segoe UI,Roboto,sans-serif;border:1px solid #e2e8f0;border-radius:10px;'>"
        "<thead style='background:#1e3c72;color:#ffffff;'>" + header + "</thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def plot_model_comparison_benchmark(current_model_name):
    """Plot a ranked heatmap of the model comparison table."""
    table_path = os.path.join(MODEL_DIR, "model_comparison_table.csv")
    if not os.path.exists(table_path):
        return None
    df_comp = pd.read_csv(table_path)
    df = df_comp.set_index("ML Model Name")
    df = df.loc[df["AUC"].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Score", "shrink": 0.7},
        ax=ax,
        annot_kws={"fontsize": 9},
    )
    if current_model_name in df.index:
        row_idx = list(df.index).index(current_model_name)
        ax.add_patch(
            plt.Rectangle((0, row_idx), df.shape[1], 1, fill=False, edgecolor="#e67e22", lw=3)
        )
    ax.set_title("Model Performance Benchmark (ranked by AUC)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Header / business framing
# --------------------------------------------------------------------------
inject_css()

st.title("📦 Late Delivery Risk Predictor")
st.markdown(
    """
**Business question:** *Can we predict, at order time, whether a supply-chain
order is at risk of arriving late - so operations teams can intervene
(expedite shipping, reallocate carrier, notify the customer) before it happens?*
"""
)

with st.expander("👨‍💻 How to use this app (read this first)"):
    st.markdown(
        """
        Follow these **4 steps**:

        1. **Pick a data source** - upload your own CSV of order data, or tick *"Use bundled test_data.csv"* in the sidebar.
        2. **Choose a model** - pick one of the 5 trained classifiers from the dropdown.
        3. **Click '🚀 Run Model'** - the app scores every order and shows the results below.
        4. **Read the results top-to-bottom** - Summary → Predictions → Evaluation Metrics → Visual Analytics.

        > **Tip:** results get richer if your CSV includes the true `Late_delivery_risk`
        > label column - then you also get Accuracy / AUC / Precision, the confusion matrix
        > and the classification report.
        """
    )

schema = load_schema()
numeric_features = schema["numeric_features"]
categorical_features = schema["categorical_features"]
target_col = schema["target"]
required_features = numeric_features + categorical_features

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    ">
        <h3 style="margin: 0; font-family: 'Segoe UI', Roboto, sans-serif; font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: 0.5px;">
            ✨ Welcome to Shivam's Predictions ✨
        </h3>
        <p style="margin: 5px 0 0 0; font-size: 0.8rem; opacity: 0.9; color: #d0e1fd;">
            ML Risk Analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Controls")

# (b) Model selection dropdown -- required by the assignment
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

# Run model button -- requires explicit user click to execute selected model
run_button = st.sidebar.button("🚀 Run Model", type="primary", use_container_width=True)

if "executed_model" not in st.session_state:
    st.session_state["executed_model"] = None

if run_button:
    st.session_state["executed_model"] = model_name

st.sidebar.markdown("---")
with st.sidebar.expander("📋 Required CSV columns"):
    st.markdown("**Features (14):**\n\n" + "\n".join(f"- `{c}`" for c in required_features))
    st.markdown(
        f"**Optional target:** `{target_col}` - include it to unlock evaluation "
        "metrics, the confusion matrix and the classification report."
    )

with st.sidebar.expander("🧠 About the models"):
    st.markdown(
        """
        - **Logistic Regression** - simple, fast linear baseline; great precision.
        - **Decision Tree** - easy-to-interpret rules, but tends to overfit.
        - **kNN** - nearest-neighbour voting; weak on mixed categorical data.
        - **Naive Bayes** - fast probabilistic model; solid despite strong assumptions.
        - **Random Forest** - ensemble of trees; the most robust overall performer (best AUC).
        """
    )

# --------------------------------------------------------------------------
# (a) Dataset selection & upload -- required by the assignment
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Source Selection")

data_mode = st.sidebar.radio(
    "Choose data source:",
    options=["Upload CSV file", "Use bundled test_data.csv"],
    index=0,
)

data = None
data_label = ""

if data_mode == "Upload CSV file":
    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            data, encoding = read_csv_robust(uploaded_file)
            data_label = f"Uploaded File ({uploaded_file.name})"
            st.success(
                f"Loaded uploaded file '{uploaded_file.name}' with {data.shape[0]} rows "
                f"(encoding: {encoding})."
            )
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.info("👈 Please upload a CSV file using the sidebar to get started, or select 'Use bundled test_data.csv'.")
else:
    if os.path.exists("test_data.csv"):
        data = pd.read_csv("test_data.csv")
        data_label = "Bundled Sample Dataset (test_data.csv)"
        st.info(f"Using bundled sample `test_data.csv` ({data.shape[0]} rows).")
    else:
        st.error("Bundled sample `test_data.csv` file not found.")

# --------------------------------------------------------------------------
# Main prediction & evaluation logic
# --------------------------------------------------------------------------
if data is not None:
    missing_cols = [c for c in required_features if c not in data.columns]
    if missing_cols:
        st.error(
            "The dataset is missing required feature columns: "
            f"{missing_cols}. Please check the column list in the sidebar."
        )
    else:
        st.subheader(f"🔍 Data Preview - {data_label}")
        show_dataframe(data.head(10))

        if st.session_state.get("executed_model") != model_name:
            st.info(
                f"👈 Click **'🚀 Run Model'** in the sidebar to score the dataset using **{model_name}**."
            )
        else:
            # Load the selected trained pipeline
            pipe = load_model(MODEL_FILES[model_name])

            X = data[required_features]
            has_labels = target_col in data.columns

            # Run predictions
            with st.spinner(f"Scoring data with {model_name} ..."):
                predictions = pipe.predict(X)
                probabilities = pipe.predict_proba(X)[:, 1]

            results_df = data.copy()
            results_df["Predicted_Late_Delivery_Risk"] = predictions
            results_df["Predicted_Risk_Probability"] = probabilities.round(4)

            st.subheader(f"📈 Predictions - {model_name}")

            # --- Quick summary cards -------------------------------------------------
            total_orders = len(predictions)
            flagged = int(predictions.sum())
            flagged_pct = predictions.mean() * 100
            avg_risk = probabilities.mean() * 100

            s1, s2, s3 = st.columns(3)
            s1.metric("📦 Orders Scored", f"{total_orders:,}", help="Total number of orders in your dataset.")
            s2.metric(
                "🚨 Flagged Late",
                f"{flagged:,} ({flagged_pct:.1f}%)",
                help="Orders the model predicts will be delivered late.",
            )
            s3.metric(
                "📈 Avg Risk Probability",
                f"{avg_risk:.1f}%",
                help="Average model confidence of late delivery across all orders.",
            )

            # --- Risk tier labels + styled preview table -----------------------------
            results_df["Risk_Tier"] = results_df["Predicted_Risk_Probability"].apply(
                lambda p: "High" if p > 0.65 else ("Medium" if p >= 0.4 else "Low")
            )

            st.markdown("**Preview (first 20 rows):**")
            st.dataframe(
                results_df.head(20),
                column_config={
                    "Predicted_Late_Delivery_Risk": st.column_config.CheckboxColumn(
                        "Predicted Late",
                        help="Checked = order flagged as at risk of late delivery.",
                    ),
                    "Predicted_Risk_Probability": st.column_config.ProgressColumn(
                        "Predicted Risk Probability",
                        help="Model's confidence the order arrives late (0.00 = on-time, 1.00 = late).",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "Risk_Tier": st.column_config.SelectboxColumn(
                        "Risk Tier",
                        options=["Low", "Medium", "High"],
                        help="Low <40% · Medium 40-65% · High >65%.",
                    ),
                },
                hide_index=True,
                width="stretch",
            )

            # --- Download + risk tier legend -----------------------------------------
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )

            st.caption(
                "🟢 **Low** <40% · 🟡 **Medium** 40-65% · 🔴 **High** >65% - risk tiers "
                "help you prioritise which orders to act on first."
            )

            # ------------------------------------------------------------
            # (c) Evaluation metrics + (d) Confusion matrix / ROC / classification
            # ------------------------------------------------------------
            if has_labels:
                y_true = data[target_col]
                y_pred = predictions
                y_proba = probabilities

                st.markdown("---")
                st.subheader("📊 Evaluation Metrics")
                st.caption(
                    "How well this model detects late-delivery risk on your data. "
                    "Hover over any metric for a plain-English explanation."
                )

                metrics = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "AUC": roc_auc_score(y_true, y_proba),
                    "Precision": precision_score(y_true, y_pred),
                    "Recall": recall_score(y_true, y_pred),
                    "F1 Score": f1_score(y_true, y_pred),
                    "MCC": matthews_corrcoef(y_true, y_pred),
                }

                cols = st.columns(len(metrics))
                for col, (metric_name, value) in zip(cols, metrics.items()):
                    col.metric(metric_name, f"{value:.3f}", help=METRIC_HELP[metric_name])

                st.success(
                    f"**In plain English:** this model correctly classified "
                    f"**{metrics['Accuracy'] * 100:.1f}%** of orders. When it flagged an order as late, "
                    f"it was right **{metrics['Precision'] * 100:.1f}%** of the time, and it caught "
                    f"**{metrics['Recall'] * 100:.1f}%** of the orders that were actually late."
                )

                with st.expander("📖 Metric cheat sheet (what does each number mean?)"):
                    st.markdown("\n".join(f"- **{k}** - {v}" for k, v in METRIC_HELP.items()))

                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("🧮 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    cm_pct = cm / cm.sum()
                    fig_cm, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Predicted On-time", "Predicted Late"],
                        yticklabels=["Actual On-time", "Actual Late"],
                        ax=ax,
                        cbar_kws={"label": "Order count"},
                    )
                    ax.set_title(f"Confusion Matrix - {model_name}")
                    st.pyplot(fig_cm)
                    st.caption(
                        f"Correct: **{cm[0, 0] + cm[1, 1]:,}** ({(cm[0, 0] + cm[1, 1]) / cm.sum() * 100:.1f}%). "
                        "**TN** = on-time right · **FP** = on-time wrongly flagged · "
                        "**FN** = late order missed · **TP** = late order caught."
                    )

                with col_right:
                    st.subheader("📈 ROC Curve")
                    fig_roc = plot_roc_curve(y_true, y_proba, model_name, metrics["AUC"])
                    st.pyplot(fig_roc)
                    st.caption(
                        f"**AUC of {metrics['AUC']:.3f}** means the model ranks a random late order "
                        f"riskier than a random on-time order **{metrics['AUC'] * 100:.1f}%** of the time "
                        "(0.50 = coin flip, 1.00 = perfect)."
                    )

                st.subheader("📋 Classification Report")
                st.caption(
                    "Per-class performance. **Precision** = of orders predicted as this class, how many were "
                    "right. **Recall** = of orders actually in this class, how many were caught. "
                    "**F1-score** = balance of the two. **Support** = number of true examples."
                )
                report = classification_report(
                    y_true, y_pred, target_names=["On-time (0)", "Late (1)"], output_dict=True
                )
                report_df = pd.DataFrame(report).transpose().round(3)
                show_dataframe(report_df)

                st.subheader("🏆 Model Benchmark Comparison")
                st.caption(
                    "All 5 models were trained on the same 20,000-order sample and evaluated on identical "
                    "test data. The model you selected is marked with ⭐, and the best score in each column "
                    "with a green ✓."
                )

                # --- best model from the benchmark results (for the recommendation) ---
                comp_path = os.path.join(MODEL_DIR, "model_comparison_table.csv")
                best_model = None
                best_auc = None
                if os.path.exists(comp_path):
                    comp_df = pd.read_csv(comp_path)
                    best_row = comp_df.loc[comp_df["AUC"].idxmax()]
                    best_model = best_row["ML Model Name"]
                    best_auc = best_row["AUC"]

                    st.markdown(render_benchmark_table(comp_df, model_name), unsafe_allow_html=True)
                    st.caption("🏅 = best overall model (highest AUC). Rows are ranked by AUC.")

                    fig_comp = plot_model_comparison_benchmark(model_name)
                    if fig_comp:
                        _, mid, _ = st.columns([1, 3, 1])
                        with mid:
                            st.pyplot(fig_comp)
                        st.caption(
                            "Darker blue = better. The orange outline highlights the model you selected "
                            "in the sidebar."
                        )

                    winners = [
                        f"{m}: **{comp_df.loc[comp_df[m].idxmax(), 'ML Model Name']}** ({comp_df[m].max():.3f})"
                        for m in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
                    ]
                    st.info("**🏅 Per-metric winners:** " + " · ".join(winners))
                    if best_model:
                        st.success(
                            f"**Best model overall: {best_model}** (AUC {best_auc:.3f}) - the strongest "
                            "ability to separate late from on-time orders across all 5 trained models."
                        )
                else:
                    st.info("Benchmark file `model/model_comparison_table.csv` not found - comparison skipped.")

                # ------------------------------------------------------------
                # Visual Risk Analytics (Distribution, Tiers & Operational Insights)
                # ------------------------------------------------------------
                st.markdown("---")
                st.subheader("📊 Visual Risk Analytics & Operational Insights")
                st.caption(
                    "Three views of where the risk sits - by severity tier, by confidence score, "
                    "and by shipping mode (an actionable operational lever)."
                )

                # --- At-a-glance risk tier summary strip ----------------------------
                low_pct = (probabilities < 0.4).mean() * 100
                med_pct = ((probabilities >= 0.4) & (probabilities <= 0.65)).mean() * 100
                high_pct = (probabilities > 0.65).mean() * 100

                t1, t2, t3 = st.columns(3)
                for col, color, label, pct, desc in [
                    (t1, "#2ecc71", "🟢 Low Risk", low_pct, "orders very likely on-time"),
                    (t2, "#f1c40f", "🟡 Medium Risk", med_pct, "orders that may slip"),
                    (t3, "#e74c3c", "🔴 High Risk", high_pct, "orders most likely late"),
                ]:
                    with col:
                        st.markdown(
                            f"""<div style="background:{color}1a;border:1px solid {color};border-radius:12px;
                            padding:14px 10px;text-align:center;">
                            <div style="font-size:0.9rem;font-weight:700;color:#1f2933;">{label}</div>
                            <div style="font-size:1.9rem;font-weight:800;color:{color};line-height:1.1;">{pct:.1f}%</div>
                            <div style="font-size:0.8rem;color:#52606d;">{desc}</div></div>""",
                            unsafe_allow_html=True,
                        )

                # --- Chart 1: which orders are risky --------------------------------
                st.markdown("**🍩 Risk tiers - how many orders are risky**")
                _, mid, _ = st.columns([1, 2, 1])
                with mid:
                    st.pyplot(plot_risk_tier_donut(probabilities))
                st.caption("Share of orders in each risk tier: 🟢 Low <40%, 🟡 Medium 40-65%, 🔴 High >65%.")

                # --- Chart 2: how confident the model is ----------------------------
                st.markdown("**📊 Risk scores - how confident the model is**")
                _, mid, _ = st.columns([1, 2, 1])
                with mid:
                    st.pyplot(plot_risk_distribution(probabilities))
                st.caption(
                    "Green/amber/red bands show the Low, Medium and High tiers. Most orders bunch near "
                    "0 or 1 - the model is usually confident, and the red dashed line marks the 0.50 "
                    "cut-off used to flag an order as late."
                )

                # --- Chart 3: what drives the risk -----------------------------------
                st.markdown("**🚚 Shipping mode - what drives the risk**")
                fig_ship = plot_shipping_mode_risk(results_df)
                if fig_ship:
                    _, mid, _ = st.columns([1, 2, 1])
                    with mid:
                        st.pyplot(fig_ship)
                    st.caption("Bars are colour-coded by risk tier - a quick way to spot which shipping option to fix first.")
                else:
                    st.info("`Shipping Mode` column not found - chart skipped.")

                # ------------------------------------------------------------
                # Answer to the business question + data-driven recommendation
                # ------------------------------------------------------------
                st.markdown("---")
                st.subheader("🎯 Answer to the Business Question & Recommendations")
                show_business_question()

                # Risk drivers observed in THIS run's actual predictions
                risk_mode = None
                drivers = []
                if "Shipping Mode" in results_df.columns:
                    ship_risk = results_df.groupby("Shipping Mode")["Predicted_Late_Delivery_Risk"].mean() * 100
                    if len(ship_risk) > 0:
                        risk_mode = ship_risk.idxmax()
                        drivers.append(
                            f"🚚 **{risk_mode}** is the riskiest way to ship here - **{ship_risk.max():.0f}%** "
                            "of those orders get flagged late. If that's your standard/budget option, it's "
                            "the first lever to pull."
                        )
                if "Days for shipment (scheduled)" in results_df.columns:
                    corr = results_df["Days for shipment (scheduled)"].corr(
                        pd.Series(probabilities, index=results_df.index)
                    )
                    if pd.notna(corr) and abs(corr) >= 0.05:
                        direction = "longer" if corr > 0 else "shorter"
                        drivers.append(
                            f"⏱️ Orders promised a **{direction}** delivery window show **higher** risk - "
                            "check that the delivery speed you promise is one your process can actually keep."
                        )
                driver_md = "\n".join(f"- {d}" for d in drivers) if drivers else "- No single driver stood out in this run."

                # ---- 1. The verdict (big, clear, no jargon) ----
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border:1px solid #81c784;
                         border-radius:14px;padding:18px 22px;margin-bottom:8px;">
                        <div style="font-size:1.25rem;font-weight:700;color:#1b5e20;">
                            ✅ YES - we can predict risky deliveries before they ship
                        </div>
                        <div style="color:#2e7d32;font-size:0.95rem;margin-top:6px;line-height:1.6;">
                            Out of every 100 orders in this test, the <b>{model_name}</b> model got
                            <b>{metrics['Accuracy'] * 100:.0f}</b> right - and when it warned that an order
                            would arrive late, it was correct <b>{metrics['Precision'] * 100:.0f}</b> out of
                            100 times. A blind guess would only be right about 50 out of 100 times, so this
                            is a real, usable edge.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- 2. The numbers in everyday words ----
                st.markdown("### 📊 The numbers, in everyday words")
                plain_cards = [
                    ("🎯", "Overall accuracy", metrics["Accuracy"] * 100,
                     "Of every 100 orders, the model gets this many right."),
                    ("🚨", "Warnings are correct", metrics["Precision"] * 100,
                     "When it warns 'this will be late', it's right this often."),
                    ("🕵️", "Late orders caught", metrics["Recall"] * 100,
                     "Of 100 orders that really end up late, it spots this many in advance."),
                    ("⚖️", "Beats a coin flip", metrics["AUC"] * 100,
                     "In a head-to-head test, it picks the late order this % of the time."),
                ]
                c1, c2, c3, c4 = st.columns(4)
                for col, (emoji, title, pct, explain) in zip([c1, c2, c3, c4], plain_cards):
                    with col:
                        st.markdown(f"**{emoji} {title}**")
                        st.progress(min(max(pct / 100, 0.0), 1.0))
                        st.markdown(
                            f"<div style='font-size:1.7rem;font-weight:700;color:#1e3c72;"
                            f"line-height:1.1;'>{pct:.0f}%</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(explain)

                # ---- 3. Plain-English analogy ----
                st.info(
                    "**Think of it like a weather forecast.** No forecast is 100% certain - but a good "
                    "one tells you which days to plan around. This model is a *delivery forecast*: at the "
                    "moment an order is placed, it looks at how it will ship, what it's worth, where it's "
                    "going, how fast delivery was promised, and more - then flags the orders most likely "
                    "to be late, **before they ever leave the warehouse**. That turns a customer complaint "
                    "into a proactive fix."
                )

                # ---- 4. What's driving the risk (plain language) ----
                st.markdown("### 🔍 What's pushing orders late (from this data)")
                st.markdown(driver_md)

                # ---- 5. Recommendations ----
                st.markdown("### 📌 What to do next")
                recs = []
                if best_model and best_auc is not None:
                    recs.append(
                        f"**Put {best_model} to work at checkout.** It was the strongest of the 5 trained "
                        f"models (picking the late order correctly in {best_auc * 100:.0f}% of head-to-head "
                        "tests), so let it score every new order automatically."
                    )
                else:
                    recs.append(
                        f"**Put {model_name} to work at checkout.** It scored well in this run "
                        f"(AUC {metrics['AUC']:.3f}) - let it score every new order automatically."
                    )
                recs.append(
                    f"**Give the {flagged_pct:.0f}% of flagged orders special care** - faster shipping, "
                    "priority packing, or a friendly heads-up to the customer *before* they ask. An early "
                    "message costs far less than a refund or a complaint."
                )
                if risk_mode:
                    recs.append(
                        f"**Fix the riskiest shipping method first - {risk_mode}.** That's where the "
                        "largest share of late deliveries in this data comes from."
                    )
                else:
                    recs.append(
                        "**Fix the riskiest shipping methods first.** Look for the shipping option with "
                        "the highest late rate and start there."
                    )
                recs.append(
                    "**Tune the alarm to your costs.** If a late order hurts more than a false warning, "
                    "make the model more sensitive so it catches more; if warnings are costly, make it "
                    "stricter. 50% is a good default."
                )
                recs.append(
                    "**Keep it fresh.** Re-run the model on new data every few months so it stays in tune "
                    "with changing order patterns."
                )
                for i, r in enumerate(recs, start=1):
                    st.markdown(f"{i}. {r}")
            else:
                st.info(
                    f"Your file doesn't include the `{target_col}` label column, so evaluation metrics, "
                    "the confusion matrix and the classification report can't be computed. Predictions "
                    "are still shown above - add the label column and re-run to unlock the full evaluation."
                )

                # Compact business answer even without labels (based on predictions only)
                st.markdown("---")
                st.subheader("🎯 Answer to the Business Question & Recommendations")
                show_business_question()
                st.markdown(
                    f"""
                    **Answer - the model already separates risk at order time.** The **{model_name}**
                    model flagged **{flagged_pct:.1f}%** of your orders as at risk of late delivery, with
                    an average predicted risk probability of **{avg_risk:.1f}%**. High-risk orders can be
                    routed to expedited shipping or proactive customer communication at checkout.
                    Upload a file with the `{target_col}` label column to also see model accuracy, the
                    confusion matrix and a model comparison, so the recommendation can be backed by
                    evaluation metrics.
                    """
                )

st.markdown("---")
st.caption(
    "Built for BITS Pilani M.Tech (AIML/DSE) Machine Learning - Assignment 2. "
    "Dataset: DataCo Smart Supply Chain."
)
