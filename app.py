import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .kpi-box {
        background: #f7f9fc;
        border-left: 5px solid #1a6fbf;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 13px; color: #666; margin: 0; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #1a6fbf; margin: 0; }
    .kpi-sub   { font-size: 12px; color: #999; margin: 0; }
    .insight-box {
        background: #eef6ff;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #1a6fbf;
    }
    .warn-box {
        background: #fff4e5;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #f0a500;
    }
    .good-box {
        background: #edfaf1;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #27ae60;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["diagnosis"] = raw.target                                      # 1=Benign, 0=Malignant
    df["diagnosis_label"] = df["diagnosis"].map({1: "Benign", 0: "Malignant"})
    return df, list(raw.feature_names)

@st.cache_resource
def train_model(test_size):
    df, features = load_data()
    X = df[features]
    y = df["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, features, scaler

df, features = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Healthcare Analytics")
st.sidebar.caption("Breast Cancer Diagnostic Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "📈 Exploratory Analysis", "🤖 Predictive Model", "📋 Business Insights"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("Breast Cancer Wisconsin (Diagnostic)")
st.sidebar.markdown(f"- Records: **{len(df):,}**")
st.sidebar.markdown(f"- Features: **{len(features)}**")
st.sidebar.markdown(f"- Classes: **Benign / Malignant**")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Plotly · Scikit-learn")


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("High-level summary of dataset quality, composition, and structure.")
    st.markdown("---")

    # KPI row
    total       = len(df)
    malignant   = (df["diagnosis"] == 0).sum()
    benign      = (df["diagnosis"] == 1).sum()
    missing     = df[features].isnull().sum().sum()
    mal_pct     = malignant / total * 100
    ben_pct     = benign / total * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-box">
            <p class="kpi-label">Total Patient Records</p>
            <p class="kpi-value">{total:,}</p>
            <p class="kpi-sub">All diagnostic entries</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-box">
            <p class="kpi-label">Malignant Cases</p>
            <p class="kpi-value" style="color:#e74c3c">{malignant}</p>
            <p class="kpi-sub">{mal_pct:.1f}% of total</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-box">
            <p class="kpi-label">Benign Cases</p>
            <p class="kpi-value" style="color:#27ae60">{benign}</p>
            <p class="kpi-sub">{ben_pct:.1f}% of total</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-box">
            <p class="kpi-label">Missing Values</p>
            <p class="kpi-value" style="color:{'#e74c3c' if missing>0 else '#27ae60'}">{missing}</p>
            <p class="kpi-sub">{'Requires imputation' if missing > 0 else 'Clean dataset ✓'}</p></div>""",
            unsafe_allow_html=True)

    st.markdown("#### Class Distribution")
    col_pie, col_bar = st.columns(2)
    with col_pie:
        pie_df = pd.DataFrame({"Diagnosis": ["Malignant", "Benign"], "Count": [malignant, benign]})
        fig = px.pie(
            pie_df, values="Count", names="Diagnosis", color="Diagnosis",
            color_discrete_map={"Malignant": "#e74c3c", "Benign": "#27ae60"},
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        diag_counts = df["diagnosis_label"].value_counts().reset_index()
        diag_counts.columns = ["Diagnosis", "Count"]
        fig2 = px.bar(
            diag_counts, x="Diagnosis", y="Count",
            color="Diagnosis",
            color_discrete_map={"Malignant": "#e74c3c", "Benign": "#27ae60"},
            text="Count",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Data Quality Report")
    quality = pd.DataFrame({
        "Feature": features,
        "Non-Null Count": [df[f].notna().sum() for f in features],
        "Missing": [df[f].isnull().sum() for f in features],
        "Missing %": [f"{df[f].isnull().mean()*100:.1f}%" for f in features],
        "Min": [f"{df[f].min():.4f}" for f in features],
        "Max": [f"{df[f].max():.4f}" for f in features],
        "Mean": [f"{df[f].mean():.4f}" for f in features],
        "Std": [f"{df[f].std():.4f}" for f in features],
    })
    st.dataframe(quality, use_container_width=True, height=400)

    st.markdown("#### Raw Data Preview")
    n_rows = st.slider("Rows to display", 5, 50, 10)
    diagnosis_filter = st.multiselect(
        "Filter by diagnosis", ["Benign", "Malignant"],
        default=["Benign", "Malignant"]
    )
    filtered = df[df["diagnosis_label"].isin(diagnosis_filter)]
    st.dataframe(filtered.head(n_rows), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Exploratory Analysis":
    st.title("📈 Exploratory Analysis")
    st.markdown("Identify trends, patterns, and distributions across clinical features.")
    st.markdown("---")

    # Feature distribution
    st.markdown("#### Feature Distribution by Diagnosis")
    selected_feature = st.selectbox("Select a feature", features, index=0)
    col_hist, col_box = st.columns(2)

    with col_hist:
        fig = px.histogram(
            df, x=selected_feature, color="diagnosis_label",
            color_discrete_map={"Malignant": "#e74c3c", "Benign": "#27ae60"},
            barmode="overlay", opacity=0.7, nbins=40,
            title=f"Distribution: {selected_feature}",
        )
        fig.update_layout(legend_title="Diagnosis")
        st.plotly_chart(fig, use_container_width=True)

    with col_box:
        fig2 = px.violin(
            df, y=selected_feature, x="diagnosis_label", color="diagnosis_label",
            color_discrete_map={"Malignant": "#e74c3c", "Benign": "#27ae60"},
            box=True, points="outliers",
            title=f"Violin Plot: {selected_feature}",
            labels={"diagnosis_label": "Diagnosis"},
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    top_n = st.slider("Number of top features", 10, 30, 15)
    corr = df[features[:top_n]].corr()
    fig3 = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Correlation Matrix (top {top_n} features)",
    )
    fig3.update_layout(height=550)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Mean Feature Values: Malignant vs Benign")
    mean_df = df.groupby("diagnosis_label")[features].mean().T.reset_index()
    mean_df.columns = ["Feature", "Benign", "Malignant"]
    mean_df_melted = mean_df.melt(id_vars="Feature", var_name="Diagnosis", value_name="Mean Value")

    top_features = st.multiselect(
        "Select features to compare",
        features,
        default=list(features[:8]),
    )
    if top_features:
        filtered_means = mean_df_melted[mean_df_melted["Feature"].isin(top_features)]
        fig4 = px.bar(
            filtered_means, x="Feature", y="Mean Value", color="Diagnosis",
            barmode="group",
            color_discrete_map={"Malignant": "#e74c3c", "Benign": "#27ae60"},
            title="Mean Clinical Measurements by Diagnosis",
        )
        fig4.update_xaxes(tickangle=30)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Scatter Analysis")
    col_x, col_y = st.columns(2)
    x_feat = col_x.selectbox("X-axis feature", features, index=0)
    y_feat = col_y.selectbox("Y-axis feature", features, index=2)
    fig5 = px.scatter(
        df, x=x_feat, y=y_feat, color="diagnosis_label",
        color_discrete_map={"Malignant": "#e74c3c", "Benign": "#27ae60"},
        opacity=0.7, title=f"{x_feat} vs {y_feat}",
    )
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE MODEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Model":
    st.title("🤖 Predictive Model")
    st.markdown("Random Forest classifier trained to predict malignant vs benign diagnosis.")
    st.markdown("---")

    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    clf, X_test, y_test, feat_names, scaler = train_model(test_size)
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 0]   # probability of malignant (class 0)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec  = recall_score(y_test, y_pred, pos_label=0)
    f1   = f1_score(y_test, y_pred, pos_label=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=0)
    roc_auc = auc(fpr, tpr)

    # KPI row
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, color in [
        (m1, "Accuracy",  f"{acc:.2%}",  "#1a6fbf"),
        (m2, "Precision", f"{prec:.2%}", "#8e44ad"),
        (m3, "Recall",    f"{rec:.2%}",  "#e74c3c"),
        (m4, "F1 Score",  f"{f1:.2%}",   "#f0a500"),
        (m5, "AUC-ROC",   f"{roc_auc:.4f}", "#27ae60"),
    ]:
        col.markdown(f"""<div class="kpi-box">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value" style="color:{color}">{val}</p></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Malignant (0)", "Benign (1)"]
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale="Blues",
            title="Predicted vs Actual",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown("#### ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {roc_auc:.4f}",
            line=dict(color="#1a6fbf", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="grey", dash="dash"),
            name="Random Classifier",
            showlegend=True,
        ))
        fig_roc.update_layout(
            title="ROC Curve — Malignant Detection",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("#### Feature Importance")
    importances = pd.DataFrame({
        "Feature": feat_names,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    top_k = st.slider("Show top N features", 5, 30, 15)
    fig_imp = px.bar(
        importances.head(top_k), x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Blues",
        title=f"Top {top_k} Predictive Features",
    )
    fig_imp.update_layout(yaxis={"autorange": "reversed"}, height=420)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Detailed Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=["Malignant", "Benign"], output_dict=True
    )
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — BUSINESS INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Business Insights":
    st.title("📋 Business Insights")
    st.markdown("Key findings, risk factor analysis, and recommendations for clinical teams.")
    st.markdown("---")

    # Compute insights
    clf, X_test, y_test, feat_names, scaler = train_model(0.20)
    importances = pd.DataFrame({
        "Feature": feat_names,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    top3 = importances.head(3)["Feature"].tolist()

    mal_df  = df[df["diagnosis"] == 0]
    ben_df  = df[df["diagnosis"] == 1]
    mal_pct = len(mal_df) / len(df) * 100
    ben_pct = len(ben_df) / len(df) * 100

    acc = accuracy_score(y_test, clf.predict(X_test))

    st.markdown("### Executive Summary")
    st.markdown(f"""
    <div class="insight-box">
    This dashboard analyses <strong>{len(df):,} patient diagnostic records</strong> from the
    Breast Cancer Wisconsin dataset. Out of all cases, <strong>{len(mal_df)} ({mal_pct:.1f}%)</strong>
    are malignant and <strong>{len(ben_df)} ({ben_pct:.1f}%)</strong> are benign.
    A Random Forest predictive model achieves <strong>{acc:.2%} accuracy</strong> in identifying
    malignant cases, providing a reliable decision-support tool for clinical teams.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Findings")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class="warn-box">
        <strong>⚠️ Class Imbalance Detected</strong><br>
        Malignant cases ({mal_pct:.1f}%) are underrepresented compared to benign ({ben_pct:.1f}%).
        Clinical workflows should account for this imbalance to avoid under-diagnosis of malignant cases.
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
        <strong>🔬 Top Predictive Risk Factors</strong><br>
        The three most clinically significant features are:<br>
        1. <strong>{top3[0]}</strong><br>
        2. <strong>{top3[1]}</strong><br>
        3. <strong>{top3[2]}</strong><br>
        These features should be prioritised in diagnostic reporting workflows.
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="good-box">
        <strong>✅ Model Performance</strong><br>
        The Random Forest model achieves <strong>{acc:.2%} overall accuracy</strong> on held-out test data,
        demonstrating strong generalisation capability and suitability as a clinical decision-support tool.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="good-box">
        <strong>✅ Data Quality</strong><br>
        Zero missing values detected across all 30 clinical features. Dataset is complete and
        ready for downstream analytics pipelines without imputation.
        </div>""", unsafe_allow_html=True)

    st.markdown("### Recommendations")
    recs = [
        ("🏥", "Prioritise Screening",
         f"Patients with high {top3[0]} and {top3[1]} readings should be flagged for immediate review by clinical staff."),
        ("📊", "Integrate into BI Reporting",
         "The predictive model output can be embedded into existing BI dashboards to surface high-risk cases automatically."),
        ("🔄", "Periodic Model Retraining",
         "Retrain the model quarterly as new patient data is collected to maintain predictive accuracy and adapt to data drift."),
        ("📋", "Standardise Data Collection",
         "Ensure all 30 clinical features are captured consistently across zones and regions to maintain data integrity and governance compliance."),
        ("⚕️", "Clinical Validation",
         "Model predictions should supplement, not replace, expert clinical judgement. All flagged cases must be reviewed by a qualified clinician."),
    ]

    for icon, title, desc in recs:
        st.markdown(f"""
        <div class="insight-box">
        <strong>{icon} {title}</strong><br>{desc}
        </div>""", unsafe_allow_html=True)

    st.markdown("### Feature Risk Profile")
    mean_diff = (
        df.groupby("diagnosis_label")[features].mean().T
        .assign(Delta=lambda x: x["Malignant"] - x["Benign"])
        .reset_index()
        .rename(columns={"index": "Feature"})
        .sort_values("Delta", ascending=False)
    )
    fig_delta = px.bar(
        mean_diff, x="Feature", y="Delta",
        color="Delta",
        color_continuous_scale=["#27ae60", "#ffffff", "#e74c3c"],
        title="Mean Value Difference (Malignant − Benign) per Feature",
    )
    fig_delta.update_xaxes(tickangle=45)
    fig_delta.update_layout(height=420)
    st.plotly_chart(fig_delta, use_container_width=True)
    st.caption("Positive values (red) indicate features elevated in malignant cases — key risk signals for clinical teams.")
