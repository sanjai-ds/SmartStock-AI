from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# SmartStock-AI | Inventory Intelligence Dashboard
# This dashboard visualizes precomputed predictions and
# recommendations stored in results/smartstock_dashboard_data.csv.
# It does not retrain or run the PyTorch model on each page load.
# ------------------------------------------------------------

st.set_page_config(
    page_title="SmartStock-AI | Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_PATH = RESULTS_DIR / "smartstock_dashboard_data.csv"
IMPORTANCE_PATH = RESULTS_DIR / "mlp_feature_importance.csv"
CONFIG_PATH = ROOT / "models" / "model_config.json"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(ellipse at 15% 0%, #172554 0%, #0b1220 38%, #070b14 100%);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111c31 100%);
        border-right: 1px solid rgba(148,163,184,.16);
    }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #f8fafc !important; }
    p, label, span { color: #dbe4f0; }
    .hero {
        padding: 1.5rem 1.7rem; border: 1px solid rgba(148,163,184,.22);
        border-radius: 22px; background: linear-gradient(120deg, rgba(30,64,175,.34), rgba(15,23,42,.78));
        margin-bottom: 1.2rem;
    }
    .eyebrow { color: #93c5fd; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
    .hero-title { color: #fff; font-family: 'Manrope',sans-serif; font-size: 2.15rem; font-weight: 800; margin: .25rem 0; }
    .hero-sub { color: #cbd5e1; font-size: 1rem; margin: 0; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20,32,54,.96), rgba(13,22,39,.96));
        border: 1px solid rgba(148,163,184,.18); border-radius: 17px;
        padding: 16px 18px; box-shadow: 0 10px 28px rgba(0,0,0,.18);
    }
    [data-testid="stMetricLabel"] { color: #a9bad0 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-family: 'Manrope',sans-serif; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] { border-radius: 10px; padding: 10px 16px; }
    .section-note { color: #9fb0c6; font-size: .9rem; }
    .status-pill { display:inline-block; padding:5px 10px; border-radius:99px; background:#123b31; color:#86efac; font-size:.78rem; font-weight:700; }
    [data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.2); border-radius: 14px; }
    .footer { color:#7f91aa; text-align:center; font-size:.8rem; padding:1.5rem 0 .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- Load data --------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "Product_Name", "Week", "Current_Stock", "Predicted_Demand",
        "Reorder_Level", "Reorder_Quantity", "Coverage_Ratio",
        "Stockout_Risk", "Recommendation", "Recommended_Quantity",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Dashboard CSV is missing columns: " + ", ".join(sorted(missing)))
    frame["Week"] = pd.to_datetime(frame["Week"], errors="coerce")
    numeric = [
        "Current_Stock", "Predicted_Demand", "Reorder_Level",
        "Reorder_Quantity", "Coverage_Ratio", "Recommended_Quantity",
        "Risk_Score",
    ]
    for col in numeric:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["Product_Name"] = frame["Product_Name"].fillna("Unknown")
    frame["Stockout_Risk"] = frame["Stockout_Risk"].fillna("UNKNOWN").str.upper()
    frame["Recommendation"] = frame["Recommendation"].fillna("REVIEW").str.upper()
    return frame

if not DATA_PATH.exists():
    st.error(f"Dashboard data file not found: {DATA_PATH}")
    st.info("Place smartstock_dashboard_data.csv inside the project's results folder.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.markdown("## 📦 SmartStock-AI")
    st.caption("Inventory intelligence workspace")
    st.divider()
    st.markdown("### Filters")

    product_options = ["All products"] + sorted(df["Product_Name"].unique().tolist())
    chosen_products = st.multiselect(
        "Products",
        options=product_options[1:],
        default=[],
        help="Leave empty to include every product.",
    )
    risk_options = sorted(df["Stockout_Risk"].dropna().unique().tolist())
    chosen_risks = st.multiselect(
        "Stockout risk",
        options=risk_options,
        default=risk_options,
    )
    action_options = sorted(df["Recommendation"].dropna().unique().tolist())
    chosen_actions = st.multiselect(
        "Recommended action",
        options=action_options,
        default=action_options,
    )

    valid_dates = df["Week"].dropna()
    if not valid_dates.empty:
        min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
        date_range = st.date_input(
            "Observation period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    st.divider()
    st.markdown("### Model information")
    st.markdown("**Forecast model**  \nPyTorch MLP")
    st.markdown("**Dashboard mode**  \nPrecomputed model outputs")
    st.markdown(
        '<span class="status-pill">● Data loaded</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Source: {DATA_PATH.name}")

filtered = df.copy()
if chosen_products:
    filtered = filtered[filtered["Product_Name"].isin(chosen_products)]
if chosen_risks:
    filtered = filtered[filtered["Stockout_Risk"].isin(chosen_risks)]
else:
    filtered = filtered.iloc[0:0]
if chosen_actions:
    filtered = filtered[filtered["Recommendation"].isin(chosen_actions)]
else:
    filtered = filtered.iloc[0:0]
if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[filtered["Week"].between(start_date, end_date)]

# -------------------------- Header --------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Retail operations · Forecasting · Replenishment</div>
      <div class="hero-title">SmartStock-AI</div>
      <p class="hero-sub">Demand forecasting and inventory decision support — turning model outputs into clear replenishment priorities.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# -------------------------- KPI row --------------------------
total_records = len(filtered)
product_count = filtered["Product_Name"].nunique()
high_count = int((filtered["Stockout_Risk"] == "HIGH").sum())
reorder_count = int((filtered["Recommendation"] == "REORDER").sum())
avg_coverage = filtered["Coverage_Ratio"].mean()
avg_demand = filtered["Predicted_Demand"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Products in view", f"{product_count:,}")
k2.metric("Inventory records", f"{total_records:,}")
k3.metric("High-risk records", f"{high_count:,}")
k4.metric("Reorder actions", f"{reorder_count:,}")
k5.metric("Average predicted demand", f"{avg_demand:,.1f}" if pd.notna(avg_demand) else "—")

st.caption(
    f"Showing {total_records} records across {product_count} products. "
    "KPIs and charts update with the selected filters."
)

# -------------------------- Main tabs --------------------------
overview_tab, demand_tab, actions_tab, data_tab, model_tab = st.tabs(
    ["Overview", "Demand analysis", "Replenishment actions", "Inventory data", "Model insights"]
)

with overview_tab:
    left, right = st.columns([1.05, 1.4], gap="large")
    with left:
        st.subheader("Stockout risk distribution")
        risk_counts = (
            filtered["Stockout_Risk"].value_counts()
            .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
            .rename_axis("Risk level").reset_index(name="Records")
        )
        fig = px.pie(
            risk_counts, names="Risk level", values="Records", hole=.68,
            color="Risk level",
            color_discrete_map={"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#34d399"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=20,l=10,r=10),
            showlegend=False, height=330,
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Inventory action mix")
        action_counts = filtered["Recommendation"].value_counts().rename_axis("Action").reset_index(name="Records")
        fig = px.bar(
            action_counts, x="Action", y="Records", color="Action",
            color_discrete_map={"REORDER": "#60a5fa", "NO REORDER": "#34d399"},
            text="Records",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=25,b=20,l=10,r=10),
            showlegend=False, height=330, yaxis_title="Number of records", xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Priority products to review")
    priority = filtered[filtered["Stockout_Risk"] == "HIGH"].copy()
    if not priority.empty:
        priority = priority.sort_values(["Coverage_Ratio", "Predicted_Demand"], ascending=[True, False])
        cols = ["Product_Name", "Week", "Current_Stock", "Predicted_Demand", "Coverage_Ratio", "Recommendation", "Recommended_Quantity"]
        st.dataframe(
            priority[cols].style.format({
                "Current_Stock": "{:,.0f}", "Predicted_Demand": "{:,.1f}",
                "Coverage_Ratio": "{:.2f}", "Recommended_Quantity": "{:,.1f}",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.success("No high-risk records in the current filter selection.")

with demand_tab:
    st.subheader("Predicted demand by observation week")
    trend = (
        filtered.dropna(subset=["Week"])
        .groupby("Week", as_index=False)
        .agg(
            Average_Predicted_Demand=("Predicted_Demand", "mean"),
            Average_Current_Stock=("Current_Stock", "mean"),
        )
        .sort_values("Week")
    )
    if not trend.empty:
        fig = px.line(
            trend, x="Week", y="Average_Predicted_Demand", markers=True,
            labels={"Average_Predicted_Demand": "Average predicted demand", "Week": "Observation week"},
        )
        fig.update_traces(line=dict(width=3, color="#60a5fa"), marker=dict(size=7))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            margin=dict(t=20,b=20,l=10,r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart shows predicted demand only. The dashboard CSV does not contain an actual-demand column, so an actual-versus-predicted comparison is not displayed.")
    st.subheader("Products with highest average predicted demand")
    product_rank = (
        filtered.groupby("Product_Name", as_index=False)["Predicted_Demand"]
        .mean().sort_values("Predicted_Demand", ascending=False).head(12)
    )
    fig = px.bar(
        product_rank.sort_values("Predicted_Demand"),
        x="Predicted_Demand", y="Product_Name", orientation="h",
        labels={"Predicted_Demand": "Average predicted demand", "Product_Name": ""},
        text_auto=".1f",
    )
    fig.update_traces(marker_color="#818cf8")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=420,
        margin=dict(t=15,b=20,l=10,r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with actions_tab:
    st.subheader("Replenishment decision queue")
    actions = filtered.copy()
    actions["_priority"] = actions["Stockout_Risk"].map({"HIGH": 0, "MEDIUM": 1, "LOW": 2}).fillna(3)
    actions = actions.sort_values(["_priority", "Coverage_Ratio"], ascending=[True, True]).drop(columns="_priority")
    cols = [
        "Product_Name", "Week", "Current_Stock", "Predicted_Demand",
        "Reorder_Level", "Reorder_Quantity", "Coverage_Ratio",
        "Stockout_Risk", "Recommendation", "Recommended_Quantity",
    ]
    st.dataframe(
        actions[cols].style.format({
            "Current_Stock": "{:,.0f}", "Predicted_Demand": "{:,.1f}",
            "Reorder_Level": "{:,.0f}", "Reorder_Quantity": "{:,.0f}",
            "Coverage_Ratio": "{:.2f}", "Recommended_Quantity": "{:,.1f}",
        }),
        use_container_width=True, hide_index=True,
    )
    csv_bytes = actions[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered recommendations",
        data=csv_bytes,
        file_name="smartstock_filtered_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

with data_tab:
    st.subheader("Inventory records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download current filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="smartstock_filtered_data.csv",
        mime="text/csv",
    )

with model_tab:
    st.subheader("Model evaluation")
    st.markdown(
        "The selected forecasting model is **PyTorch MLP**. The following metrics are read from the saved model configuration."
    )
    if CONFIG_PATH.exists():
        try:
            import json
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test MAE", f"{config.get('test_mae', float('nan')):.2f}")
            m2.metric("Test RMSE", f"{config.get('test_rmse', float('nan')):.2f}")
            m3.metric("Test WAPE", f"{config.get('test_wape', float('nan')):.2f}%")
            m4.metric("Test sMAPE", f"{config.get('test_smape', float('nan')):.2f}%")
            with st.expander("Model input features"):
                st.write(config.get("input_features", []))
        except Exception as exc:
            st.warning(f"Could not read model configuration: {exc}")
    else:
        st.info("Model configuration file was not found.")

    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH)
        if {"Feature", "Importance"}.issubset(importance.columns):
            st.subheader("Feature permutation importance")
            importance = importance.sort_values("Importance", ascending=True)
            fig = px.bar(
                importance, x="Importance", y="Feature", orientation="h",
                labels={"Importance": "Permutation importance", "Feature": ""},
            )
            fig.update_traces(marker_color="#38bdf8")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=450,
                margin=dict(t=15,b=20,l=10,r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Negative permutation importance means shuffling that feature did not worsen the evaluation score; it is not evidence of a negative causal effect.")

st.markdown(
    '<div class="footer">SMARTSTOCK-AI · Demand forecasting · Inventory risk monitoring · Replenishment decision support</div>',
    unsafe_allow_html=True,
)
