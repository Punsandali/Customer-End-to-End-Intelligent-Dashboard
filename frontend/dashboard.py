import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)

st.title("🧠 Customer Intelligence & Growth Dashboard")
st.caption("ML-powered insights for marketing, product & business teams")

API = "http://127.0.0.1:8000"

# -------------------------------------------------
# Fetch Data (Safe Calls)
# -------------------------------------------------
def fetch_data(endpoint):
    try:
        return requests.get(f"{API}/{endpoint}").json()
    except Exception:
        return []

preds = pd.DataFrame(fetch_data("predict"))
products = pd.DataFrame(fetch_data("recommendations"))
segments = pd.DataFrame(fetch_data("segments"))
heat = pd.DataFrame(fetch_data("click-heatmap"))
sentiment = pd.DataFrame(fetch_data("review-sentiment"))
churn = pd.DataFrame(fetch_data("churn-risk"))
funnel = pd.DataFrame(fetch_data("funnel-risk"))
alerts = pd.DataFrame(fetch_data("product-alerts"))
count = fetch_data("/customers/count")
# -------------------------------------------------
# KPI OVERVIEW
# -------------------------------------------------
st.markdown("## 📊 Business Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "👥 Total Customers",
    count.get("total_customers", 0)
)


k2.metric(
    "🚨 High Churn Risk",
    f"{len(churn)} customers"
)

neg_count = (
    sentiment[sentiment["sentiment"] == "Negative"]["count"].sum()
    if not sentiment.empty else 0
)
k3.metric("💬 Negative Reviews", int(neg_count))

k4.metric(
    "🛍️ Top Products",
    len(products)
)

st.divider()

# -------------------------------------------------
# PURCHASE PREDICTION
# -------------------------------------------------
st.header("🔮 Customers Likely to Purchase Soon")
st.info(
    "These customers show strong behavioral and engagement signals "
    "and are most likely to make a purchase in the next 30 days."
)

with st.expander("📄 View customer-level predictions"):
    st.dataframe(preds)

# -------------------------------------------------
# PRODUCT RECOMMENDATIONS
# -------------------------------------------------
st.header("🛍️ High-Impact Product Recommendations")
st.caption("Based on popularity and revenue contribution")

with st.expander("📄 View recommended products"):
    st.dataframe(products)

# -------------------------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------------------------
st.header("👥 Customer Segmentation (ML-driven)")
st.markdown("""
**Segment Meaning**
- 🟢 **High-Frequency Revenue Drivers** – Loyal & high spenders  
- 🟡 **Value-Conscious Regulars** – Respond to offers  
- 🔴 **At-Risk Customers** – Require re-engagement  
- 🔵 **Casual Customers** – Low engagement  
""")

fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(
    data=segments,
    x="total_transactions",
    y="total_revenue",
    hue="segment_name",
    palette="Set2",
    ax=ax
)
plt.title("Customer Value Distribution")
st.pyplot(fig)

# -------------------------------------------------
# CLICKSTREAM HEATMAP
# -------------------------------------------------
st.header("🔥 User Behavior & Friction Analysis")
st.caption("Identify where users hesitate or abandon the journey")

if not heat.empty:
    heat_pivot = heat.pivot(
        index="PageVisited",
        columns="ActionType",
        values="percentage"
    )

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(
        heat_pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        ax=ax
    )
    plt.title("User Action Distribution per Page (%)")
    st.pyplot(fig)

    worst_page = heat.sort_values("percentage").head(1)
    st.warning(
        f"⚠️ Highest friction detected on **{worst_page.iloc[0]['PageVisited']}** page"
    )

# -------------------------------------------------
# REVIEW SENTIMENT
# -------------------------------------------------
st.header("💬 Customer Voice & Brand Sentiment")
st.caption("Monitoring customer emotions from reviews")

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="sentiment",
    y="count",
    data=sentiment,
    palette=["red", "gray", "green"],
    ax=ax
)
plt.title("Sentiment Distribution")
st.pyplot(fig)

st.markdown("""
**How to use this:**
- 🔴 Rising negatives → investigate product quality
- 🟡 Neutral → improve engagement
- 🟢 Positive → promote & upsell
""")

# -------------------------------------------------
# CHURN RISK
# -------------------------------------------------
st.header("🚨 Customer Retention Watchlist")
st.error(
    "These customers are highly likely to disengage. Immediate action recommended."
)

with st.expander("📄 View high-risk customers"):
    st.dataframe(churn)

# -------------------------------------------------
# FUNNEL DROP-OFF
# -------------------------------------------------
st.header("🚪 Funnel Drop-Off Risk")
st.caption("Customers abandoning before checkout")

with st.expander("📄 View funnel risk customers"):
    st.dataframe(funnel)

st.markdown("""
### 🎯 Recommended Actions
- 🎁 Limited-time discount
- ⏰ Reminder notification
- 🛒 Simplify checkout UX
""")

# -------------------------------------------------
# PRODUCT QUALITY ALERTS
# -------------------------------------------------
st.header("🚨 Product Quality Alerts")

if not alerts.empty:
    st.error("⚠️ Immediate attention required: products with rising negative reviews")
    st.dataframe(alerts)
else:
    st.success("✅ All products operating within healthy sentiment range")

st.divider()

st.caption("© Customer Intelligence Platform | ML + Analytics")
