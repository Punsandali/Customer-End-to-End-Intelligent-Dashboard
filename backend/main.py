from fastapi import FastAPI
import pandas as pd
import joblib
from feature_engineering import prepare_features
from textblob import TextBlob

app = FastAPI(title="Customer Intelligence API")

model = joblib.load("../models/purchase_model_dynamic.pkl")
cluster_model = joblib.load("../models/customer_cluster_model.pkl")
cluster_scaler = joblib.load("../models/cluster_scaler.pkl")
sentiment_model = joblib.load("../models/sentiment_model.pkl")
churn_model = joblib.load("../models/churn_model.pkl")
funnel_model = joblib.load("../models/funnel_model.pkl")


customers = pd.read_csv("../data/synthetic_customers.csv")
products = pd.read_csv("../data/synthetic_products.csv")
transactions = pd.read_csv("../data/synthetic_transactions.csv")
clickstream = pd.read_csv("../data/synthetic_clickstream.csv")
# reviews = pd.read_csv("../data/synthetic_reviews.csv")
reviews = pd.read_csv("../data/reviews.csv")
clickstreams=pd.read_csv("../data/clickstream.csv")

# ---- ADD THIS AT TOP OF FILE ----
CLUSTER_NAMES = {
    0: "Casual Customers",
    1: "Value-Conscious Regulars",
    2: "At-Risk Customers",
    3: "High-Frequency Revenue Drivers"
}


@app.get("/")
def home():
    return {"status": "API running"}
@app.get("/customers/count")
def get_customer_count():
    return{
        "total_customers":int(customers.shape[0])
    }

@app.get("/predict")
def predict():
    data = prepare_features(customers, transactions, clickstream, reviews)

    X = data.drop(columns=["CustomerID", "will_purchase_30d"], errors="ignore")
    data["purchase_probability"] = model.predict_proba(X)[:,1]

    return data.sort_values("purchase_probability", ascending=False).head(10).to_dict("records")

@app.get("/recommendations")
def recommendations():
    products["score"] = products["PopularityScore"] * products["Price"]
    return products.sort_values("score", ascending=False).head(10).to_dict("records")

@app.get("/segments")
def segments():
    features = prepare_features(customers, transactions, clickstream, reviews)

    cluster_features = [
        "total_transactions",
        "total_revenue",
        "avg_order_value",
        "total_sessions",
        "avg_session_duration",
        "avg_rating"
    ]

    X = features[cluster_features]
    X_scaled = cluster_scaler.transform(X)

    features["cluster"] = cluster_model.predict(X_scaled)
    features["segment_name"] = features["cluster"].map(CLUSTER_NAMES)

    return features[[
        "CustomerID",
        "cluster",
        "total_revenue",
        "segment_name",
        "total_transactions",
        "total_sessions"
    ]].to_dict("records")

@app.get("/click-heatmap")
def click_heatmap():
    # Count how many times each action happened per page
    heat = clickstream.groupby(["PageVisited","ActionType"]).size().reset_index(name="count")
    
    # Optional: calculate total visits per page to get drop-off percentages
    total_visits = clickstream.groupby("PageVisited").size().reset_index(name="total_visits")
    heat = heat.merge(total_visits, on="PageVisited")
    heat["percentage"] = (heat["count"] / heat["total_visits"] * 100).round(1)
    
    # This gives 'PageVisited', 'ActionType', 'count', 'percentage'
    return heat.to_dict("records")


@app.get("/review-sentiment")
def review_sentiment():
    reviews["sentiment"] = sentiment_model.predict(reviews["ReviewText"])
    sentiment_summary = reviews.groupby("sentiment").size().reset_index(name="count")
    return sentiment_summary.to_dict("records")

@app.get("/churn-risk")
def churn_risk():
    features = prepare_features(customers, transactions, clickstream, reviews)

    churn_features = [
        "total_transactions",
        "total_revenue",
        "avg_order_value",
        "total_sessions",
        "avg_session_duration",
        "avg_rating"
    ]

    X = features[churn_features]

    # Predict churn probability
    features["churn_probability"] = churn_model.predict_proba(X)[:, 1]

    # High-risk customers
    high_risk = features.sort_values(
        "churn_probability", ascending=False
    ).head(10)

    return high_risk[[
        "CustomerID",
        "churn_probability",
        "total_transactions",
        "total_sessions",
        "total_revenue"
    ]].to_dict("records")


@app.get("/funnel-risk")
def funnel_risk():
    # Funnel features
    funnel = clickstreams.groupby("CustomerID").agg(
        total_sessions=("SessionID", "count"),
        total_time=("Duration", "sum"),
        product_page_views=("PageVisited", lambda x: (x == "Product").sum()),
        cart_views=("PageVisited", lambda x: (x == "Cart").sum())
    ).reset_index()

    features = customers.merge(
        funnel, on="CustomerID", how="left"
    ).fillna(0)

    X = features[
        ["total_sessions", "total_time", "product_page_views", "cart_views"]
    ]

    # Drop-off probability
    features["dropoff_probability"] = 1 - funnel_model.predict_proba(X)[:, 1]

    return features.sort_values(
        "dropoff_probability", ascending=False
    ).head(10)[
        ["CustomerID", "dropoff_probability", "cart_views", "total_time"]
    ].to_dict("records")


@app.get("/product-alerts")
def product_alerts():
    reviews["sentiment"] = sentiment_model.predict(reviews["ReviewText"])
    reviews["ReviewDate"] = pd.to_datetime(reviews["ReviewDate"])

    last_7_days = reviews[
        reviews["ReviewDate"] >= reviews["ReviewDate"].max() - pd.Timedelta(days=7)
    ]

    alerts = (
        last_7_days[last_7_days["sentiment"] == "Negative"]
        .groupby("ProductID")
        .size()
        .reset_index(name="negative_reviews")
    )

    alerts = alerts[alerts["negative_reviews"] > 5]


    alerts = alerts.merge(
        products[["ProductID", "Category"]],
        on="ProductID",
        how="left"
    )


    # ✅ ALWAYS return a list
    if alerts.empty:
        return []

    return alerts.to_dict("records")
