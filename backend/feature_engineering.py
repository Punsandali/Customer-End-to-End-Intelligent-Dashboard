import pandas as pd

def prepare_features(customers, transactions, clickstream, reviews):
    # Transactions
    transactions["Revenue"] = transactions["Quantity"] * transactions["Price"]
    txn = transactions.groupby("CustomerID").agg(
        total_transactions=("TransactionID", "count"),
        total_revenue=("Revenue", "sum"),
        avg_order_value=("Revenue", "mean")
    )

    # Clickstream
    click = clickstream.groupby("CustomerID").agg(
        total_sessions=("SessionID", "count"),
        avg_session_duration=("Duration", "mean")
    )

    # Reviews
    rev = reviews.groupby("CustomerID").agg(
        avg_rating=("Rating", "mean"),
        review_count=("ReviewID", "count")
    )

    # Merge
    model_data = (
        customers
        .merge(txn, on="CustomerID", how="left")
        .merge(click, on="CustomerID", how="left")
        .merge(rev, on="CustomerID", how="left")
    )

    model_data.fillna(0, inplace=True)
    return model_data
