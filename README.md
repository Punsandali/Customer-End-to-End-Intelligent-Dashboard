🧠 Customer Intelligence & Growth Analytics Platform

An end-to-end Machine Learning–powered Customer Intelligence system that transforms raw customer, transaction, clickstream, and review data into actionable business insights.

Built as a real-world analytics product, this platform helps businesses understand customer behavior, predict outcomes, and make data-driven decisions.

🚀 Key Features
🔮 Purchase Prediction

Predicts customers most likely to purchase in the next 30 days

Uses behavioral, transactional, and engagement features

👥 Customer Segmentation

ML-driven clustering of customers into meaningful business segments:

High-Frequency Revenue Drivers

Value-Conscious Regulars

At-Risk Customers

Casual Customers

🚨 Churn Risk Detection

Identifies customers likely to churn

Helps prioritize retention strategies

🚪 Funnel Drop-Off Analysis

Detects customers abandoning before checkout

Highlights friction points in the customer journey

🔥 Clickstream Heatmap

Visualizes user actions across pages

Helps identify UX bottlenecks and behavior patterns

💬 Review Sentiment Analysis

Analyzes customer reviews using NLP

Tracks positive, neutral, and negative sentiment trends

🚨 Product Quality Alerts

Flags products with a spike in negative reviews

Supports faster product quality interventions

🏗️ System Architecture
CSV Data Sources
   ↓
Feature Engineering
   ↓
Machine Learning Models (Sklearn)
   ↓
FastAPI Backend (REST APIs)
   ↓
Streamlit Interactive Dashboard

🛠️ Tech Stack

Backend

FastAPI

Scikit-learn

Pandas

Joblib

Frontend

Streamlit

Seaborn

Matplotlib

ML & NLP

Random Forest

KMeans Clustering

Text Sentiment Analysis

📊 Dashboard Highlights

KPI Overview for executives

Interactive ML-powered visualizations

Business-friendly insights (not just raw data)

Real-time API-driven analytics

📁 Project Structure
├── backend/
│   ├── main.py
│   ├── feature_engineering.py
│   └── models/
├── frontend/
│   └── dashboard.py
├── data/
│   ├── customers.csv
│   ├── transactions.csv
│   ├── clickstream.csv
│   └── reviews.csv
└── README.md

▶️ How to Run
1️⃣ Start Backend
uvicorn main:app --reload

2️⃣ Start Dashboard
streamlit run dashboard.py

🎯 Use Cases

Customer retention & growth strategy

Product quality monitoring

UX optimization

Marketing personalization

Executive-level decision support

📌 Why This Project Matters

This project goes beyond notebooks and showcases:

Production-style ML pipelines

API-driven architecture

Business-focused analytics

End-to-end deployment mindset

Ideal for Data Science, ML Engineer, and Analytics internships.

👤 Author

Punsandali
Data Science Undergraduate
Passionate about ML, Analytics, and Building Real-World Systems
