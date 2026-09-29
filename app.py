import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from google import genai

api_key = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    page_title = "Consumer Behavior Analytics",
    layout = "wide"
)

st.title("Consumer Behavior Analytics")


file = st.file_uploader(
    "Upload retail transaction data",
    type = ["csv"]
)

st.caption(
    "Required Data : Customer ID, Price, Quantity, Transaction Date, Order/Invoice ID and Product/Stock Code"
)

if file is not None:
    df = pd.read_csv(file)
    st.success("File uploaded successfully")

    st.subheader("Dataset Preview")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write("Detected Columns:")
    st.write(list(df.columns))

    st.subheader("Map Your Columns")

    customer_col = st.selectbox(
    "Customer ID column",
    df.columns)

    price_col = st.selectbox(
    "Price / Unit Price column",
    df.columns)

    quantity_col = st.selectbox(
    "Quantity column",
    df.columns)

    date_col = st.selectbox(
    "Transaction Date column",
    df.columns)

    order_col = st.selectbox(
    "Order / Invoice ID column",
    df.columns)

    product_col = st.selectbox(
    "Product / Stock Code column",
    df.columns)

    clean_df = df.rename(columns ={
        customer_col : "Customer ID",
        price_col : "Price",
        quantity_col : "Quantity",
        date_col : "InvoiceDate",
        order_col : "Invoice",
        product_col : "StockCode"
    }).copy()

    clean_df = clean_df.dropna(subset = ["Customer ID"])
    
    clean_df = clean_df[
    (clean_df["Price"] > 0) &
    (clean_df["Quantity"] > 0)]


    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])

    clean_df = clean_df.drop_duplicates()
    clean_df["total_spend"] = clean_df["Quantity"] * clean_df["Price"]

    total_revenue = clean_df["total_spend"].sum()
    total_customers = clean_df["Customer ID"].nunique()
    total_orders = clean_df["Invoice"].nunique()
    avg_order_value = total_revenue/total_orders
    price_per_customer = total_revenue/total_customers

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Revenue", f"£{total_revenue:,.0f}")
    col2.metric("Customers", f"{total_customers:,}")
    col3.metric("Orders", f"{total_orders:,}")
    col4.metric("Avg Order Value", f"£{avg_order_value:,.2f}")
    col5.metric("Avg per Customer", f"£{price_per_customer:,.2f}")

    customer_profile = clean_df.groupby("Customer ID").agg(
    TotalSpend=("total_spend", "sum"),
    Frequency=("Invoice", "nunique"),
    ProductDiversity=("StockCode", "nunique"),
    TotalQuantity=("Quantity", "sum"),
    LastPurchase=("InvoiceDate", "max"),
    FirstPurchase=("InvoiceDate", "min")).reset_index()

    customer_profile["AvgBasketValue"] = (
    customer_profile["TotalSpend"] / customer_profile["Frequency"])


    customer_profile["AvgItemsPerOrder"] = (
    customer_profile["TotalQuantity"] / customer_profile["Frequency"])


    reference_date = clean_df["InvoiceDate"].max()

    customer_profile["Recency"] = (
    reference_date - customer_profile["LastPurchase"]).dt.days

    
    st.write(f"Customers Analyzed: {len(customer_profile):,}")

    features = customer_profile[
    ["TotalSpend", "Frequency", "AvgBasketValue",
     "Recency", "ProductDiversity", "AvgItemsPerOrder"]].copy()


    features_log = np.log1p(features)


    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_log)

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    customer_profile["cluster"] = model.fit_predict(scaled_features)
    st.write("Customer Segments")
    st.write(customer_profile["cluster"].value_counts())

    segment_names = {
        0 : "Engaged / High-Value",
        1 : "Low Engagement / Lapsed"
    }

    customer_profile["segment"] = (customer_profile["cluster"].map(segment_names))

    segment_summary = customer_profile.groupby("segment").agg(
        Customers = ("cluster", "size"),
        TotalRevenue = ("TotalSpend", "sum"),
        AvgSpend = ("TotalSpend", "mean"),
        AvgOrders = ("Frequency", "mean"),
        AvgRecency = ("Recency", "mean")
    ).reset_index()

    segment_summary["CustomerShare"] = (
    segment_summary["Customers"] /
    segment_summary["Customers"].sum() * 100)

    segment_summary["RevenueShare"] = (
    segment_summary["TotalRevenue"] /
    segment_summary["TotalRevenue"].sum() * 100)

    st.subheader("Customer Segments")
    st.dataframe(segment_summary, use_container_width = True, hide_index = True)

    st.write("Revenue Contribution by Segments")
    revenue_chart = segment_summary.set_index("segment")["RevenueShare"]
    st.bar_chart(revenue_chart)


    st.subheader("AI Consumer Analyst")
    question = st.text_input("Ask a question about the consumer behavior")

    summary_text = segment_summary.to_string(index = False)

    if st.button("Ask"):
        client = genai.Client(api_key = api_key)

        prompt = f"""
You are a consumer behaviour analyst.

Use ONLY the customer analytics results provided below to answer
the user's question.

Do not invent statistics or customer characteristics that are not
supported by the data.

Customer segment results:
{summary_text}

User question:
{question}

Explain your answer clearly and provide practical marketing or
customer-retention implications when relevant.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.write(response.text)





