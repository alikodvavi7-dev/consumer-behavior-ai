# Consumer Behavior Analytics

A Streamlit application for analyzing retail transaction data and identifying different customer segments.

## What it does

- Upload retail transaction data as a CSV file
- Clean and process customer transaction data
- Calculate metrics such as revenue, number of customers and average order value
- Create customer behavioral profiles
- Segment customers using K-Means clustering
- Visualize revenue contribution from different customer segments
- Ask questions about the analysis using an AI Consumer Analyst powered by Gemini

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- K-Means Clustering
- Matplotlib
- Streamlit
- Google Gemini API

## Dataset Requirements

The dataset should contain information representing:

- Customer ID
- Price
- Quantity
- Transaction Date
- Order / Invoice ID
- Product / Stock Code

The exact column names can be different because they can be mapped inside the application.

## Running the App

Install the required packages:

pip install -r requirements.txt

Then run:

streamlit run app.py