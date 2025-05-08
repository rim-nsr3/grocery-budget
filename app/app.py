# === Imports ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(page_title="Grocery Budget Analysis", layout="wide")

# --- TITLE ---
st.title("🛒 Grocery Budget and Price Trend Analysis")

# --- LOAD CLEANED DATA ---
@st.cache_data
def load_data():
    food_prices = pd.read_excel("../data/FoodPrices_cleaned.xlsx")
    cpi_forecast = pd.read_csv("../data/CPIHistoricalForecast_cleaned.csv")
    ppi_forecast = pd.read_csv("../data/PPIForecast_cleaned.csv")
    income_summary = pd.read_excel("../data/IncomeSummary_cleaned.xlsx")
    return food_prices, cpi_forecast, ppi_forecast, income_summary

food_prices, cpi_forecast, ppi_forecast, income_summary = load_data()

# --- SIDEBAR ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview", 
    "Food Prices", 
    "CPI Trends", 
    "PPI Trends", 
    "Income Trends", 
    "ML Models", 
    "AI Grocery Advisor"
])

# --- OVERVIEW PAGE ---
if page == "Overview":
    st.subheader("Project Overview")
    st.markdown(
        """
        This app explores food prices, inflation trends, income changes, and uses machine learning and AI to deliver budget recommendations.
        
        **Contents:**
        - 📈 Food price distribution and budget analysis
        - 🛍️ Consumer Price Index (CPI) trends
        - 🏭 Producer Price Index (PPI) forecasts
        - 🏡 Income growth and affordability trends
        - 🧠 Machine Learning models
        - 🤖 AI Grocery Recommendation
        """
    )
    st.image("../data/price_distribution.png", caption="Distribution of Food Unit Prices", use_container_width=True)

# --- FOOD PRICES PAGE ---
elif page == "Food Prices":
    st.subheader("🛒 Food Prices and Budget Categories")
    st.caption("Prices are shown in dollars per unit (e.g., per pound, per gallon, per package). Budget categories (Low, Mid, High) were assigned based on relative affordability using price quantiles.")


    st.markdown("### Average Price per Budget Category Over Years")
    st.image("../data/avgPrice_bybudgetcategory.png", caption="Average Grocery Price per Unit Over Time", use_container_width=True)

    st.markdown("---")
    st.markdown("### Raw Food Prices (Sample)")

    st.dataframe(food_prices[['Product', 'Year', 'Price_per_Unit', 'Budget_Category']].head(10))

    # Dynamic plot
    st.markdown("### Trend: Price per Unit Distribution")
    fig, ax = plt.subplots(figsize=(10,6))
    food_prices['Price_per_Unit'].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7, ax=ax)
    ax.set_title('Distribution of Food Unit Prices', fontsize=16)
    ax.set_xlabel('Price per Unit ($)', fontsize=14)
    st.pyplot(fig)
    st.caption("This distribution shows how frequently food items fall into different price ranges across the dataset.")


# --- CPI TRENDS PAGE ---
elif page == "CPI Trends":
    st.subheader("📈 CPI (Consumer Price Index) Trends")
    st.caption("Forecasted annual percent changes in food prices, reflecting expected grocery inflation based on the Consumer Price Index (CPI).")

    st.markdown("### Forecasted Food Price Inflation")
    st.image("../data/cpi_trend.png", caption="CPI Forecast for All Food Items", use_container_width=True)

    st.markdown("---")
    st.markdown("### Sample CPI Forecast Data")
    st.dataframe(cpi_forecast.head(10))

# --- PPI TRENDS PAGE ---
elif page == "PPI Trends":
    st.subheader("🏭 PPI (Producer Price Index) Trends")
    st.caption("Forecasted percent changes in producer prices for food commodities. These trends represent cost changes at the supplier level, which can influence future consumer prices.")

    st.markdown("_(PPI graph optional - not shown by saved plot yet)_")
    st.dataframe(ppi_forecast.head(10))

    st.markdown("---")
    st.markdown("### Quick Dynamic Plot: PPI Over Time")

    # Extract year if available
    ppi_forecast['Year'] = ppi_forecast['Attribute'].str.extract(r'(\d{4})').astype(float)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ppi_forecast['Year'], ppi_forecast['Value'], marker='o')
    ax.set_title('Forecasted PPI Changes', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Percent Change (%)', fontsize=14)
    st.pyplot(fig)
    st.caption("Extracted year from forecast attributes for trend analysis. Values represent percent changes compared to the previous period.")


# --- INCOME TRENDS PAGE ---
elif page == "Income Trends":
    st.subheader("🏡 Household Income Growth Trends")
    st.caption("Reported median household incomes are adjusted to 2023 dollars using CPI-U inflation adjustments. Changes reflect real purchasing power growth or decline.")

    st.markdown("### Income Growth Rate Comparison (2022 - 2023)")
    st.image("../data/income_change.png", caption="Income Growth by Income Group", use_container_width=True)

    st.markdown("---")
    st.markdown("### Income Summary Table")
    st.dataframe(income_summary[['Characteristic', '2022_Estimate', '2023_Estimate']])

# --- ML MODELS PAGE ---
elif page == "ML Models":
    st.subheader("🧠 Machine Learning Models")

    st.markdown("### 1. Linear Regression: Predict CPI Inflation by Year")

    # Prepare data
    cpi_mid = cpi_forecast[cpi_forecast['Attribute'] == 'Mid point of prediction interval']
    X = cpi_mid[['Year being forecast']].values
    y = cpi_mid['Forecast percent change'].values

    # Build Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"**Model RMSE:** {rmse:.2f}")

    # Plot actual vs predicted
    st.caption("Linear regression model predicting forecasted CPI food inflation percent based on forecast year.")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color='blue', label='Actual')
    ax.plot(X, y_pred, color='red', label='Predicted', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Forecast Percent Change')
    ax.set_title('CPI Forecast vs Linear Regression Prediction')
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 2. K-Means Clustering: Grouping Foods by Price")

    # Prepare KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    food_prices['Cluster'] = kmeans.fit_predict(food_prices[['Price_per_Unit']])

    st.write("**Cluster Centers (average prices per group):**")
    st.write(kmeans.cluster_centers_)

    # Plot clusters
    st.caption("K-Means clustering groups food items into natural price tiers based on cost per unit.")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ['red', 'green', 'blue']
    for cluster in range(3):
        cluster_data = food_prices[food_prices['Cluster'] == cluster]
        ax2.scatter(cluster_data['Price_per_Unit'], [cluster]*len(cluster_data), 
                    color=colors[cluster], label=f'Cluster {cluster}')
    ax2.scatter(kmeans.cluster_centers_[:,0], [0,1,2], color='black', marker='x', s=200, label='Centers')
    ax2.set_xlabel('Price per Unit')
    ax2.set_title('K-Means Clustering of Food Prices')
    ax2.set_yticks([0,1,2])
    ax2.set_yticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'])
    ax2.legend()
    st.pyplot(fig2)
    st.caption("Cluster centers represent the average unit price of items in each group (Cluster 0 = cheapest, Cluster 2 = most expensive).")

# --- AI GROCERY ADVISOR PAGE ---
# --- AI GROCERY ADVISOR PAGE ---
elif page == "AI Grocery Advisor":
    st.subheader("🤖 AI Grocery List Advisor (Gemini API)")

    st.caption("You can either select your income group manually or enter your weekly grocery budget, and we'll match you to an income category.")

    # How user wants to input
    input_method = st.radio(
        "How would you like to provide your information?",
        ["Select Income Group", "Enter Weekly Budget"]
    )

    income_group = None
    weekly_budget = None

    if input_method == "Select Income Group":
        income_group = st.selectbox(
            "Select your income group:",
            ["Low Income", "Middle Income", "High Income"]
        )

        # Map budgets
        income_budget_map = {
            "Low Income": 25,
            "Middle Income": 50,
            "High Income": 75
        }

        weekly_budget = income_budget_map[income_group]

    else:  # Enter Weekly Budget manually
        weekly_budget = st.number_input(
            "Enter your weekly grocery budget ($)", 
            min_value=10.0, max_value=200.0, step=1.0
        )

        # Determine income group
        if weekly_budget <= 30:
            income_group = "Low Income"
        elif weekly_budget <= 60:
            income_group = "Middle Income"
        else:
            income_group = "High Income"

        st.write(f"**Detected Income Group:** {income_group}")

    # --- After getting weekly_budget and income_group, continue like normal ---

    # Filter affordable foods
    affordable_foods = food_prices[food_prices['Price_per_Unit'] <= (weekly_budget / 10)]
    affordable_list = ', '.join(affordable_foods['Product'].tolist())

    # Build prompt
    prompt = (
        f"You are a smart grocery shopping assistant.\n"
        f"The user belongs to a {income_group} group with a weekly grocery budget of ${weekly_budget:.2f}.\n"
        f"The following foods are affordable: {affordable_list}.\n"
        f"ONLY recommend foods from the list above.\n"
        f"Recommend about 10 foods and explain why they fit the budget.\n"
        f"Also mention how inflation might affect their shopping choices."
    )

    if st.checkbox("Show AI Prompt"):
        st.code(prompt)

    if st.button("Get AI Grocery List"):
        with st.spinner("Gemini is thinking... 🤔"):
       
            genai.configure(api_key=st.secrets["API_KEY"])
            model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
            response = model.generate_content(prompt)
            st.subheader("🛍️ Gemini's Recommendation:")
            st.write(response.text)
            st.caption("Recommendations prioritize budget constraints while considering inflation trends affecting food costs.")


# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("Built by Rim Nassiri")
