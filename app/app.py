import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

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
page = st.sidebar.radio("Go to:", ["Overview", "Food Prices", "CPI Trends", "PPI Trends", "Income Trends"])

# --- OVERVIEW PAGE ---
if page == "Overview":
    st.subheader("Project Overview")
    st.markdown(
        """
        This app explores food prices, inflation trends, and income changes over time.
        
        **Contents:**
        - 📈 Food price distribution and budget analysis
        - 🛍️ Consumer Price Index (CPI) trends
        - 🏭 Producer Price Index (PPI) forecasts
        - 🏡 Income growth and affordability trends
        """
    )
    st.image("../data/price_distribution.png", caption="Distribution of Food Unit Prices", use_column_width=True)

# --- FOOD PRICES PAGE ---
elif page == "Food Prices":
    st.subheader("🛒 Food Prices and Budget Categories")

    st.markdown("### Average Price per Budget Category Over Years")
    st.image("../data/avgPrice_bybudgetcategory.png", caption="Average Grocery Price per Unit Over Time", use_column_width=True)

    st.markdown("---")
    st.markdown("### Raw Food Prices (Sample)")

    st.dataframe(food_prices[['Product', 'Year', 'Price_per_Unit', 'Budget_Category']].head(10))

    # Optional dynamic plot
    st.markdown("### Trend: Price per Unit Distribution")
    fig, ax = plt.subplots(figsize=(10,6))
    food_prices['Price_per_Unit'].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7, ax=ax)
    ax.set_title('Distribution of Food Unit Prices', fontsize=16)
    ax.set_xlabel('Price per Unit ($)', fontsize=14)
    st.pyplot(fig)

# --- CPI TRENDS PAGE ---
elif page == "CPI Trends":
    st.subheader("📈 CPI (Consumer Price Index) Trends")

    st.markdown("### Forecasted Food Price Inflation")
    st.image("../data/cpi_trend.png", caption="CPI Forecast for All Food Items", use_column_width=True)

    st.markdown("---")
    st.markdown("### Sample CPI Forecast Data")
    st.dataframe(cpi_forecast.head(10))

# --- PPI TRENDS PAGE ---
elif page == "PPI Trends":
    st.subheader("🏭 PPI (Producer Price Index) Trends")

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

# --- INCOME TRENDS PAGE ---
elif page == "Income Trends":
    st.subheader("🏡 Household Income Growth Trends")

    st.markdown("### Income Growth Rate Comparison (2022 - 2023)")
    st.image("../data/income_change.png", caption="Income Growth by Income Group", use_column_width=True)

    st.markdown("---")
    st.markdown("### Income Summary Table")
    st.dataframe(income_summary[['Characteristic', '2022_Estimate', '2023_Estimate', 'Income_Group']])

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("Built by Rim 🚀")