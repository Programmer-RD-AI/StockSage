import streamlit as st
import json
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Investment Recommendations", layout="wide")

# Load the JSON file
try:
    with open("./outputs/thesis_json.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    st.error("Could not find testoutput.json file")
    st.stop()

# Page title
st.title("Investment Recommendations Dashboard")

# Extract investments
investments = data.get("investments", [])

# Time period selector
time_period = st.sidebar.selectbox(
    "Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2
)

if investments:
    for i, investment in enumerate(investments, 1):
        # Create an expander for each company
        with st.expander(
            f"{i}. {investment['company_name']} ({investment['ticker']})",
            expanded=(i == 1),
        ):
            # Display company information
            st.markdown("---")
            st.markdown(f"**Company:** {investment['company_name']}")
            st.markdown(f"**Ticker:** {investment['ticker']}")
            st.markdown("**Investment Thesis:**")
            st.markdown(f"_{investment['thesis']}_")

            # Fetch stock data
            ticker = investment["ticker"]
            stock = yf.Ticker(ticker)
            hist = stock.history(period=time_period)

            # Create stock price chart
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name="OHLC",
                    ),
                    go.Scatter(
                        x=hist.index,
                        y=hist["Close"].rolling(window=20).mean(),
                        line=dict(color="orange", width=1),
                        name="20-day MA",
                    ),
                ]
            )

            fig.update_layout(
                title=f'{investment["company_name"]} Stock Price',
                yaxis_title="Stock Price (USD)",
                xaxis_title="Date",
                template="none",
                height=600,
            )

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # Display key statistics
            info = stock.info
            col1, col2, col3 = st.columns(3)

            with col1:
                market_cap = info.get("marketCap")
                if market_cap and isinstance(market_cap, (int, float)):
                    market_cap_display = f"${market_cap:,.0f}"
                else:
                    market_cap_display = "N/A"
                st.metric("Market Cap", market_cap_display)

            with col2:
                pe_ratio = info.get("trailingPE")
                if pe_ratio and isinstance(pe_ratio, (int, float)):
                    pe_display = f"{pe_ratio:.2f}"
                else:
                    pe_display = "N/A"
                st.metric("P/E Ratio", pe_display)

            with col3:
                week_change = info.get("52WeekChange")
                if week_change and isinstance(week_change, (int, float)):
                    change_display = f"{week_change:.1%}"
                else:
                    change_display = "N/A"
                st.metric("52 Week Change", change_display)

            st.markdown("---")
else:
    st.warning("No investment recommendations found in the data.")

# Add footer
st.markdown("---")
st.markdown("*Data sourced from Yahoo Finance and investment analysis*")
