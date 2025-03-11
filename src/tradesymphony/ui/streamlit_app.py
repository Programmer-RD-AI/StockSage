import streamlit as st
import json

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
            st.markdown("---")
else:
    st.warning("No investment recommendations found in the data.")

# Add footer
st.markdown("---")
st.markdown("*Data sourced from investment analysis*")
