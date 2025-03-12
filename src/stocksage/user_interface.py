import streamlit as st
import json
import os
import glob
import time
import uuid
import webbrowser
from stocksage.utils.telemetry_tracking import verify_langsmith_setup
from stocksage.main import run, test, train, replay

# Initialize LangSmith tracking
verify_langsmith_setup()

# Page configuration
st.set_page_config(
    page_title="StockSage Dashboard", layout="wide", initial_sidebar_state="expanded"
)

# Initialize session state
if "operation_running" not in st.session_state:
    st.session_state["operation_running"] = False
if "operation_result" not in st.session_state:
    st.session_state["operation_result"] = None
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None

# Main header
st.title("StockSage Dashboard")
st.markdown("AI-powered investment analysis with LangSmith integration")

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Operations")

    # Simple operation buttons
    run_col1, run_col2 = st.columns(2)

    with run_col1:
        if st.button("📊 Run Analysis", use_container_width=True):
            st.session_state["operation_running"] = True
            st.session_state["current_operation"] = "run"
            st.session_state["run_id"] = str(uuid.uuid4())[:8]
            st.experimental_rerun()

        if st.button("🧪 Run Tests", use_container_width=True):
            st.session_state["operation_running"] = True
            st.session_state["current_operation"] = "test"
            st.session_state["run_id"] = str(uuid.uuid4())[:8]
            st.experimental_rerun()

    with run_col2:
        if st.button("🔄 Train Models", use_container_width=True):
            st.session_state["operation_running"] = True
            st.session_state["current_operation"] = "train"
            st.session_state["run_id"] = str(uuid.uuid4())[:8]
            st.experimental_rerun()

        if st.button("⏮️ Replay Run", use_container_width=True):
            st.session_state["operation_running"] = True
            st.session_state["current_operation"] = "replay"
            st.session_state["run_id"] = str(uuid.uuid4())[:8]
            st.experimental_rerun()

    st.markdown("---")

    # Basic configuration
    st.header("Configuration")
    symbols = st.text_input(
        "Stock Symbols", "AAPL,MSFT,GOOGL,AMZN", help="Enter ticker symbols to analyze"
    )

    time_period = st.select_slider(
        "Analysis Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        value="1y",
    )

    st.markdown("---")

    # Output file selection
    st.header("Results")

    # Find all available output files
    output_dir = "./outputs"
    json_files = glob.glob(f"{output_dir}/*.json")

    if not json_files:
        st.warning("No analysis files found!")
        output_files = ["No files available"]
    else:
        # Extract filenames and sort by modification time (newest first)
        output_files = sorted(
            [(os.path.basename(f), os.path.getmtime(f)) for f in json_files],
            key=lambda x: x[1],
            reverse=True,
        )
        output_files = [f[0] for f in output_files]

    selected_file = st.selectbox("Select Output File:", options=output_files, index=0)

    # LangSmith integration button
    st.markdown("---")
    st.header("LangSmith Integration")

    langsmith_project = os.environ.get("LANGSMITH_PROJECT", "stock-sage")

    if st.button("🔗 Open in LangSmith", help="View traces in LangSmith"):
        webbrowser.open(f"https://smith.langchain.com/projects/{langsmith_project}")

# Results display area
with col2:
    if st.session_state.get("operation_running"):
        operation = st.session_state.get("current_operation", "run")
        run_id = st.session_state.get("run_id", "unknown")

        st.header(f"Running {operation.title()} (ID: {run_id})")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Based on the selected operation, call the appropriate function
            if operation == "run":
                status_text.text("Starting analysis...")
                progress_bar.progress(10)
                time.sleep(0.5)
                progress_bar.progress(30)
                status_text.text("Processing market data...")
                time.sleep(0.5)
                progress_bar.progress(60)
                status_text.text("Generating recommendations...")
                time.sleep(0.5)

                # Actually run the analysis
                result = run()

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

            elif operation == "train":
                status_text.text("Training models...")
                progress_bar.progress(20)
                time.sleep(0.5)
                progress_bar.progress(50)
                status_text.text("Fine-tuning models...")
                time.sleep(0.5)

                # Actually run the training
                result = train()

                progress_bar.progress(100)
                status_text.text("Training complete!")

            elif operation == "replay":
                status_text.text("Replaying previous run...")
                progress_bar.progress(30)
                time.sleep(0.5)
                progress_bar.progress(75)
                status_text.text("Generating results...")
                time.sleep(0.5)

                # Actually replay the run
                result = replay()

                progress_bar.progress(100)
                status_text.text("Replay complete!")

            elif operation == "test":
                status_text.text("Running tests...")
                progress_bar.progress(25)
                time.sleep(0.5)
                progress_bar.progress(75)
                status_text.text("Validating results...")
                time.sleep(0.5)

                # Actually run the tests
                result = test()

                progress_bar.progress(100)
                status_text.text("Tests complete!")

            st.session_state["operation_result"] = {
                "status": "success",
                "message": f"{operation.title()} completed successfully",
                "data": result,
                "run_id": run_id,
            }

        except Exception as e:
            st.session_state["operation_result"] = {
                "status": "error",
                "message": f"Error during {operation}: {str(e)}",
                "error": str(e),
            }
            progress_bar.progress(100)
            status_text.text(f"Error: {str(e)}")

        finally:
            st.session_state["operation_running"] = False

        if st.button("Reset"):
            st.session_state["operation_running"] = False
            st.experimental_rerun()

    elif selected_file != "No files available":
        st.header("Analysis Results")

        # If we have a previous operation result, show it
        result = st.session_state.get("operation_result")
        if result:
            if result["status"] == "success":
                st.success(result["message"])

                # Show LangSmith link if we have a run ID
                if "run_id" in result:
                    langsmith_project = os.environ.get(
                        "LANGSMITH_PROJECT", "stock-sage"
                    )
                    st.info(f"Run ID: {result['run_id']}")
                    st.markdown(
                        f"[View in LangSmith](https://smith.langchain.com/projects/{langsmith_project}/runs?query={result['run_id']})"
                    )
            else:
                st.error(result["message"])

        # Load the selected JSON file
        try:
            with open(f"{output_dir}/{selected_file}", "r") as f:
                data = json.load(f)

            # Extract basic information
            if "investments" in data:
                investments = data.get("investments", [])
                analysis_date = data.get("analysis_date", "Not specified")
            elif "recommendations" in data:
                investments = data.get("recommendations", [])
                analysis_date = data.get("date", "Not specified")
            else:
                investments = []
                for key, value in data.items():
                    if (
                        isinstance(value, list)
                        and len(value) > 0
                        and isinstance(value[0], dict)
                    ):
                        if any(k in value[0] for k in ["ticker", "symbol", "company"]):
                            investments = value
                            break
                analysis_date = data.get("date", data.get("timestamp", "Not specified"))

            st.subheader(f"Analysis Date: {analysis_date}")

            # Display investment recommendations in a table
            if investments:
                # Extract relevant fields for display
                table_data = []
                for inv in investments:
                    ticker = inv.get("ticker", inv.get("symbol", "Unknown"))
                    company = inv.get("company_name", inv.get("company", ticker))
                    recommendation = inv.get(
                        "recommendation", inv.get("action", "Hold")
                    )
                    target_price = inv.get(
                        "target_price", inv.get("price_target", "N/A")
                    )

                    # Extract thesis and truncate if needed
                    thesis = inv.get(
                        "thesis", inv.get("investment_thesis", inv.get("rationale", ""))
                    )
                    if len(thesis) > 100:
                        thesis = thesis[:97] + "..."

                    table_data.append(
                        {
                            "Ticker": ticker,
                            "Company": company,
                            "Recommendation": recommendation,
                            "Target Price": target_price,
                            "Thesis": thesis,
                        }
                    )

                st.dataframe(table_data, use_container_width=True)

                # Show detailed view for a selected investment
                if table_data:
                    selected_ticker = st.selectbox(
                        "Select company for detailed view:",
                        options=[item["Ticker"] for item in table_data],
                    )

                    selected_investment = next(
                        (
                            inv
                            for inv in investments
                            if inv.get("ticker", inv.get("symbol", ""))
                            == selected_ticker
                        ),
                        None,
                    )

                    if selected_investment:
                        st.subheader(f"Details for {selected_ticker}")

                        # Extract thesis with full text
                        thesis = selected_investment.get(
                            "thesis",
                            selected_investment.get(
                                "investment_thesis",
                                selected_investment.get(
                                    "rationale", "No thesis provided"
                                ),
                            ),
                        )

                        st.markdown("#### Investment Thesis")
                        st.info(thesis)

                        # Extract pros and cons if available
                        if "pros" in selected_investment and isinstance(
                            selected_investment["pros"], list
                        ):
                            st.markdown("#### Pros")
                            for pro in selected_investment["pros"]:
                                st.markdown(f"- {pro}")

                        if "cons" in selected_investment and isinstance(
                            selected_investment["cons"], list
                        ):
                            st.markdown("#### Cons")
                            for con in selected_investment["cons"]:
                                st.markdown(f"- {con}")
            else:
                st.warning("No investment recommendations found in the selected file.")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("No analysis files available. Run an analysis to generate results.")

# Footer with LangSmith status
st.markdown("---")
langsmith_project = os.environ.get("LANGSMITH_PROJECT", "stock-sage")
langsmith_endpoint = os.environ.get(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
)

st.markdown(
    f"**LangSmith Project:** {langsmith_project} | **Endpoint:** {langsmith_endpoint}"
)
st.caption("StockSage with LangSmith Telemetry")
