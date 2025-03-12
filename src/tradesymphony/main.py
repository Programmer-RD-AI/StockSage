#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

from stocksage.crew import stocksage
from langsmith.run_helpers import traceable
from langsmith import Client

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
langsmith_client = Client()


@traceable(run_type="chain", name="stocksage Analysis")
def run():
    """
    Run the crew.
    """
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    inputs = {
        "market": "US",
        "stock_universe": "S&P 500",
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    try:
        result = stocksage().crew().kickoff(inputs=inputs)
        print("stocksage analysis complete. Results saved to outputs directory.")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


@traceable(run_type="chain", name="stocksage Training")
def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "market": "US",
        "stock_universe": "S&P 500",
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        stocksage().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


@traceable(run_type="chain", name="stocksage Replay")
def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        stocksage().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


@traceable(run_type="chain", name="stocksage Testing")
def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "market": "US",
        "stock_universe": "S&P 500",
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        stocksage().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
