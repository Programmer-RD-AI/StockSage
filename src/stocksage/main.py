import sys
import warnings
import os
from datetime import datetime
from stocksage.crew import StockSage
from langsmith.run_helpers import traceable
import stocksage.utils.telemetry_tracking  # noqa: E402, F401
import uuid


# Generate a unique session ID for tracking purposes across the application
APP_SESSION_ID = str(uuid.uuid4())
os.environ["LANGCHAIN_SOURCE_RUN_ID"] = APP_SESSION_ID
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


@traceable(
    run_type="chain",
    name="StockSage Analysis",
    tags=["parent_session"],
    run_id=os.environ.get("LANGCHAIN_SOURCE_RUN_ID"),
)
def run():
    """
    Execute the StockSage crew to perform a complete stock analysis.

    This function initializes the StockSage crew and executes the full analysis
    pipeline, including stock data collection, sentiment analysis, and investment
    thesis generation. Results are saved to the outputs directory.

    Returns:
        dict: The final result of the crew's execution including stock recommendations
              and investment theses.

    Raises:
        Exception: If any error occurs during the crew execution process.
    """
    import nest_asyncio

    nest_asyncio.apply()
    inputs = {
        "market": "US",
        "stock_universe": "S&P 500",
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    try:
        result = StockSage().crew().kickoff(inputs=inputs)
        print("stocksage analysis complete. Results saved to outputs directory.")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


@traceable(
    run_type="chain",
    name="StockSage Training",
    tags=["parent_session"],
    run_id=os.environ.get("LANGCHAIN_SOURCE_RUN_ID"),
)
def train():
    """
    Train the StockSage crew for a specified number of iterations.

    This function executes the training process for the StockSage crew, running
    the specified number of iterations and saving the training results to a file.
    It expects command-line arguments for the number of iterations and the output filename.

    Command-line Arguments:
        sys.argv[1] (int): Number of training iterations to perform
        sys.argv[2] (str): Filename to save the training results

    Raises:
        Exception: If any error occurs during the training process.
    """
    inputs = {
        "market": "US",
        "stock_universe": "S&P 500",
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    try:
        StockSage().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


@traceable(
    run_type="chain",
    name="StockSage Replay",
    tags=["parent_session"],
    run_id=os.environ.get("LANGCHAIN_SOURCE_RUN_ID"),
)
def replay():
    """
    Replay the StockSage crew execution from a specific task.

    This function replays a previous execution of the StockSage crew, starting
    from the specified task ID. It's useful for debugging or analyzing the
    execution path of a specific run.

    Command-line Arguments:
        sys.argv[1] (str): The task ID from which to replay execution

    Raises:
        Exception: If any error occurs during the replay process.
    """
    try:
        StockSage().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


@traceable(
    run_type="chain",
    name="StockSage Testing",
    tags=["parent_session"],
    run_id=os.environ.get("LANGCHAIN_SOURCE_RUN_ID"),
)
def test():
    """
    Test the StockSage crew by executing only the first task.

    This function creates a test instance of the StockSage crew and executes
    only the first task in isolation. Results are saved to a file rather than
    returned directly to avoid potential serialization issues.

    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the test completed successfully
            - result_path (str): Path to the file containing full results (on success)
            - error (str): Error message (on failure)

    Raises:
        Exception: If any error occurs during the test setup process.
    """
    try:
        # Instead of using the built-in test() method, create a test process manually
        crew_instance = StockSage().crew()

        # Get just the first task for testing
        test_task = crew_instance.tasks[0]

        print(f"Testing task: {test_task.description}")

        # Create a dedicated agent for testing rather than using the crew's agent directly
        agent = crew_instance.agents[0]

        # Execute the task with error handling for serialization
        try:
            # Run the task with task execution context to avoid serialization issues
            result = agent.execute_task(test_task)

            # Extract result string instead of returning the full object
            result_str = str(result)

            print("Test completed successfully with task:", test_task.description)
            print(
                "Result summary:",
                result_str[:200] + "..." if len(result_str) > 200 else result_str,
            )

            # Save the result to a file to avoid returning potentially problematic objects
            result_path = os.path.join("../outputs", "test_result.txt")
            with open(result_path, "w") as f:
                f.write(result_str)

            print(f"Full result saved to {result_path}")
            return {"success": True, "result_path": result_path}

        except Exception as task_error:
            print(f"Error executing task: {task_error}")
            return {"success": False, "error": str(task_error)}

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
