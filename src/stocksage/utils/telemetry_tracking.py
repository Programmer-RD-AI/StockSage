import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree
from datetime import datetime, UTC  # Use timezone-aware objects
from crewai.tasks.task_output import TaskOutput
from urllib3.util import Retry
import asyncio
from typing import Any

load_dotenv()

# Create a more robust retry configuration
retry_config = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[502, 503, 504, 408, 425, 429],
    allowed_methods=None,  # Retry on all methods
    raise_on_status=False,
)

# Initialize the client with retry and a longer timeout
langsmith_client = Client(
    auto_batch_tracing=False,
    api_key=os.getenv(
        "LANGSMITH_API_KEY", "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"
    ),
    retry_config=retry_config,
    timeout_ms=(30000, 120000),  # 30s connect timeout, 120s read timeout
)


def initialize_event_loop():
    """
    Initialize and configure an asyncio event loop.

    This function checks if an asyncio event loop is already running.
    If not, it creates and sets a new event loop. This ensures that
    asynchronous operations can be performed even within synchronous contexts.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop

    Raises:
        No exceptions are raised directly; exceptions from asyncio are caught internally
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def langsmith_task_callback(task: TaskOutput) -> None:
    """
    Callback function for CrewAI tasks that logs to LangSmith.

    This function is designed to be used as a callback for CrewAI task execution.
    It captures task details and sends them to LangSmith for logging and analysis.
    The function prints the task information and attempts to add it as an event
    to the current run tree in LangSmith.

    Args:
        task (TaskOutput): The task output object from CrewAI, containing task
            description, results, and other metadata

    Note:
        Any exceptions during execution are caught and printed, preventing
        task execution from being interrupted by telemetry failures
    """
    try:
        initialize_event_loop()

        current_run = get_current_run_tree()
        if current_run is not None:
            current_run.add_event(
                {
                    "task": task.description,
                    "time": datetime.now(UTC).isoformat(),  # Use timezone-aware objects
                    "output": task.raw,
                }
            )
    except Exception as e:
        print(f"Error in langsmith_task_callback: {e}")


def langsmith_step_callback(step: Any, info=None) -> None:
    """
    Callback function for CrewAI that logs each step.

    This function captures information about individual steps in the CrewAI
    execution flow and reports them to LangSmith. Each step is recorded with
    a timestamp and any associated contextual information.

    Args:
        step: Identifier (or number) for the step.
        info (Optional[dict]): Optional dictionary with additional context about the step.
            Defaults to None.

    Note:
        This function silently continues if the current run tree cannot be found,
        ensuring that step execution is not affected by telemetry issues.
    """
    current_run = get_current_run_tree()
    if current_run is not None:
        current_run.add_event(
            {
                "name": f"Step {step}",
                "time": datetime.now(UTC).isoformat(),  # Use timezone-aware objects
                "kwargs": info or {},
            }
        )


def verify_langsmith_setup() -> bool:
    """
    Verify LangSmith setup and create project if needed.

    This function checks if all required environment variables for LangSmith
    are present and tries to create or access the specified LangSmith project.
    It provides feedback on the success or failure of the LangSmith setup.

    Returns:
        bool: True if setup is successful, False otherwise

    Note:
        The function checks for the following environment variables:
        - LANGSMITH_API_KEY: API key for LangSmith authentication
        - LANGSMITH_PROJECT: Name of the LangSmith project to use

        If the project already exists, the function will recognize this and
        not treat it as an error.
    """
    # Check for required environment variables
    required_vars = ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("Please update your .env file with the required LangSmith variables.")
        return False

    # Initialize client and verify connection
    try:
        langsmith_client.create_project(os.getenv("LANGSMITH_PROJECT"), upsert=True)
        print("✅ Successfully connected to LangSmith")
        print(f"✅ Project '{os.getenv('LANGSMITH_PROJECT')}' is ready")
        return True
    except Exception as e:
        error_str = str(e)
        # Check if the error is due to a session already existing
        if "Session already exists" in error_str:
            print("ℹ️ LangSmith session already exists")
            print(f"✅ Project '{os.getenv('LANGSMITH_PROJECT')}' is ready")
            return True
        else:
            print(f"❌ Failed to connect to LangSmith: {error_str}")
            return False


if __name__ == "__main__":
    verify_langsmith_setup()
