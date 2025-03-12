from stocksage.utils.telemetry_tracking import (
    langsmith_task_callback,
    langsmith_step_callback,
    verify_langsmith_setup,
)
from crewai.tasks.task_output import TaskOutput


def test_langsmith_task_callback():
    task = TaskOutput(
        description="Test task",
        raw="Sample raw output",  # Changed from dict to string
        agent="test_agent",  # Added the required agent parameter
    )
    langsmith_task_callback(task)
    # No exceptions should be raised
    assert True


def test_langsmith_step_callback():
    langsmith_step_callback(1, {"key": "value"})
    # No exceptions should be raised
    assert True


def test_verify_langsmith_setup():
    verify_langsmith_setup()
    # No exceptions should be raised
    assert True
