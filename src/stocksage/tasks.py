from crewai import Task, Agent
from langsmith.run_helpers import traceable
from typing import Dict
from stocksage.tools import (
    StockSymbolFetcherTool,
    SentimentAnalysisTool,
    YFinanceTool,
    AlphaVantageTool,
    get_firecrawl_crawl_website_tool,
    get_firecrawl_scrape_website_tool,
)


@traceable
def create_analyze_sentiment_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [
        SentimentAnalysisTool(),
        get_firecrawl_crawl_website_tool(),
        get_firecrawl_scrape_website_tool(),
    ],
    default_description: str = "Analyze public sentiment and news for the selected stocks.",
    default_expected_output: str = "A comprehensive sentiment analysis report for each stock.",
):
    """
    Creates a task for analyzing public sentiment and news for selected stocks.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for sentiment analysis
    """
    analyze_sentiment_config = tasks_config.get("analyze_sentiment", {})
    return Task(
        description=analyze_sentiment_config.get("description", default_description),
        expected_output=analyze_sentiment_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_analyze_stock_data_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool(), AlphaVantageTool(), StockSymbolFetcherTool()],
    default_description: str = "Conduct detailed analysis of financial data for a list of stocks.",
    default_expected_output: str = "A comprehensive financial analysis report for each stock.",
):
    """
    Creates a task for conducting detailed analysis of financial data for stocks.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for stock data analysis
    """
    analyze_stock_data_config = tasks_config.get("analyze_stock_data", {})
    return Task(
        description=analyze_stock_data_config.get("description", default_description),
        expected_output=analyze_stock_data_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_simplified_thesis_json_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [],
    default_description: str = "Create a simplified JSON file with the investment theses for the top 5 recommended companies.",
    default_expected_output: str = "A simplified JSON file containing the investment theses for the top 5 recommended companies.",
):
    """
    Creates a task for generating a simplified JSON file with investment theses.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for creating simplified thesis JSON
    """
    create_simplified_thesis_json_config = tasks_config.get(
        "create_simplified_thesis_json", {}
    )
    return Task(
        description=create_simplified_thesis_json_config.get(
            "description", default_description
        ),
        expected_output=create_simplified_thesis_json_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        output_file=create_simplified_thesis_json_config.get("output_file", None),
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_generate_investment_thesis_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool(), AlphaVantageTool()],
    default_description: str = "Develop a comprehensive investment thesis based on the selected stocks and the selected sentiment analysis.",
    default_expected_output: str = "A detailed investment thesis for each of the top 5 selected stocks.",
):
    """
    Creates a task for developing comprehensive investment theses based on selected stocks and sentiment analysis.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for generating investment theses
    """
    generate_investment_thesis_config = tasks_config.get(
        "generate_investment_thesis", {}
    )
    return Task(
        description=generate_investment_thesis_config.get(
            "description", default_description
        ),
        expected_output=generate_investment_thesis_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_justify_sentiment_selection_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [SentimentAnalysisTool(), get_firecrawl_scrape_website_tool()],
    default_description: str = "Justify stock selections based on sentiment analysis and financial data.",
    default_expected_output: str = "Detailed justification for stock selections based on sentiment analysis.",
):
    """
    Creates a task for justifying stock selections based on sentiment analysis and financial data.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for justifying sentiment-based selections
    """
    justify_sentiment_selection_config = tasks_config.get(
        "justify_sentiment_selection", {}
    )
    return Task(
        description=justify_sentiment_selection_config.get(
            "description", default_description
        ),
        expected_output=justify_sentiment_selection_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_justify_stock_selection_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool(), AlphaVantageTool()],
    default_description: str = "Justify stock selections based on financial data and analysis.",
    default_expected_output: str = "Detailed justification for stock selections based on financial analysis.",
):
    """
    Creates a task for justifying stock selections based on financial data and analysis.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for justifying financially-based selections
    """
    justify_stock_selection_config = tasks_config.get("justify_stock_selection", {})
    return Task(
        description=justify_stock_selection_config.get(
            "description", default_description
        ),
        expected_output=justify_stock_selection_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_manage_investment_process_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool(), AlphaVantageTool()],
    default_description: str = "Monitor and manage the investment process, including portfolio allocation, risk management, and performance tracking.",
    default_expected_output: str = "A detailed report on the investment process and performance.",
):
    """
    Creates a task for monitoring and managing the investment process, including portfolio allocation,
    risk management, and performance tracking.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for managing the investment process
    """
    manage_investment_process_config = tasks_config.get("manage_investment_process", {})
    return Task(
        description=manage_investment_process_config.get(
            "description", default_description
        ),
        expected_output=manage_investment_process_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        output_file=manage_investment_process_config.get("output_file", None),
        output_json=manage_investment_process_config.get("output_json", None),
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_optimize_sentiment_selection_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [SentimentAnalysisTool(), YFinanceTool()],
    default_description: str = "Optimize stock selections based on sentiment analysis and risk-reward profiles.",
    default_expected_output: str = "Optimized stock selections based on sentiment analysis and risk-reward profiles.",
):
    """
    Creates a task for optimizing stock selections based on sentiment analysis and risk-reward profiles.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for optimizing sentiment-based selections
    """
    optimize_sentiment_selection_config = tasks_config.get(
        "optimize_sentiment_selection", {}
    )
    return Task(
        description=optimize_sentiment_selection_config.get(
            "description", default_description
        ),
        expected_output=optimize_sentiment_selection_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_optimize_stock_selection_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool(), AlphaVantageTool()],
    default_description: str = "Optimize stock selections based on financial data and risk-reward profiles.",
    default_expected_output: str = "Optimized stock selections based on financial data and risk-reward profiles.",
):
    """
    Creates a task for optimizing stock selections based on financial data and risk-reward profiles.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for optimizing financially-based selections
    """
    optimize_stock_selection_config = tasks_config.get("optimize_stock_selection", {})
    return Task(
        description=optimize_stock_selection_config.get(
            "description", default_description
        ),
        expected_output=optimize_stock_selection_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_perform_integrated_analysis_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [SentimentAnalysisTool(), YFinanceTool(), AlphaVantageTool()],
    default_description: str = "Perform detailed integrated analysis combining financial metrics and sentiment for each stock.",
    default_expected_output: str = "A comprehensive integrated analysis report for each stock.",
):
    """
    Creates a task for performing detailed integrated analysis combining financial metrics and sentiment for stocks.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for integrated analysis
    """
    perform_integrated_analysis_config = tasks_config.get(
        "perform_integrated_analysis", {}
    )
    return Task(
        description=perform_integrated_analysis_config.get(
            "description", default_description
        ),
        expected_output=perform_integrated_analysis_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_refine_stock_analysis_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool(), AlphaVantageTool()],
    default_description: str = "Refine the stock analysis based on the integrated analysis.",
    default_expected_output: str = "Refine the stock analysis based on the integrated",
):
    """
    Creates a task for refining stock analysis based on the integrated analysis results.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for refining stock analysis
    """
    refine_stock_analysis_config = tasks_config.get("refine_stock_analysis", {})
    return Task(
        description=refine_stock_analysis_config.get(
            "description", default_description
        ),
        expected_output=refine_stock_analysis_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_store_thesis_json_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [],
    default_description: str = "Store the investment thesis in a JSON file for future reference.",
    default_expected_output: str = "A JSON file containing the investment thesis for the top 5 recommended companies.",
):
    """
    Creates a task for storing investment theses in a JSON file for future reference.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for storing thesis JSON
    """
    store_thesis_json_config = tasks_config.get("store_thesis_json", {})
    return Task(
        description=store_thesis_json_config.get("description", default_description),
        expected_output=store_thesis_json_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
        output_file=store_thesis_json_config.get("output_file", None),
    )


@traceable
def create_synthesize_final_selection_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool()],
    default_description: str = "Synthesize all analyses into final investment recommendations.",
    default_expected_output: str = "Final selection of top 5 most promising investment opportunities.",
):
    """
    Creates a task for synthesizing all analyses into final investment recommendations.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for synthesizing final selections
    """
    synthesize_final_selection_config = tasks_config.get(
        "synthesize_final_selection", {}
    )
    return Task(
        description=synthesize_final_selection_config.get(
            "description", default_description
        ),
        expected_output=synthesize_final_selection_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_terminate_process_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [],
    default_description: str = "Ensure clean termination of the stocksage workflow.",
    default_expected_output: str = "All resources properly released and the process terminates cleanly.",
):
    """
    Creates a task for ensuring clean termination of the StockSage workflow.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for terminating the process
    """
    terminate_process_config = tasks_config.get("terminate_process", {})
    return Task(
        description=terminate_process_config.get("description", default_description),
        expected_output=terminate_process_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        output_file=terminate_process_config.get("output_file", None),
        context=context,
        tools=tools,
        async_execution=async_execution,
    )


@traceable
def create_verify_thesis_json_task(
    tasks_config: Dict[str, str],
    agent: Agent,
    context: list = [],
    async_execution: bool = False,
    tools: list = [YFinanceTool()],
    default_description: str = "Verify the JSON file with investment theses to ensure accuracy and correctness.",
    default_expected_output: str = "Confirmation that the JSON file is correctly formatted and contains accurate investment theses.",
):
    """
    Creates a task for verifying the JSON file with investment theses to ensure accuracy and correctness.

    Args:
        tasks_config: Dictionary containing task-specific configurations
        agent: Agent responsible for executing the task
        context: Additional context information for the task
        async_execution: Whether the task should be executed asynchronously
        tools: List of tools available for the agent to use during task execution
        default_description: Default description if not specified in config
        default_expected_output: Default expected output if not specified in config

    Returns:
        Task: A configured Task object for verifying thesis JSON
    """
    verify_thesis_json_config = tasks_config.get("verify_thesis_json", {})
    return Task(
        description=verify_thesis_json_config.get("description", default_description),
        expected_output=verify_thesis_json_config.get(
            "expected_output", default_expected_output
        ),
        agent=agent,
        context=context,
        tools=tools,
        async_execution=async_execution,
        output_file=verify_thesis_json_config.get("output_file", None),
    )
