from crewai import Agent
from stocksage.config import (
    ANALYSIS_AGENT_LLM,
    AGENT_META_CONFIG,
    VERBOSE,
    FACT_AGENT_LLM,
    JUSTIFICATION_AGENT_LLM,
    OPTIMIZATION_AGENT_LLM,
    PORTFOLIO_MANAGER_LLM,
    RECOMMENDATION_AGENT_LLM,
    SENTIMENT_AGENT_LLM,
    SYNTHESIZER_AGENT_LLM,
    THESIS_AGENT_LLM,
)
from stocksage.utils.telemetry_tracking import langsmith_step_callback
from typing import Callable, Dict, Union
from stocksage.tools import (
    StockSymbolFetcherTool,
    SentimentAnalysisTool,
    YFinanceTool,
    AlphaVantageTool,
    get_firecrawl_crawl_website_tool,
    get_firecrawl_scrape_website_tool,
)
from langsmith.run_helpers import traceable


@traceable
def create_analysis_agent(
    agents_config: Dict[str, str],
    tools: list = [
        YFinanceTool(),
        StockSymbolFetcherTool(),
        SentimentAnalysisTool(),
        AlphaVantageTool(),
    ],
    llm: str = ANALYSIS_AGENT_LLM,
    config: Dict[str, Union[str, int]] = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.4,
    default_role: str = "Investment Analyst",
    default_goal: str = "Analyze and combine quantitative and qualitative data for stock selection",
    default_backstory: str = "You are an investment analyst tasked with integrating financial and sentiment data for stock selection.",
    default_instructions: str = """
        Integrate quantitative financial data with market sentiment analysis to create comprehensive stock assessments. 
        Ensure you provide detailed analysis for each stock, combining financial metrics and sentiment data.
    """,
) -> Agent:
    """
    Create an Analysis Agent that integrates quantitative and qualitative data.

    This agent is responsible for analyzing financial data alongside market sentiment
    to produce comprehensive stock assessments. It serves as the primary analytical
    engine that combines multiple data sources into actionable insights.

    Args:
        agents_config (Dict[str, str]): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial and sentiment analysis tools.
        llm (str, optional): Language model identifier to use. Defaults to ANALYSIS_AGENT_LLM.
        config (Dict[str, Union[str, int]], optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.4.
        default_role (str, optional): Agent's role if not in config. Defaults to "Investment Analyst".
        default_goal (str, optional): Agent's goal if not in config.
            Defaults to "Analyze and combine quantitative and qualitative data for stock selection".
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Analysis Agent instance
    """
    analysis_agent_config = agents_config.get("analysis_agent", {})
    return Agent(
        role=analysis_agent_config.get("role", default_role),
        goal=analysis_agent_config.get("goal", default_goal),
        backstory=analysis_agent_config.get("backstory", default_backstory),
        verbose=verbose,
        tools=tools,
        allow_delegation=allow_delegation,
        llm=llm,
        async_execution=async_execution,
        temperature=temperature,
        instructions=analysis_agent_config.get("instructions", default_instructions),
        config=config,
        step_callback=callback,
    )


@traceable
def create_fact_agent(
    agents_config: dict,
    tools: list = [
        YFinanceTool(),
        StockSymbolFetcherTool(),
        AlphaVantageTool(),
        get_firecrawl_crawl_website_tool(),
        get_firecrawl_scrape_website_tool(),
    ],
    llm: str = FACT_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = True,
    async_execution: bool = True,
    temperature: float = 0.3,
    default_role: str = "Fact Agent",
    default_goal: str = "Analyze financial data to identify promising stocks",
    default_backstory: str = "You are a financial analyst tasked with analyzing quantitative data for stock selection.",
    default_instructions: str = """ 
        Analyze financial data for a list of stocks and identify promising candidates.
        For each stock, provide:
        - Key financial metrics
        - Ratios and performance indicators
        - Industry comparisons
        - Key investor sentiment
        - Any other relevant financial data

        Ensure you provide actual company names with correct ticker symbols.
        NEVER use placeholder names or tickers.
    """,
) -> Agent:
    """
    Create a Fact Agent that gathers and analyzes objective financial data.

    This agent is specialized in retrieving quantitative financial information and
    identifying promising stock candidates based on fundamental metrics. It provides
    the factual foundation for investment decisions.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial data tools and web scraping capabilities.
        llm (str, optional): Language model identifier to use. Defaults to FACT_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to True.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to True.
        temperature (float, optional): LLM temperature setting. Defaults to 0.3.
        default_role (str, optional): Agent's role if not in config. Defaults to "Fact Agent".
        default_goal (str, optional): Agent's goal if not in config.
            Defaults to "Analyze financial data to identify promising stocks".
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Fact Agent instance
    """
    fact_agent_config = agents_config.get("fact_agent", {})
    return Agent(
        role=fact_agent_config.get("role", default_role),
        goal=fact_agent_config.get("goal", default_goal),
        backstory=fact_agent_config.get("backstory", default_backstory),
        tools=tools,
        verbose=verbose,
        allow_delegation=allow_delegation,
        async_execution=async_execution,
        llm=llm,
        temperature=temperature,
        instructions=fact_agent_config.get("instructions", default_instructions),
        config=config,
        step_callback=callback,
    )


@traceable
def create_justification_agent(
    agents_config: dict,
    tools: list = [
        YFinanceTool(),
        StockSymbolFetcherTool(),
        SentimentAnalysisTool(),
        AlphaVantageTool(),
    ],
    llm: str = JUSTIFICATION_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    default_role: str = "Investment Justification Analyst",
    default_goal: str = "Justify stock selections based on quantitative and qualitative data",
    default_backstory: str = "You are an investment justification analyst tasked with providing detailed rationale for stock selections.",
    default_instructions: str = """
        Provide detailed justification for the most promising stocks based on fundamental and sentiment analysis.
        Ensure you include a clear rationale for each stock selection, supported by quantitative and qualitative data.
    """,
) -> Agent:
    """
    Create a Justification Agent that provides reasoned arguments for stock selections.

    This agent creates detailed rationales for why certain stocks were selected,
    combining both quantitative metrics and qualitative factors into coherent
    justifications that explain investment decisions.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial and sentiment analysis tools.
        llm (str, optional): Language model identifier to use. Defaults to JUSTIFICATION_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        default_role (str, optional): Agent's role if not in config.
            Defaults to "Investment Justification Analyst".
        default_goal (str, optional): Agent's goal if not in config.
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Justification Agent instance
    """
    justification_agent_config = agents_config.get("justification_agent", {})
    return Agent(
        role=justification_agent_config.get("role", default_role),
        goal=justification_agent_config.get("goal", default_goal),
        backstory=justification_agent_config.get("backstory", default_backstory),
        tools=tools,
        allow_delegation=allow_delegation,
        llm=llm,
        verbose=verbose,
        instructions=justification_agent_config.get(
            "instructions", default_instructions
        ),
        config=config,
        step_callback=callback,
    )


@traceable
def create_optimization_agent(
    agents_config: dict,
    tools: list = [
        YFinanceTool(),
        StockSymbolFetcherTool(),
        SentimentAnalysisTool(),
        AlphaVantageTool(),
    ],
    llm: str = OPTIMIZATION_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.3,
    default_role: str = "Investment Optimizer",
    default_goal: str = "Optimize stock selections based on risk-reward profiles",
    default_backstory: str = "You are an investment optimizer tasked with refining stock selections based on risk-reward profiles.",
    default_instructions: str = """
        Critically analyze and refine the stock selections based on risk-reward profiles.
        Ensure you provide an optimized selection of stocks with risk assessment and portfolio fit analysis.
    """,
) -> Agent:
    """
    Create an Optimization Agent that refines stock selections based on risk-reward profiles.

    This agent takes preliminary stock selections and refines them by evaluating
    their risk-reward characteristics, ensuring a balanced portfolio that aligns
    with investment objectives and risk tolerances.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial and sentiment analysis tools.
        llm (str, optional): Language model identifier to use. Defaults to OPTIMIZATION_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.3.
        default_role (str, optional): Agent's role if not in config. Defaults to "Investment Optimizer".
        default_goal (str, optional): Agent's goal if not in config.
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Optimization Agent instance
    """
    optimization_agent_config = agents_config.get("optimization_agent", {})
    return Agent(
        role=optimization_agent_config.get("role", default_role),
        goal=optimization_agent_config.get("goal", default_goal),
        backstory=optimization_agent_config.get("backstory", default_backstory),
        instructions=optimization_agent_config.get(
            "instructions", default_instructions
        ),
        verbose=verbose,
        llm=llm,
        allow_delegation=allow_delegation,
        tools=[
            YFinanceTool(),
            SentimentAnalysisTool(),
            StockSymbolFetcherTool(),
            AlphaVantageTool(),
        ],
        async_execution=async_execution,
        temperature=temperature,
        config=config,
        step_callback=callback,
    )


@traceable
def create_portfolio_manger_agent(
    agents_config: dict,
    llm: str = PORTFOLIO_MANAGER_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = True,
    default_role: str = "Portfolio Manager",
    default_goal: str = "Oversee the entire investment process and manage the team",
    default_backstory: str = "You are an experienced portfolio manager tasked with overseeing the investment process.",
) -> Agent:
    """
    Create a Portfolio Manager Agent that oversees the entire investment process.

    This agent acts as a coordinator for the other agents, managing the overall
    investment strategy and ensuring that the collective analysis leads to coherent
    investment decisions aligned with portfolio objectives.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        llm (str, optional): Language model identifier to use. Defaults to PORTFOLIO_MANAGER_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to True.
        default_role (str, optional): Agent's role if not in config. Defaults to "Portfolio Manager".
        default_goal (str, optional): Agent's goal if not in config.
            Defaults to "Oversee the entire investment process and manage the team".
        default_backstory (str, optional): Agent's backstory if not in config.

    Returns:
        Agent: Configured Portfolio Manager Agent instance

    Note:
        This agent does not have default tools as it primarily coordinates other agents
        rather than performing direct analysis.
    """
    portfolio_manager_config = agents_config.get("portfolio_manager", {})
    return Agent(
        role=portfolio_manager_config.get("role", default_role),
        goal=portfolio_manager_config.get("goal", default_goal),
        backstory=portfolio_manager_config.get("backstory", default_backstory),
        verbose=verbose,
        allow_delegation=allow_delegation,
        llm=llm,
        config=config,
        step_callback=callback,
    )


@traceable
def create_recommendation_agent(
    agents_config: dict,
    tools: list = [
        YFinanceTool(),
        SentimentAnalysisTool(),
        StockSymbolFetcherTool(),
        AlphaVantageTool(),
    ],
    llm: str = RECOMMENDATION_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.5,
    default_role: str = "Investment Recommendation Agent",
    default_goal: str = "Provide actionable investment recommendations",
    default_backstory: str = "You are an investment recommendation agent tasked with providing actionable investment recommendations.",
) -> Agent:
    """
    Create a Recommendation Agent that provides actionable investment advice.

    This agent transforms analytical insights into concrete investment recommendations,
    providing clear guidance on which stocks to invest in and why they represent
    good opportunities.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial and sentiment analysis tools.
        llm (str, optional): Language model identifier to use. Defaults to RECOMMENDATION_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.5.
        default_role (str, optional): Agent's role if not in config.
            Defaults to "Investment Recommendation Agent".
        default_goal (str, optional): Agent's goal if not in config.
            Defaults to "Provide actionable investment recommendations".
        default_backstory (str, optional): Agent's backstory if not in config.

    Returns:
        Agent: Configured Recommendation Agent instance

    Note:
        Unlike other agents, this one directly uses constant values for llm, config,
        and step_callback instead of using the parameters.
    """
    recommendation_agent_config = agents_config.get("thesis_agent", {})
    return Agent(
        role=recommendation_agent_config.get("role", default_role),
        goal=recommendation_agent_config.get("goal", default_goal),
        backstory=recommendation_agent_config.get("backstory", default_backstory),
        tools=tools,
        instructions=recommendation_agent_config.get("instructions", ""),
        verbose=verbose,
        allow_delegation=allow_delegation,
        llm=RECOMMENDATION_AGENT_LLM,
        async_execution=async_execution,
        temperature=temperature,
        config=AGENT_META_CONFIG,
        step_callback=langsmith_step_callback,
    )


@traceable
def create_sentiment_agent(
    agents_config: dict,
    tools: list = [
        StockSymbolFetcherTool(),
        SentimentAnalysisTool(),
        get_firecrawl_crawl_website_tool(),
        get_firecrawl_scrape_website_tool(),
    ],
    llm: str = SENTIMENT_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.5,
    default_role: str = "Sentiment Analyst",
    default_goal: str = "Analyze public sentiment and news for stock selection",
    default_backstory: str = "You are an expert in analyzing public sentiment and news for stock selection.",
    default_instructions: str = """
        Perform comprehensive sentiment analysis for the stocks identified by the fact agent.
        For each stock:
        1. Run the basic financial news sentiment analysis
        2. Conduct targeted searches on:
           - Product reviews and customer satisfaction
           - CEO reputation and leadership team
           - Social media mentions and trends
           - Environmental and social responsibility (ESG)
        3. Synthesize these different sentiment dimensions into a comprehensive profile
        
        Focus on creating a MULTI-DIMENSIONAL sentiment analysis that goes beyond just financial news.
    """,
) -> Agent:
    """
    Create a Sentiment Agent that analyzes public perception and news about stocks.

    This agent focuses on qualitative data, analyzing news, social media, and public
    sentiment to determine how companies are perceived in the market. It provides
    insights that complement the quantitative financial analysis.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            sentiment analysis and web scraping tools.
        llm (str, optional): Language model identifier to use. Defaults to SENTIMENT_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.5.
        default_role (str, optional): Agent's role if not in config. Defaults to "Sentiment Analyst".
        default_goal (str, optional): Agent's goal if not in config.
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Sentiment Agent instance
    """
    sentiment_agent_config = agents_config.get("sentiment_agent", {})
    return Agent(
        role=sentiment_agent_config.get("role", default_role),
        goal=sentiment_agent_config.get("goal", default_goal),
        backstory=sentiment_agent_config.get("backstory", default_backstory),
        tools=tools,
        verbose=verbose,
        allow_delegation=allow_delegation,
        llm=llm,
        async_execution=async_execution,
        temperature=temperature,
        instructions=sentiment_agent_config.get("instructions", default_instructions),
        config=config,
        step_callback=callback,
    )


@traceable
def create_synthesizer_agent(
    agents_config: dict,
    tools: list = [
        YFinanceTool(),
        SentimentAnalysisTool(),
        StockSymbolFetcherTool(),
        AlphaVantageTool(),
    ],
    llm: str = SYNTHESIZER_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.4,
    default_role: str = "Investment Synthesizer",
    default_goal: str = "Synthesize investment analyses into a coherent and unified analysis",
    default_backstory: str = "You are a senior portfolio strategist who excels at synthesizing diverse analyses into coherent investment strategies.",
    default_instructions: str = """
        For each stock:
        1. Perform comprehensive financial analysis using Yahoo Finance
        2. Conduct sentiment analysis using the sentiment analysis tool
        3. Retrieve historical price data using Alpha Vantage
        4. Fetch company information and stock symbols using the stock symbol fetcher tool
        5. Integrate all analyses to select the top 5 most promising investment opportunities
        6. Provide a final selection of top 5 stocks with a rationale for inclusion
    """,
) -> Agent:
    """
    Create a Synthesizer Agent that integrates diverse investment analyses.

    This agent combines various analytical perspectives into a coherent investment
    strategy. It aggregates the outputs from financial analysis, sentiment analysis,
    and historical data to identify the most promising investment opportunities.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial and sentiment analysis tools.
        llm (str, optional): Language model identifier to use. Defaults to SYNTHESIZER_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.4.
        default_role (str, optional): Agent's role if not in config. Defaults to "Investment Synthesizer".
        default_goal (str, optional): Agent's goal if not in config.
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Synthesizer Agent instance

    Note:
        While tools are provided as a parameter, the agent is initialized with an empty
        tools list. This is because the synthesizer primarily works with the outputs of
        other agents rather than directly using tools itself.
    """
    synthesizer_agent_config = agents_config.get("synthesizer_agent", {})
    return Agent(
        role=synthesizer_agent_config.get("role", default_role),
        goal=synthesizer_agent_config.get("goal", default_goal),
        backstory=synthesizer_agent_config.get("backstory", default_backstory),
        verbose=verbose,
        llm=llm,
        allow_delegation=allow_delegation,
        tools=[],
        async_execution=async_execution,
        temperature=temperature,
        instructions=synthesizer_agent_config.get("instructions", default_instructions),
        config=config,
        step_callback=callback,
    )


@traceable
def create_thesis_agent(
    agents_config: dict,
    tools: list = [
        YFinanceTool(),
        SentimentAnalysisTool(),
        StockSymbolFetcherTool(),
        AlphaVantageTool(),
    ],
    llm: str = THESIS_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.5,
    default_role: str = "Investment Thesis Developer",
    default_goal: str = "Develop a comprehensive investment thesis based on the selected stocks",
    default_backstory: str = "You are an investment thesis developer tasked with creating detailed investment theses for selected stocks.",
    default_instructions: str = """
        Develop comprehensive investment theses for the top 5 selected stocks.
        Ensure you provide a detailed investment thesis for each stock with a compelling narrative and supporting evidence.
    """,
) -> Agent:
    """
    Create a Thesis Agent that develops comprehensive investment narratives.

    This agent constructs detailed investment theses for selected stocks, providing
    coherent narratives that incorporate fundamental analysis, market trends,
    competitive positioning, and future growth prospects. These theses serve as
    the foundation for investment decisions.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        tools (list, optional): List of tools available to the agent. Defaults to
            financial and sentiment analysis tools.
        llm (str, optional): Language model identifier to use. Defaults to THESIS_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.5.
        default_role (str, optional): Agent's role if not in config.
            Defaults to "Investment Thesis Developer".
        default_goal (str, optional): Agent's goal if not in config.
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Thesis Agent instance

    Note:
        Unlike other agents, this one directly uses constant values for llm, config,
        and step_callback instead of using the parameters passed to the function.
    """
    thesis_agent_config = agents_config.get("thesis_agent", {})
    return Agent(
        role=thesis_agent_config.get("role", default_role),
        goal=thesis_agent_config.get("goal", default_goal),
        backstory=thesis_agent_config.get("backstory", default_backstory),
        instructions=thesis_agent_config.get("instructions", default_instructions),
        tools=tools,
        verbose=verbose,
        llm=THESIS_AGENT_LLM,
        allow_delegation=allow_delegation,
        async_execution=async_execution,
        temperature=temperature,
        config=AGENT_META_CONFIG,
        step_callback=langsmith_step_callback,
    )


@traceable
def create_terminator_agent(
    agents_config: dict,
    llm: str = ANALYSIS_AGENT_LLM,
    config: dict = AGENT_META_CONFIG,
    callback: Callable[[int], None] = langsmith_step_callback,
    verbose: bool = VERBOSE,
    allow_delegation: bool = False,
    async_execution: bool = False,
    temperature: float = 0.5,
    default_role: str = "Process Terminator",
    default_goal: str = "Ensure clean termination of the stocksage workflow",
    default_backstory: str = "You are a process terminator tasked with ensuring all resources are properly released and the process terminates cleanly.",
    default_instructions: str = """
        Ensure all resources are properly released and the process terminates cleanly.
        """,
) -> Agent:
    """
    Create a Terminator Agent that ensures proper cleanup of resources.

    This specialized utility agent is responsible for the final stage of the workflow,
    ensuring that all resources are properly released and that the process terminates
    cleanly. It helps prevent resource leaks and ensures an orderly shutdown of
    the analysis process.

    Args:
        agents_config (dict): Configuration dictionary for all agents
        llm (str, optional): Language model identifier to use. Defaults to ANALYSIS_AGENT_LLM.
        config (dict, optional): Agent configuration parameters.
            Defaults to AGENT_META_CONFIG.
        callback (Callable[[int], None], optional): Step callback function.
            Defaults to langsmith_step_callback.
        verbose (bool, optional): Whether to enable verbose output. Defaults to VERBOSE.
        allow_delegation (bool, optional): Whether agent can delegate tasks. Defaults to False.
        async_execution (bool, optional): Whether to execute tasks asynchronously. Defaults to False.
        temperature (float, optional): LLM temperature setting. Defaults to 0.5.
        default_role (str, optional): Agent's role if not in config. Defaults to "Process Terminator".
        default_goal (str, optional): Agent's goal if not in config.
        default_backstory (str, optional): Agent's backstory if not in config.
        default_instructions (str, optional): Agent's instructions if not in config.

    Returns:
        Agent: Configured Terminator Agent instance

    Note:
        This agent does not use any tools as its primary function is to manage
        process termination rather than perform analysis. It also has max_iterations
        set to 1 to ensure it runs exactly once at the end of the workflow.
    """
    terminator_agent_config = agents_config.get("terminator_agent", {})
    return Agent(
        role=terminator_agent_config.get("role", default_role),
        goal=terminator_agent_config.get("goal", default_goal),
        backstory=terminator_agent_config.get("backstory", default_backstory),
        instructions=terminator_agent_config.get("instructions", default_instructions),
        allow_delegation=allow_delegation,
        verbose=verbose,
        max_iterations=1,
        llm=llm,
        config=config,
        step_callback=callback,
    )
