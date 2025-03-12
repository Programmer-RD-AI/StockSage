from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stocksage.utils.telemetry_tracking import (
    langsmith_step_callback,
    langsmith_task_callback,
)
from dotenv import load_dotenv
from .config import (
    CHAT_LLM,
    FUNCTION_CALLING_LLM,
    TASK_DELEGATION_CONFIG,
    VERBOSE,
    MEMORY,
    CACHE,
)
from .agents import (
    create_portfolio_manger_agent,
    create_fact_agent,
    create_sentiment_agent,
    create_analysis_agent,
    create_justification_agent,
    create_optimization_agent,
    create_synthesizer_agent,
    create_thesis_agent,
    create_recommendation_agent,
    create_terminator_agent,
)
from .tasks import (
    create_manage_investment_process_task,
    create_verify_thesis_json_task,
    create_analyze_stock_data_task,
    create_analyze_sentiment_task,
    create_perform_integrated_analysis_task,
    create_justify_stock_selection_task,
    create_optimize_stock_selection_task,
    create_justify_sentiment_selection_task,
    create_optimize_sentiment_selection_task,
    create_synthesize_final_selection_task,
    create_refine_stock_analysis_task,
    create_generate_investment_thesis_task,
    create_store_thesis_json_task,
    create_simplified_thesis_json_task,
    create_terminate_process_task,
)

load_dotenv()


@CrewBase
class StockSage:
    """
    A hierarchical multi-agent system for US stock selection and analysis.

    This class orchestrates multiple specialized agents that work together to analyze
    stock data, evaluate sentiment, generate investment theses, and provide
    recommendations for top stock selections.

    Attributes:
        agents_config (str): Path to the agents configuration file
        tasks_config (str): Path to the tasks configuration file
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def portfolio_manager(self) -> Agent:
        """
        Creates the portfolio manager agent.

        This agent oversees the overall investment process and coordinates
        the work of other specialized agents.

        Returns:
            Agent: A configured portfolio manager agent
        """
        return create_portfolio_manger_agent(self.agents_config)

    @agent
    def fact_agent(self) -> Agent:
        """
        Creates the fact agent.

        This agent is responsible for gathering and analyzing factual data
        about stocks, including financial metrics and market data.

        Returns:
            Agent: A configured fact agent
        """
        return create_fact_agent(self.agents_config)

    @agent
    def sentiment_agent(self) -> Agent:
        """
        Creates the sentiment agent.

        This agent analyzes market sentiment, news, and social media data
        to gauge overall market perception of stocks.

        Returns:
            Agent: A configured sentiment agent
        """
        return create_sentiment_agent(self.agents_config)

    @agent
    def analysis_agent(self) -> Agent:
        """
        Creates the analysis agent.

        This agent performs integrated analysis of stock data combining
        both factual information and sentiment data.

        Returns:
            Agent: A configured analysis agent
        """
        return create_analysis_agent(self.agents_config)

    @agent
    def justification_agent(self) -> Agent:
        """
        Creates the justification agent.

        This agent provides reasoned justification for stock selections
        based on analyzed data and criteria.

        Returns:
            Agent: A configured justification agent
        """
        return create_justification_agent(self.agents_config)

    @agent
    def optimization_agent(self) -> Agent:
        """
        Creates the optimization agent.

        This agent optimizes stock selections based on various criteria
        including risk, return potential, and market conditions.

        Returns:
            Agent: A configured optimization agent
        """
        return create_optimization_agent(self.agents_config)

    @agent
    def synthesizer_agent(self) -> Agent:
        """
        Creates the synthesizer agent.

        This agent synthesizes information from various sources and agents
        to create a comprehensive view of the selected stocks.

        Returns:
            Agent: A configured synthesizer agent
        """
        return create_synthesizer_agent(self.agents_config)

    @agent
    def thesis_agent(self) -> Agent:
        """
        Creates the thesis agent.

        This agent generates investment theses for selected stocks based on
        synthesized information and analysis.

        Returns:
            Agent: A configured thesis agent
        """
        return create_thesis_agent(self.agents_config)

    @agent
    def recommendation_agent(self) -> Agent:
        """
        Creates the recommendation agent.

        This agent provides final investment recommendations based on the
        complete analysis and thesis development.

        Returns:
            Agent: A configured recommendation agent
        """
        return create_recommendation_agent(self.agents_config)

    @task
    def manage_investment_process(self) -> Task:
        """
        Creates a task for managing the overall investment process.

        This task oversees the coordination of all other tasks and ensures
        the overall investment strategy is properly executed.

        Returns:
            Task: A configured investment process management task
        """
        return create_manage_investment_process_task(
            self.tasks_config, self.portfolio_manager()
        )

    @task
    def verify_thesis_json(self) -> Task:
        """
        Creates a task for verifying the thesis JSON structure.

        This task ensures that the generated investment thesis is properly
        structured and contains all required information.

        Returns:
            Task: A configured thesis verification task
        """
        return create_verify_thesis_json_task(
            self.tasks_config,
            self.thesis_agent(),
            context=[self.create_simplified_thesis_json()],
        )

    @task
    def analyze_stock_data(self) -> Task:
        """
        Creates a task for analyzing stock data.

        This task gathers and analyzes factual stock data including financial
        metrics, historical performance, and market data.

        Returns:
            Task: A configured stock data analysis task
        """
        return create_analyze_stock_data_task(self.tasks_config, self.fact_agent())

    @task
    def analyze_sentiment(self) -> Task:
        """
        Creates a task for analyzing market sentiment.

        This task evaluates market sentiment from news sources, social media,
        and other inputs to gauge perception of stocks.

        Returns:
            Task: A configured sentiment analysis task
        """
        return create_analyze_sentiment_task(
            self.tasks_config, self.sentiment_agent(), [self.analyze_stock_data()]
        )

    @task
    def perform_integrated_analysis(self) -> Task:
        """
        Creates a task for performing integrated analysis.

        This task combines factual stock data with sentiment analysis
        to create a comprehensive view of each stock's potential.

        Returns:
            Task: A configured integrated analysis task
        """
        return create_perform_integrated_analysis_task(
            self.tasks_config,
            self.analysis_agent(),
            [self.analyze_stock_data(), self.analyze_sentiment()],
        )

    @task
    def justify_stock_selection(self) -> Task:
        """
        Creates a task for justifying stock selections.

        This task provides rational justification for including specific
        stocks based on the integrated analysis.

        Returns:
            Task: A configured stock selection justification task
        """
        return create_justify_stock_selection_task(
            self.tasks_config,
            self.justification_agent(),
            [self.perform_integrated_analysis()],
        )

    @task
    def optimize_stock_selection(self) -> Task:
        """
        Creates a task for optimizing stock selections.

        This task refines the stock selection based on optimization criteria
        to improve the overall portfolio composition.

        Returns:
            Task: A configured stock selection optimization task
        """
        return create_optimize_stock_selection_task(
            self.tasks_config,
            self.optimization_agent(),
            [
                self.perform_integrated_analysis(),
                self.justify_stock_selection(),
            ],
        )

    @task
    def justify_sentiment_selection(self) -> Task:
        """
        Creates a task for justifying sentiment-based selections.

        This task provides rational justification for selections based on
        sentiment analysis results.

        Returns:
            Task: A configured sentiment selection justification task
        """
        return create_justify_sentiment_selection_task(
            self.tasks_config,
            self.justification_agent(),
            [self.perform_integrated_analysis()],
        )

    @task
    def optimize_sentiment_selection(self) -> Task:
        """
        Creates a task for optimizing sentiment-based selections.

        This task refines the selections based on sentiment optimization
        to improve the reliability of sentiment-based inputs.

        Returns:
            Task: A configured sentiment selection optimization task
        """
        return create_optimize_sentiment_selection_task(
            self.tasks_config,
            self.optimization_agent(),
            [self.justify_stock_selection()],
        )

    @task
    def synthesize_final_selection(self) -> Task:
        """
        Creates a task for synthesizing the final stock selection.

        This task combines the outputs of optimization and justification tasks
        to create a final curated list of stock selections.

        Returns:
            Task: A configured final selection synthesis task
        """
        return create_synthesize_final_selection_task(
            self.tasks_config,
            self.synthesizer_agent(),
            [
                self.optimize_stock_selection(),
                self.justify_stock_selection(),
                self.justify_sentiment_selection(),
            ],
        )

    @task
    def refine_stock_analysis(self) -> Task:
        """
        Creates a task for refining the stock analysis.

        This task further refines the analysis based on optimization and
        justification outputs to improve accuracy and relevance.

        Returns:
            Task: A configured stock analysis refinement task
        """
        return create_refine_stock_analysis_task(
            self.tasks_config,
            self.analysis_agent(),
            [
                self.optimize_stock_selection(),
                self.justify_stock_selection(),
                self.justify_sentiment_selection(),
            ],
        )

    @task
    def generate_investment_thesis(self) -> Task:
        """
        Creates a task for generating investment theses.

        This task produces comprehensive investment theses for the selected
        stocks based on all prior analysis and synthesis.

        Returns:
            Task: A configured investment thesis generation task
        """
        return create_generate_investment_thesis_task(
            self.tasks_config, self.thesis_agent(), [self.synthesize_final_selection()]
        )

    @task
    def store_thesis_json(self) -> Task:
        """
        Creates a task for storing the investment thesis in JSON format.

        This task converts the generated thesis into a structured JSON format
        and handles storage for later retrieval.

        Returns:
            Task: A configured thesis storage task
        """
        return create_store_thesis_json_task(
            self.tasks_config, self.thesis_agent(), [self.generate_investment_thesis()]
        )

    @task
    def create_simplified_thesis_json(self) -> Task:
        """
        Creates a task for creating a simplified version of the thesis JSON.

        This task produces a more concise and simplified version of the investment
        thesis for easier consumption and presentation.

        Returns:
            Task: A configured simplified thesis creation task
        """
        return create_simplified_thesis_json_task(
            self.tasks_config, self.thesis_agent(), [self.generate_investment_thesis()]
        )

    @task
    def terminate_process(self) -> Task:
        """
        Creates a task for gracefully terminating the process.

        This task ensures proper cleanup and termination of all resources
        and processes when the analysis is complete.

        Returns:
            Task: A configured process termination task
        """
        return create_terminate_process_task(
            self.tasks_config,
            create_terminator_agent(self.agents_config),
            [],
        )

    # @traceable
    @crew
    def crew(self) -> Crew:
        """
        Creates and configures the complete agent crew.

        This method assembles all agents and tasks into a coordinated crew
        that will execute the entire stock analysis and selection process.

        Returns:
            Crew: A fully configured crew of agents and tasks with defined workflow
        """
        return Crew(
            agents=[
                self.fact_agent(),
                self.sentiment_agent(),
                self.analysis_agent(),
                self.recommendation_agent(),
                self.justification_agent(),
                self.optimization_agent(),
                self.synthesizer_agent(),
                self.thesis_agent(),
            ],
            tasks=[
                self.analyze_stock_data(),
                self.analyze_sentiment(),
                self.perform_integrated_analysis(),
                self.justify_stock_selection(),
                self.optimize_stock_selection(),
                self.justify_sentiment_selection(),
                self.optimize_sentiment_selection(),
                self.refine_stock_analysis(),
                self.synthesize_final_selection(),
                self.manage_investment_process(),
                self.generate_investment_thesis(),
                self.store_thesis_json(),
                self.create_simplified_thesis_json(),
                self.terminate_process(),
            ],
            manager_agent=self.portfolio_manager(),
            process=Process.hierarchical,
            verbose=VERBOSE,
            memory=MEMORY,
            cache=CACHE,
            function_calling_llm=FUNCTION_CALLING_LLM,
            chat_llm=CHAT_LLM,
            step_callback=langsmith_step_callback,
            task_callback=langsmith_task_callback,
            task_delegation_config=TASK_DELEGATION_CONFIG,
        )
