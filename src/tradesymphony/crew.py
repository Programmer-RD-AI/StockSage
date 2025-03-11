from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stocksage.tools.finance_tools import (
    YFinanceTool,
    AlphaVantageTool,
    SentimentAnalysisTool,
)
from dotenv import load_dotenv

load_dotenv()


@CrewBase
class stocksage:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def portfolio_manager(self) -> Agent:
        """Portfolio Manager agent who oversees the investment process"""
        return Agent(
            role="Portfolio Manager",
            goal="Oversee the entire investment analysis process and ensure high-quality output",
            backstory="""You are the chief investment officer of a prestigious asset management firm.
            With decades of experience across various market cycles, you excel at coordinating
            teams of specialized analysts and synthesizing their work into actionable investment strategies.
            Your expertise lies in delegating tasks effectively and making the final investment decisions.""",
            verbose=True,
            allow_delegation=True,
            max_delegation=5,
            max_rpm=20,
            llm="openai/gpt-4o",
        )

    @agent
    def fact_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_agent"],
            tools=[YFinanceTool()],
            verbose=True,
            allow_delegation=True,
            async_execution=True,
            temperature=0.3,  # Lower temperature for more factual responses
            instructions="""
            You are responsible for providing FACTUAL financial information.
            ALWAYS use real company names and accurate ticker symbols in your analysis.
            NEVER use placeholder names like 'Company A' or 'Stock B'.
            Use your tools to verify company information:
            - YFinance Stock Data Tool(ticker="SYMBOL", metrics=["PE", "PB", "ROE"])
            - Alpha Vantage Financial Data Tool(ticker="SYMBOL", function="TIME_SERIES_DAILY")
            
            If you're unsure about a company or ticker, look it up before including it in your analysis.
            If you cannot find specific information about a real company, say so explicitly
            rather than using placeholder names.
            
            IMPORTANT: Your response will be rejected if it contains placeholder company names.
            """,
        )

    @agent
    def sentiment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_agent"],
            tools=[YFinanceTool(), SentimentAnalysisTool()],
            verbose=True,
            allow_delegation=False,
            async_execution=True,  # Using async for sentiment analysis to improve performance
            temperature=0.5,
            instructions="""
            You are an expert in analyzing public perception and sentiment for companies.
            
            Your task is to gather comprehensive sentiment data for stocks by:
            1. Analyzing standard financial news using the default stock queries
            2. Performing TARGETED searches on specific aspects of company perception:
               - Use search_query="product reviews and customer satisfaction" to gauge product reception
               - Use search_query="CEO reputation and leadership team" to assess management perception
               - Use search_query="social media mentions and trends" to evaluate social buzz
               - Use search_query="environmental and social responsibility" for ESG perception
            
            For EACH company, run MULTIPLE sentiment analyses with different search queries to build a 
            complete picture of public perception across different dimensions.
            
            Synthesize these multiple sentiment analyses into a comprehensive profile that shows:
            - Overall financial sentiment (bullish, bearish, or neutral)
            - Product/service reputation among consumers
            - Leadership perception by industry and public
            - Social media presence and trends
            - ESG (Environmental, Social, Governance) reputation if applicable
            
            ALWAYS ensure you provide actual company names with correct ticker symbols.
            NEVER use placeholder names or tickers.
            """,
        )

    @agent
    def analysis_agent(self) -> Agent:
        """Analysis agent who synthesizes data from fact and sentiment agents"""
        return Agent(
            role="Financial Analyst",
            goal="Synthesize quantitative data and sentiment analysis into comprehensive stock analyses",
            backstory="""You are a seasoned financial analyst with expertise in combining fundamental data 
            with market sentiment to identify investment opportunities. Your strength lies in finding 
            patterns and insights that emerge when quantitative metrics are viewed alongside market sentiment.""",
            verbose=True,
            tools=[YFinanceTool()],
            allow_delegation=False,
            async_execution=False,  # Sequential for careful analysis
            temperature=0.4,  # Balanced temperature for analysis
            instructions="""
            Your task is to carefully analyze and combine:
            1. Quantitative data from the fact agent (financial metrics, ratios, etc.)
            2. Qualitative insights from the sentiment agent (news sentiment, social media trends, etc.)
            
            For each stock, provide:
            - Summary of key financial metrics
            - Summary of sentiment analysis
            - Integrated analysis showing how both factors affect investment potential
            - Clear recommendation based on the combined analysis
            
            Be thorough and consider all available data points.
            
            CRITICAL REQUIREMENT:
            - ALWAYS use real company names and correct ticker symbols (e.g., "Microsoft Corporation (MSFT)")
            - NEVER use placeholder names like 'Company A', 'Stock B', or generic descriptions
            - If you don't have enough information about a specific company, use YFinanceTool to look it up
            - Your analysis must reference actual companies that exist in the real market
            """,
        )

    @agent
    def justification_agent(self):
        return Agent(
            role="Investment Analyst",
            goal="Provide detailed justification for stock selections",
            backstory="You are an expert investment analyst with extensive experience in quantitative and qualitative stock analysis.",
            tools=[YFinanceTool()],  # Add this tool to verify company information
            allow_delegation=False,
            verbose=True,
            instructions="""
            Review ALL stocks from the previous analysis. You must justify all stocks that were selected.
            For each stock, provide:
            1. Ticker symbol and company name
            2. Key quantitative metrics that justified selection
            3. Comparison to industry averages
            4. Strategic alignment with investment themes
            
            MANDATORY REQUIREMENT: 
            - You MUST use real company names with correct ticker symbols
            - Example: "Apple Inc. (AAPL)" not "Company A" or "Tech Stock"
            - Use your YFinanceTool to verify company information if needed
            - Never use placeholder or generic references to companies
            
            DO NOT limit your response to just a few examples. You must cover all stocks from the input.
            Format each stock with a clear heading and bullet points for readability.
            """,
        )

    @agent
    def optimization_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["optimization_agent"],
            verbose=True,
            allow_delegation=False,
            tools=[YFinanceTool()],  # Add this tool to verify company information
            async_execution=False,  # Sequential for careful analysis
            temperature=0.3,  # Lower temperature for more precise optimization
        )

    @agent
    def synthesizer_agent(self) -> Agent:
        agent_config = self.agents_config.get("synthesizer_agent", {})
        return Agent(
            role=agent_config.get("role", "Investment Synthesizer"),
            goal=agent_config.get(
                "goal", "Synthesize all analyses into final investment recommendations"
            ),
            backstory=agent_config.get(
                "backstory",
                "You are a senior portfolio strategist who excels at integrating diverse analyses into coherent investment strategies.",
            ),
            verbose=True,
            allow_delegation=False,
            tools=[YFinanceTool()],  # Add this tool to verify company information
            async_execution=False,
            temperature=0.4,
            instructions="""
            Synthesize all previous analyses to select the top 5 most promising investment opportunities.
            
            CRITICAL INSTRUCTIONS:
            1. ALWAYS use real company names with their correct ticker symbols (e.g., "Microsoft Corporation (MSFT)")
            2. NEVER use placeholder names like 'Company A', 'Stock B', or any generic references
            3. Verify all company names and tickers using YFinanceTool before including them
            4. If you're unsure about a company, research it - don't guess or use a placeholder
            5. Your output WILL BE REJECTED if it contains any placeholder company references
            
            For each selected company, include:
            - Full company name and correct ticker symbol
            - Brief summary of why it was selected
            - Key metrics that support the investment thesis
            """,
        )

    @agent
    def thesis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["thesis_agent"],
            verbose=True,
            allow_delegation=False,
            async_execution=False,  # Sequential for coherent thesis writing
            temperature=0.5,  # Slightly higher temperature for creative writing
        )

    @task
    def manage_investment_process(self) -> Task:
        """The main task for the Portfolio Manager to oversee the entire process"""
        task_config = self.tasks_config["manage_investment_process"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.portfolio_manager(),
            output_file="outputs/investment_report.md",
        )

    @task
    def verify_thesis_json(self) -> Task:
        return Task(
            description="""Verify that the investment thesis JSON contains REAL companies with CORRECT tickers.
            1. Check each company name and ticker pair for accuracy
            2. Replace ANY generic names like "Stock A" with real companies
            3. Ensure the output structure matches the required format with "investments" array
            4. Maintain the same high-quality thesis content for each company
            """,
            expected_output="A verified JSON file with real company names and correct tickers",
            agent=self.thesis_agent(),
            context=[self.create_simplified_thesis_json()],
            output_file="outputs/verified_thesis.json",
        )

    @task
    def analyze_stock_data(self) -> Task:
        return Task(
            description="Analyze financial data for a list of stocks and identify promising candidates",
            expected_output="A detailed analysis of 10-15 stocks with quantitative metrics and initial screening",
            agent=self.fact_agent(),
            context=[],
        )

    @task
    def analyze_sentiment(self) -> Task:
        return Task(
            description="""
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
            expected_output="""
            A comprehensive sentiment analysis report for each stock with:
            - Overall financial sentiment score and interpretation
            - Product/service reputation assessment
            - Leadership perception analysis
            - Social media presence evaluation
            - ESG reputation (if applicable)
            - News items supporting each dimension
            - Integrated sentiment profile showing the company's public perception across all dimensions
            """,
            agent=self.sentiment_agent(),
            context=[self.analyze_stock_data()],
        )

    @task
    def perform_integrated_analysis(self) -> Task:
        """Integrates fact-based data with sentiment analysis"""
        return Task(
            description="Integrate quantitative financial data with market sentiment analysis to create comprehensive stock assessments",
            expected_output="Detailed integrated analysis combining financial metrics and sentiment for each stock, with clear recommendations",
            agent=self.analysis_agent(),
            context=[self.analyze_stock_data(), self.analyze_sentiment()],
            output_file="outputs/integrated_analysis.md",
        )

    @task
    def justify_stock_selection(self) -> Task:
        task_config = self.tasks_config["justify_stock_selection"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.justification_agent(),
            # Now using the integrated analysis as context
            context=[self.perform_integrated_analysis()],
            output_file="outputs/stock_selection_justification.md",
        )

    @task
    def optimize_stock_selection(self) -> Task:
        task_config = self.tasks_config["optimize_stock_selection"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.optimization_agent(),
            # Now using integrated analysis and justification as context
            context=[
                self.perform_integrated_analysis(),
                self.justify_stock_selection(),
            ],
            output_file="outputs/optimized_stock_selection.md",
        )

    @task
    def justify_sentiment_selection(self) -> Task:
        return Task(
            description="Provide detailed justification for the most promising stocks based on fundamental and sentiment analysis",
            expected_output="Comprehensive justification for 7-10 stock selections with supporting evidence",
            agent=self.justification_agent(),
            # Updated to use integrated analysis
            context=[self.perform_integrated_analysis()],
        )

    @task
    def optimize_sentiment_selection(self) -> Task:
        return Task(
            description="Critically analyze and refine the stock selections based on risk-reward profiles",
            expected_output="Optimized selection of stocks with risk assessment and portfolio fit analysis",
            agent=self.optimization_agent(),
            context=[self.justify_stock_selection()],
        )

    @task
    def synthesize_final_selection(self) -> Task:
        return Task(
            description="Integrate all analyses to select the top 5 most promising investment opportunities",
            expected_output="Final selection of top 5 stocks with rationale for inclusion",
            agent=self.synthesizer_agent(),
            # Using optimized stock selection and integrated analysis
            context=[
                self.optimize_stock_selection(),
                self.perform_integrated_analysis(),
            ],
        )

    @task
    def generate_investment_thesis(self) -> Task:
        return Task(
            description="Develop comprehensive investment theses for the top 5 selected stocks",
            expected_output="Detailed investment thesis for each stock with compelling narrative and supporting evidence",
            agent=self.thesis_agent(),
            context=[self.synthesize_final_selection()],
        )

    @task
    def store_thesis_json(self) -> Task:
        return Task(
            description="Convert the investment theses to JSON format. Extract the company names, tickers, sectors, and full investment theses from the previously generated report.",
            expected_output="A structured JSON file containing each company's name, ticker, sector, and complete investment thesis.",
            agent=self.thesis_agent(),
            context=[self.generate_investment_thesis()],
            output_file="outputs/investment_thesis.json",
        )

    @task
    def create_simplified_thesis_json(self) -> Task:
        return Task(
            description="""Create a simplified JSON output for the top 5 recommended companies.
            
            **CRITICAL REQUIREMENT**: 
            1. You MUST use REAL company names with their EXACT stock tickers (e.g., "Apple Inc." with ticker "AAPL").
            2. NEVER use generic placeholders like "Stock A" or "Company B" - this is unacceptable.
            3. Structure the output with the following fields:
               - "investments": an array of objects, each containing:
                 * "company_name": the full company name (e.g., "Microsoft Corporation")
                 * "ticker": the correct stock ticker symbol (e.g., "MSFT")
                 * "thesis": the complete investment thesis for this company
            4. Verify all ticker symbols are accurate and match their companies exactly.
            5. If you're unsure about a company name or ticker, use a well-known company instead.
            """,
            expected_output="""A JSON file with EXACTLY this structure:
            {
                "investments": [
                    {
                        "company_name": "Microsoft Corporation",
                        "ticker": "MSFT",
                        "thesis": "Microsoft continues to show strong growth in cloud services..."
                    },
                    {
                        "company_name": "Apple Inc.",
                        "ticker": "AAPL", 
                        "thesis": "Apple's ecosystem and services revenue growth..."
                    },
                    ...additional companies...
                ]
            }
            """,
            agent=self.thesis_agent(),
            context=[self.generate_investment_thesis()],
            output_file="outputs/thesis_json.json",
        )

    @task
    def terminate_process(self) -> Task:
        """Creates a task that ensures proper termination after all tasks are complete."""
        return Task(
            description="Ensure all resources are properly released and terminate the workflow cleanly.",
            expected_output="Confirmation that all processes have been terminated successfully.",
            agent=Agent(
                role="Process Terminator",
                goal="Ensure clean termination of the stocksage workflow",
                backstory="I make sure all resources are properly released and the process terminates cleanly.",
                tools=[],
                allow_delegation=False,
                verbose=True,
                max_iterations=1,  # Only run once
            ),
            output_file="outputs/termination_confirmation.md",
        )

    # Add this method to cleanly shutdown resources
    def shutdown(self):
        """Explicitly release resources and terminate any running processes."""
        # Clean up code here - close connections, join threads, etc.
        print("stocksage workflow completed. All processes terminated.")
        # You could add sys.exit(0) here if needed, but that's usually too aggressive

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.fact_agent(),
                self.sentiment_agent(),
                self.analysis_agent(),  # Added the new analysis agent
                self.justification_agent(),
                self.optimization_agent(),
                self.synthesizer_agent(),
                self.thesis_agent(),
            ],
            tasks=[
                self.analyze_stock_data(),
                self.analyze_sentiment(),
                self.perform_integrated_analysis(),  # Added new integrated analysis task
                self.justify_stock_selection(),
                self.optimize_stock_selection(),
                self.justify_sentiment_selection(),
                self.optimize_sentiment_selection(),
                self.synthesize_final_selection(),
                self.manage_investment_process(),
                self.generate_investment_thesis(),
                self.store_thesis_json(),
                self.create_simplified_thesis_json(),
                self.terminate_process(),
            ],
            manager_agent=self.portfolio_manager(),
            process=Process.hierarchical,
            verbose=True,
            task_delegation_config={
                "use_task_output": True,
                "stringify_task_and_context": True,
            },
            memory=True,
        )
