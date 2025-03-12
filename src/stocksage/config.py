PORTFOLIO_MANAGER_LLM = "openai/gpt-4o"
FACT_AGENT_LLM = "openai/gpt-4o"
SENTIMENT_AGENT_LLM = "openai/gpt-4o"
ANALYSIS_AGENT_LLM = "openai/gpt-4o"
SYNTHESIZER_AGENT_LLM = "openai/gpt-4o"
JUSTIFICATION_AGENT_LLM = "openai/gpt-4o"
OPTIMIZATION_AGENT_LLM = "openai/gpt-4o"
THESIS_AGENT_LLM = "openai/gpt-4o"
RECOMMENDATION_AGENT_LLM = "openai/gpt-4o"
CHAT_LLM = "openai/gpt-4o"
FUNCTION_CALLING_LLM = "openai/gpt-4o"

AGENT_META_CONFIG = {"timeout": 300, "retry": True}
VERBOSE = True
TASK_DELEGATION_CONFIG = {
    "use_task_output": True,
    "stringify_task_and_context": True,
    "parallel_tasks_limit": 10,
    "error_handling": "continue_on_error",
}
MEMORY = True
CACHE = True
