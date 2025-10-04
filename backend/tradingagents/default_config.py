import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": os.getenv("TRADINGAGENTS_DATA_DIR", "./data"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "anthropic",
    "deep_think_llm": "claude-sonnet-4-0",
    "quick_think_llm": "claude-sonnet-4-0",
    "backend_url": "https://api.anthropic.com",
    # Debate and discussion settings
    "max_debate_rounds": int(os.getenv("MAX_DEBATE_ROUNDS", "2")),
    "max_risk_discuss_rounds": int(os.getenv("MAX_RISK_DISCUSS_ROUNDS", "2")),
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
}
