import sys
import time
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Get command line arguments
if len(sys.argv) < 3:
    print("Usage: python main.py <TICKER> <DATE>")
    print("Example: python main.py AAPL 2024-01-15")
    sys.exit(1)

ticker = sys.argv[1].upper()
date = sys.argv[2]

print(f"🚀 TradingAgents Analysis")
print(f"=========================")
print(f"Ticker: {ticker}")
print(f"Date: {date}")
print()

start_time = time.time()

# Create custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["backend_url"] = "https://api.anthropic.com"
config["deep_think_llm"] = "claude-sonnet-4-0"
config["quick_think_llm"] = "claude-sonnet-4-0"
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["online_tools"] = True

# Initialize with custom config (excluding social analyst due to bug)
ta = TradingAgentsGraph(debug=True, config=config, selected_analysts=["market", "news", "fundamentals", "payment", "visualization"])

# Run the main analysis
print("🔄 Running analysis...")
final_state, decision = ta.propagate(ticker, date)

analysis_time = time.time() - start_time
print(f"\n🎯 FINAL DECISION FOR {ticker}: {decision}")
print(f"⏱️ Analysis completed in {analysis_time:.2f} seconds")