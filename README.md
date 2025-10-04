# TradingAgents2 - LLM + Graphs Integration

🤖 **Real-time LLM agent communication with dynamic graph generation**

## 🎯 What This Project Does

This project demonstrates a complete **Financial Visualization Agent** that combines:
- **Real LLM agents** (Claude API) communicating and analyzing data
- **Dynamic graph generation** based on LLM reasoning
- **Real-time market + payment data** integration
- **Interactive visualizations** with insights and explanations

## 🚀 Quick Start

### 1. Setup
```bash
cd backend
python setup.py  # Creates payment data
```

### 2. Run Demo
```bash
python demo.py  # Full LLM + Graphs demo
```

### 3. Test Visualization Only
```bash
python test.py  # Test just the visualization agent
```

## 📊 Features

### 🤖 LLM Agent Communication
- **Market Analyst**: Real-time technical analysis with Yahoo Finance data
- **Payment Analyst**: Payment processing metrics and correlations  
- **Visualization Agent**: LLM-driven graph selection and creation
- **All agents use Claude API** for real reasoning and decision-making

### 📈 Dynamic Graph Generation
- **LLM analyzes data** and identifies patterns
- **LLM decides which graphs to create** and explains why
- **Graphs generated dynamically** with real market + payment data
- **Interactive visualizations** with insights and explanations

### 🎨 Visualization Types
- **Time Series**: For temporal trends
- **Scatter Plots**: For correlations
- **Heatmaps**: For multi-dimensional relationships
- **Bar Charts**: For comparisons
- **Gauge Charts**: For KPI monitoring
- **Funnel Charts**: For conversion flows
- **Sankey Diagrams**: For flow analysis

## 📁 Project Structure

```
TradingAgents2/
├── backend/                 # Main application
│   ├── tradingagents/      # Core agent system
│   │   ├── agents/         # LLM agents (market, payment, visualization)
│   │   ├── dataflows/      # Data sources and utilities
│   │   └── graph/          # Workflow orchestration
│   ├── demo.py            # Main demo (LLM + Graphs)
│   ├── test.py            # Visualization test
│   ├── setup.py           # Payment data setup
│   └── main.py            # Core system entry point
├── data/                   # Data storage
├── assets/                 # Images and resources
└── README.md              # This file
```

## 🔧 Configuration

The system uses **Claude API** for LLM reasoning:
- **Provider**: Anthropic
- **Models**: claude-sonnet-4-0
- **Features**: Real-time analysis, graph reasoning, insights generation

## 📊 Data Sources

- **Market Data**: Yahoo Finance API (OHLC, Volume, Technical Indicators)
- **Payment Data**: Enhanced payment metrics (Volume, Success Rate, Fraud Rate, etc.)
- **Combined Analysis**: Cross-domain correlations and insights

## 🎯 Challenge Requirements Met

✅ **Strategic data ingestion** - Schema inference, handling large files/streams  
✅ **Visualization recommendation** - ML/LLM reasoning for choosing encodings  
✅ **Automatic feature detection** - Anomaly/cohort finding  
✅ **Effectiveness of visualizations** - Clarity, correct aggregations, appropriate scales  
✅ **Scalability and robustness** - Performance on large datasets, resilience to edge cases  
✅ **Interactive drill-downs** - Cross-filtering, export functionality  
✅ **Shareable reports** - PNG, PDF, HTML export capabilities  

## 🚀 How It Works

```
Real Data → LLM Analysis → Graph Selection → Visualization Creation → Insights
    ↓              ↓              ↓              ↓              ↓
Yahoo Finance → Claude API → Chart Types → Plotly Charts → Actionable Insights
Payment APIs → Reasoning → Justification → Interactive → Export Options
```

## 📝 Example Output

The system generates:
- **Comprehensive technical analysis** with RSI, MACD, Bollinger Bands, ATR
- **Real-time data** from Yahoo Finance (stock data)
- **Payment data integration** with success rates, fraud rates, processing times
- **LLM reasoning** for indicator selection and analysis
- **Interactive HTML visualizations** saved to files
- **Comprehensive reports** with actionable insights

## 🎉 Result

**Complete "Best Financial Visualization Agent" experience with real LLM communication, dynamic graph generation, and comprehensive market + payment data analysis!**