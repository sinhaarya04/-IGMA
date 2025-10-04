# TradingAgents2 - Project Structure

## 📁 Clean Project Layout

```
TradingAgents2/
│
├── backend/                          # FastAPI Backend Server
│   ├── api_server.py                 # Main API server with REST + WebSocket
│   ├── main.py                       # CLI entry point for running analysis
│   ├── requirements.txt              # Python dependencies
│   ├── .env                          # Environment variables (API keys)
│   │
│   └── tradingagents/                # Core trading agents package
│       ├── agents/                   # AI Agent implementations
│       │   ├── analysts/             # Analysis agents
│       │   │   ├── market_analyst.py
│       │   │   ├── fundamentals_analyst.py
│       │   │   ├── news_analyst.py
│       │   │   ├── payment_analyst.py
│       │   │   └── enhanced_visualization_agent.py
│       │   ├── managers/             # Coordination agents
│       │   │   ├── research_manager.py
│       │   │   └── risk_manager.py
│       │   ├── researchers/          # Debate agents
│       │   │   ├── bull_researcher.py
│       │   │   └── bear_researcher.py
│       │   ├── risk_mgmt/            # Risk debate agents
│       │   └── trader/               # Final decision agent
│       │       └── trader.py
│       │
│       ├── dataflows/                # Data fetching utilities
│       │   ├── yfin_utils.py         # Yahoo Finance data
│       │   ├── finnhub_utils.py      # Finnhub API
│       │   ├── googlenews_utils.py   # News scraping
│       │   ├── payment_utils.py      # Payment processing data
│       │   └── interface.py          # Unified data interface
│       │
│       └── graph/                    # LangGraph workflow
│           ├── trading_graph.py      # Main workflow definition
│           ├── setup.py              # Graph configuration
│           └── conditional_logic.py  # Routing logic
│
├── ai-trader-collab/                 # React + Vite Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx         # Main dashboard component
│   │   │   ├── AgentConversation.tsx # Agent message display
│   │   │   ├── ChartGallery.tsx      # Chart visualization
│   │   │   ├── TradingDecision.tsx   # Decision display
│   │   │   └── ui/                   # Radix UI components
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts       # WebSocket hook
│   │   └── main.tsx                  # App entry point
│   ├── package.json
│   └── vite.config.ts
│
├── venv/                             # Python virtual environment
│
├── README.md                         # Project documentation
└── LICENSE                           # MIT License

```

## 🚀 Quick Start

### Backend
```bash
cd backend
source ../venv/bin/activate
python api_server.py
# Runs on http://localhost:8002
```

### Frontend
```bash
cd ai-trader-collab
npm install
npm run dev
# Runs on http://localhost:8080
```

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `backend/api_server.py` | FastAPI server with REST API + WebSocket |
| `backend/.env` | API keys (Anthropic, etc.) |
| `backend/tradingagents/graph/trading_graph.py` | Main agent workflow |
| `ai-trader-collab/src/components/Dashboard.tsx` | Frontend UI |
| `ai-trader-collab/src/hooks/useWebSocket.ts` | WebSocket connection |

## 🗑️ Removed Files

- ❌ Test files (`test_integration.py`, `test_integration_improved.py`)
- ❌ Makefile
- ❌ Python cache files (`__pycache__/`, `*.pyc`)
- ❌ Evaluation results (`eval_results/`)
- ❌ Unused modules (alerts, backtesting, portfolio, graph_visualizer)
- ❌ Demo script (`demoScript.ts`)
- ❌ Bun lockfile (`bun.lockb`)

## 📊 System Architecture

```
User Browser (localhost:8080)
        ↓
    Frontend (React + Vite)
        ↓
    HTTP REST API (POST /api/analyze)
        ↓
    Backend (FastAPI localhost:8002)
        ↓
    WebSocket (/ws/analysis/{id})
        ↑
    Real-time Agent Messages
        ↑
    TradingAgents Graph (LangGraph)
        ↑
    AI Agents (Anthropic Claude)
```

## 🎯 Current Status

✅ Backend: Running and analyzing AAPL  
✅ Frontend: Running on port 8080  
✅ API: All endpoints working  
⚠️ WebSocket: Need to refresh browser to connect  
✅ Charts: Being generated in backend/  
✅ Analysis: Real AI agents communicating

