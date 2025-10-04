# ΣIGMA

AI-powered multi-agent trading analysis system with real-time dashboard.

## 🚀 Quick Start

### 1. Backend (Demo Mode - No API Credits Needed)

```bash
cd backend
source ../venv/bin/activate
python demo_server.py
```

### 2. Frontend

```bash
cd ai-trader-collab
npm install  # First time only
npm run dev
```

### 3. Open Browser

Visit: **http://localhost:8080**

Click "Start Analysis" with ticker "AAPL" to see the demo!

## 📊 Features

### Multi-Agent Analysis System
- **Market Analyst** - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- **News Analyst** - Sentiment analysis from news sources
- **Fundamentals Analyst** - Financial metrics (P/E, revenue, earnings, cash flow)
- **Payment Flow Analyst** - Transaction patterns and payment ecosystem health
- **Risk Manager** - Risk assessment and position sizing
- **Trader** - Final trading decision synthesis

### Real-Time Dashboard
- Clean, conversational agent messages
- Interactive Plotly charts (4 visualizations)
- Expandable detailed reports
- Trading decisions with confidence scores
- WebSocket + polling for reliable updates

### Two Modes

**Demo Mode** (`demo_server.py`)
- Hardcoded AAPL analysis based on real data
- Perfect for presentations
- No API credits needed
- All 6 agents with professional analysis

**Live Mode** (`api_server.py`)
- Real AI-powered analysis using Claude
- Requires Anthropic API credits
- LLM-powered message summarization
- Dynamic analysis for any ticker

## 🛠️ Tech Stack

**Backend:**
- Python 3.13
- FastAPI (REST + WebSocket)
- LangGraph (Agent orchestration)
- Anthropic Claude (LLM)
- TradingAgents library

**Frontend:**
- React + TypeScript
- Vite
- shadcn/ui components
- Tailwind CSS
- Plotly charts

## 📁 Project Structure

```
ΣIGMA/
├── backend/
│   ├── demo_server.py              # Demo mode server
│   ├── api_server.py               # Live analysis server  
│   ├── requirements.txt            # Python dependencies
│   ├── enhanced_visualization_*.html  # Chart files (4)
│   └── tradingagents/              # Core agent library
│       ├── agents/                 # All agent implementations
│       ├── graph/                  # LangGraph workflow
│       └── dataflows/              # Data providers
│
├── ai-trader-collab/              # React frontend
│   ├── src/
│   │   ├── components/            # UI components
│   │   └── hooks/                 # React hooks
│   └── package.json
│
└── venv/                          # Python virtual environment
```

## 🔧 Configuration

**Backend Environment Variables:**
```bash
ANTHROPIC_API_KEY=your_api_key_here
DEMO_MODE=true  # or false for live mode
```

**Frontend:**
- Connects to `http://localhost:8002` (backend)
- Runs on `http://localhost:8080` (configurable in vite.config.ts)

## 📝 API Endpoints

- `POST /api/analyze` - Start analysis
- `GET /api/analysis/{id}/messages` - Get agent messages (polling)
- `GET /api/analysis/{id}/reports` - Get detailed reports
- `GET /api/charts/{id}` - Get chart list
- `GET /api/chart/{filename}` - Serve chart HTML
- `WS /ws/analysis/{id}` - WebSocket for real-time updates

## 🎯 Usage

1. Start backend (demo or live mode)
2. Start frontend
3. Enter ticker symbol (e.g., AAPL, TSLA, MSFT)
4. Click "Start Analysis"
5. Watch agents analyze in real-time
6. View charts and reports
7. See final trading decision

## 📦 Installation

```bash
# Backend setup
cd backend
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd ../ai-trader-collab
npm install
```

## 🌟 Demo Features

The demo mode showcases real AAPL analysis data:
- RSI: 71.05 (overbought)
- Price: $258.02
- Revenue: $408.6B (9.6% growth)
- 96M monthly transactions
- Final Decision: SELL (75% confidence)

Perfect for presentations and showcasing capabilities!

## 👨‍💻 Authors

**Aryan Sinha**  
📧 sinha.arya@northeastern.edu  
🎓 Northeastern University

**Shourya Dewansh**  
🎓 Northeastern University

**Arzu Malkoch**  
🎓 Northeastern University

### Project Development
This project was developed as part of a trading analysis system showcasing multi-agent AI collaboration. The system integrates multiple specialized AI agents for comprehensive stock analysis with real-time visualization.

### Built With
- Custom multi-agent architecture using LangGraph
- Anthropic Claude AI for agent intelligence
- React + TypeScript frontend
- FastAPI backend with WebSocket support

## 📄 License

See LICENSE file for details.
