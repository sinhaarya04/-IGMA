# ΣIGMA Backend

## Quick Start

### Option 1: Demo Mode (For Presentations - No API Credits Needed)
```bash
cd backend
source ../venv/bin/activate
python demo_server.py
```
- Shows hardcoded AAPL analysis with real data
- All 6 agents with clean, professional messages
- 4 interactive charts
- Trading decision with confidence score

### Option 2: Live Mode (Real Analysis - Requires API Credits)
```bash
cd backend
source ../venv/bin/activate
DEMO_MODE=false python api_server.py
```
- Runs actual TradingAgents analysis
- Uses Anthropic API (requires credits)
- LLM-powered message summarization
- Real-time agent communications

## Frontend
```bash
cd ai-trader-collab
npm run dev
```
- Open http://localhost:8080
- Backend runs on http://localhost:8002

## Features
- 🤖 Multi-agent analysis (Market, News, Fundamentals, Payment, Risk, Trader)
- 📊 Interactive Plotly charts
- 📄 Detailed agent reports
- 💬 Clean, conversational agent messages
- 🎯 Trading decisions with confidence scores

