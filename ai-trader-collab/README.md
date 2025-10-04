# ΣIGMA Frontend

Modern, real-time trading analysis dashboard built with React, TypeScript, and shadcn/ui.

## Quick Start

```bash
npm install
npm run dev
```

Frontend will run on: **http://localhost:8080**

## Features

- 🤖 **Multi-Agent Analysis** - See real-time communications from 6 specialized agents:
  - Market Analyst (Technical Analysis)
  - News Analyst (Sentiment Analysis)
  - Fundamentals Analyst (Financial Metrics)
  - Payment Flow Analyst (Transaction Analysis)
  - Risk Manager (Risk Assessment)
  - Trader (Final Decision)

- 📊 **Interactive Charts** - Plotly-powered visualizations:
  - Returns vs Volume (Scatter)
  - Price & VWAP (Line Chart)
  - Multi-Line Momentum
  - Bollinger Bands

- 📄 **Detailed Reports** - Expandable agent reports with full analysis

- 🎯 **Trading Decisions** - Clear buy/sell/hold recommendations with confidence scores

- 🔌 **Real-time Updates** - WebSocket + polling for guaranteed message delivery

## Backend Connection

The frontend connects to the backend API at `http://localhost:8002`

**Demo Mode**: Hardcoded AAPL analysis for presentations
**Live Mode**: Real AI-powered analysis (requires Anthropic API credits)

## Tech Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **shadcn/ui** - Component library
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **Sonner** - Toast notifications

## Project Structure

```
src/
├── components/
│   ├── Dashboard.tsx           # Main dashboard orchestration
│   ├── AgentConversation.tsx   # Agent message display
│   ├── ChartGallery.tsx        # Chart visualization grid
│   ├── ReportsPanel.tsx        # Expandable reports
│   ├── TradingDecision.tsx     # Decision display with confidence
│   ├── ControlPanel.tsx        # Analysis controls
│   └── ui/                     # shadcn/ui components
├── hooks/
│   └── useWebSocket.ts         # WebSocket management
└── main.tsx                    # App entry point
```

## Development

- Run `npm run dev` for development server
- Backend must be running on port 8002
- Hot reload enabled for instant updates
