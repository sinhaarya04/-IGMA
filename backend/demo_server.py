#!/usr/bin/env python3
"""
Simple demo server with hardcoded AAPL analysis for presentation
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os

# Data Models
class AnalysisRequest(BaseModel):
    ticker: str
    timeframe: str
    date: str = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str

# Global State
active_connections: Dict[str, WebSocket] = {}
analysis_messages: Dict[str, List[Dict[str, Any]]] = {}

# FastAPI App
app = FastAPI(title="ΣIGMA Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded analysis messages based on REAL AAPL terminal output
DEMO_AGENTS = {
    "Market Analyst": "AAPL is showing strong bullish momentum - currently trading at $258.02, significantly above the 50-day SMA ($232.67) and 200-day SMA ($221.79). RSI is at 71.05, indicating overbought conditions, while MACD at 7.42 shows continued positive momentum. The stock has recovered 28% from August lows around $201, with Bollinger Bands showing room to $267 on the upside but suggesting caution at current levels.",
    
    "News Analyst": "Recent news sentiment for AAPL is positive with steady momentum in trading. One article from Yahoo Finance highlights positive trading patterns. Overall media coverage shows optimistic sentiment about the company's performance, though no major catalyst events are driving headlines. The sentiment is constructive but not overly exuberant.",
    
    "Fundamentals Analyst": "AAPL demonstrates exceptional financial strength with $408.6 billion in TTM revenue (9.6% growth) and $99.3 billion net income (12.1% earnings growth). Key metrics: 24.30% profit margin, 149.81% ROE, P/E ratio of 39.21, and forward P/E of 31.05. The company maintains strong cash flow ($94.9B free cash flow) with 63.63% institutional ownership, though trading at premium valuations with debt-to-equity at 154.49.",
    
    "Payment Flow Analyst": "AAPL's payment ecosystem shows robust operational efficiency with 96M monthly transactions and strong correlation to stock performance (0.200). Payment metrics: 0.30% fraud rate (vs industry 0.60%), 93% success rate, 1.7s processing time, 13% cross-border transactions, and 2.0% refund rate. Risk score of 11.5/100 indicates exceptionally low payment processing risk with 1,000 active merchants supporting the ecosystem.",
    
    "Risk Manager": "After evaluating bull and bear perspectives, the risk assessment indicates: Growth has decelerated from 15-30% historically to current 9.6%. Technical indicators show overextension with RSI at 71 and price 16.4% above 200-day MA. Analyst targets at $246 are below current $258 price. Forward P/E of 31x for single-digit growth presents valuation risk in a rising rate environment. Recommendation: Gradual position reduction to manage downside risk while the $245 support level holds.",
    
    "Trader": "Final trading decision based on comprehensive analysis: The convergence of technical overextension (RSI 71, overbought), fundamental growth deceleration, and analyst price targets below current levels creates asymmetric risk. While AAPL remains a quality company, the valuation risk outweighs potential upside. Recommended action: SELL with gradual scaling strategy to capture remaining upside while managing downside exposure. Key support level to monitor: $245."
}

FINAL_DECISION = "SELL - While AAPL demonstrates strong fundamentals with $408.6B revenue and robust profitability, the stock is technically overextended (RSI 71, trading 16.4% above 200-day MA). Analyst price targets at $246 suggest limited upside from current $258 levels. The forward P/E of 31x for 9.6% growth presents unfavorable risk/reward. Recommendation: Scale out gradually, monitor $245 support level."

async def send_message(analysis_id: str, message: Dict):
    """Store and send message"""
    if analysis_id not in analysis_messages:
        analysis_messages[analysis_id] = []
    analysis_messages[analysis_id].append(message)
    
    if analysis_id in active_connections:
        try:
            await active_connections[analysis_id].send_text(json.dumps(message))
        except:
            pass

@app.get("/")
async def root():
    return {"message": "ΣIGMA Demo API", "status": "healthy"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def start_analysis(analysis_request: AnalysisRequest):
    """Start demo analysis"""
    analysis_id = str(uuid.uuid4())
    
    async def run_demo():
        await asyncio.sleep(2)
        
        # Send each agent message
        for i, (agent, content) in enumerate(DEMO_AGENTS.items(), 1):
            progress = 15 + (i * 12)
            
            print(f"\n{'='*80}")
            print(f"🤖 {agent}")
            print(f"{'='*80}")
            print(f"{content[:200]}...")
            print(f"{'='*80}\n")
            
            await send_message(analysis_id, {
                "type": "message",
                "data": {
                    "agent": agent,
                    "agent_name": agent,
                    "content": content,
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "stage": f"Analysis Phase {i}",
                    "sentiment": "neutral"
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            await send_message(analysis_id, {
                "type": "progress",
                "data": {
                    "current": progress,
                    "total": 100,
                    "stage": "Analyzing",
                    "message": f"{agent} complete"
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            await asyncio.sleep(1.5)
        
        # Send chart notification
        await send_message(analysis_id, {
            "type": "chart",
            "data": {
                "message": "Generated 4 visualization charts",
                "status": "complete",
                "count": 4
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
        
        # Send decision
        await send_message(analysis_id, {
            "type": "decision",
            "data": {
                "text": FINAL_DECISION,
                "confidence": 0.75,
                "rationale": "Technical overextension, valuation concerns, and analyst targets below current price"
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
        
        await send_message(analysis_id, {
            "type": "progress",
            "data": {
                "current": 100,
                "total": 100,
                "stage": "Complete",
                "message": "Analysis complete"
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
        
        await send_message(analysis_id, {
            "type": "complete",
            "data": {
                "decision": {"text": FINAL_DECISION, "confidence": 0.75},
                "charts_generated": 4,
                "reports_generated": 6
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
    
    asyncio.create_task(run_demo())
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="started",
        message=f"Demo analysis started for {analysis_request.ticker}"
    )

@app.get("/api/analysis/{analysis_id}/messages")
async def get_messages(analysis_id: str, since: int = 0):
    """Get messages for polling"""
    if analysis_id not in analysis_messages:
        return {"messages": [], "total": 0, "since": since, "new_count": 0}
    
    messages = analysis_messages[analysis_id]
    new_messages = messages[since:] if since < len(messages) else []
    
    return {
        "messages": new_messages,
        "total": len(messages),
        "since": since,
        "new_count": len(new_messages)
    }

@app.get("/api/charts/{analysis_id}")
async def get_charts(analysis_id: str):
    """Get demo charts"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    charts = [
        {"id": "scatter", "title": "Returns vs Volume", "url": "/api/chart/enhanced_visualization_AAPL_scatter_plot_1.html", "type": "scatter", "explanation": "Interactive scatter visualization showing correlation between returns and trading volume"},
        {"id": "line", "title": "Price & VWAP", "url": "/api/chart/enhanced_visualization_AAPL_line_chart_3.html", "type": "line", "explanation": "Interactive line visualization comparing price movements with volume-weighted average price"},
        {"id": "multi", "title": "Multi-Line Momentum", "url": "/api/chart/enhanced_visualization_AAPL_multi_line_chart_2.html", "type": "multi", "explanation": "Interactive multi-line visualization displaying momentum indicators and trend analysis"},
        {"id": "bbands", "title": "Bollinger Bands", "url": "/api/chart/enhanced_visualization_AAPL_bollinger_bands_4.html", "type": "bbands", "explanation": "Interactive Bollinger Bands visualization showing price volatility and trading ranges"},
    ]
    
    return {"charts": charts}

@app.get("/api/chart/{filename}")
async def serve_chart(filename: str):
    """Serve chart files"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(backend_dir, filename)
    
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="text/html")
    else:
        return JSONResponse({"error": "Chart not found"}, status_code=404)

@app.get("/api/analysis/{analysis_id}/reports")
async def get_reports(analysis_id: str):
    """Get demo reports"""
    timestamp = datetime.now().isoformat()
    return {
        "analysis_id": analysis_id,
        "ticker": "AAPL",
        "timeframe": "1D",
        "reports": [
            {
                "id": "market-report",
                "agentName": "Market Analyst",
                "title": "Technical Analysis Report",
                "preview": "AAPL showing strong bullish momentum with RSI at 71.05 indicating overbought conditions...",
                "content": DEMO_AGENTS["Market Analyst"],
                "timestamp": timestamp
            },
            {
                "id": "fundamentals-report",
                "agentName": "Fundamentals Analyst",
                "title": "Fundamental Analysis Report",
                "preview": "Exceptional financial strength with $408.6B revenue and 24.30% profit margin...",
                "content": DEMO_AGENTS["Fundamentals Analyst"],
                "timestamp": timestamp
            },
            {
                "id": "news-report",
                "agentName": "News Analyst",
                "title": "News Sentiment Report",
                "preview": "Positive sentiment with steady momentum, Yahoo Finance highlighting positive patterns...",
                "content": DEMO_AGENTS["News Analyst"],
                "timestamp": timestamp
            },
            {
                "id": "payment-report",
                "agentName": "Payment Flow Analyst",
                "title": "Payment Flow Analysis",
                "preview": "96M monthly transactions with 0.30% fraud rate and 93% success rate...",
                "content": DEMO_AGENTS["Payment Flow Analyst"],
                "timestamp": timestamp
            },
            {
                "id": "risk-report",
                "agentName": "Risk Manager",
                "title": "Risk Assessment",
                "preview": "Growth deceleration and technical overextension suggest valuation risk...",
                "content": DEMO_AGENTS["Risk Manager"],
                "timestamp": timestamp
            },
            {
                "id": "trader-report",
                "agentName": "Trader",
                "title": "Final Trading Decision",
                "preview": "SELL recommendation based on technical overextension and unfavorable risk/reward...",
                "content": DEMO_AGENTS["Trader"],
                "timestamp": timestamp
            }
        ]
    }

@app.websocket("/ws/analysis/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint"""
    await websocket.accept()
    active_connections[analysis_id] = websocket
    print(f"✅ WebSocket connected: {analysis_id}")
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "data": {"message": f"Connected to analysis {analysis_id}"},
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
        
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        if analysis_id in active_connections:
            del active_connections[analysis_id]

if __name__ == "__main__":
    print("🎭 Starting ΣIGMA Demo Server...")
    print("🔌 API: http://localhost:8002")
    print("📊 Frontend: http://localhost:8080")
    
    uvicorn.run(
        "demo_server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )

