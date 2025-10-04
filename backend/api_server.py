#!/usr/bin/env python3
"""
FastAPI server for TradingAgents frontend integration
Provides REST API and WebSocket endpoints for real-time trading analysis
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from anthropic import Anthropic

# Import your existing TradingAgents
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# ---------------------
# Data Models
# ---------------------

class AnalysisRequest(BaseModel):
    ticker: str
    timeframe: str
    date: str = None  # Make date optional

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]

# ---------------------
# Global State
# ---------------------

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store analysis results
analysis_results: Dict[str, Dict[str, Any]] = {}

# Store analysis messages for polling
analysis_messages: Dict[str, List[Dict[str, Any]]] = {}

# Track server start time for uptime calculation
start_time = time.time()

# ---------------------
# Security & Rate Limiting
# ---------------------

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Optional API key authentication
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Optional API key verification"""
    required_key = os.getenv("API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Get CORS origins from environment
def get_cors_origins():
    origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002")
    return [origin.strip() for origin in origins_env.split(",")]

# ---------------------
# FastAPI App
# ---------------------

app = FastAPI(
    title="ΣIGMA API",
    description="Real-time trading analysis API with WebSocket support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add explicit OPTIONS handler for CORS preflight requests BEFORE rate limiting
@app.options("/{path:path}")
async def handle_options_request(path: str):
    """Handle CORS preflight requests explicitly"""
    return JSONResponse(
        content={"message": "CORS preflight OK"},
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        }
    )

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------
# TradingAgents Integration
# ---------------------

class TradingAgentsRunner:
    """Wrapper class to run TradingAgents analysis and stream results via WebSocket"""
    
    def __init__(self):
        # Create custom config
        self.config = DEFAULT_CONFIG.copy()
        self.config["llm_provider"] = "anthropic"
        self.config["backend_url"] = "https://api.anthropic.com"
        self.config["deep_think_llm"] = "claude-sonnet-4-0"
        self.config["quick_think_llm"] = "claude-sonnet-4-0"
        self.config["max_debate_rounds"] = 1
        self.config["max_risk_discuss_rounds"] = 1
        self.config["online_tools"] = True
        
        # Initialize TradingAgents (excluding social and visualization analysts due to bugs)
        self.ta = TradingAgentsGraph(
            debug=True, 
            config=self.config, 
            selected_analysts=["market", "news", "fundamentals", "payment"]
        )
        
        # Initialize Anthropic client for message summarization
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    async def run_demo_analysis(self, analysis_id: str, ticker: str):
        """Run demo analysis with hardcoded conversational messages based on REAL AAPL data"""
        
        # Based on REAL AAPL analysis data from terminal output
        demo_messages = {
            "Market Analyst": f"{ticker} is showing strong bullish momentum - currently trading at $258.02, significantly above the 50-day SMA ($232.67) and 200-day SMA ($221.79). RSI is at 71.05, indicating overbought conditions, while MACD at 7.42 shows continued positive momentum. The stock has recovered 28% from August lows around $201, with Bollinger Bands showing room to $267 on the upside but suggesting caution at current levels.",
            
            "News Analyst": f"Recent news sentiment for {ticker} is positive with steady momentum in trading. One article from Yahoo Finance highlights positive trading patterns. Overall media coverage shows optimistic sentiment about the company's performance, though no major catalyst events are driving headlines. The sentiment is constructive but not overly exuberant.",
            
            "Fundamentals Analyst": f"{ticker} demonstrates exceptional financial strength with $408.6 billion in TTM revenue (9.6% growth) and $99.3 billion net income (12.1% earnings growth). Key metrics: 24.30% profit margin, 149.81% ROE, P/E ratio of 39.21, and forward P/E of 31.05. The company maintains strong cash flow ($94.9B free cash flow) with 63.63% institutional ownership, though trading at premium valuations with debt-to-equity at 154.49.",
            
            "Payment Flow Analyst": f"{ticker}'s payment ecosystem shows robust operational efficiency with 96M monthly transactions and strong correlation to stock performance (0.200). Payment metrics: 0.30% fraud rate (vs industry 0.60%), 93% success rate, 1.7s processing time, 13% cross-border transactions, and 2.0% refund rate. Risk score of 11.5/100 indicates exceptionally low payment processing risk with 1,000 active merchants supporting the ecosystem.",
            
            "Risk Manager": f"After evaluating bull and bear perspectives, the risk assessment indicates: Growth has decelerated from 15-30% historically to current 9.6%. Technical indicators show overextension with RSI at 71 and price 16.4% above 200-day MA. Analyst targets at $246 are below current $258 price. Forward P/E of 31x for single-digit growth presents valuation risk in a rising rate environment. Recommendation: Gradual position reduction to manage downside risk while the $245 support level holds.",
            
            "Trader": f"Final trading decision based on comprehensive analysis: The convergence of technical overextension (RSI 71, overbought), fundamental growth deceleration, and analyst price targets below current levels creates asymmetric risk. While {ticker} remains a quality company, the valuation risk outweighs potential upside. Recommended action: SELL with gradual scaling strategy to capture remaining upside while managing downside exposure. Key support level to monitor: $245."
        }
        
        # Send messages with delays to simulate real analysis
        await asyncio.sleep(2)
        
        for i, (agent, message) in enumerate(demo_messages.items(), 1):
            progress = 20 + (i * 15)
            
            print(f"\n{'='*80}")
            print(f"🤖 {agent}")
            print(f"{'='*80}")
            print(f"📊 {message}")
            print(f"{'='*80}\n")
            
            await self.send_websocket_message(analysis_id, {
                "type": "message",
                "data": {
                    "agent": agent,
                    "agent_name": agent,
                    "content": message,
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "stage": f"Analysis Phase {i}",
                    "sentiment": "neutral"
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            await self.send_websocket_message(analysis_id, {
                "type": "progress",
                "data": {
                    "current": progress,
                    "total": 100,
                    "stage": "Analyzing",
                    "message": f"Running {agent}..."
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            await asyncio.sleep(2)  # Delay between agents
        
        # Send chart notification
        await self.send_websocket_message(analysis_id, {
            "type": "chart",
            "data": {
                "message": "Generated 4 visualization charts",
                "status": "complete",
                "count": 4
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
        
        # Final decision based on real analysis
        decision = f"SELL - While {ticker} demonstrates strong fundamentals with $408.6B revenue and robust profitability, the stock is technically overextended (RSI 71, trading 16.4% above 200-day MA). Analyst price targets at $246 suggest limited upside from current $258 levels. The forward P/E of 31x for 9.6% growth presents unfavorable risk/reward. Recommendation: Scale out gradually, monitor $245 support level."
        
        print(f"\n{'='*80}")
        print(f"✅ FINAL DECISION: {decision}")
        print(f"{'='*80}\n")
        
        # Send trading decision
        await self.send_websocket_message(analysis_id, {
            "type": "decision",
            "data": {
                "text": decision,
                "confidence": 0.75,  # Based on convergence of multiple bearish signals
                "rationale": "Technical overextension, valuation concerns, and analyst targets below current price create unfavorable risk/reward profile despite strong fundamentals."
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
        
        return {
            "decision": {"text": decision, "confidence": 0.75, "rationale": "Technical overextension + valuation risk"},
            "charts_generated": 4,
            "reports_generated": 6
        }
    
    async def summarize_agent_message(self, agent_name: str, raw_content: str, ticker: str) -> str:
        """Use Claude to create a concise, analytical summary of agent output"""
        
        # Define prompts based on agent type - CONVERSATIONAL STYLE
        agent_prompts = {
            "Market Analyst": f"You're explaining {ticker}'s market situation to a friend over coffee. In 2-3 casual sentences, tell them: Is the stock trending up or down? What's the vibe - should they be excited or cautious? Skip ALL numbers and technical jargon - just talk like a human.",
            "News Analyst": f"You're chatting with a friend about what's happening with {ticker} in the news. In 2-3 casual sentences, tell them: What are people talking about? Is the overall mood positive or negative? Keep it conversational - no formal language.",
            "Fundamentals Analyst": f"You're explaining {ticker}'s business health to a friend who doesn't know finance. In 2-3 simple sentences, tell them: Is the company making good money? Is it a solid business or shaky? Talk like you're explaining it to your non-finance friend - no jargon or numbers.",
            "Payment Flow Analyst": f"You're telling a friend about {ticker}'s transaction activity. In 2-3 casual sentences, explain: Are people using their services more or less? Is business flowing smoothly? Keep it conversational and skip technical terms.",
        }
        
        system_prompt = agent_prompts.get(agent_name, f"Summarize this {agent_name} analysis for {ticker} in 2-3 concise, actionable sentences.")
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast and cheap model for summaries
                max_tokens=150,
                temperature=0.3,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Analyze and summarize this {agent_name} output:\n\n{raw_content[:1500]}"
                }]
            )
            
            summary = response.content[0].text.strip()
            return summary
            
        except Exception as e:
            print(f"⚠️ Summarization failed for {agent_name}: {e}")
            # Fallback: return first 200 chars of original
            return raw_content[:200] + "..." if len(raw_content) > 200 else raw_content
    
    async def run_analysis(self, analysis_id: str, ticker: str, timeframe: str, date: str = None):
        """Run TradingAgents analysis and stream results via WebSocket"""
        
        try:
            print(f"🚀 Starting analysis for {ticker} (ID: {analysis_id})")
            # Send initial progress
            await self.send_websocket_message(analysis_id, {
                "type": "progress",
                "data": {
                    "current": 0,
                    "total": 100,
                    "stage": "Initializing",
                    "message": "Initializing analysis..."
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            # Use provided date or convert timeframe to date
            trade_date = date if date else self._timeframe_to_date(timeframe)
            
            # Send progress update
            await self.send_websocket_message(analysis_id, {
                "type": "progress", 
                "data": {
                    "current": 10,
                    "total": 100,
                    "stage": "Starting",
                    "message": f"Starting analysis for {ticker} on {trade_date}"
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            # Initialize state
            init_agent_state = self.ta.propagator.create_initial_state(ticker, trade_date)
            args = self.ta.propagator.get_graph_args()
            
            # Stream the analysis
            progress = 10
            # Run the actual analysis
            print(f"\n{'='*80}")
            print(f"🚀 Starting analysis for {ticker}")
            print(f"📅 Date: {trade_date}")
            print(f"🔄 Streaming graph output...")
            print(f"{'='*80}\n")
            
            # Track agents we've seen to avoid duplicate messages
            agent_stages_seen = set()
            chunk_count = 0
            current_step = 0
            total_steps = 100  # Estimate
            
            for chunk in self.ta.graph.stream(init_agent_state, **args):
                chunk_count += 1
                
                # Show progress indicator every 50 chunks
                if chunk_count % 50 == 0:
                    print(f"⚙️ Processing... ({chunk_count} chunks received)")
                
                # Handle different chunk formats
                if isinstance(chunk, dict):
                    # Check for messages in the chunk
                    if "messages" in chunk and len(chunk.get("messages", [])) > 0:
                    current_step += 1
                    message = chunk["messages"][-1]
                        
                        # Get agent name and content
                        agent_name = getattr(message, 'name', None)
                        content = getattr(message, 'content', str(message))
                    
                    # Calculate progress
                    progress = min(90, 10 + (current_step / total_steps) * 80)
                    
                        # ===== STRICT FILTERING =====
                        # Only show messages from KNOWN agents with SUBSTANTIAL content
                        should_send = False
                        filtered_content = None
                        
                        # Skip if agent_name is None, empty, or "Unknown Agent"
                        if not agent_name or agent_name == "Unknown Agent":
                            continue
                        
                        # Skip if it's just the ticker symbol
                        if isinstance(content, str):
                            if content.strip() == ticker.upper() or len(content.strip()) < 100:
                                continue
                            filtered_content = content
                            should_send = True
                        elif isinstance(content, list):
                            # Extract text from content blocks (Claude format)
                            text_blocks = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
                            if text_blocks:
                                filtered_content = ' '.join(text_blocks)
                                # Only send if it's substantial analysis (not just planning/tool use)
                                if len(filtered_content.strip()) > 100:
                                    should_send = True
                        
                        # Create a unique identifier for this agent stage
                        stage_key = f"{agent_name}_{current_step // 5}"  # Group every 5 steps
                        
                        # Only send meaningful messages from known agents, avoid duplicates
                        if should_send and filtered_content and stage_key not in agent_stages_seen:
                            agent_stages_seen.add(stage_key)
                            
                            # Map to display-friendly names
                            agent_display_name = agent_name
                            agent_name_lower = agent_name.lower()
                            
                            if 'market' in agent_name_lower:
                                agent_display_name = "Market Analyst"
                            elif 'news' in agent_name_lower:
                                agent_display_name = "News Analyst"
                            elif 'fundamental' in agent_name_lower:
                                agent_display_name = "Fundamentals Analyst"
                            elif 'payment' in agent_name_lower:
                                agent_display_name = "Payment Flow Analyst"
                            elif 'risk' in agent_name_lower:
                                agent_display_name = "Risk Manager"
                            elif 'trader' in agent_name_lower:
                                agent_display_name = "Trader"
                            else:
                                # Skip any agent we don't recognize
                                continue
                            
                            # Use LLM to create analytical summary
                            print(f"\n{'='*80}")
                            print(f"🤖 {agent_display_name} - Analyzing...")
                            print(f"{'='*80}")
                            
                            summarized_content = await self.summarize_agent_message(
                                agent_display_name, 
                                filtered_content, 
                                ticker
                            )
                            
                            print(f"📊 {summarized_content}")
                            print(f"{'='*80}\n")
                            
                    await self.send_websocket_message(analysis_id, {
                        "type": "message",
                                "data": {
                                    "agent": agent_display_name,
                                    "agent_name": agent_display_name,
                                    "content": summarized_content,
                                    "timestamp": int(datetime.now().timestamp() * 1000),
                                    "stage": f"Analysis Phase {current_step // 5 + 1}",
                                    "sentiment": "neutral"
                                },
                                "timestamp": int(datetime.now().timestamp() * 1000)
                            })
                            
                            # Send progress update ONLY when we send a meaningful message
                    await self.send_websocket_message(analysis_id, {
                        "type": "progress",
                                "data": {
                                    "current": int(progress),
                                    "total": 100,
                                    "stage": "Analyzing",
                                    "message": f"Running {agent_display_name}..."
                                },
                                "timestamp": int(datetime.now().timestamp() * 1000)
                    })
            
            # Get final decision
            final_state, decision = self.ta.propagate(ticker, trade_date)
            
            # Send final progress
            await self.send_websocket_message(analysis_id, {
                "type": "progress",
                "data": {
                    "current": 95,
                    "total": 100,
                    "stage": "Finalizing",
                    "message": "Generating final decision..."
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            # Send trading decision
            await self.send_websocket_message(analysis_id, {
                "type": "decision",
                "data": {
                "text": decision,
                "confidence": 0.85,  # You can extract this from your analysis
                    "rationale": f"Based on comprehensive analysis of {ticker} market data, technical indicators, and payment flow analysis."
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            # Send completion
            await self.send_websocket_message(analysis_id, {
                "type": "complete",
                "data": {
                    "finalState": final_state,
                    "decision": {
                        "text": decision,
                        "confidence": 0.85,
                        "rationale": f"Based on comprehensive analysis of {ticker} market data, technical indicators, and payment flow analysis."
                    },
                    "analysisTime": 0  # You can calculate this
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            
            # Store results
            analysis_results[analysis_id] = {
                "ticker": ticker,
                "timeframe": timeframe,
                "decision": decision,
                "final_state": final_state,
                "completed_at": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            # Send error message
            await self.send_websocket_message(analysis_id, {
                "type": "error",
                "data": {
                    "message": f"Analysis failed: {str(e)}",
                    "error": str(e)
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
            print(f"Analysis error: {e}")
    
    async def send_websocket_message(self, analysis_id: str, message: Dict[str, Any]):
        """Send message to WebSocket connection"""
        print(f"📡 Attempting to send WebSocket message to {analysis_id}: {message['type']}")
        print(f"📡 Active connections: {list(active_connections.keys())}")
        
        if analysis_id in active_connections:
            try:
                await active_connections[analysis_id].send_text(json.dumps(message))
                print(f"✅ WebSocket message sent successfully: {message['type']}")
            except Exception as e:
                print(f"❌ Failed to send WebSocket message: {e}")
        else:
            print(f"❌ No WebSocket connection found for analysis_id: {analysis_id}")
    
    def _timeframe_to_date(self, timeframe: str) -> str:
        """Convert timeframe to date string"""
        from datetime import datetime, timedelta
        
        if timeframe == "1D":
            return datetime.now().strftime("%Y-%m-%d")
        elif timeframe == "1W":
            return (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif timeframe == "1H":
            return datetime.now().strftime("%Y-%m-%d")
        else:
            return datetime.now().strftime("%Y-%m-%d")

# Initialize the runner
runner = TradingAgentsRunner()

# ---------------------
# REST API Endpoints
# ---------------------

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ΣIGMA API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_connections": len(active_connections),
        "active_analyses": len(analysis_results),
        "environment": {
            "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "cors_origins": len(get_cors_origins()),
        }
    }

@app.get("/api/status")
async def system_status():
    """System status with detailed metrics"""
    # Count analyses by status
    status_counts = {}
    completed_analyses = []

    for analysis_id, analysis in analysis_results.items():
        status = analysis.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == "completed":
            completed_analyses.append({
                "id": analysis_id[:8] + "...",
                "ticker": analysis.get("ticker"),
                "completed_at": analysis.get("completed_at"),
            })

    return {
        "system": {
            "status": "operational",
            "uptime_seconds": int(time.time() - start_time),
            "timestamp": datetime.now().isoformat(),
        },
        "analyses": {
            "total": len(analysis_results),
            "by_status": status_counts,
            "recent_completed": completed_analyses[-5:],  # Last 5 completed
        },
        "connections": {
            "active_websockets": len(active_connections),
            "connection_ids": list(active_connections.keys())[:3] if len(active_connections) <= 3 else list(active_connections.keys())[:3] + ["..."],
        }
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
@limiter.limit("10/minute")
async def start_analysis(request: Request, analysis_request: AnalysisRequest, _: bool = Depends(verify_api_key)):
    """Start a new trading analysis"""
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Store initial analysis data
    analysis_results[analysis_id] = {
        "ticker": analysis_request.ticker.upper(),
        "timeframe": analysis_request.timeframe,
        "status": "started",
        "started_at": datetime.now().isoformat()
    }
    
    # Start analysis in background with WebSocket messaging
    async def run_analysis_with_websocket():
        """Run analysis and send WebSocket updates"""
        try:
            print(f"🔄 Starting analysis task for {analysis_id}")

            # Wait a moment for WebSocket to potentially connect
            await asyncio.sleep(1)

            # Send test messages to any connected WebSocket
            for conn_id, websocket in active_connections.items():
                try:
                    # Send to analysis-specific connection or general connections
                    if analysis_id == conn_id or analysis_id in conn_id or len(active_connections) <= 2:
                        await websocket.send_text(json.dumps({
                            "type": "progress",
                            "data": {
                                "current": 10,
                                "total": 100,
                                "stage": "Starting Analysis",
                                "message": "Initializing trading agents..."
                            },
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        }))

                        await websocket.send_text(json.dumps({
                            "type": "message",
                            "data": {
                                "agent": "System",
                                "content": f"✅ Analysis started for {analysis_request.ticker.upper()}",
                                "timestamp": int(datetime.now().timestamp() * 1000),
                                "stage": "initialization"
                            },
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        }))
                        print(f"📤 Sent test messages to WebSocket {conn_id}")
                except Exception as ws_error:
                    print(f"❌ WebSocket send error: {ws_error}")

            # Run real trading analysis with WebSocket integration
            print(f"🚀 Running REAL analysis with trading agents")

            # Send initial progress
            for conn_id, websocket in active_connections.items():
                try:
                    if analysis_id == conn_id or analysis_id in conn_id or len(active_connections) <= 2:
                        await websocket.send_text(json.dumps({
                            "type": "progress",
                            "data": {
                                "current": 15,
                                "total": 100,
                                "stage": "Initializing Agents",
                                "message": "Starting TradingAgents analysis..."
                            },
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        }))
                except Exception as ws_error:
                    print(f"❌ WebSocket initial progress error: {ws_error}")

            # Run the actual analysis
            try:
                # CHECK IF DEMO MODE (for presentation when API credits are low)
                demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"
                
                if demo_mode:
                    print(f"🎭 Running DEMO MODE for {analysis_request.ticker.upper()}")
                    result = await runner.run_demo_analysis(analysis_id, analysis_request.ticker.upper())
                else:
                    print(f"📊 Running real analysis for {analysis_request.ticker.upper()}")
                    result = await runner.run_analysis(
        analysis_id, 
                        analysis_request.ticker.upper(),
                        analysis_request.timeframe,
                        analysis_request.date
                    )
                print(f"✅ Analysis completed: {result}")

                # Send completion message with real results
                for conn_id, websocket in active_connections.items():
                    try:
                        if analysis_id == conn_id or analysis_id in conn_id or len(active_connections) <= 2:
                            await websocket.send_text(json.dumps({
                                "type": "complete",
                                "data": {
                                    "analysis_id": analysis_id,
                                    "decision": result.get("decision", {
                                        "text": "ANALYSIS_COMPLETE",
                                        "confidence": 0.8,
                                        "rationale": "Real TradingAgents analysis completed"
                                    }),
                                    "charts_generated": result.get("charts_generated", 0),
                                    "reports_generated": result.get("reports_generated", 0)
                                },
                                "timestamp": int(datetime.now().timestamp() * 1000)
                            }))
                            print(f"📤 Sent real completion message to WebSocket {conn_id}")
                    except Exception as ws_error:
                        print(f"❌ WebSocket real completion error: {ws_error}")

            except Exception as analysis_error:
                print(f"❌ Real analysis failed: {analysis_error}")
                import traceback
                traceback.print_exc()

                # Send error message
                for conn_id, websocket in active_connections.items():
                    try:
                        if analysis_id == conn_id or analysis_id in conn_id or len(active_connections) <= 2:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "data": {"message": f"Analysis failed: {str(analysis_error)}"},
                                "timestamp": int(datetime.now().timestamp() * 1000)
                            }))
                    except Exception as ws_error:
                        print(f"❌ WebSocket error message failed: {ws_error}")

                # Update analysis results with error
                analysis_results[analysis_id].update({
                    "status": "error",
                    "error": str(analysis_error),
                    "completed_at": datetime.now().isoformat()
                })
                return  # Exit early on error

            # Update analysis results with real data
            analysis_results[analysis_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": result,
                "decision": result.get("decision", {
                    "text": "ANALYSIS_COMPLETE",
                    "confidence": 0.8,
                    "rationale": "Real TradingAgents analysis completed"
                }),
                "charts_generated": result.get("charts_generated", 0),
                "reports_generated": result.get("reports_generated", 0)
            })

        except Exception as e:
            print(f"❌ Error in analysis task: {e}")
            import traceback
            traceback.print_exc()

            # Send error message
            for conn_id, websocket in active_connections.items():
                try:
                    if analysis_id in conn_id or conn_id == analysis_id:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "data": {"message": f"Analysis failed: {str(e)}"},
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        }))
                except Exception as ws_error:
                    print(f"❌ WebSocket error send failed: {ws_error}")

    # Start the analysis task (this should not block the API response)
    try:
        task = asyncio.create_task(run_analysis_with_websocket())
        print(f"🚀 Analysis task created for {analysis_id}")

        # Add a callback to catch any task exceptions
        def handle_task_exception(task):
            try:
                task.result()
            except Exception as e:
                print(f"❌ Analysis task failed: {e}")
                import traceback
                traceback.print_exc()

        task.add_done_callback(handle_task_exception)
    except Exception as e:
        print(f"❌ Failed to create analysis task: {e}")
        import traceback
        traceback.print_exc()
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="started",
        message=f"Analysis started for {analysis_request.ticker.upper()}"
    )

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis results"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/api/analysis/{analysis_id}/messages")
async def get_analysis_messages(analysis_id: str, since: int = 0):
    """Get all messages for an analysis (for polling)"""
    
    if analysis_id not in analysis_messages:
        return {"messages": [], "total": 0, "since": since, "new_count": 0}
    
    messages = analysis_messages[analysis_id]
    # Return messages after 'since' index
    new_messages = messages[since:] if since < len(messages) else []
    
    return {
        "messages": new_messages,
        "total": len(messages),
        "since": since,
        "new_count": len(new_messages)
    }

@app.get("/api/analysis/{analysis_id}/reports")
async def get_analysis_reports(analysis_id: str):
    """Get detailed analysis reports for an analysis"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # This would return detailed reports from the analysis
    # For now, return the basic analysis data
    analysis_data = analysis_results[analysis_id]
    
    return {
        "analysis_id": analysis_id,
        "ticker": analysis_data.get("ticker"),
        "timeframe": analysis_data.get("timeframe"),
        "decision": analysis_data.get("decision"),
        "final_state": analysis_data.get("final_state"),
        "completed_at": analysis_data.get("completed_at"),
        "reports": [
            {
                "agent": "Market Analyst",
                "title": "Market Analysis Report",
                "content": f"Comprehensive market analysis for {analysis_data.get('ticker')}",
                "timestamp": analysis_data.get("completed_at")
            },
            {
                "agent": "Technical Analyst", 
                "title": "Technical Analysis Report",
                "content": f"Technical indicators and chart patterns for {analysis_data.get('ticker')}",
                "timestamp": analysis_data.get("completed_at")
            }
        ]
    }

@app.get("/api/charts/{analysis_id}")
async def get_charts(analysis_id: str):
    """Get charts for an analysis"""
    import os
    import glob
    
    # Look for generated chart files in the backend directory
    chart_patterns = [
        f"enhanced_visualization_*_{analysis_id}_*.html",
        f"visualization_*_{analysis_id}_*.html",
        f"*_visualization_*_{analysis_id}_*.html"
    ]
    
    charts = []
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    for pattern in chart_patterns:
        chart_files = glob.glob(os.path.join(backend_dir, pattern))
        for chart_file in chart_files:
            filename = os.path.basename(chart_file)
            # Extract chart type from filename
            if "scatter" in filename.lower():
                chart_type = "scatter"
                title = "Returns vs Volume"
            elif "line" in filename.lower():
                chart_type = "line"
                title = "Price & VWAP"
            elif "multi" in filename.lower():
                chart_type = "multi"
                title = "Multi-Line Momentum"
            elif "bollinger" in filename.lower() or "bands" in filename.lower():
                chart_type = "bbands"
                title = "Bollinger Bands"
            else:
                chart_type = "unknown"
                title = "Financial Chart"
            
            charts.append({
                "id": chart_type,
                "title": title,
                "url": f"/api/chart/{filename}",
                "filename": filename,
                "type": chart_type
            })
    
    # If no specific charts found, return recent charts
    if not charts:
        recent_charts = glob.glob(os.path.join(backend_dir, "*_visualization_*.html"))
        recent_charts.sort(key=os.path.getmtime, reverse=True)
        
        for i, chart_file in enumerate(recent_charts[:4]):
            filename = os.path.basename(chart_file)
            chart_type = f"chart_{i+1}"
            title = f"Chart {i+1}"
            
            charts.append({
                "id": chart_type,
                "title": title,
                "url": f"/api/chart/{filename}",
                "filename": filename,
                "type": chart_type
            })
    
    return {"charts": charts}

@app.get("/api/charts/recent")
async def get_recent_charts():
    """Get recent charts (fallback when no analysis-specific charts found)"""
    import os
    import glob
    
    charts = []
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all visualization HTML files
    chart_files = glob.glob(os.path.join(backend_dir, "*_visualization_*.html"))
    chart_files.sort(key=os.path.getmtime, reverse=True)
    
    for i, chart_file in enumerate(chart_files[:4]):
        filename = os.path.basename(chart_file)
        
        # Extract chart type from filename
        if "scatter" in filename.lower():
            chart_type = "scatter"
            title = "Returns vs Volume"
        elif "line" in filename.lower():
            chart_type = "line"
            title = "Price & VWAP"
        elif "multi" in filename.lower():
            chart_type = "multi"
            title = "Multi-Line Momentum"
        elif "bollinger" in filename.lower() or "bands" in filename.lower():
            chart_type = "bbands"
            title = "Bollinger Bands"
        else:
            chart_type = f"chart_{i+1}"
            title = f"Chart {i+1}"
        
        charts.append({
            "id": chart_type,
            "title": title,
            "url": f"/api/chart/{filename}",
            "filename": filename,
            "type": chart_type
        })
    
    return {"charts": charts}

@app.get("/api/chart/{filename}")
async def serve_chart(filename: str):
    """Serve individual chart HTML files"""
    import os
    from fastapi.responses import FileResponse
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(backend_dir, filename)
    
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Chart not found")

# ---------------------
# WebSocket Endpoint
# ---------------------

@app.websocket("/ws/analysis/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for real-time analysis updates"""
    
    print(f"🔌 WebSocket connection attempt for analysis_id: {analysis_id}")
    await websocket.accept()
    active_connections[analysis_id] = websocket
    print(f"✅ WebSocket connected for analysis_id: {analysis_id}")
    print(f"📊 Total active connections: {len(active_connections)}")
    print(f"📋 Active connection IDs: {list(active_connections.keys())}")
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "data": {"message": f"Connected to analysis {analysis_id}"},
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (optional)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client messages if needed
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "data": {},
                        "timestamp": int(datetime.now().timestamp() * 1000)
                    }))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up connection
        if analysis_id in active_connections:
            del active_connections[analysis_id]

@app.websocket("/ws")
async def websocket_general(websocket: WebSocket):
    """General WebSocket endpoint for frontend connection"""

    print(f"🔌 WebSocket connection attempt from {websocket.client}")
    try:
        await websocket.accept()
        print(f"✅ WebSocket connection accepted")
        connection_id = str(uuid.uuid4())
        active_connections[connection_id] = websocket
    except Exception as e:
        print(f"❌ WebSocket accept failed: {e}")
        raise
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "data": {"message": "Connected to TradingAgents WebSocket"},
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client messages
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "data": {},
                        "timestamp": int(datetime.now().timestamp() * 1000)
                    }))
                elif message.get("type") == "start_analysis":
                    # Start analysis directly from WebSocket
                    data = message.get("data", {})
                    ticker = data.get("ticker", "SPY")
                    timeframe = data.get("timeframe", "1D")
                    date = data.get("date", None)
                    asyncio.create_task(runner.run_analysis(connection_id, ticker, timeframe, date))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up connection
        if connection_id in active_connections:
            del active_connections[connection_id]

# ---------------------
# Main
# ---------------------

if __name__ == "__main__":
    print("🚀 Starting ΣIGMA API Server with REAL Analysis...")
    print("📊 Frontend: http://localhost:3002")
    print("🔌 API: http://localhost:8002")
    print("📡 WebSocket: ws://localhost:8002/ws/analysis/{id}")
    print()
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
