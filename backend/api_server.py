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
    title="TradingAgents API",
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
        
        # Initialize TradingAgents (excluding social analyst due to bug, including payment analyst)
        self.ta = TradingAgentsGraph(
            debug=True, 
            config=self.config, 
            selected_analysts=["market", "news", "fundamentals", "payment", "visualization"]
        )
    
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
            total_steps = 0
            
            # First pass to count total steps
            for chunk in self.ta.graph.stream(init_agent_state, **args):
                if len(chunk.get("messages", [])) > 0:
                    total_steps += 1
            
            # Reset state for actual run
            init_agent_state = self.ta.propagator.create_initial_state(ticker, trade_date)
            current_step = 0
            
            # Run the actual analysis
            print(f"🔍 Starting to stream analysis for {ticker}")
            for chunk in self.ta.graph.stream(init_agent_state, **args):
                print(f"📦 Received chunk: {type(chunk)} - {list(chunk.keys()) if isinstance(chunk, dict) else 'Not a dict'}")
                
                # Handle different chunk formats
                if isinstance(chunk, dict):
                    # Check for messages in the chunk
                    if "messages" in chunk and len(chunk.get("messages", [])) > 0:
                        current_step += 1
                        message = chunk["messages"][-1]
                        
                        print(f"💬 Agent message from {getattr(message, 'name', 'Unknown')}: {getattr(message, 'content', str(message))[:100]}...")
                        
                        # Calculate progress
                        progress = min(90, 10 + (current_step / total_steps) * 80)
                        
                        # Send agent message
                        await self.send_websocket_message(analysis_id, {
                            "type": "message",
                            "data": {
                                "agent": getattr(message, 'name', 'Unknown Agent'),
                                "content": getattr(message, 'content', str(message)),
                                "timestamp": int(datetime.now().timestamp() * 1000),
                                "stage": f"Step {current_step}/{total_steps}",
                                "sentiment": "neutral"
                            },
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        })
                        
                        # Send progress update
                        await self.send_websocket_message(analysis_id, {
                            "type": "progress",
                            "data": {
                                "current": int(progress),
                                "total": 100,
                                "stage": "Processing",
                                "message": f"Processing step {current_step}/{total_steps}"
                            },
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        })
                    else:
                        # Send progress for any chunk without messages
                        current_step += 1
                        progress = min(90, 10 + (current_step / total_steps) * 80)
                        
                        await self.send_websocket_message(analysis_id, {
                            "type": "progress",
                            "data": {
                                "current": int(progress),
                                "total": 100,
                                "stage": "Processing",
                                "message": f"Processing step {current_step}/{total_steps}"
                            },
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        })
                else:
                    # Handle non-dict chunks
                    current_step += 1
                    progress = min(90, 10 + (current_step / total_steps) * 80)
                    
                    await self.send_websocket_message(analysis_id, {
                        "type": "progress",
                        "data": {
                            "current": int(progress),
                            "total": 100,
                            "stage": "Processing",
                            "message": f"Processing step {current_step}/{total_steps}"
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
    return {"message": "TradingAgents API is running", "status": "healthy"}

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
                print(f"📊 Running real analysis for {analysis_request.ticker.upper()}")
                result = await runner.run_analysis(
                    analysis_id,
                    analysis_request.ticker.upper(),
                    analysis_request.timeframe,
                    analysis_request.date
                )
                print(f"✅ Real analysis completed: {result}")

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
    print("🚀 Starting TradingAgents API Server with REAL Analysis...")
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
