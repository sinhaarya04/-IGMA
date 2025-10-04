"""
REST API for TradingAgents - External integrations and web interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime, timedelta
import uvicorn

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.analysts.decision_confidence_analyzer import DecisionConfidenceAnalyzer, DecisionConfidence
from tradingagents.portfolio.position_sizer import PositionSizer, PositionSize, PortfolioConfig
from tradingagents.backtesting.backtester import TradingAgentsBacktester, BacktestResult, BacktestConfig
from tradingagents.alerts.alert_system import TradingAlertSystem, Alert, AlertConfig


# Pydantic models for API
class AnalysisRequest(BaseModel):
    ticker: str
    date: str
    include_confidence: bool = True
    include_position_sizing: bool = False
    portfolio_capital: float = 100000.0


class AnalysisResponse(BaseModel):
    ticker: str
    date: str
    decision: str
    confidence: Optional[DecisionConfidence] = None
    position_size: Optional[PositionSize] = None
    analysis_time: float
    timestamp: datetime


class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0


class AlertSubscriptionRequest(BaseModel):
    tickers: List[str]
    email_recipients: List[str] = []
    webhook_url: str = ""
    min_confidence_threshold: float = 0.7
    check_interval_minutes: int = 15


class AlertResponse(BaseModel):
    success: bool
    message: str
    alert_id: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="TradingAgents API",
    description="Advanced AI-powered trading analysis and portfolio management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
trading_graph = None
confidence_analyzer = None
position_sizer = None
backtester = None
alert_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize the trading system on startup."""
    global trading_graph, confidence_analyzer, position_sizer, backtester, alert_system
    
    print("🚀 Initializing TradingAgents API...")
    
    # Initialize trading graph
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["backend_url"] = "https://api.anthropic.com"
    config["deep_think_llm"] = "claude-sonnet-4-0"
    config["quick_think_llm"] = "claude-sonnet-4-0"
    config["max_debate_rounds"] = 2
    config["max_risk_discuss_rounds"] = 2
    config["online_tools"] = True
    
    trading_graph = TradingAgentsGraph(debug=False, config=config)
    
    # Initialize other components
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-3-sonnet-20240229")
    
    confidence_analyzer = DecisionConfidenceAnalyzer(llm)
    position_sizer = PositionSizer(PortfolioConfig())
    backtester = TradingAgentsBacktester(BacktestConfig())
    alert_system = TradingAlertSystem(AlertConfig())
    
    print("✅ TradingAgents API initialized successfully!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TradingAgents API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analysis": "/analyze",
            "backtest": "/backtest",
            "alerts": "/alerts",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "trading_graph": trading_graph is not None,
            "confidence_analyzer": confidence_analyzer is not None,
            "position_sizer": position_sizer is not None,
            "backtester": backtester is not None,
            "alert_system": alert_system is not None
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticker(request: AnalysisRequest):
    """
    Analyze a ticker and return trading decision with optional confidence and position sizing.
    """
    try:
        start_time = datetime.now()
        
        print(f"📊 Analyzing {request.ticker} for {request.date}")
        
        # Run trading analysis
        final_state, decision = trading_graph.run_once(request.ticker, request.date)
        
        # Calculate analysis time
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        response = AnalysisResponse(
            ticker=request.ticker,
            date=request.date,
            decision=decision,
            analysis_time=analysis_time,
            timestamp=datetime.now()
        )
        
        # Add confidence analysis if requested
        if request.include_confidence:
            try:
                confidence = confidence_analyzer.analyze_decision_confidence(
                    final_decision=decision,
                    market_report=final_state.get('market_report', ''),
                    fundamentals_report=final_state.get('fundamentals_report', ''),
                    sentiment_report=final_state.get('sentiment_report', ''),
                    news_report=final_state.get('news_report', ''),
                    payment_report=final_state.get('payment_report', ''),
                    risk_analysis=final_state.get('final_trade_decision', ''),
                    trader_plan=final_state.get('trader_investment_plan', '')
                )
                response.confidence = confidence
            except Exception as e:
                print(f"⚠️ Error calculating confidence: {e}")
        
        # Add position sizing if requested
        if request.include_position_sizing and response.confidence:
            try:
                # Get current price and ATR (simplified)
                import yfinance as yf
                stock_data = yf.download(request.ticker, period="1mo", progress=False)
                if not stock_data.empty:
                    current_price = float(stock_data['Close'].iloc[-1])
                    atr = float(stock_data['High'].rolling(14).max().iloc[-1] - stock_data['Low'].rolling(14).min().iloc[-1])
                    volatility = float(stock_data['Close'].pct_change().std() * 100)
                    
                    position_size = position_sizer.calculate_position_size(
                        ticker=request.ticker,
                        decision=decision,
                        confidence_score=response.confidence.confidence_score,
                        current_price=current_price,
                        atr=atr,
                        volatility=volatility,
                        expected_return=0.0,  # Could be calculated from analysis
                        technical_strength=response.confidence.technical_strength,
                        fundamental_strength=response.confidence.fundamental_strength
                    )
                    response.position_size = position_size
            except Exception as e:
                print(f"⚠️ Error calculating position size: {e}")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest for a ticker over a specified date range.
    """
    try:
        print(f"📈 Starting backtest for {request.ticker} from {request.start_date} to {request.end_date}")
        
        # Update backtest config
        backtester.config.initial_capital = request.initial_capital
        backtester.config.commission_per_trade = request.commission_per_trade
        
        # Run backtest
        result = backtester.run_backtest(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            debug=False
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.post("/alerts/subscribe", response_model=AlertResponse)
async def subscribe_alerts(request: AlertSubscriptionRequest):
    """
    Subscribe to real-time trading alerts for specified tickers.
    """
    try:
        print(f"🔔 Setting up alerts for {request.tickers}")
        
        # Update alert system config
        alert_system.config.email_recipients = request.email_recipients
        alert_system.config.webhook_url = request.webhook_url
        alert_system.config.min_confidence_threshold = request.min_confidence_threshold
        alert_system.config.check_interval_minutes = request.check_interval_minutes
        
        # Start monitoring
        alert_system.start_monitoring(request.tickers)
        
        return AlertResponse(
            success=True,
            message=f"Successfully subscribed to alerts for {request.tickers}",
            alert_id=f"alert_{int(datetime.now().timestamp())}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert subscription failed: {str(e)}")


@app.get("/alerts/history")
async def get_alert_history(hours: int = 24):
    """
    Get alert history for the specified number of hours.
    """
    try:
        alerts = alert_system.get_alert_history(hours)
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "ticker": alert.ticker,
                    "decision": alert.decision,
                    "confidence": alert.confidence,
                    "price": alert.price,
                    "timestamp": alert.timestamp.isoformat(),
                    "priority": alert.priority,
                    "message": alert.message
                }
                for alert in alerts
            ],
            "summary": alert_system.get_alert_summary(hours)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert history: {str(e)}")


@app.get("/alerts/summary")
async def get_alert_summary(hours: int = 24):
    """
    Get alert summary for the specified number of hours.
    """
    try:
        return {"summary": alert_system.get_alert_summary(hours)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert summary: {str(e)}")


@app.post("/alerts/stop")
async def stop_alerts():
    """
    Stop all alert monitoring.
    """
    try:
        alert_system.stop_monitoring()
        return {"success": True, "message": "Alert monitoring stopped"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop alerts: {str(e)}")


@app.get("/portfolio/position-size")
async def calculate_position_size(
    ticker: str,
    decision: str,
    confidence: float,
    current_price: float,
    atr: float,
    volatility: float,
    portfolio_capital: float = 100000.0
):
    """
    Calculate optimal position size for a trading decision.
    """
    try:
        # Update portfolio config
        position_sizer.config.total_capital = portfolio_capital
        
        position_size = position_sizer.calculate_position_size(
            ticker=ticker,
            decision=decision,
            confidence_score=confidence,
            current_price=current_price,
            atr=atr,
            volatility=volatility,
            expected_return=0.0,
            technical_strength=0.5,
            fundamental_strength=0.5
        )
        
        return {
            "ticker": position_size.ticker,
            "decision": position_size.decision,
            "confidence": position_size.confidence_score,
            "recommended_shares": position_size.recommended_shares,
            "recommended_value": position_size.recommended_value,
            "position_size_percent": position_size.position_size_percent,
            "stop_loss_price": position_size.stop_loss_price,
            "take_profit_price": position_size.take_profit_price,
            "risk_amount": position_size.risk_amount,
            "risk_reward_ratio": position_size.risk_reward_ratio,
            "summary": position_sizer.get_position_summary(position_size)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Position sizing failed: {str(e)}")


@app.get("/market/status")
async def get_market_status():
    """
    Get current market status and system metrics.
    """
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational",
            "active_alerts": len(alert_system.alert_history),
            "recent_alerts": len(alert_system.get_alert_history(1)),  # Last hour
            "components": {
                "trading_graph": "active",
                "confidence_analyzer": "active",
                "position_sizer": "active",
                "backtester": "active",
                "alert_system": "active" if alert_system.running else "stopped"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market status: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "rest_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
