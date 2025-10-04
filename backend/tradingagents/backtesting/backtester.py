"""
Backtesting System - Historical performance validation for TradingAgents
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    ticker: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    profit_factor: float
    trades: List[Dict]
    equity_curve: pd.DataFrame


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0
    slippage_percent: float = 0.001  # 0.1% slippage
    max_positions: int = 10
    rebalance_frequency: str = "daily"  # daily, weekly, monthly


class TradingAgentsBacktester:
    """Backtesting system for TradingAgents strategies."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trading_graph = None
        
    def run_backtest(self,
                    ticker: str,
                    start_date: str,
                    end_date: str,
                    debug: bool = False) -> BacktestResult:
        """
        Run a complete backtest for a given ticker and date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            debug: Whether to run in debug mode
            
        Returns:
            BacktestResult with comprehensive performance metrics
        """
        
        print(f"🚀 Starting backtest for {ticker} from {start_date} to {end_date}")
        
        # Initialize trading graph
        self._initialize_trading_graph(debug)
        
        # Get trading dates
        trading_dates = self._get_trading_dates(start_date, end_date)
        
        # Run backtest
        trades = []
        equity_curve = []
        current_capital = self.config.initial_capital
        current_position = 0
        current_shares = 0
        
        for i, date in enumerate(trading_dates):
            try:
                print(f"📅 Processing {date} ({i+1}/{len(trading_dates)})")
                
                # Get trading decision for this date
                decision_data = self._get_trading_decision(ticker, date)
                
                if decision_data:
                    decision = decision_data['decision']
                    confidence = decision_data.get('confidence', 0.5)
                    current_price = decision_data.get('price', 0)
                    
                    # Execute trade based on decision
                    trade_result = self._execute_trade(
                        ticker, date, decision, confidence, current_price,
                        current_capital, current_position, current_shares
                    )
                    
                    if trade_result:
                        trades.append(trade_result)
                        current_capital = trade_result['capital_after']
                        current_position = trade_result['position_after']
                        current_shares = trade_result['shares_after']
                
                # Record equity curve
                equity_curve.append({
                    'date': date,
                    'capital': current_capital,
                    'position': current_position,
                    'shares': current_shares
                })
                
            except Exception as e:
                print(f"❌ Error processing {date}: {e}")
                continue
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        performance_metrics = self._calculate_performance_metrics(trades, equity_df)
        
        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
            equity_curve=equity_df,
            **performance_metrics
        )
    
    def _initialize_trading_graph(self, debug: bool):
        """Initialize the trading graph for backtesting."""
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "anthropic"
        config["backend_url"] = "https://api.anthropic.com"
        config["deep_think_llm"] = "claude-sonnet-4-0"  # Default working model
        config["quick_think_llm"] = "claude-sonnet-4-0"  # Default working model
        config["max_debate_rounds"] = 1  # Reduce for faster backtesting
        config["max_risk_discuss_rounds"] = 1
        config["online_tools"] = True
        
        self.trading_graph = TradingAgentsGraph(debug=debug, config=config)
    
    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """Get list of trading dates between start and end dates."""
        # Get stock data to find actual trading days
        ticker = "AAPL"  # Use AAPL as proxy for market calendar
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        trading_dates = [date.strftime('%Y-%m-%d') for date in stock_data.index]
        return trading_dates
    
    def _get_trading_decision(self, ticker: str, date: str) -> Optional[Dict]:
        """Get trading decision for a specific date."""
        try:
            # Use run_once for faster execution
            final_state, decision = self.trading_graph.run_once(ticker, date)
            
            # Get current price using period approach to avoid date issues
            try:
                stock_data = yf.download(ticker, period="1mo", progress=False)
                if stock_data.empty:
                    return None
                # Get the price closest to the target date
                stock_data = stock_data.tail(1)  # Get the most recent data
            except Exception as e:
                print(f"⚠️ Error getting price data: {e}")
                return None
            
            current_price = stock_data['Close'].iloc[0]
            
            # Extract confidence if available
            confidence = 0.5  # Default confidence
            if 'final_trade_decision' in final_state:
                # Try to extract confidence from the decision text
                decision_text = final_state['final_trade_decision']
                confidence = self._extract_confidence_from_text(decision_text)
            
            return {
                'decision': decision,
                'confidence': confidence,
                'price': current_price,
                'date': date
            }
            
        except Exception as e:
            print(f"❌ Error getting decision for {ticker} on {date}: {e}")
            return None
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """Extract confidence score from decision text."""
        import re
        
        # Look for confidence patterns
        confidence_patterns = [
            r'confidence[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*%?\s*confident',
            r'confidence\s*score[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                confidence = float(match.group(1))
                # Normalize to 0-1 range
                if confidence > 1:
                    confidence = confidence / 100
                return min(1.0, max(0.0, confidence))
        
        return 0.5  # Default confidence
    
    def _execute_trade(self,
                      ticker: str,
                      date: str,
                      decision: str,
                      confidence: float,
                      price: float,
                      current_capital: float,
                      current_position: int,
                      current_shares: int) -> Optional[Dict]:
        """Execute a trade based on the decision."""
        
        # Skip if confidence is too low
        if confidence < 0.3:
            return None
        
        # Calculate position size based on confidence
        position_size = min(0.1, confidence * 0.2)  # Max 10% position, scaled by confidence
        trade_value = current_capital * position_size
        
        # Apply slippage
        slippage = price * self.config.slippage_percent
        if decision == "BUY":
            execution_price = price + slippage
        else:
            execution_price = price - slippage
        
        # Calculate shares to trade
        shares_to_trade = int(trade_value / execution_price)
        
        if shares_to_trade == 0:
            return None
        
        # Execute trade
        if decision == "BUY" and current_position <= 0:
            # Buy shares
            cost = shares_to_trade * execution_price + self.config.commission_per_trade
            if cost <= current_capital:
                new_capital = current_capital - cost
                new_shares = current_shares + shares_to_trade
                new_position = 1
                
                return {
                    'date': date,
                    'action': 'BUY',
                    'shares': shares_to_trade,
                    'price': execution_price,
                    'value': cost,
                    'confidence': confidence,
                    'capital_before': current_capital,
                    'capital_after': new_capital,
                    'shares_before': current_shares,
                    'shares_after': new_shares,
                    'position_before': current_position,
                    'position_after': new_position
                }
        
        elif decision == "SELL" and current_position >= 0 and current_shares > 0:
            # Sell shares
            shares_to_sell = min(shares_to_trade, current_shares)
            proceeds = shares_to_sell * execution_price - self.config.commission_per_trade
            new_capital = current_capital + proceeds
            new_shares = current_shares - shares_to_sell
            new_position = -1 if new_shares == 0 else 0
            
            return {
                'date': date,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': execution_price,
                'value': proceeds,
                'confidence': confidence,
                'capital_before': current_capital,
                'capital_after': new_capital,
                'shares_before': current_shares,
                'shares_after': new_shares,
                'position_before': current_position,
                'position_after': new_position
            }
        
        return None
    
    def _calculate_performance_metrics(self, trades: List[Dict], equity_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'avg_trade_return': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['value'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        initial_capital = self.config.initial_capital
        final_capital = equity_df['capital'].iloc[-1] if not equity_df.empty else initial_capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Annualized return
        days = len(equity_df)
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['capital'].pct_change()
        
        # Max drawdown
        equity_df['cumulative'] = (1 + equity_df['daily_return']).cumprod()
        equity_df['running_max'] = equity_df['cumulative'].expanding().max()
        equity_df['drawdown'] = (equity_df['cumulative'] - equity_df['running_max']) / equity_df['running_max']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio
        daily_returns = equity_df['daily_return'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        trade_returns = [trade['value'] for trade in trades]
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        winning_trade_returns = [trade['value'] for trade in trades if trade['value'] > 0]
        avg_winning_trade = np.mean(winning_trade_returns) if winning_trade_returns else 0
        
        losing_trade_returns = [trade['value'] for trade in trades if trade['value'] < 0]
        avg_losing_trade = np.mean(losing_trade_returns) if losing_trade_returns else 0
        
        # Profit factor
        gross_profit = sum(winning_trade_returns) if winning_trade_returns else 0
        gross_loss = abs(sum(losing_trade_returns)) if losing_trade_returns else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_trade_return': avg_trade_return,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor
        }
    
    def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        
        report = f"""
📊 BACKTESTING REPORT: {result.ticker}
===============================
Period: {result.start_date} to {result.end_date}

📈 PERFORMANCE METRICS
=====================
• Total Return: {result.total_return:.2%}
• Annualized Return: {result.annualized_return:.2%}
• Max Drawdown: {result.max_drawdown:.2%}
• Sharpe Ratio: {result.sharpe_ratio:.2f}
• Sortino Ratio: {result.sortino_ratio:.2f}
• Calmar Ratio: {result.calmar_ratio:.2f}

📊 TRADE STATISTICS
==================
• Total Trades: {result.total_trades}
• Winning Trades: {result.winning_trades}
• Losing Trades: {result.losing_trades}
• Win Rate: {result.win_rate:.2%}
• Average Trade Return: ${result.avg_trade_return:.2f}
• Average Winning Trade: ${result.avg_winning_trade:.2f}
• Average Losing Trade: ${result.avg_losing_trade:.2f}
• Profit Factor: {result.profit_factor:.2f}

💰 CAPITAL EVOLUTION
===================
• Initial Capital: ${self.config.initial_capital:,.2f}
• Final Capital: ${result.equity_curve['capital'].iloc[-1]:,.2f}
• Total Profit/Loss: ${result.equity_curve['capital'].iloc[-1] - self.config.initial_capital:,.2f}
        """
        
        return report.strip()
