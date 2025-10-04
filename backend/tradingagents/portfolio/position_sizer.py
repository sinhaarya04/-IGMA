"""
Portfolio Position Sizing - Advanced position sizing based on risk metrics
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class PositionSize:
    """Represents a calculated position size with risk metrics."""
    ticker: str
    decision: str  # BUY, SELL, HOLD
    confidence_score: float
    recommended_shares: int
    recommended_value: float
    risk_amount: float
    position_size_percent: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    max_loss_amount: float
    expected_return: float
    volatility_adjustment: float


@dataclass
class PortfolioConfig:
    """Portfolio configuration for position sizing."""
    total_capital: float = 100000.0
    max_position_size_percent: float = 0.10  # 10% max per position
    max_portfolio_risk_percent: float = 0.02  # 2% max portfolio risk per trade
    confidence_threshold: float = 0.5  # Minimum confidence for position sizing (lowered from 0.6)
    atr_multiplier: float = 2.0  # ATR multiplier for stop loss
    risk_reward_ratio_min: float = 1.5  # Minimum risk/reward ratio


class PositionSizer:
    """Advanced position sizing calculator based on risk metrics."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
    
    def calculate_position_size(self,
                              ticker: str,
                              decision: str,
                              confidence_score: float,
                              current_price: float,
                              atr: float,
                              volatility: float,
                              expected_return: float = 0.0,
                              technical_strength: float = 0.5,
                              fundamental_strength: float = 0.5) -> PositionSize:
        """
        Calculate optimal position size based on risk metrics.
        
        Args:
            ticker: Stock ticker symbol
            decision: Trading decision (BUY/SELL/HOLD)
            confidence_score: Decision confidence (0.0 to 1.0)
            current_price: Current stock price
            atr: Average True Range for volatility
            volatility: Historical volatility
            expected_return: Expected return percentage
            technical_strength: Technical analysis strength (0.0 to 1.0)
            fundamental_strength: Fundamental analysis strength (0.0 to 1.0)
            
        Returns:
            PositionSize object with detailed calculations
        """
        
        # Skip position sizing for HOLD decisions
        if decision == "HOLD":
            return self._create_hold_position(ticker, current_price)
        
        # Skip if confidence is too low
        if confidence_score < self.config.confidence_threshold:
            return self._create_low_confidence_position(ticker, decision, current_price, confidence_score)
        
        # Calculate position size based on Kelly Criterion and risk metrics
        position_size = self._calculate_kelly_position_size(
            confidence_score, expected_return, volatility, technical_strength, fundamental_strength
        )
        
        # Apply risk-based position sizing
        risk_based_size = self._calculate_risk_based_position_size(atr, current_price)
        
        # Use the more conservative approach
        final_position_percent = min(position_size, risk_based_size, self.config.max_position_size_percent)
        
        # Calculate position details
        position_value = self.config.total_capital * final_position_percent
        shares = int(position_value / current_price)
        actual_position_value = shares * current_price
        
        # Calculate stop loss and take profit
        stop_loss_price, take_profit_price = self._calculate_stop_loss_take_profit(
            current_price, atr, decision, expected_return
        )
        
        # Calculate risk metrics
        risk_amount = abs(current_price - stop_loss_price) * shares
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            current_price, stop_loss_price, take_profit_price
        )
        
        # Volatility adjustment
        volatility_adjustment = self._calculate_volatility_adjustment(volatility)
        
        return PositionSize(
            ticker=ticker,
            decision=decision,
            confidence_score=confidence_score,
            recommended_shares=shares,
            recommended_value=actual_position_value,
            risk_amount=risk_amount,
            position_size_percent=final_position_percent,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_reward_ratio=risk_reward_ratio,
            max_loss_amount=risk_amount,
            expected_return=expected_return,
            volatility_adjustment=volatility_adjustment
        )
    
    def _calculate_kelly_position_size(self,
                                     confidence: float,
                                     expected_return: float,
                                     volatility: float,
                                     technical_strength: float,
                                     fundamental_strength: float) -> float:
        """Calculate position size using Kelly Criterion."""
        
        # Adjust expected return based on confidence and analysis strength
        adjusted_return = expected_return * confidence * (technical_strength + fundamental_strength) / 2
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        win_probability = confidence
        loss_probability = 1 - confidence
        odds_received = abs(adjusted_return) / 100  # Convert percentage to decimal
        
        if odds_received <= 0:
            return 0.0
        
        kelly_fraction = (odds_received * win_probability - loss_probability) / odds_received
        
        # Apply volatility adjustment
        volatility_penalty = max(0.1, 1 - volatility / 100)  # Higher volatility = smaller position
        
        # Cap Kelly fraction to reasonable levels
        kelly_fraction = max(0.0, min(kelly_fraction * volatility_penalty, 0.25))
        
        return kelly_fraction
    
    def _calculate_risk_based_position_size(self, atr: float, current_price: float) -> float:
        """Calculate position size based on risk (ATR)."""
        
        # Risk per share
        risk_per_share = atr * self.config.atr_multiplier
        
        # Maximum risk amount
        max_risk_amount = self.config.total_capital * self.config.max_portfolio_risk_percent
        
        # Calculate position size based on risk
        max_shares_by_risk = max_risk_amount / risk_per_share
        max_position_value = max_shares_by_risk * current_price
        position_percent = max_position_value / self.config.total_capital
        
        return min(position_percent, self.config.max_position_size_percent)
    
    def _calculate_stop_loss_take_profit(self,
                                       current_price: float,
                                       atr: float,
                                       decision: str,
                                       expected_return: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        
        # Stop loss based on ATR
        stop_distance = atr * self.config.atr_multiplier
        
        if decision == "BUY":
            stop_loss_price = current_price - stop_distance
            # Take profit based on expected return or risk/reward ratio
            if expected_return > 0:
                take_profit_price = current_price * (1 + expected_return / 100)
            else:
                take_profit_price = current_price + (stop_distance * self.config.risk_reward_ratio_min)
        else:  # SELL
            stop_loss_price = current_price + stop_distance
            if expected_return < 0:
                take_profit_price = current_price * (1 + expected_return / 100)
            else:
                take_profit_price = current_price - (stop_distance * self.config.risk_reward_ratio_min)
        
        return stop_loss_price, take_profit_price
    
    def _calculate_risk_reward_ratio(self,
                                   current_price: float,
                                   stop_loss_price: float,
                                   take_profit_price: float) -> float:
        """Calculate risk/reward ratio."""
        
        risk = abs(current_price - stop_loss_price)
        reward = abs(take_profit_price - current_price)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate volatility adjustment factor."""
        # Higher volatility = lower adjustment factor
        return max(0.5, 1 - (volatility - 20) / 100)  # Base volatility of 20%
    
    def _create_hold_position(self, ticker: str, current_price: float) -> PositionSize:
        """Create position size for HOLD decision."""
        return PositionSize(
            ticker=ticker,
            decision="HOLD",
            confidence_score=0.0,
            recommended_shares=0,
            recommended_value=0.0,
            risk_amount=0.0,
            position_size_percent=0.0,
            stop_loss_price=current_price,
            take_profit_price=current_price,
            risk_reward_ratio=0.0,
            max_loss_amount=0.0,
            expected_return=0.0,
            volatility_adjustment=1.0
        )
    
    def _create_low_confidence_position(self, ticker: str, decision: str, current_price: float, confidence: float) -> PositionSize:
        """Create position size for low confidence decisions."""
        return PositionSize(
            ticker=ticker,
            decision=decision,
            confidence_score=confidence,
            recommended_shares=0,
            recommended_value=0.0,
            risk_amount=0.0,
            position_size_percent=0.0,
            stop_loss_price=current_price,
            take_profit_price=current_price,
            risk_reward_ratio=0.0,
            max_loss_amount=0.0,
            expected_return=0.0,
            volatility_adjustment=1.0
        )
    
    def get_position_summary(self, position: PositionSize) -> str:
        """Generate a human-readable position summary."""
        
        if position.decision == "HOLD":
            return f"""
📊 POSITION SIZING: {position.ticker}
==================
Decision: HOLD - No position recommended
Current Price: ${position.stop_loss_price:.2f}
Reason: Hold decision - maintain current position
            """
        
        if position.recommended_shares == 0:
            return f"""
📊 POSITION SIZING: {position.ticker}
==================
Decision: {position.decision}
Confidence: {position.confidence_score:.2f}
Current Price: ${position.stop_loss_price:.2f}
Reason: Confidence too low ({position.confidence_score:.2f} < {self.config.confidence_threshold})
No position recommended
            """
        
        return f"""
📊 POSITION SIZING: {position.ticker}
==================
Decision: {position.decision}
Confidence: {position.confidence_score:.2f}

💰 Position Details:
• Recommended Shares: {position.recommended_shares:,}
• Position Value: ${position.recommended_value:,.2f}
• Position Size: {position.position_size_percent:.1%} of portfolio

🎯 Price Levels:
• Current Price: ${position.stop_loss_price + (position.stop_loss_price * 0.02):.2f}
• Stop Loss: ${position.stop_loss_price:.2f}
• Take Profit: ${position.take_profit_price:.2f}

⚖️ Risk Metrics:
• Risk Amount: ${position.risk_amount:,.2f}
• Risk/Reward Ratio: {position.risk_reward_ratio:.2f}
• Max Loss: ${position.max_loss_amount:,.2f}
• Expected Return: {position.expected_return:.1%}
• Volatility Adjustment: {position.volatility_adjustment:.2f}
        """
