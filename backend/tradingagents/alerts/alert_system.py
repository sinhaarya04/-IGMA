"""
Alert System - Real-time notifications for trading signals
"""

import asyncio
import smtplib
import json
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


@dataclass
class Alert:
    """Represents a trading alert."""
    id: str
    ticker: str
    decision: str
    confidence: float
    price: float
    timestamp: datetime
    message: str
    priority: str  # HIGH, MEDIUM, LOW
    alert_type: str  # SIGNAL, THRESHOLD, CUSTOM
    metadata: Dict


@dataclass
class AlertConfig:
    """Configuration for alert system."""
    email_enabled: bool = True
    webhook_enabled: bool = False
    sms_enabled: bool = False
    email_recipients: List[str] = None
    webhook_url: str = ""
    sms_provider: str = ""
    min_confidence_threshold: float = 0.7
    high_priority_threshold: float = 0.9
    check_interval_minutes: int = 15
    max_alerts_per_hour: int = 10


class TradingAlertSystem:
    """Real-time alert system for trading signals."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.trading_graph = None
        self.alert_history = []
        self.subscribers = []
        self.running = False
        
        if self.config.email_recipients is None:
            self.config.email_recipients = []
    
    def start_monitoring(self, tickers: List[str]):
        """Start monitoring the specified tickers."""
        print(f"🚨 Starting alert monitoring for {tickers}")
        self.running = True
        
        # Initialize trading graph
        self._initialize_trading_graph()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop(tickers))
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        print("🛑 Alert monitoring stopped")
    
    def subscribe(self, callback: Callable[[Alert], None]):
        """Subscribe to alerts with a callback function."""
        self.subscribers.append(callback)
    
    def send_alert(self, alert: Alert):
        """Send an alert through all configured channels."""
        
        # Check rate limiting
        if self._is_rate_limited():
            print(f"⚠️ Rate limited - skipping alert for {alert.ticker}")
            return
        
        # Add to history
        self.alert_history.append(alert)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                print(f"❌ Error in alert callback: {e}")
        
        # Send through configured channels
        if self.config.email_enabled:
            self._send_email_alert(alert)
        
        if self.config.webhook_enabled:
            self._send_webhook_alert(alert)
        
        if self.config.sms_enabled:
            self._send_sms_alert(alert)
        
        print(f"📢 Alert sent: {alert.ticker} {alert.decision} (Confidence: {alert.confidence:.2f})")
    
    async def _monitoring_loop(self, tickers: List[str]):
        """Main monitoring loop."""
        while self.running:
            try:
                for ticker in tickers:
                    await self._check_ticker(ticker)
                
                # Wait before next check
                await asyncio.sleep(self.config.check_interval_minutes * 60)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _check_ticker(self, ticker: str):
        """Check a specific ticker for trading signals."""
        try:
            # Get current date
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get trading decision
            final_state, decision = self.trading_graph.run_once(ticker, current_date)
            
            # Extract confidence and price
            confidence = self._extract_confidence(final_state)
            price = self._get_current_price(ticker)
            
            # Check if alert should be sent
            if self._should_send_alert(decision, confidence):
                alert = Alert(
                    id=f"{ticker}_{int(time.time())}",
                    ticker=ticker,
                    decision=decision,
                    confidence=confidence,
                    price=price,
                    timestamp=datetime.now(),
                    message=self._generate_alert_message(ticker, decision, confidence, price),
                    priority=self._determine_priority(confidence),
                    alert_type="SIGNAL",
                    metadata={
                        'final_state': final_state,
                        'analysis_date': current_date
                    }
                )
                
                self.send_alert(alert)
        
        except Exception as e:
            print(f"❌ Error checking {ticker}: {e}")
    
    def _initialize_trading_graph(self):
        """Initialize the trading graph for alerts."""
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "anthropic"
        config["backend_url"] = "https://api.anthropic.com"
        config["deep_think_llm"] = "claude-sonnet-4-0"
        config["quick_think_llm"] = "claude-sonnet-4-0"
        config["max_debate_rounds"] = 1  # Faster for alerts
        config["max_risk_discuss_rounds"] = 1
        config["online_tools"] = True
        
        self.trading_graph = TradingAgentsGraph(debug=False, config=config)
    
    def _extract_confidence(self, final_state: Dict) -> float:
        """Extract confidence score from final state."""
        try:
            # Try to extract from decision text
            if 'final_trade_decision' in final_state:
                decision_text = final_state['final_trade_decision']
                return self._parse_confidence_from_text(decision_text)
            
            # Default confidence
            return 0.5
            
        except Exception:
            return 0.5
    
    def _parse_confidence_from_text(self, text: str) -> float:
        """Parse confidence score from decision text."""
        import re
        
        # Look for confidence patterns
        patterns = [
            r'confidence[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*%?\s*confident',
            r'confidence\s*score[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                confidence = float(match.group(1))
                if confidence > 1:
                    confidence = confidence / 100
                return min(1.0, max(0.0, confidence))
        
        return 0.5
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return 0.0
    
    def _should_send_alert(self, decision: str, confidence: float) -> bool:
        """Determine if an alert should be sent."""
        
        # Skip HOLD decisions
        if decision == "HOLD":
            return False
        
        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            return False
        
        # Check if we already sent an alert for this ticker recently
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_alerts) >= self.config.max_alerts_per_hour:
            return False
        
        return True
    
    def _determine_priority(self, confidence: float) -> str:
        """Determine alert priority based on confidence."""
        if confidence >= self.config.high_priority_threshold:
            return "HIGH"
        elif confidence >= self.config.min_confidence_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_alert_message(self, ticker: str, decision: str, confidence: float, price: float) -> str:
        """Generate alert message."""
        
        priority_emoji = {
            "HIGH": "🚨",
            "MEDIUM": "⚠️",
            "LOW": "ℹ️"
        }
        
        priority = self._determine_priority(confidence)
        emoji = priority_emoji.get(priority, "📢")
        
        message = f"""
{emoji} TRADING ALERT - {ticker}
===============================

Decision: {decision}
Confidence: {confidence:.2%}
Current Price: ${price:.2f}
Priority: {priority}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from TradingAgents.
Please review the analysis before making any trading decisions.
        """
        
        return message.strip()
    
    def _is_rate_limited(self) -> bool:
        """Check if we're rate limited."""
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        return len(recent_alerts) >= self.config.max_alerts_per_hour
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        if not self.config.email_recipients:
            return
        
        try:
            # Email configuration (you'll need to set these)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            email_user = "your-email@gmail.com"  # Set your email
            email_password = "your-app-password"  # Set your app password
            
            msg = MimeMultipart()
            msg['From'] = email_user
            msg['To'] = ", ".join(self.config.email_recipients)
            msg['Subject'] = f"Trading Alert: {alert.ticker} {alert.decision}"
            
            msg.attach(MimeText(alert.message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, self.config.email_recipients, text)
            server.quit()
            
        except Exception as e:
            print(f"❌ Error sending email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert."""
        if not self.config.webhook_url:
            return
        
        try:
            payload = asdict(alert)
            payload['timestamp'] = alert.timestamp.isoformat()
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending webhook alert: {e}")
    
    def _send_sms_alert(self, alert: Alert):
        """Send SMS alert (placeholder - implement with your SMS provider)."""
        # This is a placeholder - implement with your preferred SMS provider
        # (Twilio, AWS SNS, etc.)
        print(f"📱 SMS Alert: {alert.ticker} {alert.decision}")
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def get_alert_summary(self, hours: int = 24) -> str:
        """Get a summary of recent alerts."""
        recent_alerts = self.get_alert_history(hours)
        
        if not recent_alerts:
            return f"No alerts in the last {hours} hours"
        
        summary = f"""
📊 ALERT SUMMARY (Last {hours} hours)
=====================================
Total Alerts: {len(recent_alerts)}

By Decision:
• BUY: {sum(1 for a in recent_alerts if a.decision == 'BUY')}
• SELL: {sum(1 for a in recent_alerts if a.decision == 'SELL')}
• HOLD: {sum(1 for a in recent_alerts if a.decision == 'HOLD')}

By Priority:
• HIGH: {sum(1 for a in recent_alerts if a.priority == 'HIGH')}
• MEDIUM: {sum(1 for a in recent_alerts if a.priority == 'MEDIUM')}
• LOW: {sum(1 for a in recent_alerts if a.priority == 'LOW')}

Average Confidence: {sum(a.confidence for a in recent_alerts) / len(recent_alerts):.2%}

Recent Alerts:
{chr(10).join(f"• {a.ticker} {a.decision} ({a.confidence:.2%}) - {a.timestamp.strftime('%H:%M')}" for a in recent_alerts[-5:])}
        """
        
        return summary.strip()
