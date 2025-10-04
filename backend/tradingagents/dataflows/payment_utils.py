"""
Payment Data Integration for TradingAgents
Provides payment processing data and correlations for agent analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import os

class PaymentDataProvider:
    """Provides payment processing data for trading analysis"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("../../data/outputs")
        self.enhanced_df = None
        self._load_enhanced_data()
    
    def _load_enhanced_data(self):
        """Load enhanced market-payment data"""
        try:
            enhanced_file = self.data_dir / "enhanced_market_payment_data.parquet"
            if enhanced_file.exists():
                self.enhanced_df = pd.read_parquet(enhanced_file)
                print(f"✅ Loaded payment data: {len(self.enhanced_df)} records")
            else:
                print(f"❌ Payment data not found at {enhanced_file}")
                self.enhanced_df = None
        except Exception as e:
            print(f"❌ Error loading payment data: {e}")
            self.enhanced_df = None
    
    def get_payment_metrics(self, symbol: str, date: str = None) -> Dict[str, Any]:
        """Get payment processing metrics for a symbol"""
        if self.enhanced_df is None:
            return self._get_default_metrics()
        
        symbol_data = self.enhanced_df[self.enhanced_df['symbol'] == symbol.upper()]
        if symbol_data.empty:
            return self._get_default_metrics()
        
        # Get latest data or specific date
        if date:
            # Convert date to match data format
            from datetime import datetime
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d')
            symbol_data = symbol_data[symbol_data['date'] <= date]
        
        if symbol_data.empty:
            return self._get_default_metrics()
        
        latest_data = symbol_data.tail(1).iloc[0]
        
        return {
            'payment_volume': float(latest_data.get('payment_volume', 0)),
            'payment_success_rate': float(latest_data.get('payment_success_rate', 95.0)),
            'fraud_rate': float(latest_data.get('fraud_rate', 0.5)),
            'processing_time_avg': float(latest_data.get('processing_time_avg', 2.0)),
            'payment_methods': int(latest_data.get('payment_methods', 5)),
            'merchant_count': int(latest_data.get('merchant_count', 1000)),
            'cross_border_rate': float(latest_data.get('cross_border_rate', 15.0)),
            'refund_rate': float(latest_data.get('refund_rate', 2.0)),
            'chargeback_rate': float(latest_data.get('chargeback_rate', 0.1)),
            'market_payment_correlation': float(latest_data.get('market_payment_correlation', 0.2))
        }
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default payment metrics when data is not available"""
        return {
            'payment_volume': 100.0,
            'payment_success_rate': 95.0,
            'fraud_rate': 0.5,
            'processing_time_avg': 2.0,
            'payment_methods': 5,
            'merchant_count': 1000,
            'cross_border_rate': 15.0,
            'refund_rate': 2.0,
            'chargeback_rate': 0.1,
            'market_payment_correlation': 0.2
        }
    
    def get_payment_analysis(self, symbol: str, date: str = None) -> str:
        """Get comprehensive payment analysis for a symbol"""
        metrics = self.get_payment_metrics(symbol, date)
        
        analysis = f"""
Payment Processing Analysis for {symbol.upper()}:

📊 Payment Volume: {metrics['payment_volume']:.1f}M transactions
✅ Success Rate: {metrics['payment_success_rate']:.1f}%
🚨 Fraud Rate: {metrics['fraud_rate']:.2f}%
⏱️ Processing Time: {metrics['processing_time_avg']:.1f}s average
💳 Payment Methods: {metrics['payment_methods']} available
🏪 Merchant Count: {metrics['merchant_count']:,} merchants
🌍 Cross-border Rate: {metrics['cross_border_rate']:.1f}%
↩️ Refund Rate: {metrics['refund_rate']:.1f}%
💸 Chargeback Rate: {metrics['chargeback_rate']:.2f}%
📈 Market Correlation: {metrics['market_payment_correlation']:.3f}

Risk Assessment:
- {'LOW RISK' if metrics['fraud_rate'] < 0.8 else 'MODERATE RISK' if metrics['fraud_rate'] < 1.5 else 'HIGH RISK'} fraud exposure
- {'EXCELLENT' if metrics['payment_success_rate'] > 98 else 'GOOD' if metrics['payment_success_rate'] > 95 else 'MODERATE'} payment reliability
- {'FAST' if metrics['processing_time_avg'] < 2.0 else 'MODERATE' if metrics['processing_time_avg'] < 3.0 else 'SLOW'} processing speed
"""
        return analysis.strip()
    
    def get_payment_risk_score(self, symbol: str, date: str = None) -> float:
        """Calculate payment risk score (0-100, lower is better)"""
        metrics = self.get_payment_metrics(symbol, date)
        
        # Risk factors
        fraud_risk = metrics['fraud_rate'] * 20  # 0-20 points
        success_risk = (100 - metrics['payment_success_rate']) * 0.5  # 0-2.5 points
        chargeback_risk = metrics['chargeback_rate'] * 10  # 0-1 points
        refund_risk = metrics['refund_rate'] * 0.5  # 0-1 points
        
        total_risk = fraud_risk + success_risk + chargeback_risk + refund_risk
        return min(100, max(0, total_risk))
    
    def get_payment_sentiment(self, symbol: str, date: str = None) -> str:
        """Get payment sentiment based on metrics"""
        metrics = self.get_payment_metrics(symbol, date)
        risk_score = self.get_payment_risk_score(symbol, date)
        
        if risk_score < 20:
            return "POSITIVE - Low payment risk, high reliability"
        elif risk_score < 40:
            return "NEUTRAL - Moderate payment risk, acceptable reliability"
        else:
            return "NEGATIVE - High payment risk, reliability concerns"

# Global instance
payment_provider = PaymentDataProvider()

def get_payment_metrics(symbol: str, date: str = None) -> Dict[str, Any]:
    """Get payment metrics for a symbol"""
    return payment_provider.get_payment_metrics(symbol, date)

def get_payment_analysis(symbol: str, date: str = None) -> str:
    """Get payment analysis for a symbol"""
    return payment_provider.get_payment_analysis(symbol, date)

def get_payment_risk_score(symbol: str, date: str = None) -> float:
    """Get payment risk score for a symbol"""
    return payment_provider.get_payment_risk_score(symbol, date)

def get_payment_sentiment(symbol: str, date: str = None) -> str:
    """Get payment sentiment for a symbol"""
    return payment_provider.get_payment_sentiment(symbol, date)
