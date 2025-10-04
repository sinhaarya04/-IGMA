"""
Decision Confidence Analyzer - Provides confidence scores for trading decisions
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic


@dataclass
class DecisionConfidence:
    """Represents a trading decision with confidence metrics."""
    decision: str  # BUY, SELL, or HOLD
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    risk_factors: List[str]
    supporting_evidence: List[str]
    conflicting_evidence: List[str]
    market_conditions: str
    technical_strength: float  # 0.0 to 1.0
    fundamental_strength: float  # 0.0 to 1.0
    sentiment_strength: float  # 0.0 to 1.0


class DecisionConfidenceAnalyzer:
    """Analyzes trading decisions and provides confidence scores."""
    
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        
    def analyze_decision_confidence(self, 
                                  final_decision: str,
                                  market_report: str,
                                  fundamentals_report: str,
                                  sentiment_report: str,
                                  news_report: str,
                                  payment_report: str,
                                  risk_analysis: str,
                                  trader_plan: str) -> DecisionConfidence:
        """
        Analyze the confidence of a trading decision based on all available data.
        
        Args:
            final_decision: The final trading decision (BUY/SELL/HOLD)
            market_report: Technical analysis report
            fundamentals_report: Fundamental analysis report
            sentiment_report: Social media sentiment report
            news_report: News analysis report
            payment_report: Payment data analysis report
            risk_analysis: Risk management analysis
            trader_plan: Trader's investment plan
            
        Returns:
            DecisionConfidence object with detailed confidence metrics
        """
        
        # Combine all reports for analysis
        combined_analysis = f"""
        FINAL DECISION: {final_decision}
        
        TECHNICAL ANALYSIS:
        {market_report}
        
        FUNDAMENTAL ANALYSIS:
        {fundamentals_report}
        
        SENTIMENT ANALYSIS:
        {sentiment_report}
        
        NEWS ANALYSIS:
        {news_report}
        
        PAYMENT ANALYSIS:
        {payment_report}
        
        RISK ANALYSIS:
        {risk_analysis}
        
        TRADER PLAN:
        {trader_plan}
        """
        
        # Analyze confidence using LLM
        confidence_analysis = self._analyze_with_llm(combined_analysis, final_decision)
        
        # Calculate technical strength
        technical_strength = self._calculate_technical_strength(market_report)
        
        # Calculate fundamental strength
        fundamental_strength = self._calculate_fundamental_strength(fundamentals_report)
        
        # Calculate sentiment strength
        sentiment_strength = self._calculate_sentiment_strength(sentiment_report, news_report)
        
        # Overall confidence score
        overall_confidence = self._calculate_overall_confidence(
            confidence_analysis, technical_strength, fundamental_strength, sentiment_strength
        )
        
        return DecisionConfidence(
            decision=final_decision,
            confidence_score=overall_confidence,
            reasoning=confidence_analysis.get('reasoning', ''),
            risk_factors=confidence_analysis.get('risk_factors', []),
            supporting_evidence=confidence_analysis.get('supporting_evidence', []),
            conflicting_evidence=confidence_analysis.get('conflicting_evidence', []),
            market_conditions=confidence_analysis.get('market_conditions', ''),
            technical_strength=technical_strength,
            fundamental_strength=fundamental_strength,
            sentiment_strength=sentiment_strength
        )
    
    def _analyze_with_llm(self, combined_analysis: str, decision: str) -> Dict:
        """Use LLM to analyze decision confidence."""
        
        prompt = f"""
        As a quantitative analyst, analyze the confidence level of this trading decision: {decision}
        
        Based on the comprehensive analysis below, provide:
        
        1. CONFIDENCE SCORE (0.0 to 1.0): How confident should we be in this decision?
        2. REASONING: Brief explanation of the confidence level
        3. SUPPORTING EVIDENCE: List 3-5 key factors that support this decision
        4. CONFLICTING EVIDENCE: List any factors that contradict this decision
        5. RISK FACTORS: List 3-5 potential risks to this decision
        6. MARKET CONDITIONS: Current market environment assessment
        
        Format your response as JSON:
        {{
            "confidence_score": 0.85,
            "reasoning": "Strong technical and fundamental alignment with moderate sentiment support",
            "supporting_evidence": ["RSI showing oversold conditions", "Strong earnings growth", "Low debt levels"],
            "conflicting_evidence": ["High P/E ratio", "Market volatility concerns"],
            "risk_factors": ["Interest rate sensitivity", "Competition risk", "Market correction risk"],
            "market_conditions": "Bullish with moderate volatility"
        }}
        
        ANALYSIS:
        {combined_analysis}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            import json
            return json.loads(response.content)
        except Exception as e:
            print(f"Error in LLM confidence analysis: {e}")
            return {
                "confidence_score": 0.5,
                "reasoning": "Unable to analyze confidence due to processing error",
                "supporting_evidence": [],
                "conflicting_evidence": [],
                "risk_factors": ["Analysis error"],
                "market_conditions": "Unknown"
            }
    
    def _calculate_technical_strength(self, market_report: str) -> float:
        """Calculate technical analysis strength score."""
        strength_indicators = [
            'strong', 'bullish', 'uptrend', 'breakout', 'support', 'momentum',
            'oversold', 'oversold', 'golden cross', 'higher highs', 'higher lows'
        ]
        
        weakness_indicators = [
            'weak', 'bearish', 'downtrend', 'breakdown', 'resistance', 'divergence',
            'overbought', 'death cross', 'lower highs', 'lower lows', 'consolidation'
        ]
        
        text_lower = market_report.lower()
        strength_count = sum(1 for indicator in strength_indicators if indicator in text_lower)
        weakness_count = sum(1 for indicator in weakness_indicators if indicator in text_lower)
        
        if strength_count + weakness_count == 0:
            return 0.5
        
        return min(1.0, max(0.0, (strength_count - weakness_count) / (strength_count + weakness_count) + 0.5))
    
    def _calculate_fundamental_strength(self, fundamentals_report: str) -> float:
        """Calculate fundamental analysis strength score."""
        positive_indicators = [
            'strong', 'growth', 'profit', 'revenue', 'earnings', 'cash flow',
            'low debt', 'high margin', 'competitive', 'market share'
        ]
        
        negative_indicators = [
            'weak', 'decline', 'loss', 'debt', 'high valuation', 'overvalued',
            'competition', 'risk', 'concern', 'challenge'
        ]
        
        text_lower = fundamentals_report.lower()
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5
        
        return min(1.0, max(0.0, (positive_count - negative_count) / (positive_count + negative_count) + 0.5))
    
    def _calculate_sentiment_strength(self, sentiment_report: str, news_report: str) -> float:
        """Calculate sentiment analysis strength score."""
        combined_text = (sentiment_report + " " + news_report).lower()
        
        positive_sentiment = [
            'positive', 'bullish', 'optimistic', 'strong', 'growth', 'upbeat',
            'favorable', 'supportive', 'encouraging', 'confident'
        ]
        
        negative_sentiment = [
            'negative', 'bearish', 'pessimistic', 'weak', 'decline', 'downbeat',
            'unfavorable', 'concerning', 'worrisome', 'uncertain'
        ]
        
        positive_count = sum(1 for word in positive_sentiment if word in combined_text)
        negative_count = sum(1 for word in negative_sentiment if word in combined_text)
        
        if positive_count + negative_count == 0:
            return 0.5
        
        return min(1.0, max(0.0, (positive_count - negative_count) / (positive_count + negative_count) + 0.5))
    
    def _calculate_overall_confidence(self, 
                                    llm_analysis: Dict,
                                    technical_strength: float,
                                    fundamental_strength: float,
                                    sentiment_strength: float) -> float:
        """Calculate overall confidence score."""
        
        # Weighted average of different components
        weights = {
            'llm_confidence': 0.4,
            'technical': 0.3,
            'fundamental': 0.2,
            'sentiment': 0.1
        }
        
        llm_confidence = llm_analysis.get('confidence_score', 0.5)
        
        overall = (
            llm_confidence * weights['llm_confidence'] +
            technical_strength * weights['technical'] +
            fundamental_strength * weights['fundamental'] +
            sentiment_strength * weights['sentiment']
        )
        
        return min(1.0, max(0.0, overall))
    
    def get_confidence_summary(self, confidence: DecisionConfidence) -> str:
        """Generate a human-readable confidence summary."""
        
        confidence_level = "HIGH" if confidence.confidence_score >= 0.7 else "MEDIUM" if confidence.confidence_score >= 0.5 else "LOW"
        
        summary = f"""
🎯 DECISION CONFIDENCE ANALYSIS
==============================

Decision: {confidence.decision}
Confidence Level: {confidence_level} ({confidence.confidence_score:.2f})

📊 Component Scores:
• Technical Strength: {confidence.technical_strength:.2f}
• Fundamental Strength: {confidence.fundamental_strength:.2f}
• Sentiment Strength: {confidence.sentiment_strength:.2f}

💡 Reasoning: {confidence.reasoning}

✅ Supporting Evidence:
{chr(10).join(f"• {evidence}" for evidence in confidence.supporting_evidence)}

⚠️ Conflicting Evidence:
{chr(10).join(f"• {evidence}" for evidence in confidence.conflicting_evidence)}

🚨 Risk Factors:
{chr(10).join(f"• {risk}" for risk in confidence.risk_factors)}

📈 Market Conditions: {confidence.market_conditions}
        """
        
        return summary.strip()
