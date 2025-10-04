# TradingAgents/graph/signal_processing.py

from langchain_anthropic import ChatAnthropic


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatAnthropic):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision with improved reliability.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        # First try to extract using regex patterns
        import re
        
        # Look for explicit decision patterns
        decision_patterns = [
            r'FINAL\s+DECISION:\s*(BUY|SELL|HOLD)',
            r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*(BUY|SELL|HOLD)\*\*',
            r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*(BUY|SELL|HOLD)',
            r'RECOMMENDATION:\s*(BUY|SELL|HOLD)',
            r'DECISION:\s*(BUY|SELL|HOLD)',
            r'\*\*(BUY|SELL|HOLD)\*\*',
            r'(BUY|SELL|HOLD)\s*$'
        ]
        
        for i, pattern in enumerate(decision_patterns):
            # Find ALL matches and take the LAST one to get the final decision
            matches = re.findall(pattern, full_signal, re.IGNORECASE | re.MULTILINE)
            if matches:
                decision = matches[-1].upper()  # Take the last match
                print(f"✅ Decision extracted via regex pattern {i+1}: {decision}")
                print(f"   Pattern: {pattern}")
                print(f"   Found {len(matches)} matches, using last: {decision}")
                return decision
        
        # Fallback to LLM extraction with improved prompt
        messages = [
            (
                "system",
                """You are a precise decision extraction assistant. Your ONLY task is to find the final trading decision in the provided text.

Look for these patterns in order of priority:
1. "FINAL DECISION: [BUY/SELL/HOLD]"
2. "FINAL TRANSACTION PROPOSAL: [BUY/SELL/HOLD]"
3. "RECOMMENDATION: [BUY/SELL/HOLD]"
4. Any clear decision statement ending with BUY, SELL, or HOLD

CRITICAL: Return ONLY the decision word (BUY, SELL, or HOLD) with no additional text, punctuation, or formatting.""",
            ),
            ("human", full_signal),
        ]

        try:
            result = self.quick_thinking_llm.invoke(messages).content.strip().upper()
            # Clean up the result
            if result in ['BUY', 'SELL', 'HOLD']:
                print(f"✅ Decision extracted via LLM: {result}")
                return result
            else:
                print(f"⚠️ LLM returned invalid decision: {result}, defaulting to HOLD")
                return "HOLD"
        except Exception as e:
            print(f"❌ Error in LLM decision extraction: {e}, defaulting to HOLD")
            return "HOLD"
