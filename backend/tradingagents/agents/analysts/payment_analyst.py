"""
Payment Analyst Agent
Analyzes payment processing data and correlations for trading decisions
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.dataflows.payment_utils import get_payment_analysis, get_payment_risk_score, get_payment_sentiment

def create_payment_analyst(llm, toolkit):
    """Create a payment analyst agent that analyzes payment processing data"""
    
    def payment_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        
        # Get payment analysis
        payment_analysis = get_payment_analysis(ticker, current_date)
        payment_risk_score = get_payment_risk_score(ticker, current_date)
        payment_sentiment = get_payment_sentiment(ticker, current_date)
        
        system_message = f"""You are a Payment Processing Analyst specializing in analyzing payment data correlations with stock performance. Your role is to:

1. Analyze payment processing metrics and their correlation with stock performance
2. Assess payment-related risks that could impact stock valuation
3. Evaluate payment ecosystem health and merchant adoption trends
4. Identify payment processing efficiency indicators
5. Provide insights on cross-border payment trends and their market implications

Current Analysis for {ticker}:
{payment_analysis}

Payment Risk Score: {payment_risk_score:.1f}/100
Payment Sentiment: {payment_sentiment}

Key Analysis Areas:
- Payment volume trends and stock correlation
- Fraud rate impact on market confidence
- Payment success rate as operational efficiency indicator
- Processing time as competitive advantage metric
- Merchant adoption as market penetration indicator
- Cross-border payments as global expansion metric
- Refund/chargeback rates as customer satisfaction indicators

Provide a comprehensive analysis focusing on how payment processing data correlates with and predicts stock performance. Include specific metrics and their implications for trading decisions."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(messages)
        
        return {
            "messages": [result],
            "payment_report": result.content,
            "sender": "payment_analyst",
        }
    
    return payment_analyst_node
