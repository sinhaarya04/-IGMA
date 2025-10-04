import { AgentMessage, ChartData, DecisionData, ReportData } from "@/components/Dashboard";

export interface DemoScriptStep {
  delay: number;
  type: "message" | "chart" | "decision" | "report" | "progress";
  data: any;
}

export const generateDemoScript = (ticker: string): DemoScriptStep[] => {
  return [
    {
      delay: 500,
      type: "progress",
      data: { progress: 10 },
    },
    {
      delay: 1000,
      type: "message",
      data: {
        agent: "system",
        agentName: "System",
        stage: "Initialization",
        content: `🚀 Analysis started for ${ticker}. Initializing TradingAgents...`,
      },
    },
    {
      delay: 1500,
      type: "progress",
      data: { progress: 20 },
    },
    {
      delay: 2000,
      type: "message",
      data: {
        agent: "market_analyst",
        agentName: "Market Analyst",
        stage: "Technical Analysis",
        sentiment: "bullish",
        content: `📊 Analyzing ${ticker} technical indicators. RSI at 71 showing overbought momentum. MACD histogram positive with bullish crossover detected. Price trading above 20-day and 50-day moving averages, indicating strong uptrend.`,
      },
    },
    {
      delay: 3000,
      type: "chart",
      data: {
        title: "Price vs Volume Analysis",
        html: `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#1a1a2e;color:#16a34a;border-radius:8px;"><div style="text-align:center;"><div style="font-size:48px;margin-bottom:10px;">📈</div><div style="font-size:24px;font-weight:bold;">Returns vs Volume</div><div style="font-size:14px;margin-top:10px;color:#888;">Interactive Plotly chart showing correlation</div></div></div>`,
        explanation: "Shows strong positive correlation between volume spikes and price increases, suggesting institutional buying.",
      },
    },
    {
      delay: 3500,
      type: "progress",
      data: { progress: 35 },
    },
    {
      delay: 4000,
      type: "message",
      data: {
        agent: "fundamentals",
        agentName: "Fundamentals Analyst",
        stage: "Fundamental Analysis",
        sentiment: "bullish",
        content: `💰 ${ticker} fundamental analysis complete. P/E ratio of 39.2 vs sector avg 32. Q3 earnings beat by 12%, revenue growth YoY at 9.6%. Market cap $3.83T. Strong profitability metrics with 24.3% profit margin.`,
      },
    },
    {
      delay: 4500,
      type: "chart",
      data: {
        title: "Multi-Line Momentum",
        html: `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#1a1a2e;color:#3b82f6;border-radius:8px;"><div style="text-align:center;"><div style="font-size:48px;margin-bottom:10px;">📉</div><div style="font-size:24px;font-weight:bold;">RSI & MACD Trends</div><div style="font-size:14px;margin-top:10px;color:#888;">Multi-line momentum indicators</div></div></div>`,
        explanation: "RSI and MACD showing divergence, indicating potential momentum shift.",
      },
    },
    {
      delay: 5000,
      type: "progress",
      data: { progress: 50 },
    },
    {
      delay: 5500,
      type: "message",
      data: {
        agent: "news_analyst",
        agentName: "News Analyst",
        stage: "News Analysis",
        sentiment: "neutral",
        content: `📰 Recent news sentiment for ${ticker}: Mostly positive coverage on product launches. No adverse headlines detected. Social media sentiment trending positive with increased engagement.`,
      },
    },
    {
      delay: 6000,
      type: "message",
      data: {
        agent: "payment_analyst",
        agentName: "Payment Analyst",
        stage: "Payment Analysis",
        sentiment: "bullish",
        content: `💳 Payment processing analysis: 96M monthly transactions with 93% success rate. Fraud rate at 0.30% (excellent). Processing time 1.7s (competitive advantage). Strong payment ecosystem health.`,
      },
    },
    {
      delay: 6500,
      type: "progress",
      data: { progress: 65 },
    },
    {
      delay: 7000,
      type: "chart",
      data: {
        title: "Price & VWAP",
        html: `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#1a1a2e;color:#f59e0b;border-radius:8px;"><div style="text-align:center;"><div style="font-size:48px;margin-bottom:10px;">📊</div><div style="font-size:24px;font-weight:bold;">Price vs VWAP</div><div style="font-size:14px;margin-top:10px;color:#888;">Volume-weighted average price</div></div></div>`,
        explanation: "Price trading above VWAP indicates strong institutional support and bullish sentiment.",
      },
    },
    {
      delay: 7500,
      type: "message",
      data: {
        agent: "bull_researcher",
        agentName: "Bull Researcher",
        stage: "Research",
        sentiment: "bullish",
        content: `🐂 Bull Case: Strong technical momentum with RSI at 71. Fundamentals support premium valuation with 24% profit margins and $95B free cash flow. Payment ecosystem shows robust growth. Recommendation: BUY.`,
      },
    },
    {
      delay: 8000,
      type: "message",
      data: {
        agent: "bear_researcher",
        agentName: "Bear Researcher",
        stage: "Research",
        sentiment: "bearish",
        content: `🐻 Bear Case: Overbought conditions with RSI > 70. P/E of 39 too high for 9.6% revenue growth. Current ratio below 1.0 raises liquidity concerns. Risk-reward favors profit-taking. Recommendation: SELL.`,
      },
    },
    {
      delay: 8500,
      type: "progress",
      data: { progress: 80 },
    },
    {
      delay: 9000,
      type: "chart",
      data: {
        title: "Bollinger Bands",
        html: `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#1a1a2e;color:#ef4444;border-radius:8px;"><div style="text-align:center;"><div style="font-size:48px;margin-bottom:10px;">📉</div><div style="font-size:24px;font-weight:bold;">Bollinger Bands</div><div style="font-size:14px;margin-top:10px;color:#888;">Volatility and breakout analysis</div></div></div>`,
        explanation: "Price approaching upper Bollinger Band, indicating potential resistance and overbought conditions.",
      },
    },
    {
      delay: 9500,
      type: "message",
      data: {
        agent: "risk_manager",
        agentName: "Risk Manager",
        stage: "Risk Assessment",
        sentiment: "neutral",
        content: `⚖️ Risk Assessment: VIX at 15.2 suggests controlled volatility. Position size recommended: 2% of portfolio. Stop loss at $253 (-1.9% from current). Upside target $267 (+3.5%). Risk-reward ratio: 1:1.8.`,
      },
    },
    {
      delay: 10000,
      type: "progress",
      data: { progress: 95 },
    },
    {
      delay: 10500,
      type: "report",
      data: {
        title: "Technical Analysis Report",
        agent: "market_analyst",
        agentName: "Market Analyst",
        preview: "Comprehensive technical indicator breakdown with trend analysis and momentum signals...",
        fullText: "Full technical analysis report with detailed indicator analysis, support/resistance levels, and trading recommendations.",
      },
    },
    {
      delay: 11000,
      type: "report",
      data: {
        title: "Fundamental Deep Dive",
        agent: "fundamentals",
        agentName: "Fundamentals Analyst",
        preview: "Q3 earnings analysis with growth metrics and valuation assessment...",
        fullText: "Comprehensive fundamental analysis including financial metrics, growth projections, and competitive positioning.",
      },
    },
    {
      delay: 11500,
      type: "report",
      data: {
        title: "Risk Assessment",
        agent: "risk_manager",
        agentName: "Risk Manager",
        preview: "Position sizing and stop loss recommendations with volatility analysis...",
        fullText: "Complete risk analysis with position sizing, stop loss levels, and portfolio allocation recommendations.",
      },
    },
    {
      delay: 12000,
      type: "progress",
      data: { progress: 100 },
    },
    {
      delay: 12500,
      type: "decision",
      data: {
        decision: "HOLD" as const,
        confidence: 0.68,
        rationale: `After comprehensive analysis of ${ticker}, the trading decision is HOLD. While technical indicators show overbought conditions (RSI 71), strong fundamentals (24% profit margins, $95B FCF) and robust payment ecosystem (96M transactions, 93% success rate) provide support. The bull-bear debate reveals valid arguments on both sides. Given the P/E of 39.2 vs 9.6% growth, we recommend maintaining current positions while monitoring the $253 support level. Risk-reward ratio of 1:1.8 is acceptable for existing positions.`,
      },
    },
  ];
};

export const runDemoScript = (
  ticker: string,
  onStep: (step: DemoScriptStep) => void,
  onComplete: () => void
) => {
  const steps = generateDemoScript(ticker);
  let currentStep = 0;

  const executeNextStep = () => {
    if (currentStep >= steps.length) {
      onComplete();
      return;
    }

    const step = steps[currentStep];
    onStep(step);
    currentStep++;

    if (currentStep < steps.length) {
      setTimeout(executeNextStep, steps[currentStep].delay);
    } else {
      onComplete();
    }
  };

  // Start the script
  setTimeout(executeNextStep, steps[0].delay);
};

