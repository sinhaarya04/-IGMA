import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Target, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { DecisionData } from "./Dashboard";

interface TradingDecisionProps {
  decision: DecisionData | null;
}

const decisionConfig = {
  BUY: {
    color: "bg-bullish text-bullish-foreground",
    icon: TrendingUp,
    label: "BUY",
  },
  SELL: {
    color: "bg-bearish text-bearish-foreground",
    icon: TrendingDown,
    label: "SELL",
  },
  HOLD: {
    color: "bg-yellow-500 text-white",
    icon: Minus,
    label: "HOLD",
  },
  NO_TRADE: {
    color: "bg-muted text-muted-foreground",
    icon: Minus,
    label: "NO TRADE",
  },
};

export const TradingDecision = ({ decision }: TradingDecisionProps) => {
  if (!decision) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Trading Decision
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Target className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">
              Decision will appear after analysis
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Extract decision type from text (BUY, SELL, HOLD)
  const decisionText = decision.text || decision.decision || "";
  const decisionType = decisionText.toUpperCase().startsWith("BUY") ? "BUY" :
                      decisionText.toUpperCase().startsWith("SELL") ? "SELL" :
                      decisionText.toUpperCase().startsWith("HOLD") ? "HOLD" : "NO_TRADE";
  
  const config = decisionConfig[decisionType];
  const Icon = config.icon;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="h-5 w-5" />
          Trading Decision
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center gap-8 mb-6">
          <div
            className={`${config.color} rounded-lg p-8 flex items-center justify-center`}
          >
            <div className="text-center">
              <Icon className="h-12 w-12 mx-auto mb-2" />
              <span className="text-3xl font-bold">{config.label}</span>
            </div>
          </div>

          <div className="text-center">
            <div className="text-5xl font-bold mb-2">{Math.round((decision.confidence || 0) * 100)}%</div>
            <p className="text-muted-foreground">Confidence</p>
          </div>
        </div>

        <div>
          <h4 className="font-semibold mb-2">Decision:</h4>
          <p className="text-sm leading-relaxed mb-4">
            {decision.text || decision.decision || "No decision available"}
          </p>
          
          {decision.rationale && (
            <>
              <h4 className="font-semibold mb-2">Rationale:</h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {decision.rationale}
              </p>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
