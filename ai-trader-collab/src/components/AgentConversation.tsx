import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { ScrollArea } from "./ui/scroll-area";
import { MessageSquare, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { AgentMessage } from "./Dashboard";

interface AgentConversationProps {
  messages: AgentMessage[];
}

const agentAvatars: Record<string, { initials: string; color: string; name: string }> = {
  "Market Analyst": { initials: "MA", color: "bg-blue-500", name: "Market Analyst" },
  "Fundamentals Analyst": { initials: "FA", color: "bg-purple-500", name: "Fundamentals" },
  "News Analyst": { initials: "NA", color: "bg-yellow-500", name: "News" },
  "Payment Flow Analyst": { initials: "PA", color: "bg-green-500", name: "Payment Flow" },
  "Risk Manager": { initials: "RM", color: "bg-red-500", name: "Risk Manager" },
  "Visualization Agent": { initials: "VA", color: "bg-pink-500", name: "Visualization" },
  "Bull Researcher": { initials: "BU", color: "bg-emerald-500", name: "Bull" },
  "Bear Researcher": { initials: "BE", color: "bg-orange-500", name: "Bear" },
  "Trader": { initials: "TR", color: "bg-indigo-500", name: "Trader" },
  "System": { initials: "SY", color: "bg-gray-500", name: "System" },
};

// Helper to format message content with proper spacing and structure
const formatMessageContent = (content: string): string => {
  if (!content) return "";
  
  // Remove excessive whitespace
  let formatted = content.trim();
  
  // Break up long sentences at punctuation for better readability
  formatted = formatted
    .replace(/\. /g, ".\n")
    .replace(/\! /g, "!\n")
    .replace(/\? /g, "?\n");
  
  // Remove multiple consecutive newlines
  formatted = formatted.replace(/\n{3,}/g, "\n\n");
  
  return formatted;
};

const getSentimentIcon = (sentiment?: string) => {
  switch (sentiment) {
    case "bullish":
      return <TrendingUp className="h-3 w-3" />;
    case "bearish":
      return <TrendingDown className="h-3 w-3" />;
    case "neutral":
      return <Minus className="h-3 w-3" />;
    default:
      return null;
  }
};

const getSentimentColor = (sentiment?: string) => {
  switch (sentiment) {
    case "bullish":
      return "bullish";
    case "bearish":
      return "bearish";
    case "neutral":
      return "neutral";
    default:
      return "secondary";
  }
};

export const AgentConversation = ({ messages }: AgentConversationProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Agent Conversation
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full px-4">
          <div ref={scrollRef} className="space-y-4 pb-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-8">
                <MessageSquare className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">
                  Start an analysis to see agents communicate
                </p>
              </div>
            ) : (
              messages.map((message, index) => {
                // Look up avatar by agentName (which now comes from backend)
                const avatar = agentAvatars[message.agentName] || 
                              agentAvatars[message.agent] || {
                  initials: message.agentName?.substring(0, 2).toUpperCase() || "AG",
                  color: "bg-gray-500",
                  name: message.agentName || "Agent"
                };

                // Format the message content for better readability
                const formattedContent = formatMessageContent(message.content);
                
                // Check if this is a continuation of the same agent
                const prevMessage = index > 0 ? messages[index - 1] : null;
                const isSameAgent = prevMessage && prevMessage.agentName === message.agentName;

                return (
                  <div
                    key={message.id}
                    className={`p-4 rounded-lg border border-border bg-card hover:bg-accent/50 transition-all duration-200 ${
                      isSameAgent ? "mt-2" : "mt-4"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`${avatar.color} text-white rounded-full h-10 w-10 flex items-center justify-center font-semibold text-sm flex-shrink-0 shadow-md`}
                      >
                        {avatar.initials}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-2">
                          <span className="font-semibold text-base">
                            {message.agentName}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {message.timestamp}
                          </span>
                          {message.stage && (
                            <Badge variant="outline" className="text-xs">
                              {message.stage}
                            </Badge>
                          )}
                          {message.sentiment && (
                            <Badge
                              variant={getSentimentColor(message.sentiment) as any}
                              className="text-xs gap-1"
                            >
                              {getSentimentIcon(message.sentiment)}
                              {message.sentiment}
                            </Badge>
                          )}
                        </div>

                        <div className="text-sm leading-relaxed whitespace-pre-line text-foreground/90">
                          {formattedContent}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
