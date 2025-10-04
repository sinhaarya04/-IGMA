import { useState, useEffect, useRef } from "react";
import { DashboardHeader } from "./DashboardHeader";
import { ControlPanel } from "./ControlPanel";
import { AgentConversation } from "./AgentConversation";
import { ChartGallery } from "./ChartGallery";
import { TradingDecision } from "./TradingDecision";
import { ReportsPanel } from "./ReportsPanel";
import { useWebSocket } from "../hooks/useWebSocket";
import { runDemoScript, DemoScriptStep } from "../utils/demoScript";
import { toast } from "sonner";

export interface AgentMessage {
  id: string;
  agent: string;
  agentName: string;
  timestamp: string;
  stage: string;
  sentiment?: "bullish" | "bearish" | "neutral";
  content: string;
}

export interface ChartData {
  id: string;
  title: string;
  html: string;
  explanation: string;
}

export interface DecisionData {
  decision: "BUY" | "SELL" | "HOLD" | "NO_TRADE";
  confidence: number;
  rationale: string;
}

export interface ReportData {
  id: string;
  title: string;
  agent: string;
  agentName: string;
  timestamp: string;
  preview: string;
  fullText: string;
}

const Dashboard = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("1D");
  const [selectedDate, setSelectedDate] = useState<Date | undefined>();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [charts, setCharts] = useState<ChartData[]>([]);
  const [decision, setDecision] = useState<DecisionData | null>(null);
  const [reports, setReports] = useState<ReportData[]>([]);
  const [messageIndex, setMessageIndex] = useState(0); // Track which messages we've fetched
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const { status, connect, disconnect, sendMessage } = useWebSocket({
    onMessage: (data) => {
      console.log("📨 WebSocket message received:", data);

      // Handle "message" type from backend (agent messages)
      if (data.type === "message" || data.type === "agent_message") {
        const messageData = data.data || data;
        
        // Convert content to string if it's an object/array
        let content = messageData.content || messageData.message || "";
        if (typeof content === "object") {
          // If content is an array (like Claude's tool calls), extract text
          if (Array.isArray(content)) {
            content = content
              .filter((item) => item?.text || typeof item === "string")
              .map((item) => (typeof item === "string" ? item : item.text))
              .join(" ");
          } else {
            // If it's an object, stringify it for display
            content = JSON.stringify(content);
          }
        }
        
        const newMessage: AgentMessage = {
          id: `${messageData.agent || 'agent'}-${Date.now()}-${Math.random()}`,
          agent: messageData.agent || "system",
          agentName: messageData.agent_name || messageData.agent || "System",
          timestamp: new Date().toLocaleTimeString(),
          stage: messageData.stage || "Analysis",
          sentiment: messageData.sentiment,
          content: content,
        };
        setMessages((prev) => [...prev, newMessage]);
        console.log("✅ Added agent message:", newMessage.agentName);
      }

      // Handle progress updates
      if (data.type === "progress") {
        const progressData = data.data || data;
        const current = progressData.current || progressData.progress || 0;
        const total = progressData.total || 100;
        const calculatedProgress = Math.round((current / total) * 100);
        setProgress(calculatedProgress);
        console.log("📊 Progress:", calculatedProgress + "%");
      }

      // Handle completion
      if (data.type === "complete") {
        setIsAnalyzing(false);
        setProgress(100);
        toast.success("✅ Analysis complete!");
        console.log("🎉 Analysis completed!");
        
        // Fetch charts and reports from backend
        if (analysisId) {
          fetchChartsAndReports(analysisId);
        }
      }

      // Handle charts
      if (data.type === "chart") {
        const chartData = data.data || data;
        
        // If it's a chart generation notification (not actual HTML), show a toast
        if (chartData.status === "generating") {
          toast.info("📊 Generating visualizations...");
          console.log("📊 Chart generation started");
        } else if (chartData.status === "complete") {
          toast.success(`✅ Generated ${chartData.count || 'multiple'} charts!`);
          console.log(`📈 Charts complete: ${chartData.count || 0}`);
          // Trigger chart fetch after a brief delay
          if (analysisId) {
            setTimeout(() => fetchChartsAndReports(analysisId), 1000);
          }
        }
        
        // If it includes actual HTML, add the chart
        if (chartData.html) {
          const newChart: ChartData = {
            id: `chart-${Date.now()}-${Math.random()}`,
            title: chartData.title || "Chart",
            html: chartData.html || "",
            explanation: chartData.explanation || "Chart generated by analysis agent",
          };
          setCharts((prev) => [...prev, newChart]);
          console.log("📈 Added chart:", newChart.title);
        }
      }

      // Handle decision
      if (data.type === "decision") {
        const decisionData = data.data || data;
        setDecision({
          decision: decisionData.decision || decisionData.text || "HOLD",
          confidence: decisionData.confidence || 0.5,
          rationale: decisionData.rationale || "Analysis completed",
        });
        console.log("🎯 Decision:", decisionData.decision || decisionData.text);
      }

      // Handle reports
      if (data.type === "report") {
        const reportData = data.data || data;
        const newReport: ReportData = {
          id: `report-${Date.now()}-${Math.random()}`,
          title: reportData.title || "Analysis Report",
          agent: reportData.agent || "system",
          agentName: reportData.agent_name || reportData.agent || "System",
          timestamp: new Date().toLocaleTimeString(),
          preview: reportData.preview || reportData.content?.slice(0, 150) + "...",
          fullText: reportData.content || reportData.fullText || "",
        };
        setReports((prev) => [...prev, newReport]);
        console.log("📄 Added report:", newReport.title);
      }

      // Handle errors
      if (data.type === "error") {
        const errorData = data.data || data;
        toast.error(errorData.message || "An error occurred");
        setIsAnalyzing(false);
        console.error("❌ Error:", errorData.message);
      }

      // Handle connected status
      if (data.type === "connected") {
        toast.success("🔌 WebSocket connected!");
        console.log("🔌 WebSocket connection established");
      }
    },
  });

  // Poll for new messages
  const pollMessages = async (id: string) => {
    try {
      const response = await fetch(
        `http://localhost:8002/api/analysis/${id}/messages?since=${messageIndex}`
      );
      
      if (!response.ok) return;
      
      const data = await response.json();
      
      if (data.new_count > 0) {
        console.log(`📬 Polled ${data.new_count} new messages`);
        
        // Process each new message
        data.messages.forEach((msg: any) => {
          if (msg.type === "message") {
            const messageData = msg.data || msg;
            
            // Convert content to string if it's an object/array
            let content = messageData.content || messageData.message || "";
            if (typeof content === "object") {
              // If content is an array (like Claude's tool calls), extract text
              if (Array.isArray(content)) {
                content = content
                  .filter((item) => item?.text || typeof item === "string")
                  .map((item) => (typeof item === "string" ? item : item.text))
                  .join(" ");
              } else {
                // If it's an object, stringify it for display
                content = JSON.stringify(content);
              }
            }
            
            const newMessage: AgentMessage = {
              id: `${messageData.agent || 'agent'}-${Date.now()}-${Math.random()}`,
              agent: messageData.agent || "system",
              agentName: messageData.agent_name || messageData.agent || "System",
              timestamp: new Date().toLocaleTimeString(),
              stage: messageData.stage || "Analysis",
              sentiment: messageData.sentiment,
              content: content,
            };
            setMessages((prev) => [...prev, newMessage]);
          }
          
          if (msg.type === "progress") {
            const progressData = msg.data || msg;
            const current = progressData.current || progressData.progress || 0;
            const total = progressData.total || 100;
            setProgress(Math.round((current / total) * 100));
          }
          
          if (msg.type === "complete") {
            setIsAnalyzing(false);
            setProgress(100);
            toast.success("✅ Analysis complete!");
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
            }
            // Fetch charts and reports after completion
            setTimeout(() => fetchChartsAndReports(id), 500);
          }
          
          if (msg.type === "chart") {
            const chartData = msg.data || msg;
            if (chartData.status === "generating") {
              toast.info("📊 Generating visualizations...");
            } else if (chartData.status === "complete") {
              toast.success(`✅ Generated ${chartData.count || 'multiple'} charts!`);
              // Fetch charts immediately
              setTimeout(() => fetchChartsAndReports(id), 1000);
            }
          }
          
          if (msg.type === "decision") {
            const decisionData = msg.data || msg;
            setDecision({
              decision: decisionData.decision || decisionData.text || "HOLD",
              confidence: decisionData.confidence || 0.5,
              rationale: decisionData.rationale || "Analysis completed",
            });
          }
        });
        
        setMessageIndex(data.total); // Update index for next poll
      }
    } catch (error) {
      console.error("Polling error:", error);
    }
  };

  // Fetch charts and reports after analysis completes
  const fetchChartsAndReports = async (id: string) => {
    try {
      // Fetch charts
      const chartsResponse = await fetch(`http://localhost:8002/api/charts/${id}`);
      if (chartsResponse.ok) {
        const chartsData = await chartsResponse.json();
        if (chartsData.charts && chartsData.charts.length > 0) {
          const fetchedCharts = await Promise.all(
            chartsData.charts.map(async (chart: any) => {
              const htmlResponse = await fetch(`http://localhost:8002${chart.url}`);
              const html = await htmlResponse.text();
              return {
                id: chart.id,
                title: chart.title || chart.type,
                html: html,
                explanation: chart.explanation || `Interactive ${chart.type} visualization`,
              };
            })
          );
          setCharts(fetchedCharts);
          console.log(`📊 Loaded ${fetchedCharts.length} charts from backend`);
        }
      }

      // Fetch reports
      const reportsResponse = await fetch(`http://localhost:8002/api/analysis/${id}/reports`);
      if (reportsResponse.ok) {
        const reportsData = await reportsResponse.json();
        if (reportsData.reports && reportsData.reports.length > 0) {
          const fetchedReports = reportsData.reports.map((report: any, index: number) => ({
            id: `report-${index}`,
            title: report.title || "Analysis Report",
            agent: report.agent || "system",
            agentName: report.agent_name || report.agent || "System",
            timestamp: new Date().toLocaleTimeString(),
            preview: report.preview || report.content?.slice(0, 150) + "...",
            fullText: report.content || "",
          }));
          setReports(fetchedReports);
          console.log(`📄 Loaded ${fetchedReports.length} reports from backend`);
        }
      }
    } catch (error) {
      console.error("Failed to fetch charts/reports:", error);
    }
  };

  const handleStartAnalysis = async () => {
    if (!ticker.trim()) {
      toast.error("Please enter a ticker symbol");
      return;
    }

    // Clear previous data
    setMessages([]);
    setCharts([]);
    setDecision(null);
    setReports([]);
    setProgress(0);

    setIsAnalyzing(true);

    try {
      // Call REAL backend API
      toast.info(`Starting REAL analysis for ${ticker.toUpperCase()}...`);
      
      const response = await fetch("http://localhost:8002/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          timeframe,
          date: selectedDate ? selectedDate.toISOString().split("T")[0] : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const data = await response.json();
      const realAnalysisId = data.analysis_id;
      setAnalysisId(realAnalysisId);

      toast.success(`🚀 Analysis started! Polling for updates...`);

      // Reset message index
      setMessageIndex(0);

      // Connect WebSocket (optional, polling is primary)
      connect(realAnalysisId);
      
      // Start polling for messages every 1 second
      pollingIntervalRef.current = setInterval(() => {
        pollMessages(realAnalysisId);
      }, 1000); // Poll every second for smooth updates

      console.log("✅ Polling started for:", realAnalysisId);

    } catch (error) {
      console.error("Failed to start analysis:", error);
      toast.error(`Failed to start analysis: ${error}`);
      setIsAnalyzing(false);
    }
  };

  const handleStopAnalysis = () => {
    // Stop polling
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    disconnect();
    setIsAnalyzing(false);
    toast.info("Analysis stopped");
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const handleExportPNG = () => {
    toast.info("Exporting as PNG (feature coming soon)");
  };

  const handleExportPDF = () => {
    toast.info("Exporting as PDF (feature coming soon)");
  };

  return (
    <div className="min-h-screen bg-background dark flex flex-col">
      <DashboardHeader
        status={status}
        progress={progress}
        isAnalyzing={isAnalyzing}
        onExportPNG={handleExportPNG}
        onExportPDF={handleExportPDF}
      />

      <div className="flex-1 flex flex-col">
        <ControlPanel
          ticker={ticker}
          setTicker={setTicker}
          timeframe={timeframe}
          setTimeframe={setTimeframe}
          selectedDate={selectedDate}
          setSelectedDate={setSelectedDate}
          isAnalyzing={isAnalyzing}
          onStartAnalysis={handleStartAnalysis}
          onStopAnalysis={handleStopAnalysis}
          analysisId={analysisId}
          messageCount={messages.length}
          chartCount={charts.length}
        />

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
          {/* Left Column: Agent Conversation */}
          <div className="lg:col-span-1">
            <AgentConversation messages={messages} />
          </div>

          {/* Right Columns: Charts and Decision */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <ChartGallery charts={charts} />
            <TradingDecision decision={decision} />
          </div>
        </div>

        {/* Bottom: Reports Panel */}
        <ReportsPanel reports={reports} />
      </div>

      <footer className="border-t border-border p-2 text-center text-xs text-muted-foreground">
        Backend: http://localhost:8002 • WebSocket: ws://localhost:8002 • Status: {status}
      </footer>
    </div>
  );
};

export default Dashboard;
