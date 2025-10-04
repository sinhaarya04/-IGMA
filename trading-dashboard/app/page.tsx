"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Activity,
  TrendingUp,
  Users,
  MessageSquare,
  BarChart3,
  Play,
  Square,
  Download,
  FileText,
  Wifi,
  WifiOff,
  AlertCircle,
  Trash2,
} from "lucide-react"

import { useTradingAnalysis } from "@/hooks/use-trading-analysis"

interface StatusDotProps {
  connected: boolean
}

function StatusDot({ connected }: StatusDotProps) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-500" : "bg-gray-400"} animate-pulse`} />
      <span className="text-sm font-medium">{connected ? "Connected" : "Disconnected"}</span>
      {connected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
    </div>
  )
}

interface StatChipProps {
  label: string
  value: string | number
  icon?: React.ReactNode
}

function StatChip({ label, value, icon }: StatChipProps) {
  return (
    <Badge variant="secondary" className="px-4 py-2 gap-2 bg-gradient-to-r from-secondary to-secondary/80 hover:from-secondary/80 hover:to-secondary/60 transition-all duration-200 shadow-sm hover:shadow-md border border-secondary-foreground/10">
      {icon}
      <span className="text-sm font-medium">{label}:</span>
      <span className="text-sm font-bold">{value}</span>
    </Badge>
  )
}

interface AgentBubbleProps {
  agent: string
  content: string
  timestamp: number
  stage?: string
  sentiment?: "bullish" | "bearish" | "neutral"
}

const TRADING_AGENTS = ["market", "news", "fundamentals", "payment", "visualization"]

function AgentBubble({ agent, content, timestamp, stage, sentiment }: AgentBubbleProps) {
  const getAgentColor = (agentName: string) => {
    const colors = {
      market: "bg-blue-500",
      news: "bg-purple-500",
      fundamentals: "bg-green-500",
      payment: "bg-orange-500",
      visualization: "bg-pink-500",
      default: "bg-primary",
    }
    const key = agentName.toLowerCase()
    return colors[key as keyof typeof colors] || colors.default
  }

  const initials = agent
    .split(" ")
    .map((word) => word[0])
    .join("")
    .toUpperCase()
  const timeStr = new Date(timestamp).toLocaleTimeString()

  const sentimentColors = {
    bullish: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
    bearish: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    neutral: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
  }

  return (
    <div className="flex gap-3 p-3 hover:bg-muted/50 rounded-lg transition-colors">
      <div
        className={`w-8 h-8 rounded-full ${getAgentColor(agent)} text-white flex items-center justify-center text-xs font-bold`}
      >
        {initials}
      </div>
      <div className="flex-1 space-y-1">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-medium text-foreground capitalize">{agent}</span>
          <span>{timeStr}</span>
          {stage && (
            <Badge variant="outline" className="text-xs">
              {stage}
            </Badge>
          )}
          {sentiment && <Badge className={`text-xs ${sentimentColors[sentiment]}`}>{sentiment}</Badge>}
        </div>
        <p className="text-sm leading-relaxed">{content}</p>
      </div>
    </div>
  )
}

interface ChartFrameProps {
  title: string
  url?: string
  html?: string
  loading?: boolean
  explanation?: string
  type?: string
}

function ChartFrame({ title, url, html, loading, explanation, type }: ChartFrameProps) {
  if (loading) {
    return (
      <Card className="h-[450px] shadow-lg border-2 border-dashed border-primary/20">
        <CardHeader className="pb-3 bg-gradient-to-r from-primary/5 to-primary/10">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <Skeleton className="w-full h-80 rounded-lg" />
          <div className="mt-3 space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-3/4" />
          </div>
        </CardContent>
      </Card>
    )
  }

  const getChartExplanation = (chartType: string, title: string) => {
    const explanations = {
      scatter: "Shows the relationship between returns and trading volume, helping identify volume-price correlations and potential breakouts.",
      line: "Displays price movements and volume-weighted average price (VWAP) to identify trend direction and support/resistance levels.",
      multi: "Multi-line chart showing various technical indicators and price action to identify momentum and trend changes.",
      bbands: "Bollinger Bands chart showing price volatility and potential overbought/oversold conditions based on standard deviations.",
      chart_1: "Technical analysis chart showing price patterns and key support/resistance levels for trend identification.",
      chart_2: "Volume analysis chart displaying trading volume patterns and their correlation with price movements.",
      chart_3: "Momentum indicators chart showing technical signals for potential entry and exit points.",
      chart_4: "Correlation analysis chart displaying relationships between different market factors and price movements."
    }
    
    return explanations[chartType as keyof typeof explanations] || 
           `Financial analysis chart showing ${title.toLowerCase()} patterns and trends for trading decision support.`
  }

  return (
    <Card className="h-[450px] hover:shadow-xl transition-all duration-300 border-0 shadow-lg bg-gradient-to-br from-card to-card/95 group">
      <CardHeader className="pb-3 bg-gradient-to-r from-primary/5 to-primary/10 rounded-t-lg">
        <CardTitle className="text-base font-semibold flex items-center gap-3">
          <div className="w-3 h-3 bg-gradient-to-br from-primary to-primary/70 rounded-full" />
          <BarChart3 className="w-5 h-5 text-primary" />
          {title}
          <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex flex-col h-full">
        <div className="flex-1 min-h-0 relative p-4">
          {url ? (
            <iframe
              src={url}
              className="w-full h-full border-0 rounded-lg shadow-inner absolute inset-4"
              title={title}
              style={{
                minHeight: '300px',
                height: 'calc(100% - 2rem)',
                width: 'calc(100% - 2rem)'
              }}
              sandbox="allow-scripts allow-same-origin"
            />
          ) : (
            <div className="w-full h-full min-h-[300px] bg-gradient-to-br from-muted/30 to-muted/60 rounded-lg flex items-center justify-center text-muted-foreground border-2 border-dashed border-muted-foreground/30">
              <div className="text-center space-y-3">
                <BarChart3 className="w-12 h-12 mx-auto text-muted-foreground/60" />
                <div>
                  <p className="text-base font-medium">Chart Preview</p>
                  <p className="text-sm">Waiting for analysis data</p>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="mt-auto p-4 pt-2 bg-gradient-to-r from-muted/30 to-muted/50 rounded-b-lg border-t border-muted/50">
          <div className="text-xs text-muted-foreground leading-relaxed">
            <strong className="text-foreground/80">Purpose:</strong> {explanation || getChartExplanation(type || 'chart', title)}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

interface DecisionBadgeProps {
  decision: string
  confidence?: number
  rationale?: string
}

function DecisionBadge({ decision, confidence, rationale }: DecisionBadgeProps) {
  const getDecisionColor = (decisionText: string) => {
    const text = decisionText.toUpperCase()
    if (text.includes("BUY") || text.includes("LONG")) return "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-lg shadow-emerald-500/25"
    if (text.includes("SELL") || text.includes("SHORT")) return "bg-gradient-to-r from-red-500 to-red-600 text-white shadow-lg shadow-red-500/25"
    return "bg-gradient-to-r from-yellow-500 to-yellow-600 text-white shadow-lg shadow-yellow-500/25"
  }

  return (
    <Card className="border-0 shadow-xl bg-gradient-to-br from-card to-card/95 overflow-hidden">
      <CardHeader className="pb-3 bg-gradient-to-r from-primary/10 to-primary/5">
        <CardTitle className="text-lg flex items-center gap-3">
          <div className="w-3 h-3 bg-gradient-to-br from-primary to-primary/70 rounded-full" />
          <TrendingUp className="w-6 h-6 text-primary" />
          Trading Decision
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 p-6">
        <div className="flex items-center justify-between">
          <Badge className={`text-xl font-bold px-6 py-3 rounded-xl border-0 ${getDecisionColor(decision)}`}>
            {decision.toUpperCase()}
          </Badge>
          {confidence && (
            <div className="text-right space-y-1">
              <div className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                {Math.round(confidence * 100)}%
              </div>
              <div className="text-sm text-muted-foreground font-medium">Confidence Level</div>
            </div>
          )}
        </div>
        {rationale && (
          <div className="bg-muted/30 rounded-lg p-4 border-l-4 border-primary/50">
            <p className="text-sm leading-relaxed text-muted-foreground">
              <span className="font-semibold text-foreground">Rationale:</span> {rationale}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function SkeletonCard() {
  return (
    <Card className="h-[450px] shadow-lg border-2 border-dashed border-primary/20">
      <CardHeader className="pb-3 bg-gradient-to-r from-primary/5 to-primary/10">
        <Skeleton className="h-5 w-48" />
      </CardHeader>
      <CardContent className="p-4">
        <Skeleton className="w-full h-80 rounded-lg" />
        <div className="mt-3 space-y-2">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-3/4" />
        </div>
      </CardContent>
    </Card>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
      <Activity className="w-12 h-12 mb-4" />
      <p className="text-center">{message}</p>
    </div>
  )
}

export default function TradingAgentsDashboard() {
  const [ticker, setTicker] = useState("SPY")
  const [timeframe, setTimeframe] = useState("1D")
  const [date, setDate] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const {
    isAnalyzing,
    progress,
    messages,
    decision,
    charts,
    reports,
    error,
    analysisId,
    startAnalysis,
    stopAnalysis,
    clearMessages,
    isConnected,
    connectionError,
  } = useTradingAnalysis()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleStartAnalysis = async () => {
    try {
      await startAnalysis(ticker, timeframe, date || undefined)
    } catch (error) {
      console.error("Failed to start analysis:", error)
    }
  }

  const handleStopAnalysis = () => {
    stopAnalysis()
  }

  const handleClearMessages = () => {
    clearMessages()
  }

  const handleExportPNG = () => {
    // Export current dashboard state as PNG
    if (analysisId) {
      const exportData = {
        ticker,
        timeframe,
        analysisId,
        timestamp: new Date().toISOString(),
        decision: decision?.text,
        confidence: decision?.confidence,
        charts: displayCharts.map(chart => chart.filename),
        messages: messages.length
      }
      
      // Create downloadable JSON file with analysis data
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `trading-analysis-${ticker}-${analysisId.slice(0, 8)}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  const handleExportPDF = () => {
    // Export analysis report as PDF
    if (analysisId) {
      const reportData = {
        ticker,
        timeframe,
        analysisId,
        timestamp: new Date().toISOString(),
        decision: decision,
        charts: displayCharts,
        agentReports: realReports,
        messages: messages
      }
      
      // Create downloadable JSON file with full analysis report
      const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `trading-report-${ticker}-${analysisId.slice(0, 8)}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  // Use real charts from backend instead of mock charts
  const displayCharts = charts.length > 0 ? charts : []

  // Use real reports from backend instead of mock data
  const realReports = reports.length > 0 ? reports : messages
    .filter(msg => msg.content && msg.content.length > 50) // Filter substantial messages
    .map(msg => ({
      agent: msg.agent,
      title: `${msg.agent} Analysis`,
      content: msg.content,
      timestamp: new Date(msg.timestamp).toISOString(),
    }))
    .slice(-6) // Show last 6 substantial reports

  const activeAgents = TRADING_AGENTS.length

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gradient-to-r from-background via-background/95 to-background backdrop-blur-xl border-b border-border/50 px-6 py-4 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary/70 rounded-xl flex items-center justify-center shadow-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                TradingAgents Dashboard
              </h1>
            </div>
            <StatusDot connected={isConnected} />
            {isAnalyzing && progress && (
              <div className="flex items-center gap-2">
                <Progress value={progress.current} className="w-32" />
                <span className="text-sm text-muted-foreground">{progress.current}%</span>
                <span className="text-xs text-muted-foreground">{progress.stage}</span>
              </div>
            )}
            {analysisId && (
              <Badge variant="outline" className="text-xs">
                ID: {analysisId.slice(0, 8)}...
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleExportPNG} disabled={!analysisId}>
              <Download className="w-4 h-4 mr-2" />
              Export Data
            </Button>
            <Button variant="outline" size="sm" onClick={handleExportPDF} disabled={!analysisId}>
              <FileText className="w-4 h-4 mr-2" />
              Export Report
            </Button>
          </div>
        </div>
      </header>

      {(error || connectionError) && (
        <div className="px-4 pt-4">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error || connectionError}</AlertDescription>
          </Alert>
        </div>
      )}

      {/* Controls */}
      <div className="px-6 py-6 bg-gradient-to-r from-muted/20 via-muted/30 to-muted/20 border-b border-border/50 backdrop-blur-sm">
        <div className="flex flex-wrap items-center gap-6 mb-6">
          <div className="flex items-center gap-2">
            <Input
              placeholder="Ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-24 uppercase"
            />
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1D">1D</SelectItem>
                <SelectItem value="5D">5D</SelectItem>
                <SelectItem value="1M">1M</SelectItem>
                <SelectItem value="3M">3M</SelectItem>
                <SelectItem value="6M">6M</SelectItem>
                <SelectItem value="1Y">1Y</SelectItem>
                <SelectItem value="YTD">YTD</SelectItem>
                <SelectItem value="MAX">MAX</SelectItem>
              </SelectContent>
            </Select>
            <Input
              type="date"
              placeholder="Date (optional)"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="w-40"
            />
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={handleStartAnalysis} disabled={isAnalyzing || !isConnected} className="gap-2">
              <Play className="w-4 h-4" />
              Start Analysis
            </Button>
            <Button variant="secondary" onClick={handleStopAnalysis} disabled={!isAnalyzing} className="gap-2">
              <Square className="w-4 h-4" />
              Stop
            </Button>
            <Button
              variant="outline"
              onClick={handleClearMessages}
              disabled={isAnalyzing}
              className="gap-2 bg-transparent"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="flex flex-wrap gap-3">
          <StatChip label="Ticker" value={ticker} />
          <StatChip label="Timeframe" value={timeframe} />
          <StatChip label="Active Agents" value={activeAgents} icon={<Users className="w-3 h-3" />} />
          <StatChip label="Messages" value={messages.length} icon={<MessageSquare className="w-3 h-3" />} />
          <StatChip label="Charts" value={displayCharts.length} icon={<BarChart3 className="w-3 h-3" />} />
          <StatChip
            label="Status"
            value={isConnected ? "Connected" : "Disconnected"}
            icon={isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6 grid grid-cols-1 xl:grid-cols-4 gap-8">
        {/* Agent Conversation */}
        <div className="xl:col-span-1">
          <Card className="h-[700px] flex flex-col shadow-xl bg-gradient-to-br from-card to-card/95">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageSquare className="w-5 h-5" />
                Agent Conversation
                {progress && (
                  <Badge variant="secondary" className="text-xs">
                    {progress.message}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              <div className="h-full overflow-y-auto space-y-2">
                {messages.length === 0 && !isAnalyzing ? (
                  <EmptyState message="Start an analysis to see agent conversations" />
                ) : isAnalyzing && messages.length === 0 ? (
                  <div className="space-y-4">
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <AgentBubble
                      key={index}
                      agent={message.agent}
                      content={message.content}
                      timestamp={message.timestamp}
                      stage={message.stage}
                      sentiment={message.sentiment}
                    />
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts and Decision */}
        <div className="xl:col-span-3 space-y-8">
          {/* Chart Gallery */}
          <div>
            <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary/70 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              Chart Gallery
              {displayCharts.length > 0 && (
                <Badge variant="outline" className="ml-auto text-sm">
                  {displayCharts.length} charts
                </Badge>
              )}
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-3 gap-6">
              {displayCharts.length === 0 && !isAnalyzing
                ? Array.from({ length: 6 }).map((_, i) => (
                    <Card key={i} className="h-[450px] shadow-lg bg-gradient-to-br from-muted/20 to-muted/40 border-2 border-dashed border-primary/20">
                      <CardContent className="h-full flex items-center justify-center">
                        <EmptyState message="Charts will appear here after analysis" />
                      </CardContent>
                    </Card>
                  ))
                : isAnalyzing && displayCharts.length === 0
                  ? Array.from({ length: 6 }).map((_, i) => (
                      <Card key={i} className="h-[450px] shadow-lg border-2 border-dashed border-primary/20">
                        <CardHeader className="pb-3 bg-gradient-to-r from-primary/5 to-primary/10">
                          <Skeleton className="h-5 w-48" />
                        </CardHeader>
                        <CardContent className="p-4">
                          <Skeleton className="w-full h-80 rounded-lg" />
                          <div className="mt-3 space-y-2">
                            <Skeleton className="h-3 w-full" />
                            <Skeleton className="h-3 w-3/4" />
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  : displayCharts.map((chart) => (
                      <ChartFrame 
                        key={chart.id} 
                        title={chart.title} 
                        url={chart.url} 
                        type={chart.type}
                        explanation={`Generated by TradingAgents visualization system for ${ticker} analysis`}
                      />
                    ))}
            </div>
          </div>

          {/* Decision Card */}
          <div>
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Trading Decision
            </h2>
            {decision ? (
              <DecisionBadge decision={decision.text} confidence={decision.confidence} rationale={decision.rationale} />
            ) : (
              <Card className="h-32">
                <CardContent className="h-full">
                  <EmptyState message="Decision will appear after analysis" />
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* Reports Panel */}
      <div className="px-6 pb-8">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary/70 rounded-lg flex items-center justify-center">
            <FileText className="w-5 h-5 text-white" />
          </div>
          Recent Reports
          {realReports.length > 0 && (
            <Badge variant="outline" className="ml-auto text-sm">
              {realReports.length} reports
            </Badge>
          )}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {realReports.length === 0 ? (
            <div className="col-span-full">
              <Card className="h-32">
                <CardContent className="h-full flex items-center justify-center">
                  <EmptyState message="Agent reports will appear here after analysis" />
                </CardContent>
              </Card>
            </div>
          ) : (
            realReports.map((report, index) => (
              <Card key={index} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">{report.title}</CardTitle>
                  <div className="text-xs text-muted-foreground">
                    {report.agent} • {new Date(report.timestamp).toLocaleTimeString()}
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{report.content}</p>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="px-4 py-3 border-t border-border bg-muted/30">
        <p className="text-xs text-muted-foreground text-center">
          TradingAgents Backend: {process.env.NEXT_PUBLIC_BACKEND_HTTP || "http://localhost:8002"} • Active Agents:{" "}
          {TRADING_AGENTS.join(", ")} • WebSocket: {process.env.NEXT_PUBLIC_BACKEND_WS || "ws://localhost:8002"}
        </p>
      </footer>
    </div>
  )
}
