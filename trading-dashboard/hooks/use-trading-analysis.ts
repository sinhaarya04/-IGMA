"use client"

import { useState, useCallback, useEffect } from "react"
import { useWebSocket } from "./use-websocket"
import { apiClient } from "@/lib/api-client"
import {
  AgentMessage,
  TradingDecision,
  AnalysisProgress,
  Chart,
  AnalysisReport,
  WebSocketMessage,
} from "@/lib/types"

export interface AnalysisState {
  isAnalyzing: boolean
  progress: AnalysisProgress | null
  messages: AgentMessage[]
  decision: TradingDecision | null
  charts: Chart[]
  reports: AnalysisReport[]
  error: string | null
  analysisId: string | null
  finalDecisionReceived: boolean // Track if we got the final decision
}

export interface TradingAnalysisHookReturn extends AnalysisState {
  startAnalysis: (ticker: string, timeframe: string, date?: string) => Promise<void>
  stopAnalysis: () => void
  clearMessages: () => void
  isConnected: boolean
  connectionError: string | null
}

export function useTradingAnalysis(): TradingAnalysisHookReturn {
  const [state, setState] = useState<AnalysisState>({
    isAnalyzing: false,
    progress: null,
    messages: [],
    decision: null,
    charts: [],
    reports: [],
    error: null,
    analysisId: null,
    finalDecisionReceived: false,
  })

  const { isConnected, lastMessage, sendMessage, connect, disconnect, connectionError } = useWebSocket()

  // Fetch charts for an analysis
  const fetchCharts = useCallback(async (analysisId: string) => {
    try {
      console.log("[v0] Fetching charts for analysis:", analysisId)

      const data = await apiClient.getCharts(analysisId)
      const charts = data.charts.map((chart) => ({
        ...chart,
        url: `${apiClient["baseUrl"]}${chart.url}`
      }))

      setState((prev) => ({
        ...prev,
        charts: charts.filter((chart, index, self) =>
          index === self.findIndex(c => c.id === chart.id) // Dedupe by ID
        ),
      }))

      console.log("[v0] Charts loaded:", charts.length)
    } catch (error) {
      console.error("[v0] Error fetching charts:", error)
      // Try to fetch recent charts as fallback
      await fetchRecentCharts()
    }
  }, [])

  // Fetch recent charts as fallback
  const fetchRecentCharts = useCallback(async () => {
    try {
      console.log("[v0] Fetching recent charts as fallback")

      const data = await apiClient.getRecentCharts()
      const charts = data.charts.map((chart) => ({
        ...chart,
        url: `${apiClient["baseUrl"]}${chart.url}`
      }))

      setState((prev) => ({
        ...prev,
        charts: charts.filter((chart, index, self) =>
          index === self.findIndex(c => c.id === chart.id) // Dedupe by ID
        ),
      }))

      console.log("[v0] Recent charts loaded:", charts.length)
    } catch (error) {
      console.error("[v0] Error fetching recent charts:", error)
    }
  }, [])

  // Fetch reports for an analysis
  const fetchReports = useCallback(async (analysisId: string) => {
    try {
      console.log("[v0] Fetching reports for analysis:", analysisId)

      const data = await apiClient.getReports(analysisId)

      setState((prev) => ({
        ...prev,
        reports: data.reports || [],
      }))

      console.log("[v0] Reports loaded:", data.reports?.length || 0)
    } catch (error) {
      console.error("[v0] Error fetching reports:", error)
    }
  }, [])

  // Load recent charts on mount
  useEffect(() => {
    fetchRecentCharts()
  }, [fetchRecentCharts])

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    console.log("[v0] Processing WebSocket message:", lastMessage.type)

    switch (lastMessage.type) {
      case "connected":
        console.log("[v0] WebSocket connected:", lastMessage.data.message)
        break

      case "progress":
        setState((prev) => ({
          ...prev,
          progress: lastMessage.data,
        }))
        break

      case "message":
        const agentMessage: AgentMessage = {
          agent: lastMessage.data.agent || "Unknown Agent",
          content: lastMessage.data.content || String(lastMessage.data),
          timestamp: lastMessage.data.timestamp || Date.now(),
          stage: lastMessage.data.stage,
          sentiment: lastMessage.data.sentiment || "neutral",
        }

        setState((prev) => ({
          ...prev,
          messages: [...prev.messages, agentMessage],
        }))
        break

      case "decision":
        // Only accept the first decision to prevent duplicates
        setState((prev) => {
          if (prev.finalDecisionReceived) {
            console.log("[v0] Ignoring duplicate decision")
            return prev
          }

          const decision: TradingDecision = {
            text: lastMessage.data.text || "HOLD",
            confidence: lastMessage.data.confidence || 0,
            rationale: lastMessage.data.rationale || "Analysis completed",
          }

          return {
            ...prev,
            decision,
            finalDecisionReceived: true,
          }
        })
        break

      case "complete":
        console.log("[v0] Analysis completed")
        const finalData = lastMessage.data
        let currentAnalysisId = null

        setState((prev) => {
          currentAnalysisId = prev.analysisId
          return {
            ...prev,
            isAnalyzing: false,
            progress: {
              current: 100,
              total: 100,
              stage: "Complete",
              message: "Analysis completed successfully",
            },
            // Update decision if provided in completion data and we haven't received one yet
            decision: finalData.decision && !prev.finalDecisionReceived
              ? {
                  text: finalData.decision.text || finalData.decision,
                  confidence: finalData.decision.confidence || 0.85,
                  rationale: finalData.decision.rationale || "Based on comprehensive TradingAgents analysis",
                }
              : prev.decision,
            finalDecisionReceived: prev.finalDecisionReceived || !!finalData.decision,
          }
        })

        // Fetch charts and reports after analysis completion
        if (currentAnalysisId) {
          // Wait a moment for charts to be generated, then fetch them
          setTimeout(() => {
            fetchCharts(currentAnalysisId)
            fetchReports(currentAnalysisId)
          }, 2000)
        }
        break

      case "error":
        console.error("[v0] Analysis error:", lastMessage.data.message)
        setState((prev) => ({
          ...prev,
          isAnalyzing: false,
          error: lastMessage.data.message,
          progress: null,
        }))
        break

      case "pong":
        // Handle ping/pong for connection health
        break

      default:
        console.warn("[v0] Unknown message type:", lastMessage.type)
    }
  }, [lastMessage])

  const startAnalysis = useCallback(
    async (ticker: string, timeframe: string, date?: string) => {
      try {
        console.log("[v0] Starting analysis for:", ticker, timeframe, date)

        // Reset state
        setState((prev) => ({
          ...prev,
          isAnalyzing: true,
          progress: {
            current: 0,
            total: 100,
            stage: "Initializing",
            message: "Starting analysis...",
          },
          messages: [],
          decision: null,
          charts: [],
          reports: [],
          error: null,
          analysisId: null,
          finalDecisionReceived: false,
        }))

        // Start analysis via REST API
        const result = await apiClient.startAnalysis({
          ticker: ticker.toUpperCase(),
          timeframe,
          date,
        })
        console.log("[v0] Analysis started:", result)

        setState((prev) => ({
          ...prev,
          analysisId: result.analysis_id,
        }))

        // Connect to WebSocket for real-time updates
        // Use the analysis-specific WebSocket endpoint
        console.log("[v0] Connecting to analysis-specific WebSocket:", result.analysis_id)
        connect(result.analysis_id)

        // Add a small delay to ensure WebSocket connection is established
        // before the analysis starts sending messages
        setTimeout(() => {
          console.log("[v0] WebSocket connection should be ready for analysis:", result.analysis_id)
        }, 1000)
      } catch (error) {
        console.error("[v0] Failed to start analysis via REST API, trying WebSocket:", error)
        
        // Fallback: Try starting analysis directly via WebSocket
        try {
          // Connect to general WebSocket endpoint first
          connect()
          
          // Wait a moment for connection to establish, then send analysis request
          setTimeout(() => {
            sendMessage({
              type: "start_analysis",
              data: {
                ticker: ticker.toUpperCase(),
                timeframe,
                date,
              },
            })
          }, 1000)
          
        } catch (wsError) {
          console.error("[v0] Failed to start analysis via WebSocket:", wsError)
          setState((prev) => ({
            ...prev,
            isAnalyzing: false,
            error: error instanceof Error ? error.message : "Failed to start analysis",
            progress: null,
          }))
        }
      }
    },
    [connect, sendMessage],
  )

  const stopAnalysis = useCallback(() => {
    console.log("[v0] Stopping analysis")
    setState((prev) => ({
      ...prev,
      isAnalyzing: false,
      progress: null,
    }))
    disconnect()
  }, [disconnect])

  const clearMessages = useCallback(() => {
    setState((prev) => ({
      ...prev,
      messages: [],
      decision: null,
      charts: [],
      reports: [],
      error: null,
      progress: null,
      finalDecisionReceived: false,
    }))
  }, [])

  // Alternative method: Start analysis directly via WebSocket
  const startAnalysisViaWebSocket = useCallback(
    (ticker: string, timeframe: string, date?: string) => {
      console.log("[v0] Starting analysis via WebSocket")

      setState((prev) => ({
        ...prev,
        isAnalyzing: true,
        messages: [],
        decision: null,
        error: null,
        progress: {
          current: 0,
          total: 100,
          stage: "Initializing",
          message: "Starting analysis...",
        },
      }))

      // Connect to general WebSocket endpoint
      connect()

      // Send analysis request via WebSocket
      setTimeout(() => {
        sendMessage({
          type: "start_analysis",
          data: {
            ticker: ticker.toUpperCase(),
            timeframe,
            date,
          },
        })
      }, 1000) // Wait for connection to establish
    },
    [connect, sendMessage],
  )

  return {
    ...state,
    startAnalysis,
    stopAnalysis,
    clearMessages,
    isConnected,
    connectionError,
  }
}
