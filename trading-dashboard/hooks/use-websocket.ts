"use client"

import { useEffect, useRef, useState, useCallback } from "react"

export interface WebSocketMessage {
  type: "connected" | "progress" | "message" | "decision" | "complete" | "error" | "pong"
  data: any
  timestamp: number
}

export interface WebSocketHookReturn {
  isConnected: boolean
  lastMessage: WebSocketMessage | null
  sendMessage: (message: any) => void
  connect: (analysisId?: string) => void
  disconnect: () => void
  connectionError: string | null
}

export function useWebSocket(url?: string): WebSocketHookReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(
    (analysisId?: string) => {
      try {
        const wsUrl =
          url ||
          (analysisId
            ? `${process.env.NEXT_PUBLIC_BACKEND_WS || "ws://localhost:8002"}/ws/analysis/${analysisId}`
            : `${process.env.NEXT_PUBLIC_BACKEND_WS || "ws://localhost:8002"}/ws`)

        console.log("[v0] Connecting to WebSocket:", wsUrl)

        ws.current = new WebSocket(wsUrl)

        ws.current.onopen = () => {
          console.log("[v0] WebSocket connected")
          setIsConnected(true)
          setConnectionError(null)
          reconnectAttempts.current = 0

          // Send initial connected message to trigger frontend state
          setLastMessage({
            type: "connected",
            data: { message: "WebSocket connected successfully" },
            timestamp: Date.now()
          })

          // Send ping every 30 seconds to keep connection alive
          const pingInterval = setInterval(() => {
            if (ws.current?.readyState === WebSocket.OPEN) {
              ws.current.send(JSON.stringify({ type: "ping", data: {} }))
            } else {
              clearInterval(pingInterval)
            }
          }, 30000)
        }

        ws.current.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            console.log("[v0] WebSocket message received:", message.type)
            setLastMessage(message)
          } catch (error) {
            console.error("[v0] Failed to parse WebSocket message:", error)
          }
        }

        ws.current.onclose = (event) => {
          console.log("[v0] WebSocket disconnected:", event.code, event.reason)
          setIsConnected(false)

          // Attempt to reconnect if not a normal closure
          if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000)
            console.log(
              `[v0] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`,
            )

            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttempts.current++
              connect(analysisId)
            }, delay)
          } else if (reconnectAttempts.current >= maxReconnectAttempts) {
            setConnectionError("Failed to reconnect after multiple attempts")
          }
        }

        ws.current.onerror = (error) => {
          console.error("[v0] WebSocket error:", error)
          setConnectionError("WebSocket connection error")
        }
      } catch (error) {
        console.error("[v0] Failed to create WebSocket connection:", error)
        setConnectionError("Failed to create WebSocket connection")
      }
    },
    [url],
  )

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (ws.current) {
      ws.current.close(1000, "Manual disconnect")
      ws.current = null
    }

    setIsConnected(false)
    setLastMessage(null)
    setConnectionError(null)
    reconnectAttempts.current = 0
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
      console.log("[v0] WebSocket message sent:", message.type)
    } else {
      console.warn("[v0] WebSocket not connected, cannot send message")
    }
  }, [])

  // Auto-connect on mount
  useEffect(() => {
    // Try to connect to general WebSocket endpoint on mount
    connect()
  }, [connect])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    connectionError,
  }
}
