import { useEffect, useRef, useState, useCallback } from "react";

interface UseWebSocketOptions {
  onMessage?: (data: any) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

export const useWebSocket = ({
  onMessage,
  onConnect,
  onDisconnect,
  onError,
}: UseWebSocketOptions = {}) => {
  const [status, setStatus] = useState<"connected" | "connecting" | "disconnected">(
    "disconnected"
  );
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback((analysisId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log("WebSocket already connected");
      return;
    }

    setStatus("connecting");

    // In a real implementation, this would connect to your backend WebSocket
    // For now, we'll create a mock WebSocket connection
    const wsUrl = `ws://localhost:8002/ws/analysis/${analysisId}`;
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("🔌✅ WebSocket connected successfully!");
        setStatus("connected");
        onConnect?.();
      };

      ws.onmessage = (event) => {
        console.log("📨 WebSocket message received (raw):", event.data);
        try {
          const data = JSON.parse(event.data);
          console.log("📨 Parsed WebSocket data:", data);
          onMessage?.(data);
        } catch (error) {
          console.error("❌ Failed to parse WebSocket message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        onError?.(error);
        setStatus("disconnected");
      };

      ws.onclose = (event) => {
        console.log("🔌❌ WebSocket disconnected. Code:", event.code, "Reason:", event.reason);
        setStatus("disconnected");
        onDisconnect?.();

        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          if (wsRef.current?.readyState !== WebSocket.OPEN) {
            console.log("Attempting to reconnect...");
            connect(analysisId);
          }
        }, 3000);
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      setStatus("disconnected");
    }
  }, [onMessage, onConnect, onDisconnect, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus("disconnected");
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn("WebSocket is not connected. Message not sent:", data);
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    status,
    connect,
    disconnect,
    sendMessage,
  };
};
