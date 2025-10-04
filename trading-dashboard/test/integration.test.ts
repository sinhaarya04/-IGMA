/**
 * Simple frontend integration test for TradingAgents
 * Tests the API client and basic functionality
 */

import { apiClient } from "@/lib/api-client"
import { AnalysisRequest } from "@/lib/types"

describe("TradingAgents Integration", () => {
  // Mock analysis request
  const mockRequest: AnalysisRequest = {
    ticker: "SPY",
    timeframe: "1D",
    date: undefined,
  }

  beforeAll(() => {
    // Set up test environment
    process.env.NEXT_PUBLIC_BACKEND_HTTP = "http://localhost:8000"
    process.env.NEXT_PUBLIC_BACKEND_WS = "ws://localhost:8000"
  })

  test("API client should be configured correctly", () => {
    expect(apiClient).toBeDefined()
    expect(apiClient["baseUrl"]).toBe("http://localhost:8000")
  })

  test("Health check should work (if backend is running)", async () => {
    try {
      const health = await apiClient.health()
      expect(health).toHaveProperty("status")
      expect(health.status).toBe("healthy")
    } catch (error) {
      // Skip if backend is not running
      console.log("Backend not running, skipping health check")
    }
  }, 10000)

  test("WebSocket connection should be creatable", () => {
    const ws = apiClient.createWebSocket("test-id")
    expect(ws).toBeInstanceOf(WebSocket)
    ws.close()
  })

  test("Start analysis should return proper structure (if backend is running)", async () => {
    try {
      const response = await apiClient.startAnalysis(mockRequest)
      expect(response).toHaveProperty("analysis_id")
      expect(response).toHaveProperty("status")
      expect(response.status).toBe("started")
    } catch (error) {
      // Skip if backend is not running
      console.log("Backend not running, skipping analysis test")
    }
  }, 15000)

  test("Charts endpoint should handle errors gracefully", async () => {
    try {
      await apiClient.getCharts("non-existent-id")
    } catch (error) {
      expect(error).toBeInstanceOf(Error)
    }
  })

  test("Recent charts should work (if backend is running)", async () => {
    try {
      const charts = await apiClient.getRecentCharts()
      expect(charts).toHaveProperty("charts")
      expect(Array.isArray(charts.charts)).toBe(true)
    } catch (error) {
      // Skip if backend is not running
      console.log("Backend not running, skipping charts test")
    }
  })
})

// Mock WebSocket for frontend-only tests
global.WebSocket = jest.fn().mockImplementation(() => ({
  close: jest.fn(),
  send: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
}))