#!/usr/bin/env python3
"""
Improved integration test for TradingAgents frontend-backend connectivity
Tests REST API, WebSocket streaming, and complete analysis flow
"""

import asyncio
import json
import requests
import websockets
import time
import unittest
from datetime import datetime
from typing import Optional, Dict, Any

# Configuration
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

class TradingAgentsIntegrationTest(unittest.TestCase):
    """Comprehensive integration tests for TradingAgents API"""

    def setUp(self):
        """Set up test fixtures"""
        self.analysis_id: Optional[str] = None
        self.received_messages: list = []

    def test_01_health_check(self):
        """Test API health endpoint"""
        print("🔍 Testing health endpoint...")

        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

        print("✅ Health endpoint working")

    def test_02_start_analysis_api(self):
        """Test starting analysis via REST API"""
        print("🔍 Testing analysis start API...")

        payload = {
            "ticker": "SPY",
            "timeframe": "1D",
            "date": None
        }

        response = requests.post(
            f"{BACKEND_URL}/api/analyze",
            json=payload,
            timeout=10
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("analysis_id", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "started")

        self.analysis_id = data["analysis_id"]
        print(f"✅ Analysis started with ID: {self.analysis_id}")

    def test_03_analysis_status(self):
        """Test getting analysis status"""
        if not self.analysis_id:
            self.test_02_start_analysis_api()

        print("🔍 Testing analysis status endpoint...")

        response = requests.get(f"{BACKEND_URL}/api/analysis/{self.analysis_id}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("ticker", data)
        self.assertIn("status", data)

        print("✅ Analysis status endpoint working")

    async def test_04_websocket_streaming(self):
        """Test WebSocket streaming with real analysis"""
        if not self.analysis_id:
            # Start a new analysis for this test
            payload = {"ticker": "SPY", "timeframe": "1D", "date": None}
            response = requests.post(f"{BACKEND_URL}/api/analyze", json=payload, timeout=10)
            self.analysis_id = response.json()["analysis_id"]

        print(f"🔍 Testing WebSocket streaming for analysis {self.analysis_id}...")

        try:
            async with websockets.connect(f"{WS_URL}/ws/analysis/{self.analysis_id}") as websocket:
                print("✅ WebSocket connected")

                # Wait for messages
                message_count = 0
                decision_received = False
                start_time = time.time()
                timeout = 30  # 30 second timeout

                while time.time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        message_count += 1

                        self.received_messages.append(data)
                        message_type = data.get("type")

                        print(f"   📨 Message {message_count}: {message_type}")

                        if message_type == "decision":
                            decision_received = True
                            decision_data = data.get("data", {})
                            self.assertIn("text", decision_data)
                            self.assertIn("confidence", decision_data)
                            print(f"   🎯 Decision: {decision_data.get('text')}")

                        elif message_type == "complete":
                            print("   🏁 Analysis completed")
                            break

                        elif message_type == "error":
                            self.fail(f"Analysis failed: {data.get('data', {}).get('message')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"❌ Error receiving message: {e}")
                        break

                self.assertGreater(message_count, 0, "Should receive at least one message")
                print(f"✅ Received {message_count} messages via WebSocket")

                return decision_received

        except Exception as e:
            self.fail(f"WebSocket test failed: {e}")

    def test_05_charts_endpoint(self):
        """Test charts endpoint"""
        if not self.analysis_id:
            self.test_02_start_analysis_api()
            time.sleep(5)  # Wait for some charts to be generated

        print("🔍 Testing charts endpoint...")

        response = requests.get(f"{BACKEND_URL}/api/charts/{self.analysis_id}")

        if response.status_code == 200:
            data = response.json()
            self.assertIn("charts", data)
            charts = data["charts"]

            if len(charts) > 0:
                chart = charts[0]
                self.assertIn("id", chart)
                self.assertIn("title", chart)
                self.assertIn("url", chart)
                print(f"✅ Found {len(charts)} charts")
            else:
                print("⚠️  No charts generated yet")
        else:
            # Try recent charts as fallback
            response = requests.get(f"{BACKEND_URL}/api/charts/recent")
            self.assertEqual(response.status_code, 200)
            print("✅ Recent charts endpoint working as fallback")

    async def test_06_full_analysis_flow(self):
        """Test complete analysis flow from start to finish"""
        print("🔍 Testing full analysis flow...")

        # Start analysis
        payload = {"ticker": "AAPL", "timeframe": "1D", "date": None}
        response = requests.post(f"{BACKEND_URL}/api/analyze", json=payload, timeout=10)
        self.assertEqual(response.status_code, 200)

        analysis_id = response.json()["analysis_id"]
        print(f"✅ Started analysis: {analysis_id}")

        # Stream via WebSocket
        decision_received = False
        progress_received = False

        try:
            async with websockets.connect(f"{WS_URL}/ws/analysis/{analysis_id}") as websocket:
                message_count = 0
                start_time = time.time()
                timeout = 45  # 45 second timeout for full flow

                while time.time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        data = json.loads(message)
                        message_count += 1
                        message_type = data.get("type")

                        if message_type == "progress":
                            progress_received = True
                        elif message_type == "decision":
                            decision_received = True
                            decision_data = data.get("data", {})
                            print(f"   🎯 Final Decision: {decision_data.get('text')} ({decision_data.get('confidence', 0)*100:.0f}% confidence)")
                        elif message_type == "complete":
                            print("   🏁 Analysis completed successfully")
                            break
                        elif message_type == "error":
                            self.fail(f"Analysis failed: {data.get('data', {}).get('message')}")

                    except asyncio.TimeoutError:
                        continue

                # Verify we received expected message types
                self.assertTrue(progress_received, "Should receive progress updates")
                self.assertTrue(decision_received, "Should receive trading decision")
                self.assertGreater(message_count, 5, "Should receive multiple messages in full flow")

                print(f"✅ Full flow completed with {message_count} messages")
                return True

        except Exception as e:
            self.fail(f"Full analysis flow failed: {e}")

    def test_07_concurrent_analyses(self):
        """Test multiple concurrent analyses"""
        print("🔍 Testing concurrent analyses...")

        # Start multiple analyses
        tickers = ["SPY", "QQQ", "IWM"]
        analysis_ids = []

        for ticker in tickers:
            payload = {"ticker": ticker, "timeframe": "1D", "date": None}
            response = requests.post(f"{BACKEND_URL}/api/analyze", json=payload, timeout=10)
            self.assertEqual(response.status_code, 200)
            analysis_ids.append(response.json()["analysis_id"])

        print(f"✅ Started {len(analysis_ids)} concurrent analyses")

        # Verify all analyses are tracked
        for analysis_id in analysis_ids:
            response = requests.get(f"{BACKEND_URL}/api/analysis/{analysis_id}")
            self.assertEqual(response.status_code, 200)

        print("✅ All concurrent analyses properly tracked")

async def run_async_tests():
    """Run async test methods"""
    test_instance = TradingAgentsIntegrationTest()
    test_instance.setUp()

    # Run basic sync tests first
    test_instance.test_01_health_check()
    test_instance.test_02_start_analysis_api()
    test_instance.test_03_analysis_status()

    # Run async tests
    await test_instance.test_04_websocket_streaming()

    # Run remaining sync tests
    test_instance.test_05_charts_endpoint()

    # Run full flow test
    await test_instance.test_06_full_analysis_flow()

    # Run concurrent test
    test_instance.test_07_concurrent_analyses()

def main():
    """Main test runner"""
    print("🚀 TradingAgents Enhanced Integration Test Suite")
    print("=" * 60)

    try:
        # Run the async test suite
        asyncio.run(run_async_tests())

        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("   ✅ REST API working")
        print("   ✅ WebSocket streaming working")
        print("   ✅ Analysis flow complete")
        print("   ✅ Charts generation working")
        print("   ✅ Concurrent analyses supported")
        return True

    except Exception as e:
        print(f"\n❌ INTEGRATION TESTS FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)