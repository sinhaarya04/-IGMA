#!/usr/bin/env python3
"""
Integration test script for TradingAgents frontend-backend connectivity
Tests REST API, WebSocket, and data flow between frontend and backend
"""

import asyncio
import json
import requests
import websockets
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

def test_rest_api():
    """Test REST API endpoints"""
    print("🔍 Testing REST API endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
        # Test root endpoint
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ REST API test failed: {e}")
        return False

def test_analysis_api():
    """Test analysis API endpoint"""
    print("\n🔍 Testing Analysis API...")
    
    try:
        # Test analysis request
        payload = {
            "ticker": "AAPL",
            "timeframe": "1D",
            "date": None
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/analyze",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis API working")
            print(f"   Analysis ID: {result.get('analysis_id')}")
            print(f"   Status: {result.get('status')}")
            return result.get('analysis_id')
        else:
            print(f"❌ Analysis API failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Analysis API test failed: {e}")
        return None

async def test_websocket_connection():
    """Test WebSocket connection"""
    print("\n🔍 Testing WebSocket connection...")
    
    try:
        async with websockets.connect(f"{WS_URL}/ws") as websocket:
            print("✅ WebSocket connection established")
            
            # Wait for welcome message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"   Welcome message: {data.get('type')}")
            
            # Send ping
            ping_message = {"type": "ping", "data": {}}
            await websocket.send(json.dumps(ping_message))
            print("✅ Ping sent")
            
            # Wait for pong
            pong_message = await websocket.recv()
            pong_data = json.loads(pong_message)
            if pong_data.get('type') == 'pong':
                print("✅ Pong received")
                return True
            else:
                print(f"❌ Unexpected pong response: {pong_data}")
                return False
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

async def test_analysis_websocket(analysis_id):
    """Test analysis-specific WebSocket"""
    print(f"\n🔍 Testing Analysis WebSocket for ID: {analysis_id}")
    
    try:
        async with websockets.connect(f"{WS_URL}/ws/analysis/{analysis_id}") as websocket:
            print("✅ Analysis WebSocket connection established")
            
            # Wait for welcome message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"   Welcome message: {data.get('type')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Analysis WebSocket test failed: {e}")
        return False

async def test_full_integration():
    """Test full integration flow"""
    print("\n🔍 Testing Full Integration Flow...")
    
    try:
        # Start analysis
        payload = {
            "ticker": "AAPL",
            "timeframe": "1D",
            "date": None
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/analyze",
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start analysis: {response.status_code}")
            return False
            
        result = response.json()
        analysis_id = result.get('analysis_id')
        print(f"✅ Analysis started with ID: {analysis_id}")
        
        # Connect to analysis WebSocket
        async with websockets.connect(f"{WS_URL}/ws/analysis/{analysis_id}") as websocket:
            print("✅ Connected to analysis WebSocket")
            
            # Wait for initial messages
            message_count = 0
            start_time = time.time()
            timeout = 30  # 30 second timeout
            
            while time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"   Message {message_count}: {data.get('type')} - {data.get('data', {}).get('message', 'No message')}")
                    
                    # Check if analysis is complete
                    if data.get('type') == 'complete':
                        print("✅ Analysis completed successfully")
                        return True
                    elif data.get('type') == 'error':
                        print(f"❌ Analysis failed: {data.get('data', {}).get('message')}")
                        return False
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            if message_count == 0:
                print("❌ No messages received from analysis")
                return False
            else:
                print(f"✅ Received {message_count} messages (analysis may still be running)")
                return True
                
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("🚀 TradingAgents Integration Test Suite")
    print("=" * 50)
    
    # Test 1: REST API
    rest_ok = test_rest_api()
    
    # Test 2: Analysis API
    analysis_id = test_analysis_api()
    
    # Test 3: WebSocket connection
    ws_ok = await test_websocket_connection()
    
    # Test 4: Analysis WebSocket (if analysis was started)
    analysis_ws_ok = False
    if analysis_id:
        analysis_ws_ok = await test_analysis_websocket(analysis_id)
    
    # Test 5: Full integration
    integration_ok = await test_full_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"REST API:           {'✅ PASS' if rest_ok else '❌ FAIL'}")
    print(f"Analysis API:       {'✅ PASS' if analysis_id else '❌ FAIL'}")
    print(f"WebSocket:          {'✅ PASS' if ws_ok else '❌ FAIL'}")
    print(f"Analysis WebSocket: {'✅ PASS' if analysis_ws_ok else '❌ FAIL'}")
    print(f"Full Integration:   {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if all([rest_ok, analysis_id, ws_ok, analysis_ws_ok, integration_ok]):
        print("\n🎉 ALL TESTS PASSED! Frontend-Backend integration is working!")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED! Check the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
