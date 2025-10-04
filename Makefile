.PHONY: install run-backend run-frontend run-dev test clean help

help:
	@echo "TradingAgents2 Development Commands"
	@echo "=================================="
	@echo "install       - Install dependencies for both backend and frontend"
	@echo "run-backend   - Start the backend server"
	@echo "run-frontend  - Start the frontend development server"
	@echo "run-dev       - Start both backend and frontend in parallel"
	@echo "test          - Run integration tests"
	@echo "clean         - Clean up build artifacts and caches"

install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd trading-dashboard && npm install

run-backend:
	@echo "Starting backend server on http://localhost:8000..."
	cd backend && python3 api_server.py

run-frontend:
	@echo "Starting frontend server on http://localhost:3000..."
	cd trading-dashboard && npm run dev

run-dev:
	@echo "Starting both servers..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "Press Ctrl+C to stop both servers"
	@( \
		trap 'kill 0' INT; \
		cd backend && python3 api_server.py & \
		cd trading-dashboard && npm run dev & \
		wait \
	)

test:
	@echo "Running integration tests..."
	python test_integration_improved.py

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	cd trading-dashboard && rm -rf .next out 2>/dev/null || true