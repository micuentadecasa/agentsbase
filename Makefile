.PHONY: help dev-frontend dev-backend dev dev-backend-gen gen

help:
	@echo "Available commands:"
	@echo "  make dev-frontend    - Starts the frontend development server (Vite)"
	@echo "  make dev-backend     - Starts the backend development server (Uvicorn with reload)"
	@echo "  make dev             - Starts both frontend and backend development servers"
	@echo "  make dev-backend-gen - Starts the backend_gen development server (Uvicorn with reload)"
	@echo "  make gen             - Starts both frontend and backend_gen development servers"

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev

dev-backend:
	@echo "Starting backend development server..."
	@cd backend && langgraph dev

dev-backend-gen:
	@echo "Starting backend_gen development server..."
	@cd backend_gen && langgraph dev

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@make dev-frontend & make dev-backend 

# Run frontend and backend_gen concurrently
gen:
	@echo "Starting both frontend and backend_gen development servers..."
	@make dev-frontend & make dev-backend-gen 