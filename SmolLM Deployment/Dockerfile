# ---- Build React frontend ----
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Build FastAPI backend ----
FROM python:3.10-slim
WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn gunicorn transformers torch

# Copy backend code
COPY backend/ ./backend

# Copy built frontend into backend/static
COPY --from=frontend-builder /app/frontend/build ./frontend_build

# Expose port
EXPOSE 7860

# Run FastAPI + serve React build as static
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "backend.app:app", "--bind", "0.0.0.0:7860"]
