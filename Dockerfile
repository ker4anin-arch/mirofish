# Production image: the backend serves the API and the built frontend on one
# port ($PORT, default 5001), which is what single-port hosts like Render need.

# ---- frontend build ----
FROM node:20-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm ci --prefix frontend
COPY frontend ./frontend
COPY locales ./locales
RUN npm run build --prefix frontend

# ---- backend runtime ----
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=5001

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the default local embedding model into the image so the first start
# does not depend on downloading it.
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY backend ./
COPY locales /app/locales
COPY --from=frontend /app/frontend/dist /app/frontend/dist

EXPOSE 5001
CMD [".venv/bin/python", "run.py"]
