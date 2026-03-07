# ============================================
# ECommerce-IA — Dockerfile Multi-stage
# ============================================
# Stage 1 : Build Next.js frontend
# Stage 2 : Python API + Streamlit + modèles IA
#
# Auteur : BAWANA Théodore — Projet SAHELYS
# ============================================

# ============================================
# STAGE 1 : Build frontend Next.js 14
# ============================================
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Installer les dépendances
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --legacy-peer-deps || npm install

# Copier et builder
COPY frontend/ .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ============================================
# STAGE 2 : Backend Python (API + IA + Streamlit)
# ============================================
FROM python:3.10-slim AS production

# Métadonnées
LABEL maintainer="BAWANA Théodore <theodore8bawana@gmail.com>"
LABEL description="ECommerce-IA — Plateforme e-commerce avec 3 modules IA"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY database/ ./database/
COPY data/ ./data/
COPY .env.example ./.env

# Copier le frontend buildé (Next.js standalone output)
COPY --from=frontend-builder /app/frontend/.next/standalone /app/frontend-server
COPY --from=frontend-builder /app/frontend/.next/static /app/frontend-server/.next/static
COPY --from=frontend-builder /app/frontend/public /app/frontend-server/public

# Configuration Nginx — proxy vers Next.js (port 3000) et FastAPI (port 8000)
RUN echo 'server { \
    listen 80; \
    server_name _; \
    location / { \
        proxy_pass http://127.0.0.1:3000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
    location /api { \
        proxy_pass http://127.0.0.1:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
    location /docs { \
        proxy_pass http://127.0.0.1:8000; \
    } \
    location /ws { \
        proxy_pass http://127.0.0.1:8000; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
    } \
}' > /etc/nginx/sites-available/default

# Créer les répertoires nécessaires
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/results /app/logs

# Script de démarrage
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Ports
EXPOSE 80 3000 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée
ENTRYPOINT ["/app/docker-entrypoint.sh"]
