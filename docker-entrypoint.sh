#!/bin/bash
# ============================================
# ECommerce-IA — Script de démarrage Docker
# ============================================

set -e

echo "🚀 Démarrage ECommerce-IA..."

# Initialiser la base de données
echo "📦 Initialisation de la base de données..."
python -c "
import sys
sys.path.insert(0, '/app')
try:
    from database.models import init_database
    init_database()
    print('✅ Base de données initialisée')
except Exception as e:
    print(f'⚠️  Base de données : {e}')
"

# Démarrer Next.js frontend en arrière-plan
echo "🌐 Démarrage du frontend Next.js..."
if [ -d /app/frontend-server ]; then
    cd /app/frontend-server && node server.js &
    cd /app
fi

# Démarrer Nginx (proxy)
echo "🔄 Démarrage Nginx (proxy)..."
nginx

# Démarrer Streamlit en arrière-plan
echo "📊 Démarrage du dashboard Streamlit..."
streamlit run /app/app/streamlit_demo.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false &

# Démarrer l'API FastAPI
echo "🔌 Démarrage de l'API FastAPI..."
echo "📖 Documentation : http://localhost:8000/docs"
echo "📊 Dashboard    : http://localhost:8501"
echo "🌐 Frontend     : http://localhost:3000"

exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
