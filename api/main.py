"""
============================================
ECommerce-IA — API FastAPI Complète
============================================
API REST + WebSocket pour la plateforme e-commerce.

Endpoints :
- Classification : POST /classify, POST /classify/batch
- Recommandation : GET /recommend/{user_id}, GET /recommend/similar/{product_id}
- Chatbot        : POST /chat, GET /chat/history/{session_id}
- Produits       : GET /products, GET /products/{id}, POST /products/search
- Auth           : POST /auth/register, POST /auth/token
- WebSocket      : WS /ws/chat/{session_id}

Documentation Swagger : /docs

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import io
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File,
    WebSocket, WebSocketDisconnect, Query, Body, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr

# JWT
from jose import JWTError, jwt
from passlib.context import CryptContext

# Image
from PIL import Image
import numpy as np

# ============================================
# Configuration
# ============================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Variables d'environnement
JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "***REMOVED***")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION = int(os.environ.get("JWT_EXPIRATION_MINUTES", "60"))

# ============================================
# Application FastAPI
# ============================================
app = FastAPI(
    title="ECommerce-IA API",
    description=(
        "API e-commerce avec 3 modules IA intégrés : "
        "Classification visuelle (EfficientNet-B4, 94% accuracy), "
        "Recommandation hybride (4 facteurs), "
        "Chatbot RAG (LangChain + ChromaDB). "
        "Projet réalisé chez SAHELYS par BAWANA Théodore."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "BAWANA Théodore",
        "email": "theodore8bawana@gmail.com",
        "url": "https://theo.portefolio.io"
    }
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, limiter aux domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Sécurité JWT
# ============================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# Stockage en mémoire (remplacer par DB en production)
users_db: Dict[str, Dict] = {}
sessions_db: Dict[str, Dict] = {}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict]:
    if token is None:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        if email and email in users_db:
            return users_db[email]
    except JWTError:
        pass
    return None


# ============================================
# Modèles Pydantic
# ============================================

# Auth
class UserRegister(BaseModel):
    nom: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=6)
    ville: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


# Classification
class ClassificationResult(BaseModel):
    categorie: str
    confiance: float
    top_k: List[Dict[str, Any]]
    inference_ms: float


# Recommandation
class RecommendationRequest(BaseModel):
    user_location: Optional[List[float]] = None  # [lat, lon]
    user_budget: Optional[List[float]] = None     # [min, max]
    produit_consulte: Optional[str] = None
    n: int = Field(default=10, ge=1, le=50)


class FeedbackRequest(BaseModel):
    user_id: str
    product_id: str
    type: str  # "click", "purchase", "like"
    recommendation_id: Optional[int] = None


# Chat
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reponse: str
    confiance: float
    sources: List[str]
    escalade_humain: bool


# Produits
class ProductSearch(BaseModel):
    query: Optional[str] = None
    categorie: Optional[str] = None
    prix_min: Optional[float] = None
    prix_max: Optional[float] = None
    marque: Optional[str] = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)


# ============================================
# Pipeline IA (chargement paresseux)
# ============================================
_pipeline = None


def get_pipeline():
    """Retourne le pipeline IA (chargé une seule fois)."""
    global _pipeline
    if _pipeline is None:
        try:
            from src.pipeline import EcommerceAIPipeline
            _pipeline = EcommerceAIPipeline()
            _pipeline.initialiser(
                charger_classifier=True,
                charger_recommender=True,
                charger_chatbot=True
            )
            logger.info("✅ Pipeline IA chargé")
        except Exception as e:
            logger.warning(f"⚠️  Pipeline IA non disponible : {e}")
            # Créer un pipeline minimal
            from src.pipeline import EcommerceAIPipeline
            _pipeline = EcommerceAIPipeline()
            try:
                _pipeline.initialiser(
                    charger_classifier=False,
                    charger_recommender=True,
                    charger_chatbot=True
                )
            except Exception:
                pass
    return _pipeline


# ============================================
# ENDPOINTS — Santé & Status
# ============================================
@app.get("/", tags=["Status"])
async def root():
    """Point d'entrée de l'API."""
    return {
        "message": "🚀 ECommerce-IA API",
        "version": "1.0.0",
        "docs": "/docs",
        "auteur": "BAWANA Théodore — SAHELYS"
    }


@app.get("/health", tags=["Status"])
async def health_check():
    """Vérification de santé de l'API."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "modules": pipeline.status() if pipeline else {}
    }


# ============================================
# ENDPOINTS — Authentification
# ============================================
@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(user: UserRegister):
    """
    Inscription d'un nouvel utilisateur.
    Retourne un token JWT.
    """
    if user.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cet email est déjà utilisé"
        )
    
    user_data = {
        "id": str(uuid.uuid4()),
        "nom": user.nom,
        "email": user.email,
        "password_hash": hash_password(user.password),
        "ville": user.ville,
        "latitude": user.latitude,
        "longitude": user.longitude,
        "role": "client",
        "created_at": datetime.utcnow().isoformat()
    }
    
    users_db[user.email] = user_data
    
    token = create_token({"sub": user.email})
    
    return TokenResponse(
        access_token=token,
        user={k: v for k, v in user_data.items() if k != "password_hash"}
    )


@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Connexion et obtention d'un token JWT.
    """
    user = users_db.get(form_data.username)
    
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = create_token({"sub": form_data.username})
    
    return TokenResponse(
        access_token=token,
        user={k: v for k, v in user.items() if k != "password_hash"}
    )


# ============================================
# ENDPOINTS — Classification
# ============================================
@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
async def classify_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    Classifie une image de produit.
    
    Upload une image → retourne la catégorie prédite avec le score de confiance.
    Modèle : EfficientNet-B4 (94% accuracy).
    """
    pipeline = get_pipeline()
    
    if not pipeline or not pipeline.is_classifier_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle de classification n'est pas disponible"
        )
    
    # Lire l'image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image invalide : {str(e)}"
        )
    
    # Classification
    result = pipeline.classifier_image(image, top_k=top_k)
    
    if "erreur" in result:
        raise HTTPException(status_code=500, detail=result["erreur"])
    
    return ClassificationResult(**result)


@app.post("/classify/batch", tags=["Classification"])
async def classify_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    Classifie plusieurs images en batch.
    """
    pipeline = get_pipeline()
    
    if not pipeline or not pipeline.is_classifier_ready:
        raise HTTPException(
            status_code=503,
            detail="Classificateur non disponible"
        )
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = pipeline.classifier_image(image, top_k=top_k)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "erreur": str(e)
            })
    
    return {"results": results, "total": len(results)}


# ============================================
# ENDPOINTS — Recommandation
# ============================================
@app.get("/recommend/{user_id}", tags=["Recommandation"])
async def get_recommendations(
    user_id: str,
    n: int = Query(default=10, ge=1, le=50),
    produit_consulte: Optional[str] = None
):
    """
    Retourne les top-N recommandations personnalisées pour un utilisateur.
    
    Algorithme hybride à 4 facteurs :
    - Historique (40%) — Collaborative Filtering
    - Similarité (30%) — Content-Based
    - Géographie (15%) — Distance Haversine
    - Prix (15%) — Budget utilisateur
    """
    pipeline = get_pipeline()
    
    if not pipeline or not pipeline.is_recommender_ready:
        raise HTTPException(status_code=503, detail="Recommandeur non disponible")
    
    recs = pipeline.recommander(
        user_id=user_id,
        n=n,
        produit_consulte=produit_consulte
    )
    
    return {
        "user_id": user_id,
        "recommendations": recs,
        "count": len(recs)
    }


@app.get("/recommend/similar/{product_id}", tags=["Recommandation"])
async def get_similar_products(
    product_id: str,
    n: int = Query(default=10, ge=1, le=50)
):
    """
    Retourne les produits similaires (content-based).
    """
    pipeline = get_pipeline()
    
    if not pipeline or not pipeline.is_recommender_ready:
        raise HTTPException(status_code=503, detail="Recommandeur non disponible")
    
    similar = pipeline.produits_similaires(product_id, n)
    
    return {
        "product_id": product_id,
        "similar_products": similar,
        "count": len(similar)
    }


@app.post("/recommend/feedback", tags=["Recommandation"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Soumet un feedback utilisateur sur une recommandation.
    """
    logger.info(
        f"📊 Feedback : user={feedback.user_id}, "
        f"product={feedback.product_id}, type={feedback.type}"
    )
    
    return {
        "status": "recorded",
        "feedback": feedback.dict()
    }


# ============================================
# ENDPOINTS — Chatbot
# ============================================
@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(message: ChatMessage):
    """
    Envoie un message au chatbot IA.
    
    Architecture RAG : recherche dans la base de connaissances
    + génération de réponse contextuelle.
    """
    pipeline = get_pipeline()
    
    if not pipeline or not pipeline.is_chatbot_ready:
        raise HTTPException(status_code=503, detail="Chatbot non disponible")
    
    response = pipeline.chat(
        message=message.message,
        session_id=message.session_id
    )
    
    return ChatResponse(
        session_id=response.get("session_id", ""),
        reponse=response.get("reponse", ""),
        confiance=response.get("confiance", 0.0),
        sources=response.get("sources", []),
        escalade_humain=response.get("escalade_humain", False)
    )


@app.get("/chat/history/{session_id}", tags=["Chatbot"])
async def get_chat_history(session_id: str):
    """
    Retourne l'historique d'une session de chat.
    """
    pipeline = get_pipeline()
    
    if not pipeline or not pipeline.is_chatbot_ready:
        raise HTTPException(status_code=503, detail="Chatbot non disponible")
    
    history = pipeline.chat_historique(session_id)
    
    return {
        "session_id": session_id,
        "messages": history,
        "count": len(history)
    }


# ============================================
# ENDPOINTS — Produits
# ============================================
# Catalogue en mémoire (remplacer par DB)
_products_catalog: List[Dict] = []


def _init_catalog():
    """Initialise le catalogue depuis le recommandeur."""
    global _products_catalog
    if not _products_catalog:
        pipeline = get_pipeline()
        if pipeline and pipeline.is_recommender_ready and pipeline.recommender.products_df is not None:
            _products_catalog = pipeline.recommender.products_df.to_dict("records")


@app.get("/products", tags=["Produits"])
async def list_products(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    categorie: Optional[str] = None,
    prix_min: Optional[float] = None,
    prix_max: Optional[float] = None
):
    """
    Retourne le catalogue de produits avec pagination et filtres.
    """
    _init_catalog()
    
    filtered = _products_catalog.copy()
    
    if categorie:
        filtered = [p for p in filtered if p.get("categorie") == categorie]
    if prix_min is not None:
        filtered = [p for p in filtered if p.get("prix", 0) >= prix_min]
    if prix_max is not None:
        filtered = [p for p in filtered if p.get("prix", 0) <= prix_max]
    
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "products": filtered[start:end],
        "total": len(filtered),
        "page": page,
        "limit": limit,
        "pages": (len(filtered) + limit - 1) // limit
    }


@app.get("/products/{product_id}", tags=["Produits"])
async def get_product(product_id: str):
    """
    Retourne le détail d'un produit.
    """
    _init_catalog()
    
    for product in _products_catalog:
        if str(product.get("id")) == product_id:
            return {"product": product}
    
    raise HTTPException(status_code=404, detail="Produit non trouvé")


@app.post("/products/search", tags=["Produits"])
async def search_products_by_image(
    file: UploadFile = File(...),
    n: int = Query(default=10, ge=1, le=50)
):
    """
    Recherche de produits par image (classification visuelle).
    Upload une image → classification → produits correspondants.
    """
    pipeline = get_pipeline()
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline non disponible")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide")
    
    result = {}
    
    if pipeline.is_classifier_ready:
        result["classification"] = pipeline.classifier_image(image, top_k=5)
    
    _init_catalog()
    
    # Filtrer le catalogue par catégorie prédite
    if result.get("classification"):
        categorie = result["classification"]["categorie"]
        matching = [
            p for p in _products_catalog
            if p.get("categorie", "").lower() == categorie.lower()
        ]
        result["products"] = matching[:n]
        result["total_matching"] = len(matching)
    
    return result


# ============================================
# WEBSOCKET — Chat temps réel
# ============================================
class ConnectionManager:
    """Gère les connexions WebSocket."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"🔌 WebSocket connecté : {session_id}")
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        logger.info(f"🔌 WebSocket déconnecté : {session_id}")
    
    async def send_json(self, session_id: str, data: dict):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_json(data)


ws_manager = ConnectionManager()


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    Chat en temps réel via WebSocket.
    
    Protocole :
    - Client envoie : {"message": "texte"}
    - Serveur répond : {"reponse": "...", "confiance": 0.95, ...}
    """
    await ws_manager.connect(websocket, session_id)
    
    pipeline = get_pipeline()
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                await websocket.send_json({"erreur": "Message vide"})
                continue
            
            if pipeline and pipeline.is_chatbot_ready:
                response = pipeline.chat(message, session_id)
            else:
                response = {
                    "reponse": "Le chatbot n'est pas disponible.",
                    "confiance": 0.0,
                    "sources": [],
                    "escalade_humain": True,
                    "session_id": session_id
                }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"❌ Erreur WebSocket : {e}")
        ws_manager.disconnect(session_id)


# ============================================
# ÉVÉNEMENT DE DÉMARRAGE
# ============================================
@app.on_event("startup")
async def startup():
    """Initialise les ressources au démarrage de l'API."""
    logger.info("🚀 Démarrage de l'API ECommerce-IA...")
    # Le pipeline sera chargé au premier appel (lazy loading)


@app.on_event("shutdown")
async def shutdown():
    """Nettoie les ressources à l'arrêt."""
    logger.info("🛑 Arrêt de l'API ECommerce-IA")


# ============================================
# Point d'entrée
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("API_PORT", 8000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    
    logger.info(f"🌐 API démarrée sur http://{host}:{port}")
    logger.info(f"📖 Documentation : http://{host}:{port}/docs")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        workers=1
    )
