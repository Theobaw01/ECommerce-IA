"""
============================================
ECommerce-IA — Modèles de Base de Données
============================================
Définition des tables PostgreSQL avec SQLAlchemy ORM.

Tables :
- products       : Catalogue de produits
- users          : Utilisateurs de la plateforme
- orders         : Commandes passées
- interactions   : Interactions utilisateur-produit
- chat_sessions  : Sessions de chatbot
- recommendations: Recommandations générées

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, JSON, ForeignKey, Index, Enum, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool

# ============================================
# Configuration de la base de données
# ============================================
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://ecommerce_user:ecommerce_pass@localhost:5432/ecommerce_ia"
)

# Pour les tests sans PostgreSQL, utiliser SQLite
SQLITE_URL = "sqlite:///ecommerce_ia.db"

Base = declarative_base()


# ============================================
# TABLE : products
# ============================================
class Product(Base):
    """
    Table des produits du catalogue e-commerce.
    
    Stocke les informations produit ainsi que les embeddings
    visuels pour la recherche par image.
    """
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    categorie = Column(String(100), nullable=False, index=True)
    sous_categorie = Column(String(100), nullable=True)
    prix = Column(Float, nullable=False)
    prix_promo = Column(Float, nullable=True)
    stock = Column(Integer, default=0)
    marque = Column(String(100), nullable=True, index=True)
    image_path = Column(String(500), nullable=True)
    image_url = Column(String(500), nullable=True)
    
    # Embeddings visuels (stockés en JSON pour simplicité)
    embedding_visuel = Column(JSON, nullable=True)
    
    # Localisation du vendeur
    vendeur_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Métriques
    note_moyenne = Column(Float, default=0.0)
    nb_avis = Column(Integer, default=0)
    nb_vues = Column(Integer, default=0)
    nb_achats = Column(Integer, default=0)
    
    # Promotion
    en_promotion = Column(Boolean, default=False)
    discount_percent = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    actif = Column(Boolean, default=True)
    
    # Relations
    vendeur = relationship("User", back_populates="produits_vendus", foreign_keys=[vendeur_id])
    order_items = relationship("Order", back_populates="product")
    interactions = relationship("Interaction", back_populates="product")
    
    # Index composites
    __table_args__ = (
        Index("idx_product_categorie_prix", "categorie", "prix"),
        Index("idx_product_actif", "actif"),
    )
    
    def to_dict(self) -> dict:
        """Convertit le produit en dictionnaire."""
        return {
            "id": self.id,
            "nom": self.nom,
            "description": self.description,
            "categorie": self.categorie,
            "sous_categorie": self.sous_categorie,
            "prix": self.prix,
            "prix_promo": self.prix_promo,
            "stock": self.stock,
            "marque": self.marque,
            "image_url": self.image_url,
            "note_moyenne": self.note_moyenne,
            "nb_avis": self.nb_avis,
            "en_promotion": self.en_promotion,
            "discount_percent": self.discount_percent,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
    
    def __repr__(self):
        return f"<Product(id={self.id}, nom='{self.nom}', prix={self.prix})>"


# ============================================
# TABLE : users
# ============================================
class User(Base):
    """
    Table des utilisateurs de la plateforme.
    Stocke les informations de profil et préférences.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profil
    telephone = Column(String(20), nullable=True)
    adresse = Column(Text, nullable=True)
    ville = Column(String(100), nullable=True)
    code_postal = Column(String(10), nullable=True)
    pays = Column(String(50), default="France")
    
    # Localisation
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Préférences
    budget_moyen = Column(Float, nullable=True)
    categories_favorites = Column(JSON, nullable=True)  # Liste de catégories
    historique_categories = Column(JSON, nullable=True)
    
    # Rôle
    role = Column(String(20), default="client")  # client, vendeur, admin
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relations
    orders = relationship("Order", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")
    produits_vendus = relationship("Product", back_populates="vendeur")
    
    def to_dict(self) -> dict:
        """Convertit l'utilisateur en dictionnaire (sans mot de passe)."""
        return {
            "id": self.id,
            "nom": self.nom,
            "prenom": self.prenom,
            "email": self.email,
            "ville": self.ville,
            "role": self.role,
            "budget_moyen": self.budget_moyen,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


# ============================================
# TABLE : orders
# ============================================
class Order(Base):
    """
    Table des commandes passées sur la plateforme.
    """
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    quantite = Column(Integer, default=1)
    prix_unitaire = Column(Float, nullable=False)
    prix_total = Column(Float, nullable=False)
    
    statut = Column(
        String(30),
        default="en_preparation",
        index=True
    )  # en_preparation, expedie, en_transit, livre, retourne, annule
    
    # Livraison
    mode_livraison = Column(String(50), nullable=True)
    frais_livraison = Column(Float, default=0.0)
    adresse_livraison = Column(Text, nullable=True)
    numero_suivi = Column(String(100), nullable=True)
    
    # Paiement
    mode_paiement = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    
    # Relations
    user = relationship("User", back_populates="orders")
    product = relationship("Product", back_populates="order_items")
    
    __table_args__ = (
        Index("idx_order_user_date", "user_id", "created_at"),
    )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "quantite": self.quantite,
            "prix_total": self.prix_total,
            "statut": self.statut,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<Order(id={self.id}, user={self.user_id}, statut='{self.statut}')>"


# ============================================
# TABLE : interactions
# ============================================
class Interaction(Base):
    """
    Table des interactions utilisateur-produit.
    Utilisée pour le système de recommandation.
    """
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    
    # Type d'interaction
    type_interaction = Column(
        String(20),
        nullable=False
    )  # vue, like, ajout_panier, achat, avis
    
    # Score/rating (pour les avis)
    rating = Column(Float, nullable=True)
    
    # Session
    session_id = Column(String(100), nullable=True)
    
    # Durée de consultation (secondes)
    duree_consultation = Column(Integer, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relations
    user = relationship("User", back_populates="interactions")
    product = relationship("Product", back_populates="interactions")
    
    __table_args__ = (
        Index("idx_interaction_user_product", "user_id", "product_id"),
        Index("idx_interaction_type", "type_interaction"),
    )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "type": self.type_interaction,
            "rating": self.rating,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ============================================
# TABLE : chat_sessions
# ============================================
class ChatSession(Base):
    """
    Table des sessions de chatbot.
    Stocke les conversations et le score de satisfaction.
    """
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_uuid = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Messages stockés en JSON
    messages = Column(JSON, nullable=True)  # [{role, content, timestamp}, ...]
    
    # Métriques
    nb_messages = Column(Integer, default=0)
    satisfaction_score = Column(Float, nullable=True)  # 1-5
    escalade_humain = Column(Boolean, default=False)
    
    # Contexte
    sujet_principal = Column(String(200), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relations
    user = relationship("User", back_populates="chat_sessions")
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_uuid": self.session_uuid,
            "user_id": self.user_id,
            "nb_messages": self.nb_messages,
            "satisfaction_score": self.satisfaction_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================
# TABLE : recommendations
# ============================================
class Recommendation(Base):
    """
    Table des recommandations générées.
    Stocke les recommandations et les scores par facteur.
    """
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Produits recommandés (IDs en JSON)
    product_ids = Column(JSON, nullable=False)
    
    # Scores détaillés (JSON)
    scores = Column(JSON, nullable=True)
    
    # Facteurs utilisés (poids)
    facteurs = Column(JSON, nullable=True)
    
    # Contexte
    produit_source = Column(Integer, nullable=True)  # Produit consulté
    type_recommandation = Column(String(50), default="personnalise")
    
    # Feedback
    clicked = Column(Boolean, default=False)
    purchased = Column(Boolean, default=False)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relations
    user = relationship("User", back_populates="recommendations")
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_ids": self.product_ids,
            "scores": self.scores,
            "type": self.type_recommandation,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ============================================
# GESTION DE LA BASE DE DONNÉES
# ============================================
def get_engine(database_url: Optional[str] = None, echo: bool = False):
    """
    Crée le moteur SQLAlchemy.
    
    Args:
        database_url: URL de connexion (défaut: variable d'environnement)
        echo: Afficher les requêtes SQL
    
    Returns:
        Engine SQLAlchemy
    """
    url = database_url or DATABASE_URL
    
    try:
        engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=echo
        )
        return engine
    except Exception:
        # Fallback SQLite si PostgreSQL non disponible
        print(f"⚠️  PostgreSQL non disponible, utilisation de SQLite")
        return create_engine(SQLITE_URL, echo=echo)


def get_session(engine=None):
    """Crée une session de base de données."""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database(engine=None) -> None:
    """
    Initialise la base de données (crée toutes les tables).
    
    Args:
        engine: Engine SQLAlchemy (optionnel)
    """
    if engine is None:
        engine = get_engine()
    
    Base.metadata.create_all(engine)
    print("✅ Base de données initialisée — toutes les tables créées")


def peupler_donnees_demo(session) -> None:
    """
    Peuple la base de données avec des données de démonstration.
    
    Args:
        session: Session SQLAlchemy
    """
    import random
    import hashlib
    
    random.seed(42)
    
    # Vérifier si les données existent déjà
    if session.query(User).count() > 0:
        print("ℹ️  Données de démo déjà présentes")
        return
    
    # Catégories
    categories = [
        "Électronique", "Vêtements", "Maison", "Sport",
        "Alimentation", "Beauté", "Jouets", "Livres"
    ]
    
    marques = [
        "Samsung", "Nike", "IKEA", "Adidas", "Apple",
        "Sony", "Zara", "Decathlon"
    ]
    
    # Créer des utilisateurs
    users = []
    for i in range(20):
        user = User(
            nom=f"Utilisateur_{i}",
            prenom=f"Prénom_{i}",
            email=f"user{i}@example.com",
            password_hash=hashlib.sha256(f"password{i}".encode()).hexdigest(),
            ville=random.choice(["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux"]),
            latitude=round(random.uniform(43.0, 49.0), 4),
            longitude=round(random.uniform(-1.0, 7.0), 4),
            budget_moyen=round(random.uniform(20, 300), 2),
            role=random.choice(["client", "client", "client", "vendeur"]),
        )
        users.append(user)
        session.add(user)
    
    session.flush()
    
    # Créer des produits
    products = []
    for i in range(100):
        cat = random.choice(categories)
        prix = round(random.uniform(5, 500), 2)
        en_promo = random.random() < 0.2
        
        product = Product(
            nom=f"Produit {cat} #{i}",
            description=f"Description du produit {i} dans la catégorie {cat}.",
            categorie=cat,
            prix=prix,
            prix_promo=round(prix * 0.8, 2) if en_promo else None,
            stock=random.randint(0, 100),
            marque=random.choice(marques),
            note_moyenne=round(random.uniform(1, 5), 1),
            nb_avis=random.randint(0, 200),
            en_promotion=en_promo,
            discount_percent=20 if en_promo else 0,
            latitude=round(random.uniform(43.0, 49.0), 4),
            longitude=round(random.uniform(-1.0, 7.0), 4),
            vendeur_id=random.choice(users).id if users else None,
        )
        products.append(product)
        session.add(product)
    
    session.flush()
    
    # Créer des interactions
    for _ in range(500):
        interaction = Interaction(
            user_id=random.choice(users).id,
            product_id=random.choice(products).id,
            type_interaction=random.choice(["vue", "vue", "like", "ajout_panier", "achat"]),
            rating=round(random.uniform(1, 5), 1) if random.random() > 0.5 else None,
        )
        session.add(interaction)
    
    # Créer des commandes
    for _ in range(50):
        product = random.choice(products)
        qty = random.randint(1, 3)
        order = Order(
            user_id=random.choice(users).id,
            product_id=product.id,
            quantite=qty,
            prix_unitaire=product.prix,
            prix_total=product.prix * qty,
            statut=random.choice([
                "en_preparation", "expedie", "en_transit", "livre"
            ]),
            mode_livraison=random.choice(["standard", "express", "point_relais"]),
            frais_livraison=random.choice([0, 3.99, 4.99, 9.99]),
        )
        session.add(order)
    
    session.commit()
    
    print(f"✅ Données de démo insérées :")
    print(f"   {len(users)} utilisateurs")
    print(f"   {len(products)} produits")
    print(f"   500 interactions")
    print(f"   50 commandes")


# ============================================
# Point d'entrée
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🗄️  ECommerce-IA — Initialisation Base de Données")
    print("=" * 60)
    
    engine = get_engine()
    init_database(engine)
    
    session = get_session(engine)
    peupler_donnees_demo(session)
    session.close()
    
    print("\n✅ Base de données prête !")
