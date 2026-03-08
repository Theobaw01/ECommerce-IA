"""
============================================
ECommerce-IA — Tests Unitaires : Chatbot RAG
============================================
Tests pour le chatbot RAG (LangChain + ChromaDB) :
- Base de connaissances
- Gestion de session
- Historique de conversation
- Pipeline NLP intégré

Auteur : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chatbot import (
    creer_base_connaissances,
    CHATBOT_CONFIG,
)


# ============================================
# Tests : Configuration
# ============================================
class TestChatbotConfig:
    """Tests de la configuration du chatbot."""

    def test_embedding_model_defined(self):
        assert "embedding_model" in CHATBOT_CONFIG
        assert "MiniLM" in CHATBOT_CONFIG["embedding_model"]

    def test_generation_model_defined(self):
        assert "generation_model" in CHATBOT_CONFIG
        assert "Mistral" in CHATBOT_CONFIG["generation_model"]

    def test_fallback_model_defined(self):
        assert "fallback_model" in CHATBOT_CONFIG
        assert "flan-t5" in CHATBOT_CONFIG["fallback_model"]

    def test_chunk_size_reasonable(self):
        assert 100 <= CHATBOT_CONFIG["chunk_size"] <= 2000

    def test_top_k_retrieval(self):
        assert 1 <= CHATBOT_CONFIG["top_k_retrieval"] <= 10

    def test_confidence_threshold(self):
        assert 0 < CHATBOT_CONFIG["confidence_threshold"] < 1

    def test_temperature(self):
        assert 0 <= CHATBOT_CONFIG["temperature"] <= 1


# ============================================
# Tests : Base de connaissances
# ============================================
class TestKnowledgeBase:
    """Tests de la base de connaissances FAQ."""

    def test_knowledge_base_not_empty(self):
        docs = creer_base_connaissances()
        assert len(docs) > 0

    def test_knowledge_base_minimum_docs(self):
        docs = creer_base_connaissances()
        assert len(docs) >= 10  # Au moins 10 documents FAQ

    def test_document_structure(self):
        docs = creer_base_connaissances()
        for doc in docs:
            assert "titre" in doc
            assert "contenu" in doc
            assert "categorie" in doc

    def test_document_not_empty(self):
        docs = creer_base_connaissances()
        for doc in docs:
            assert len(doc["titre"]) > 0
            assert len(doc["contenu"]) > 0
            assert len(doc["categorie"]) > 0

    def test_categories_diverse(self):
        docs = creer_base_connaissances()
        categories = {doc["categorie"] for doc in docs}
        assert len(categories) >= 4  # Au moins 4 catégories différentes

    def test_covers_payment(self):
        docs = creer_base_connaissances()
        payment_docs = [
            d for d in docs
            if "paiement" in d["contenu"].lower() or "paiement" in d["categorie"].lower()
        ]
        assert len(payment_docs) > 0

    def test_covers_delivery(self):
        docs = creer_base_connaissances()
        delivery_docs = [
            d for d in docs
            if "livraison" in d["contenu"].lower() or "livraison" in d["categorie"].lower()
        ]
        assert len(delivery_docs) > 0

    def test_covers_returns(self):
        docs = creer_base_connaissances()
        return_docs = [
            d for d in docs
            if "retour" in d["contenu"].lower() or "retour" in d["categorie"].lower()
        ]
        assert len(return_docs) > 0


# ============================================
# Tests : Intégration NLP dans le chatbot
# ============================================
class TestChatbotNLPIntegration:
    """Tests de l'intégration NLP → Chatbot."""

    def test_nlp_engine_importable(self):
        from src.nlp_engine import get_nlp_engine
        engine = get_nlp_engine()
        assert engine is not None

    def test_chatbot_uses_nlp(self):
        """Vérifie que le chatbot importe le NLP engine."""
        import src.chatbot as chatbot_module
        assert hasattr(chatbot_module, 'get_nlp_engine') or \
               'nlp_engine' in dir(chatbot_module) or \
               'NLPEngine' in str(chatbot_module.__dict__)

    def test_nlp_analysis_on_greeting(self):
        from src.nlp_engine import get_nlp_engine
        nlp = get_nlp_engine()
        result = nlp.analyser("Bonjour !")
        assert result["intent"]["intent"] == "salutation"

    def test_nlp_routing_for_chatbot(self):
        from src.nlp_engine import get_nlp_engine
        nlp = get_nlp_engine()
        route = nlp.router_requete("Quels sont vos délais de livraison ?")
        assert route["requires_rag"] is True
