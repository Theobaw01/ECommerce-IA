"""
============================================
ECommerce-IA — Pipeline Complet Intégré
============================================
Intègre les 3 modules IA de la plateforme :
1. Classification visuelle (EfficientNet-B4)
2. Système de recommandation hybride (4 facteurs)
3. Chatbot RAG (LangChain + ChromaDB)

Fournit une interface unifiée pour l'API et le dashboard.

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image

# ============================================
# Configuration
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import charger_modele, predire_image
from src.recommendation import HybridRecommender, generer_donnees_demo
from src.chatbot import EcommerceChatbot
from src.dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


class EcommerceAIPipeline:
    """
    Pipeline IA complet pour la plateforme e-commerce.
    
    Intègre les 3 modules :
    - Classification : image → catégorie + confiance
    - Recommandation : utilisateur → top-N produits
    - Chatbot : question → réponse contextuelle
    
    Usage :
        pipeline = EcommerceAIPipeline()
        pipeline.initialiser()
        
        # Classification
        result = pipeline.classifier_image("photo.jpg")
        
        # Recommandation
        recs = pipeline.recommander("user_123")
        
        # Chatbot
        response = pipeline.chat("Quels sont les délais de livraison ?")
    """
    
    def __init__(self):
        """Initialise le pipeline (sans charger les modèles)."""
        self.classifier_model = None
        self.classifier_class_to_idx = None
        self.classifier_device = None
        self.classifier_transform = None
        
        self.recommender = None
        self.chatbot = None
        
        self.is_classifier_ready = False
        self.is_recommender_ready = False
        self.is_chatbot_ready = False
        
        logger.info("🏗️  Pipeline IA initialisé (modèles non chargés)")
    
    def initialiser(
        self,
        charger_classifier: bool = True,
        charger_recommender: bool = True,
        charger_chatbot: bool = True
    ) -> None:
        """
        Charge tous les modules IA.
        
        Args:
            charger_classifier: Charger le modèle de classification
            charger_recommender: Charger le système de recommandation
            charger_chatbot: Charger le chatbot RAG
        """
        logger.info("=" * 60)
        logger.info("🚀 Initialisation du Pipeline IA Complet")
        logger.info("=" * 60)
        
        start = time.time()
        
        if charger_classifier:
            self._init_classifier()
        
        if charger_recommender:
            self._init_recommender()
        
        if charger_chatbot:
            self._init_chatbot()
        
        elapsed = time.time() - start
        
        logger.info("")
        logger.info(f"✅ Pipeline initialisé en {elapsed:.1f}s")
        logger.info(f"   Classification : {'✅' if self.is_classifier_ready else '❌'}")
        logger.info(f"   Recommandation : {'✅' if self.is_recommender_ready else '❌'}")
        logger.info(f"   Chatbot        : {'✅' if self.is_chatbot_ready else '❌'}")
    
    def _init_classifier(self) -> None:
        """Initialise le module de classification visuelle."""
        logger.info("📌 Chargement du classificateur EfficientNet-B4...")
        try:
            self.classifier_model, self.classifier_class_to_idx, self.classifier_device = \
                charger_modele()
            self.classifier_transform = get_transforms("test")
            self.is_classifier_ready = True
            logger.info("   ✅ Classificateur prêt")
        except FileNotFoundError:
            logger.warning("   ⚠️  Modèle non trouvé — entraînez d'abord avec train_classification.py")
        except Exception as e:
            logger.warning(f"   ⚠️  Erreur chargement classificateur : {e}")
    
    def _init_recommender(self) -> None:
        """Initialise le système de recommandation."""
        logger.info("📌 Chargement du système de recommandation...")
        try:
            recommender_path = PROJECT_ROOT / "models" / "recommendation" / "recommender.pkl"
            if recommender_path.exists():
                self.recommender = HybridRecommender.load(recommender_path)
                self.is_recommender_ready = True
                logger.info("   ✅ Recommandeur chargé depuis le fichier")
            else:
                # Initialiser avec des données de démo
                logger.info("   ℹ️  Pas de modèle sauvegardé — initialisation démo")
                products_df, users_df, interactions_df = generer_donnees_demo()
                self.recommender = HybridRecommender()
                self.recommender.fit(products_df, interactions_df, users_df)
                self.is_recommender_ready = True
                logger.info("   ✅ Recommandeur initialisé (mode démo)")
        except Exception as e:
            logger.warning(f"   ⚠️  Erreur recommandation : {e}")
    
    def _init_chatbot(self) -> None:
        """Initialise le chatbot RAG."""
        logger.info("📌 Chargement du chatbot RAG...")
        try:
            self.chatbot = EcommerceChatbot()
            self.chatbot.initialiser()
            self.is_chatbot_ready = True
            logger.info("   ✅ Chatbot prêt")
        except Exception as e:
            logger.warning(f"   ⚠️  Erreur chatbot : {e}")
    
    # ============================================
    # MODULE 1 — Classification
    # ============================================
    def classifier_image(
        self,
        image_input,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Classifie une image de produit.
        
        Args:
            image_input: Chemin (str/Path) ou objet PIL Image
            top_k: Nombre de prédictions top-K
        
        Returns:
            {
                "categorie": str,
                "confiance": float,
                "top_k": [(categorie, confiance), ...],
                "inference_ms": float
            }
        """
        if not self.is_classifier_ready:
            return {"erreur": "Classificateur non disponible"}
        
        # Charger l'image
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            return {"erreur": "Format d'image non supporté"}
        
        # Prétraitement
        img_tensor = self.classifier_transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.classifier_device)
        
        # Inférence
        self.classifier_model.eval()
        start = time.perf_counter()
        
        with torch.no_grad():
            output = self.classifier_model(img_tensor)
        
        inference_ms = (time.perf_counter() - start) * 1000
        
        # Top-K prédictions
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(min(top_k, probs.size(1)), dim=1)
        
        idx_to_class = {v: k for k, v in self.classifier_class_to_idx.items()}
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = idx_to_class.get(idx.item(), f"class_{idx.item()}")
            predictions.append({
                "categorie": class_name,
                "confiance": float(prob)
            })
        
        return {
            "categorie": predictions[0]["categorie"],
            "confiance": predictions[0]["confiance"],
            "top_k": predictions,
            "inference_ms": inference_ms,
            "image_size": f"{img.width}×{img.height}"
        }
    
    def classifier_batch(
        self,
        images: List,
        top_k: int = 5,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Classifie un batch d'images.
        
        Args:
            images: Liste de chemins ou d'images PIL
            top_k: Nombre de prédictions top-K
            batch_size: Taille du batch
        
        Returns:
            Liste de résultats de classification
        """
        results = []
        for img in images:
            results.append(self.classifier_image(img, top_k))
        return results
    
    # ============================================
    # MODULE 2 — Recommandation
    # ============================================
    def recommander(
        self,
        user_id: str,
        n: int = 10,
        produit_consulte: Optional[str] = None,
        user_location: Optional[Tuple[float, float]] = None,
        user_budget: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Génère des recommandations personnalisées.
        
        Args:
            user_id: ID utilisateur
            n: Nombre de recommandations
            produit_consulte: ID du produit actuellement consulté
            user_location: Coordonnées GPS (lat, lon)
            user_budget: Fourchette de budget (min, max)
        
        Returns:
            Liste de recommandations avec scores détaillés
        """
        if not self.is_recommender_ready:
            return [{"erreur": "Système de recommandation non disponible"}]
        
        return self.recommender.recommander(
            user_id=user_id,
            n=n,
            produit_consulte=produit_consulte,
            user_location=user_location,
            user_budget=user_budget
        )
    
    def produits_similaires(
        self,
        product_id: str,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retourne les produits similaires.
        
        Args:
            product_id: ID du produit source
            n: Nombre de résultats
        
        Returns:
            Liste de produits similaires
        """
        if not self.is_recommender_ready:
            return [{"erreur": "Système non disponible"}]
        
        return self.recommender.produits_similaires(product_id, n)
    
    # ============================================
    # MODULE 3 — Chatbot
    # ============================================
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Envoie un message au chatbot.
        
        Args:
            message: Message de l'utilisateur
            session_id: ID de session
        
        Returns:
            Réponse du chatbot avec confiance et sources
        """
        if not self.is_chatbot_ready:
            return {
                "reponse": "Le chatbot n'est pas disponible pour le moment.",
                "confiance": 0.0,
                "erreur": True
            }
        
        return self.chatbot.generer_reponse(message, session_id)
    
    def chat_historique(self, session_id: str) -> List[Dict]:
        """Retourne l'historique d'une session de chat."""
        if not self.is_chatbot_ready:
            return []
        return self.chatbot.get_historique(session_id)
    
    def nouvelle_session_chat(self) -> str:
        """Crée une nouvelle session de chat."""
        if not self.is_chatbot_ready:
            return ""
        return self.chatbot.nouvelle_session()
    
    # ============================================
    # UTILITAIRES
    # ============================================
    def status(self) -> Dict[str, Any]:
        """Retourne le statut de tous les modules."""
        return {
            "classification": {
                "ready": self.is_classifier_ready,
                "model": "EfficientNet-B4",
                "device": str(self.classifier_device) if self.classifier_device else None,
                "num_classes": len(self.classifier_class_to_idx) if self.classifier_class_to_idx else 0
            },
            "recommendation": {
                "ready": self.is_recommender_ready,
                "type": "Hybrid (SVD + Content-Based + Geo + Price)"
            },
            "chatbot": {
                "ready": self.is_chatbot_ready,
                "type": "RAG (LangChain + ChromaDB)"
            }
        }


# ============================================
# Singleton global pour l'API
# ============================================
_pipeline_instance: Optional[EcommerceAIPipeline] = None


def get_pipeline() -> EcommerceAIPipeline:
    """
    Retourne l'instance singleton du pipeline.
    Initialise si nécessaire.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EcommerceAIPipeline()
        _pipeline_instance.initialiser()
    return _pipeline_instance


# ============================================
# Point d'entrée principal
# ============================================
def main():
    """
    Démonstration du pipeline complet.
    """
    logger.info("=" * 60)
    logger.info("🚀 ECommerce-IA — Pipeline Complet")
    logger.info("=" * 60)
    
    pipeline = EcommerceAIPipeline()
    pipeline.initialiser()
    
    # Status
    status = pipeline.status()
    logger.info(f"\n📊 Status : {json.dumps(status, indent=2, default=str)}")
    
    # Test recommandation
    if pipeline.is_recommender_ready:
        logger.info("\n🛍️  Test Recommandation :")
        recs = pipeline.recommander("U0001", n=3, user_budget=(10, 200))
        for r in recs:
            logger.info(f"   → {r['nom']} ({r['categorie']}) — Score: {r['score_final']:.3f}")
    
    # Test chatbot
    if pipeline.is_chatbot_ready:
        logger.info("\n🤖 Test Chatbot :")
        response = pipeline.chat("Comment suivre ma commande ?")
        logger.info(f"   → Confiance: {response['confiance']:.1%}")
        logger.info(f"   → {response['reponse'][:200]}...")
    
    # Test classification (si modèle disponible)
    if pipeline.is_classifier_ready:
        logger.info("\n🖼️  Test Classification :")
        # Créer une image de test
        test_img = Image.new("RGB", (380, 380), (128, 64, 200))
        result = pipeline.classifier_image(test_img)
        logger.info(f"   → Catégorie: {result['categorie']}")
        logger.info(f"   → Confiance: {result['confiance']:.1%}")
        logger.info(f"   → Inférence: {result['inference_ms']:.1f}ms")
    
    logger.info("\n✅ Pipeline complet opérationnel !")


if __name__ == "__main__":
    main()
