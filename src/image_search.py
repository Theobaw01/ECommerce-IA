"""
============================================
ECommerce-IA — Recherche visuelle par FAISS
============================================
Module de recherche d'images similaires basé sur FAISS (Facebook AI).
Utilise les embeddings extraits par EfficientNet-B4 pour construire
un index de recherche rapide par similarité cosine.

Fonctionnalités :
- Construction d'un index FAISS à partir des embeddings produits
- Recherche des N produits les plus similaires à une image query
- Extraction d'embedding à la volée pour une nouvelle image
- Persistance de l'index sur disque

Architecture :
    Image query → EfficientNet-B4 (backbone) → embedding 1792-d
    → FAISS IndexFlatIP (cosine similarity) → Top-N produits similaires

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

try:
    import faiss
except ImportError:
    faiss = None
    print("⚠️  FAISS non installé. Installez avec : pip install faiss-cpu")

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
MODELS_DIR = PROJECT_ROOT / "models" / "classification"

sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset import get_transforms, IMAGE_SIZE


class FAISSImageSearch:
    """
    Moteur de recherche d'images par similarité visuelle.
    
    Utilise FAISS (Facebook AI Similarity Search) pour effectuer
    des recherches efficaces dans un espace d'embeddings de haute dimension.
    
    Workflow :
        1. Charger les embeddings produits (pré-extraits par le notebook Colab)
        2. Construire un index FAISS (cosine similarity via Inner Product)
        3. Pour une image query : extraire l'embedding → rechercher les plus proches
    
    Usage :
        searcher = FAISSImageSearch()
        searcher.charger_index()
        
        # Recherche par image
        results = searcher.rechercher_par_image("query.jpg", top_k=10)
        
        # Recherche par embedding
        results = searcher.rechercher(embedding_vector, top_k=10)
    """
    
    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialise le moteur de recherche FAISS.
        
        Args:
            embeddings_path: Chemin vers le fichier product_embeddings.pkl
            model_path: Chemin vers le modèle EfficientNet-B4 (.pth)
            device: Device PyTorch (auto-détecté si None)
        """
        if faiss is None:
            raise ImportError(
                "FAISS est requis pour la recherche d'images.\n"
                "Installation : pip install faiss-cpu"
            )
        
        self.embeddings_path = Path(embeddings_path) if embeddings_path else \
            MODELS_DIR / "product_embeddings.pkl"
        self.model_path = Path(model_path) if model_path else \
            MODELS_DIR / "efficientnet_b4_best.pth"
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # État interne
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[List[str]] = None
        self.paths: Optional[List[str]] = None
        self.embedding_dim: int = 0
        self.feature_extractor: Optional[nn.Module] = None
        self.transform = None
        
        self.is_ready = False
        
        logger.info("🔍 FAISSImageSearch initialisé")
        logger.info(f"   Device           : {self.device}")
        logger.info(f"   Embeddings       : {self.embeddings_path}")
    
    def charger_index(self) -> None:
        """
        Charge les embeddings et construit l'index FAISS.
        
        L'index utilise la similarité cosine (via normalisation L2 + Inner Product).
        """
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings non trouvés : {self.embeddings_path}\n"
                "Exécutez d'abord le notebook d'entraînement sur Colab "
                "pour extraire les embeddings."
            )
        
        logger.info("📦 Chargement des embeddings produits...")
        
        with open(self.embeddings_path, "rb") as f:
            data = pickle.load(f)
        
        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"]
        self.paths = data["paths"]
        self.embedding_dim = data["embedding_dim"]
        
        n_vectors = self.embeddings.shape[0]
        logger.info(f"   {n_vectors} embeddings chargés (dim={self.embedding_dim})")
        
        # Normaliser L2 pour la similarité cosine
        faiss.normalize_L2(self.embeddings)
        
        # Construire l'index FAISS
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product = cosine après normalisation
        self.index.add(self.embeddings)
        
        logger.info(f"   ✅ Index FAISS construit ({self.index.ntotal} vecteurs)")
        
        # Charger le feature extractor pour les nouvelles images
        self._charger_feature_extractor()
        
        self.is_ready = True
        logger.info("✅ Recherche d'images prête")
    
    def _charger_feature_extractor(self) -> None:
        """Charge le backbone EfficientNet-B4 comme extracteur de features."""
        logger.info("📌 Chargement du feature extractor...")
        
        # Créer le modèle sans classifieur (num_classes=0 → pooling global)
        self.feature_extractor = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0  # Retourne les features avant le classifieur
        )
        
        # Charger les poids du backbone depuis le modèle entraîné
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"]
            
            # Filtrer les poids du classifieur
            backbone_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("classifier")
            }
            self.feature_extractor.load_state_dict(backbone_dict, strict=False)
            logger.info("   ✅ Poids du backbone chargés depuis le modèle entraîné")
        else:
            logger.warning("   ⚠️  Modèle non trouvé — utilisation des poids ImageNet")
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Transformation pour les images query
        self.transform = get_transforms("test")
    
    @torch.no_grad()
    def extraire_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extrait l'embedding d'une image.
        
        Args:
            image: Image PIL en mode RGB
        
        Returns:
            Vecteur embedding normalisé (1, embedding_dim)
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor non chargé. Appelez charger_index() d'abord.")
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.feature_extractor(img_tensor)
        embedding = embedding.cpu().numpy().astype(np.float32)
        
        # Normaliser L2 pour la similarité cosine
        faiss.normalize_L2(embedding)
        
        return embedding
    
    def rechercher(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recherche les produits les plus similaires à un embedding.
        
        Args:
            query_embedding: Vecteur embedding (1, dim) normalisé L2
            top_k: Nombre de résultats
        
        Returns:
            Liste de résultats avec score, label et path
        """
        if not self.is_ready:
            raise RuntimeError("Index non chargé. Appelez charger_index() d'abord.")
        
        top_k = min(top_k, self.index.ntotal)
        
        # Recherche FAISS
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "rang": len(results) + 1,
                "score_similarite": float(score),
                "categorie": self.labels[idx],
                "image": self.paths[idx],
                "index": int(idx)
            })
        
        return results
    
    def rechercher_par_image(
        self,
        image_input,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Recherche les produits visuellement similaires à une image.
        
        Args:
            image_input: Chemin (str/Path) ou objet PIL Image
            top_k: Nombre de résultats
        
        Returns:
            Dictionnaire avec résultats et métadonnées
        """
        start = time.perf_counter()
        
        # Charger l'image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            return {"erreur": "Format d'image non supporté"}
        
        # Extraire l'embedding
        query_embedding = self.extraire_embedding(image)
        
        # Rechercher
        results = self.rechercher(query_embedding, top_k)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "resultats": results,
            "total": len(results),
            "recherche_ms": round(elapsed_ms, 1),
            "embedding_dim": self.embedding_dim,
            "index_size": self.index.ntotal
        }
    
    def rechercher_par_categorie(
        self,
        categorie: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retourne les images d'une catégorie spécifique.
        
        Args:
            categorie: Nom de la catégorie
            top_k: Nombre maximum de résultats
        
        Returns:
            Liste des produits de cette catégorie
        """
        results = []
        for i, label in enumerate(self.labels):
            if label.lower() == categorie.lower():
                results.append({
                    "rang": len(results) + 1,
                    "categorie": label,
                    "image": self.paths[i],
                    "index": i
                })
                if len(results) >= top_k:
                    break
        
        return results
    
    def categories_disponibles(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des catégories avec leur nombre d'images.
        """
        from collections import Counter
        counts = Counter(self.labels)
        return [
            {"categorie": cat, "nombre_images": count}
            for cat, count in sorted(counts.items(), key=lambda x: -x[1])
        ]
    
    def status(self) -> Dict[str, Any]:
        """Retourne le statut du moteur de recherche."""
        return {
            "ready": self.is_ready,
            "index_type": "FAISS IndexFlatIP (cosine similarity)",
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
            "categories": len(set(self.labels)) if self.labels else 0
        }


# ============================================
# Singleton
# ============================================
_search_instance: Optional[FAISSImageSearch] = None


def get_image_search() -> FAISSImageSearch:
    """
    Retourne l'instance singleton du moteur de recherche.
    """
    global _search_instance
    if _search_instance is None:
        _search_instance = FAISSImageSearch()
        try:
            _search_instance.charger_index()
        except Exception as e:
            logger.warning(f"⚠️  Recherche FAISS non disponible : {e}")
    return _search_instance


# ============================================
# Point d'entrée — Démonstration
# ============================================
def main():
    """Démonstration du moteur de recherche FAISS."""
    logger.info("=" * 60)
    logger.info("🔍 ECommerce-IA — Recherche d'Images FAISS")
    logger.info("=" * 60)
    
    searcher = FAISSImageSearch()
    
    try:
        searcher.charger_index()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Afficher le statut
    status = searcher.status()
    logger.info(f"\n📊 Statut : {json.dumps(status, indent=2, default=str)}")
    
    # Afficher les catégories
    cats = searcher.categories_disponibles()
    logger.info(f"\n📂 {len(cats)} catégories disponibles :")
    for cat in cats[:10]:
        logger.info(f"   {cat['categorie']}: {cat['nombre_images']} images")
    
    # Test avec une image synthétique
    logger.info("\n🧪 Test avec image synthétique...")
    test_img = Image.new("RGB", (380, 380), (128, 64, 200))
    result = searcher.rechercher_par_image(test_img, top_k=5)
    
    logger.info(f"   Temps de recherche : {result['recherche_ms']:.1f}ms")
    logger.info(f"   Résultats :")
    for r in result["resultats"]:
        logger.info(
            f"     #{r['rang']} {r['categorie']} "
            f"(similarité: {r['score_similarite']:.4f}) — {r['image']}"
        )
    
    logger.info("\n✅ Recherche FAISS opérationnelle !")


if __name__ == "__main__":
    main()
