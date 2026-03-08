"""
============================================
ECommerce-IA — Classifieur Vision Transformer (ViT)
============================================
Module de classification d'images par Vision Transformer,
en complément du classifieur CNN (EfficientNet-B4).

Démontre la maîtrise des architectures Transformers appliquées
à la vision par ordinateur (exigence BCEAO).

Architecture :
    Image 384×384 → Patch Embedding (16×16) → Transformer Encoder (12 layers)
    → [CLS] token → MLP Head → Classification

Modèle : ViT-Base/16 (vit_base_patch16_384)
- 86M paramètres
- Pré-entraîné ImageNet-21k + fine-tuné ImageNet-1k
- Resolution : 384×384
- Patch size : 16×16 → 576 patches + 1 CLS token

Référence :
    Dosovitskiy et al., "An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale", ICLR 2021.

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as T

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

# Constantes ViT
VIT_IMAGE_SIZE = 384
VIT_MODEL_NAME = "vit_base_patch16_384"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViTClassifier:
    """
    Classifieur d'images basé sur Vision Transformer (ViT).
    
    Utilise un ViT-Base/16 pré-entraîné ImageNet comme backbone.
    Peut être utilisé en mode :
    - Zero-shot (poids ImageNet) pour 1000 classes ImageNet
    - Fine-tuné (poids personnalisés) pour les classes e-commerce
    
    Usage :
        classifier = ViTClassifier()
        classifier.charger_modele()
        result = classifier.classifier_image("produit.jpg")
    """
    
    def __init__(
        self,
        model_name: str = VIT_MODEL_NAME,
        device: Optional[torch.device] = None
    ):
        """
        Initialise le classifieur ViT.
        
        Args:
            model_name: Nom du modèle timm (défaut: vit_base_patch16_384)
            device: Device PyTorch
        """
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model: Optional[nn.Module] = None
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_class: Optional[Dict[int, str]] = None
        self.transform: Optional[T.Compose] = None
        
        self.is_ready = False
        self.mode = "pretrained"  # "pretrained" ou "finetuned"
        
        logger.info(f"🤖 ViTClassifier initialisé ({model_name})")
        logger.info(f"   Device : {self.device}")
    
    def charger_modele(
        self,
        model_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None
    ) -> None:
        """
        Charge le modèle ViT.
        
        Si model_path est fourni, charge un modèle fine-tuné.
        Sinon, utilise le modèle pré-entraîné ImageNet avec
        le mapping des classes e-commerce.
        
        Args:
            model_path: Chemin vers un checkpoint fine-tuné (.pth)
            class_mapping_path: Chemin vers class_mapping.json
        """
        # Charger le class_mapping e-commerce
        if class_mapping_path is None:
            class_mapping_path = MODELS_DIR / "class_mapping.json"
        else:
            class_mapping_path = Path(class_mapping_path)
        
        if class_mapping_path.exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                self.class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = len(self.class_to_idx)
            logger.info(f"   📂 {num_classes} classes e-commerce chargées")
        else:
            num_classes = 1000  # ImageNet par défaut
            logger.info("   📂 Utilisation des 1000 classes ImageNet")
        
        # Charger le modèle
        vit_path = MODELS_DIR / "vit_best.pth" if model_path is None else Path(model_path)
        
        if vit_path.exists():
            # Charger le modèle fine-tuné
            logger.info(f"📦 Chargement du ViT fine-tuné : {vit_path}")
            self.model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=num_classes
            )
            checkpoint = torch.load(vit_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.mode = "finetuned"
            logger.info("   ✅ Modèle fine-tuné chargé")
        else:
            # Utiliser le modèle pré-entraîné ImageNet
            logger.info(f"📦 Chargement du ViT pré-entraîné ({self.model_name})...")
            self.model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=num_classes
            )
            self.mode = "pretrained"
            logger.info("   ✅ Modèle pré-entraîné ImageNet chargé")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"   Paramètres : {total_params:,}")
        
        # Transformation
        self.transform = T.Compose([
            T.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
            T.CenterCrop(VIT_IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        self.is_ready = True
        logger.info("✅ ViT Classifier prêt")
    
    @torch.no_grad()
    def classifier_image(
        self,
        image_input,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Classifie une image avec le ViT.
        
        Args:
            image_input: Chemin (str/Path) ou objet PIL Image
            top_k: Nombre de prédictions top-K
        
        Returns:
            Dictionnaire avec catégorie, confiance, top_k, temps d'inférence
        """
        if not self.is_ready:
            return {"erreur": "Modèle ViT non chargé"}
        
        # Charger l'image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            return {"erreur": "Format d'image non supporté"}
        
        # Prétraitement
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inférence
        start = time.perf_counter()
        output = self.model(img_tensor)
        inference_ms = (time.perf_counter() - start) * 1000
        
        # Top-K prédictions
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(min(top_k, probs.size(1)), dim=1)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            if self.idx_to_class:
                class_name = self.idx_to_class.get(idx.item(), f"class_{idx.item()}")
            else:
                class_name = f"class_{idx.item()}"
            predictions.append({
                "categorie": class_name,
                "confiance": float(prob)
            })
        
        return {
            "modele": "ViT-Base/16 (Transformer)",
            "categorie": predictions[0]["categorie"],
            "confiance": predictions[0]["confiance"],
            "top_k": predictions,
            "inference_ms": round(inference_ms, 1),
            "image_size": f"{image.width}×{image.height}",
            "mode": self.mode
        }
    
    @torch.no_grad()
    def extraire_features(self, image_input) -> np.ndarray:
        """
        Extrait les features (embedding) d'une image via le ViT.
        
        Le ViT utilise le token [CLS] comme représentation globale,
        ce qui est fondamentalement différent du pooling spatial des CNN.
        
        Args:
            image_input: Image PIL ou chemin
        
        Returns:
            Vecteur de features (768-d pour ViT-Base)
        """
        if not self.is_ready:
            raise RuntimeError("Modèle non chargé")
        
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extraire les features avant le classifieur
        features = self.model.forward_features(img_tensor)
        
        # Le token [CLS] est le premier token
        cls_token = features[:, 0]  # (1, 768) pour ViT-Base
        
        return cls_token.cpu().numpy().flatten()
    
    def comparer_avec_cnn(
        self,
        image_input,
        cnn_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare les prédictions ViT vs CNN (EfficientNet-B4).
        
        Utile pour le rapport et la démonstration de la maîtrise
        des deux architectures (CNN vs Transformer).
        
        Args:
            image_input: Image à classifier
            cnn_result: Résultat du classifieur CNN
        
        Returns:
            Comparaison détaillée des deux modèles
        """
        vit_result = self.classifier_image(image_input, top_k=5)
        
        # Concordance
        meme_prediction = (
            vit_result.get("categorie", "").lower() ==
            cnn_result.get("categorie", "").lower()
        )
        
        return {
            "concordance": meme_prediction,
            "cnn": {
                "modele": "EfficientNet-B4 (CNN)",
                "architecture": "Convolutional Neural Network",
                "categorie": cnn_result.get("categorie"),
                "confiance": cnn_result.get("confiance"),
                "inference_ms": cnn_result.get("inference_ms"),
                "forces": [
                    "Convolutions locales efficaces",
                    "Moins de paramètres (19.3M vs 86M)",
                    "Inférence plus rapide",
                    "Excellente sur petits datasets"
                ]
            },
            "vit": {
                "modele": "ViT-Base/16 (Transformer)",
                "architecture": "Vision Transformer",
                "categorie": vit_result.get("categorie"),
                "confiance": vit_result.get("confiance"),
                "inference_ms": vit_result.get("inference_ms"),
                "forces": [
                    "Attention globale (toute l'image)",
                    "Meilleure généralisation sur grands datasets",
                    "Architecture unifiée NLP + Vision",
                    "Transfer learning plus flexible"
                ]
            },
            "analyse": (
                "Les deux modèles convergent vers la même prédiction."
                if meme_prediction
                else "Divergence entre CNN et Transformer — "
                     "utile pour l'analyse d'incertitude (ensemble)."
            )
        }
    
    def status(self) -> Dict[str, Any]:
        """Retourne le statut du classifieur ViT."""
        return {
            "ready": self.is_ready,
            "modele": self.model_name,
            "architecture": "Vision Transformer",
            "mode": self.mode,
            "device": str(self.device),
            "num_classes": len(self.class_to_idx) if self.class_to_idx else 0,
            "parametres": f"{sum(p.numel() for p in self.model.parameters()):,}" if self.model else "N/A",
            "image_size": VIT_IMAGE_SIZE,
            "patch_size": 16,
            "reference": "Dosovitskiy et al., ICLR 2021"
        }


# ============================================
# Singleton
# ============================================
_vit_instance: Optional[ViTClassifier] = None


def get_vit_classifier() -> ViTClassifier:
    """Retourne l'instance singleton du classifieur ViT."""
    global _vit_instance
    if _vit_instance is None:
        _vit_instance = ViTClassifier()
        try:
            _vit_instance.charger_modele()
        except Exception as e:
            logger.warning(f"⚠️  ViT non disponible : {e}")
    return _vit_instance


# ============================================
# Point d'entrée — Démonstration
# ============================================
def main():
    """Démonstration du classifieur ViT."""
    logger.info("=" * 60)
    logger.info("🤖 ECommerce-IA — Vision Transformer (ViT)")
    logger.info("=" * 60)
    
    classifier = ViTClassifier()
    classifier.charger_modele()
    
    # Statut
    status = classifier.status()
    logger.info(f"\n📊 Statut : {json.dumps(status, indent=2, default=str)}")
    
    # Test avec image synthétique
    logger.info("\n🧪 Test avec image synthétique...")
    test_img = Image.new("RGB", (400, 400), (64, 128, 200))
    result = classifier.classifier_image(test_img, top_k=5)
    
    logger.info(f"   Modèle    : {result.get('modele')}")
    logger.info(f"   Catégorie : {result['categorie']}")
    logger.info(f"   Confiance : {result['confiance']:.2%}")
    logger.info(f"   Inférence : {result['inference_ms']:.1f}ms")
    logger.info(f"   Top-5 :")
    for pred in result["top_k"]:
        logger.info(f"     → {pred['categorie']}: {pred['confiance']:.2%}")
    
    # Extraction de features
    logger.info("\n🔍 Extraction de features ViT...")
    features = classifier.extraire_features(test_img)
    logger.info(f"   Dimension : {features.shape}")
    logger.info(f"   Norme L2  : {np.linalg.norm(features):.4f}")
    
    logger.info("\n✅ ViT Classifier opérationnel !")


if __name__ == "__main__":
    main()
