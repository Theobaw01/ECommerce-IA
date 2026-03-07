"""
============================================
ECommerce-IA — Dataset PyTorch
============================================
Classe ProductDataset pour charger les images prétraitées.
Gère les transformations PyTorch (ToTensor, Normalisation ImageNet).

⚠️ Règles strictes :
- Normalisation ImageNet uniquement
- Pas d'augmentation ici (déjà faite dans preprocess.py)
- Train : images augmentées (×5)
- Val/Test : images originales uniquement

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ============================================
# Configuration
# ============================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

# Constantes de normalisation ImageNet
IMAGE_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ProductDataset(Dataset):
    """
    Dataset PyTorch pour la classification de produits e-commerce.
    
    Charge les images depuis les dossiers split (train/val/test),
    organisés par catégorie.
    
    Structure attendue :
        splits/train/categorie_1/image1.jpg
        splits/train/categorie_1/image1_aug0.jpg
        splits/train/categorie_2/image2.jpg
        ...
    
    Attributes:
        root_dir: Chemin racine du split
        transform: Transformations PyTorch à appliquer
        images: Liste des chemins d'images
        labels: Liste des labels (index de catégorie)
        classes: Liste ordonnée des noms de catégories
        class_to_idx: Dictionnaire {nom_catégorie: index}
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[T.Compose] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialise le dataset.
        
        Args:
            root_dir: Chemin vers le dossier du split (train/val/test)
            transform: Transformations PyTorch (ToTensor, Normalize)
            class_mapping: Dictionnaire de mapping classe→index
                           Si None, généré automatiquement
            max_samples: Nombre maximum d'échantillons (pour debug)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_samples = max_samples
        
        # Extensions d'images supportées
        self.extensions = {".jpg", ".jpeg", ".png", ".webp"}
        
        # Charger ou générer le mapping des classes
        if class_mapping is not None:
            self.class_to_idx = class_mapping
        else:
            self.class_to_idx = self._generate_class_mapping()
        
        self.classes = sorted(self.class_to_idx.keys())
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.classes)
        
        # Charger la liste des images et labels
        self.images, self.labels = self._load_images()
        
        logger.info(
            f"📦 Dataset chargé : {len(self)} images, "
            f"{self.num_classes} classes depuis {self.root_dir.name}/"
        )
    
    def _generate_class_mapping(self) -> Dict[str, int]:
        """Génère le mapping classe→index à partir des sous-dossiers."""
        categories = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        return {cat: idx for idx, cat in enumerate(categories)}
    
    def _load_images(self) -> Tuple[List[Path], List[int]]:
        """
        Charge la liste de toutes les images avec leurs labels.
        
        Returns:
            Tuple (liste_chemins, liste_labels)
        """
        images = []
        labels = []
        
        for cat_name, cat_idx in self.class_to_idx.items():
            cat_dir = self.root_dir / cat_name
            if not cat_dir.exists():
                continue
            
            for img_path in sorted(cat_dir.iterdir()):
                if img_path.suffix.lower() in self.extensions:
                    images.append(img_path)
                    labels.append(cat_idx)
        
        # Limiter le nombre d'échantillons si demandé
        if self.max_samples and len(images) > self.max_samples:
            indices = np.random.choice(
                len(images), self.max_samples, replace=False
            )
            images = [images[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        return images, labels
    
    def __len__(self) -> int:
        """Nombre total d'images dans le dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retourne une image et son label.
        
        Args:
            idx: Index de l'image
        
        Returns:
            Tuple (image_tensor, label)
            image_tensor : torch.Tensor de shape (3, 380, 380)
            label : int (index de la catégorie)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Charger l'image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"⚠️  Erreur chargement {img_path}: {e}")
            # Retourner une image noire en cas d'erreur
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        
        # Appliquer les transformations
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Transformation par défaut
            image = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """Retourne le nom de la catégorie pour un index donné."""
        return self.idx_to_class.get(idx, f"unknown_{idx}")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Retourne la distribution des classes dans le dataset."""
        distribution = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return dict(sorted(distribution.items()))
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Calcule les poids d'échantillonnage pour gérer le déséquilibre de classes.
        Utile avec WeightedRandomSampler.
        
        Returns:
            Tensor de poids pour chaque échantillon
        """
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total = len(self.labels)
        class_weights = {
            cls: total / count for cls, count in class_counts.items()
        }
        
        sample_weights = [class_weights[label] for label in self.labels]
        return torch.tensor(sample_weights, dtype=torch.float32)


def get_transforms(split: str = "train") -> T.Compose:
    """
    Retourne les transformations PyTorch pour un split donné.
    
    ⚠️ L'augmentation a déjà été faite dans preprocess.py.
    Ici, on applique uniquement ToTensor + Normalisation.
    
    Args:
        split: "train", "val" ou "test"
    
    Returns:
        Compose de transformations PyTorch
    """
    if split == "train":
        # Train : resize + tensor + normalisation
        # (l'augmentation est déjà dans les fichiers)
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Val/Test : resize + center crop + tensor + normalisation
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    class_mapping: Optional[Dict[str, int]] = None,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], int]:
    """
    Crée les DataLoaders pour Train, Val et Test.
    
    Args:
        batch_size: Taille du batch (défaut: 32)
        num_workers: Nombre de workers pour le chargement (défaut: 4)
        pin_memory: Pin memory pour GPU (défaut: True)
        class_mapping: Mapping des classes (optionnel)
        max_samples: Limite d'échantillons par split (debug)
    
    Returns:
        Tuple (train_loader, val_loader, test_loader, class_to_idx, num_classes)
    """
    # Charger le mapping des classes depuis le fichier si disponible
    if class_mapping is None:
        mapping_path = DATA_DIR / "class_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                class_mapping = json.load(f)
            logger.info(f"📄 Mapping chargé : {len(class_mapping)} classes")
    
    # Créer les datasets
    train_dataset = ProductDataset(
        root_dir=str(SPLITS_DIR / "train"),
        transform=get_transforms("train"),
        class_mapping=class_mapping,
        max_samples=max_samples
    )
    
    val_dataset = ProductDataset(
        root_dir=str(SPLITS_DIR / "val"),
        transform=get_transforms("val"),
        class_mapping=class_mapping,
        max_samples=max_samples
    )
    
    test_dataset = ProductDataset(
        root_dir=str(SPLITS_DIR / "test"),
        transform=get_transforms("test"),
        class_mapping=class_mapping,
        max_samples=max_samples
    )
    
    # Utiliser le même mapping pour tous les splits
    class_to_idx = train_dataset.class_to_idx
    num_classes = train_dataset.num_classes
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"🔄 DataLoaders créés (batch_size={batch_size})")
    logger.info(f"   Train : {len(train_dataset)} images → {len(train_loader)} batches")
    logger.info(f"   Val   : {len(val_dataset)} images → {len(val_loader)} batches")
    logger.info(f"   Test  : {len(test_dataset)} images → {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, class_to_idx, num_classes


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Dé-normalise un tensor pour la visualisation.
    
    Args:
        tensor: Image normalisée (C, H, W)
    
    Returns:
        Image dé-normalisée (C, H, W) dans [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# ============================================
# Test du module
# ============================================
if __name__ == "__main__":
    """Test rapide du dataset et des DataLoaders."""
    
    logger.info("=" * 60)
    logger.info("🧪 Test du module Dataset")
    logger.info("=" * 60)
    
    try:
        train_loader, val_loader, test_loader, class_to_idx, num_classes = \
            get_dataloaders(batch_size=8, num_workers=0)
        
        # Tester un batch
        images, labels = next(iter(train_loader))
        logger.info(f"\n📦 Batch Train :")
        logger.info(f"   Images shape : {images.shape}")    # (8, 3, 380, 380)
        logger.info(f"   Labels shape : {labels.shape}")     # (8,)
        logger.info(f"   Labels       : {labels.tolist()}")
        logger.info(f"   Min pixel    : {images.min():.4f}")
        logger.info(f"   Max pixel    : {images.max():.4f}")
        
        # Afficher la distribution des classes
        train_dataset = train_loader.dataset
        dist = train_dataset.get_class_distribution()
        logger.info(f"\n📊 Distribution (top 10) :")
        for i, (name, count) in enumerate(dist.items()):
            if i >= 10:
                break
            logger.info(f"   {name}: {count} images")
        
        logger.info(f"\n✅ Test réussi — {num_classes} classes détectées")
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        logger.info("   Vérifiez que les données sont prétraitées :")
        logger.info("   1. python data/download_dataset.py")
        logger.info("   2. python src/preprocess.py")
