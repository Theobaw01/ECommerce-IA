"""
============================================
ECommerce-IA — Prétraitement & Augmentation
============================================
Prétraite les images brutes et applique la data augmentation
sur le set d'entraînement uniquement.

Stratégie :
- Resize 380×380 (standard EfficientNet-B4)
- Normalisation ImageNet : mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Data augmentation ×5 sur Train uniquement :
  * Rotation (±15°)
  * Flip horizontal
  * Variation luminosité/contraste (±20%)
  * Zoom aléatoire (0.8-1.2)
  * Cutout / RandomErasing

Séparation stricte :
- 70% Train (2 100 images brutes → ~10 500 après augmentation)
- 15% Val   (450 images — PAS d'augmentation)
- 15% Test  (450 images — PAS d'augmentation, évalué UNE SEULE FOIS)

⚠️ fit_transform() sur Train uniquement
⚠️ Data augmentation sur Train uniquement
⚠️ Ne jamais toucher au Test set avant évaluation finale

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import shutil
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ============================================
# Configuration du logging
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================
# Chemins du projet
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "val"
TEST_DIR = SPLITS_DIR / "test"

# ============================================
# Constantes de prétraitement
# ============================================
IMAGE_SIZE = 380                      # EfficientNet-B4 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42

# Ratios de split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentation
NB_AUGMENTATIONS = 5  # Facteur de multiplication des images Train


def get_augmentation_transform() -> Optional[A.Compose]:
    """
    Définit le pipeline d'augmentation pour le set d'entraînement.
    
    Augmentations appliquées :
    - Rotation aléatoire (±15°)
    - Flip horizontal (50% de chance)
    - Variation de luminosité et contraste (±20%)
    - Zoom aléatoire (RandomResizedCrop 0.8-1.0)
    - Cutout / CoarseDropout (RandomErasing)
    
    Returns:
        Pipeline Albumentations ou None si non disponible
    """
    if not HAS_ALBUMENTATIONS:
        logger.warning("⚠️  Albumentations non installé — augmentation PIL basique")
        return None
    
    transform = A.Compose([
        # Resize + Crop
        A.RandomResizedCrop(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            scale=(0.8, 1.0),       # Zoom aléatoire 0.8-1.0
            ratio=(0.9, 1.1),
            p=1.0
        ),
        # Rotation ±15°
        A.Rotate(limit=15, p=0.7, border_mode=0),
        # Flip horizontal
        A.HorizontalFlip(p=0.5),
        # Variation luminosité / contraste ±20%
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        # Légère modification de teinte/saturation
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        # Cutout / CoarseDropout (RandomErasing)
        A.CoarseDropout(
            max_holes=3,
            max_height=int(IMAGE_SIZE * 0.1),
            max_width=int(IMAGE_SIZE * 0.1),
            min_holes=1,
            fill_value=0,
            p=0.5
        ),
    ])
    
    return transform


def get_preprocessing_transform() -> Optional[A.Compose]:
    """
    Définit le prétraitement standard (sans augmentation).
    Utilisé pour Val et Test sets.
    
    Returns:
        Pipeline Albumentations pour le prétraitement simple
    """
    if not HAS_ALBUMENTATIONS:
        return None
    
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
    ])
    
    return transform


def preprocess_image_pil(image_path: Path, target_size: int = IMAGE_SIZE) -> Image.Image:
    """
    Prétraite une image avec PIL (fallback si Albumentations non dispo).
    
    Args:
        image_path: Chemin vers l'image
        target_size: Taille cible (carré)
    
    Returns:
        Image PIL redimensionnée
    """
    img = Image.open(image_path).convert("RGB")
    
    # Resize en gardant le ratio, puis center crop
    w, h = img.size
    ratio = max(target_size / w, target_size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    
    return img


def augment_image_pil(img: Image.Image, idx: int) -> Image.Image:
    """
    Applique une augmentation simple avec PIL (fallback).
    
    Args:
        img: Image PIL source
        idx: Index de l'augmentation (détermine la transformation)
    
    Returns:
        Image PIL augmentée
    """
    from PIL import ImageEnhance, ImageFilter
    
    augmented = img.copy()
    
    if idx == 0:
        # Rotation légère
        angle = random.uniform(-15, 15)
        augmented = augmented.rotate(angle, fillcolor=(0, 0, 0))
    elif idx == 1:
        # Flip horizontal
        augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
    elif idx == 2:
        # Variation luminosité
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(augmented)
        augmented = enhancer.enhance(factor)
    elif idx == 3:
        # Variation contraste
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(augmented)
        augmented = enhancer.enhance(factor)
    elif idx == 4:
        # Combinaison rotation + flip + luminosité
        angle = random.uniform(-10, 10)
        augmented = augmented.rotate(angle, fillcolor=(0, 0, 0))
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
        factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Brightness(augmented)
        augmented = enhancer.enhance(factor)
    
    return augmented


def split_dataset(
    raw_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """
    Sépare le dataset en Train/Val/Test (70/15/15).
    
    ⚠️ RÈGLE STRICTE : La séparation est faite AVANT toute augmentation.
    Les images d'un même produit ne peuvent PAS être dans deux splits différents.
    
    Args:
        raw_dir: Chemin vers les images brutes organisées par catégorie
        train_ratio: Ratio du set d'entraînement (0.70)
        val_ratio: Ratio du set de validation (0.15)
        test_ratio: Ratio du set de test (0.15)
        seed: Graine aléatoire
    
    Returns:
        Tuple (train_files, val_files, test_files)
        Chaque élément est une liste de (chemin_image, catégorie)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Les ratios doivent sommer à 1.0"
    
    random.seed(seed)
    np.random.seed(seed)
    
    train_files = []
    val_files = []
    test_files = []
    
    # Parcourir chaque catégorie
    categories = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    
    if len(categories) == 0:
        raise FileNotFoundError(
            f"Aucune catégorie trouvée dans {raw_dir}. "
            "Exécutez d'abord data/download_dataset.py"
        )
    
    logger.info(f"📊 Séparation du dataset ({len(categories)} catégories)")
    
    for cat in categories:
        cat_dir = raw_dir / cat
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = [
            f for f in cat_dir.iterdir()
            if f.suffix.lower() in extensions
        ]
        
        # Mélanger
        random.shuffle(images)
        
        n = len(images)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Le reste va dans test
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Si test est vide, prendre la dernière image de val
        if len(test_imgs) == 0 and len(val_imgs) > 1:
            test_imgs = [val_imgs.pop()]
        elif len(test_imgs) == 0:
            test_imgs = [train_imgs.pop()]
        
        train_files.extend([(img, cat) for img in train_imgs])
        val_files.extend([(img, cat) for img in val_imgs])
        test_files.extend([(img, cat) for img in test_imgs])
    
    logger.info(f"   Train : {len(train_files)} images")
    logger.info(f"   Val   : {len(val_files)} images")
    logger.info(f"   Test  : {len(test_files)} images")
    
    return train_files, val_files, test_files


def traiter_et_sauvegarder(
    files: List[Tuple[Path, str]],
    output_dir: Path,
    augment: bool = False,
    nb_augmentations: int = NB_AUGMENTATIONS,
    transform_aug: Optional[A.Compose] = None,
    transform_pre: Optional[A.Compose] = None
) -> int:
    """
    Prétraite les images et les sauvegarde dans le dossier de sortie.
    
    Args:
        files: Liste de (chemin_image, catégorie)
        output_dir: Dossier de sortie
        augment: Appliquer la data augmentation (Train uniquement)
        nb_augmentations: Nombre d'augmentations par image
        transform_aug: Pipeline d'augmentation Albumentations
        transform_pre: Pipeline de prétraitement Albumentations
    
    Returns:
        Nombre total d'images sauvegardées
    """
    compteur = 0
    
    desc = "Augmentation Train" if augment else "Prétraitement"
    
    for img_path, cat in tqdm(files, desc=desc):
        cat_dir = output_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if HAS_ALBUMENTATIONS and HAS_CV2:
                # Charger avec OpenCV (BGR → RGB)
                img_cv = cv2.imread(str(img_path))
                if img_cv is None:
                    logger.warning(f"⚠️  Impossible de lire : {img_path}")
                    continue
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Prétraitement de base
                if transform_pre is not None:
                    preprocessed = transform_pre(image=img_cv)["image"]
                else:
                    preprocessed = cv2.resize(img_cv, (IMAGE_SIZE, IMAGE_SIZE))
                
                # Sauvegarder l'image prétraitée
                stem = img_path.stem
                out_path = cat_dir / f"{stem}.jpg"
                img_pil = Image.fromarray(preprocessed)
                img_pil.save(out_path, quality=95)
                compteur += 1
                
                # Augmentation (Train uniquement)
                if augment and transform_aug is not None:
                    for aug_idx in range(nb_augmentations):
                        augmented = transform_aug(image=preprocessed)["image"]
                        aug_path = cat_dir / f"{stem}_aug{aug_idx}.jpg"
                        img_aug = Image.fromarray(augmented)
                        img_aug.save(aug_path, quality=90)
                        compteur += 1
                elif augment:
                    # Fallback PIL augmentation
                    img_pil_base = Image.fromarray(preprocessed)
                    for aug_idx in range(nb_augmentations):
                        img_aug = augment_image_pil(img_pil_base, aug_idx)
                        aug_path = cat_dir / f"{stem}_aug{aug_idx}.jpg"
                        img_aug.save(aug_path, quality=90)
                        compteur += 1
            else:
                # Fallback PIL complet
                img_pil = preprocess_image_pil(img_path)
                stem = img_path.stem
                out_path = cat_dir / f"{stem}.jpg"
                img_pil.save(out_path, quality=95)
                compteur += 1
                
                if augment:
                    for aug_idx in range(nb_augmentations):
                        img_aug = augment_image_pil(img_pil, aug_idx)
                        aug_path = cat_dir / f"{stem}_aug{aug_idx}.jpg"
                        img_aug.save(aug_path, quality=90)
                        compteur += 1
        
        except Exception as e:
            logger.warning(f"⚠️  Erreur traitement {img_path.name} : {e}")
            continue
    
    return compteur


def calculer_statistiques_normalisation(train_dir: Path) -> Dict[str, List[float]]:
    """
    Calcule les statistiques de normalisation SUR LE TRAIN SET UNIQUEMENT.
    
    ⚠️ RÈGLE STRICTE : fit_transform() uniquement sur Train.
    On utilise les statistiques ImageNet par défaut (standard),
    mais on peut aussi calculer les stats spécifiques au dataset.
    
    Args:
        train_dir: Chemin vers le dossier Train
    
    Returns:
        Dictionnaire avec mean et std calculés
    """
    logger.info("📊 Calcul des statistiques de normalisation (Train set)...")
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    
    extensions = {".jpg", ".jpeg", ".png"}
    images = list(train_dir.rglob("*"))
    images = [f for f in images if f.suffix.lower() in extensions]
    
    # Échantillonner pour la vitesse (max 500 images)
    if len(images) > 500:
        random.seed(SEED)
        images = random.sample(images, 500)
    
    for img_path in tqdm(images, desc="Calcul stats normalisation"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.float64) / 255.0
            pixel_sum += img_array.sum(axis=(0, 1))
            pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
            n_pixels += img_array.shape[0] * img_array.shape[1]
        except Exception:
            continue
    
    if n_pixels == 0:
        logger.warning("⚠️  Aucune image pour calculer les stats, utilisation ImageNet")
        return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    
    mean = (pixel_sum / n_pixels).tolist()
    std = np.sqrt(pixel_sq_sum / n_pixels - np.array(mean) ** 2).tolist()
    
    logger.info(f"   Mean calculé : {[f'{m:.4f}' for m in mean]}")
    logger.info(f"   Std calculé  : {[f'{s:.4f}' for s in std]}")
    logger.info(f"   Mean ImageNet: {IMAGENET_MEAN}")
    logger.info(f"   Std ImageNet : {IMAGENET_STD}")
    logger.info("   → Utilisation des stats ImageNet (standard pour transfer learning)")
    
    return {
        "mean_dataset": mean,
        "std_dataset": std,
        "mean_imagenet": IMAGENET_MEAN,
        "std_imagenet": IMAGENET_STD,
        "note": "Utiliser les stats ImageNet pour le transfer learning"
    }


# ============================================
# Point d'entrée principal
# ============================================
def main():
    """
    Pipeline complet de prétraitement.
    
    Étapes :
    1. Séparer le dataset en Train/Val/Test (70/15/15)
    2. Prétraiter les images (Resize 380×380)
    3. Appliquer la data augmentation ×5 sur Train uniquement
    4. Calculer les statistiques de normalisation sur Train
    5. Sauvegarder les métadonnées de split
    """
    logger.info("=" * 60)
    logger.info("🔧 ECommerce-IA — Prétraitement & Augmentation")
    logger.info("=" * 60)
    
    # Vérifier que le dataset brut existe
    if not RAW_DIR.exists() or len(list(RAW_DIR.iterdir())) == 0:
        logger.error("❌ Aucune donnée brute trouvée dans data/raw/")
        logger.info("   Exécutez d'abord : python data/download_dataset.py")
        sys.exit(1)
    
    # Nettoyer les dossiers de sortie
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # Étape 1 : Séparer le dataset
    logger.info("")
    logger.info("📌 ÉTAPE 1 : Séparation Train/Val/Test (70/15/15)")
    logger.info("-" * 40)
    train_files, val_files, test_files = split_dataset(RAW_DIR)
    
    # Préparer les pipelines de transformation
    transform_aug = get_augmentation_transform()
    transform_pre = get_preprocessing_transform()
    
    if transform_aug is not None:
        logger.info("✅ Pipeline Albumentations chargé")
    else:
        logger.info("⚠️  Fallback PIL pour les transformations")
    
    # Étape 2 : Prétraiter et augmenter le Train set
    logger.info("")
    logger.info("📌 ÉTAPE 2 : Prétraitement + Augmentation Train")
    logger.info("-" * 40)
    n_train = traiter_et_sauvegarder(
        train_files, TRAIN_DIR,
        augment=True,
        nb_augmentations=NB_AUGMENTATIONS,
        transform_aug=transform_aug,
        transform_pre=transform_pre
    )
    logger.info(f"   ✅ {n_train} images Train (après augmentation ×{NB_AUGMENTATIONS})")
    
    # Étape 3 : Prétraiter Val (PAS d'augmentation)
    logger.info("")
    logger.info("📌 ÉTAPE 3 : Prétraitement Val (SANS augmentation)")
    logger.info("-" * 40)
    n_val = traiter_et_sauvegarder(
        val_files, VAL_DIR,
        augment=False,
        transform_pre=transform_pre
    )
    logger.info(f"   ✅ {n_val} images Val")
    
    # Étape 4 : Prétraiter Test (PAS d'augmentation)
    logger.info("")
    logger.info("📌 ÉTAPE 4 : Prétraitement Test (SANS augmentation)")
    logger.info("-" * 40)
    n_test = traiter_et_sauvegarder(
        test_files, TEST_DIR,
        augment=False,
        transform_pre=transform_pre
    )
    logger.info(f"   ✅ {n_test} images Test")
    
    # Étape 5 : Statistiques de normalisation
    logger.info("")
    logger.info("📌 ÉTAPE 5 : Statistiques de normalisation (Train)")
    logger.info("-" * 40)
    norm_stats = calculer_statistiques_normalisation(TRAIN_DIR)
    
    # Sauvegarder les statistiques
    stats_path = DATA_DIR / "normalization_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"   💾 Stats sauvegardées : {stats_path}")
    
    # Sauvegarder les informations de split
    split_info = {
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "train_images_brutes": len(train_files),
        "train_images_augmentees": n_train,
        "val_images": n_val,
        "test_images": n_test,
        "augmentation_factor": NB_AUGMENTATIONS,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "augmentations": [
            "Rotation ±15°",
            "Flip horizontal",
            "RandomBrightnessContrast ±20%",
            "RandomResizedCrop (zoom 0.8-1.0)",
            "CoarseDropout (Cutout)"
        ]
    }
    
    split_path = DATA_DIR / "split_info.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    # Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ DU PRÉTRAITEMENT")
    logger.info("=" * 60)
    logger.info(f"   Image size       : {IMAGE_SIZE}×{IMAGE_SIZE}")
    logger.info(f"   Train (brut)     : {len(train_files)} images")
    logger.info(f"   Train (augmenté) : {n_train} images (×{NB_AUGMENTATIONS + 1})")
    logger.info(f"   Validation       : {n_val} images (pas d'augmentation)")
    logger.info(f"   Test             : {n_test} images (pas d'augmentation)")
    logger.info(f"   Normalisation    : ImageNet (mean={IMAGENET_MEAN})")
    logger.info("=" * 60)
    logger.info("✅ Prochaine étape : python src/train_classification.py")


if __name__ == "__main__":
    main()
