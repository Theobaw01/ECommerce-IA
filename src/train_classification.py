"""
============================================
ECommerce-IA — Fine-Tuning EfficientNet-B4
============================================
Entraînement complet du modèle de classification visuelle
de produits e-commerce.

Modèle : EfficientNet-B4 (timm)
- Meilleur compromis accuracy/vitesse pour 500 catégories
- Input size : 380×380
- Accuracy cible : ≥ 94% sur Test set

Paramètres :
- Epochs : 30 (early stopping patience=7)
- Learning rate : 3e-4 (CosineAnnealingLR)
- Batch size : 32
- Optimizer : AdamW (weight_decay=1e-4)
- Loss : CrossEntropyLoss + label smoothing 0.1
- Dropout : 0.3 avant la couche finale
- Unfreeze progressif :
  * Epochs 1-5   : seule la tête (classifier)
  * Epochs 6-15  : derniers 2 blocs
  * Epochs 16-30 : tout le réseau

Compatible Google Colab (GPU T4).

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import timm
from tqdm import tqdm

# Import local
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import get_dataloaders

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
MODELS_DIR = PROJECT_ROOT / "models" / "classification"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================
# Hyperparamètres
# ============================================
CONFIG = {
    "model_name": "efficientnet_b4",
    "image_size": 380,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "dropout_rate": 0.3,
    "early_stopping_patience": 7,
    "num_workers": 4,
    "seed": 42,
    # Phases d'unfreeze progressif
    "unfreeze_schedule": {
        "phase_1": {"start": 1, "end": 5, "description": "Tête uniquement"},
        "phase_2": {"start": 6, "end": 15, "description": "Derniers 2 blocs"},
        "phase_3": {"start": 16, "end": 30, "description": "Réseau complet"},
    },
    # Mixed precision pour accélérer sur GPU
    "use_amp": True,
}


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les graines pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def creer_modele(num_classes: int, dropout_rate: float = 0.3) -> nn.Module:
    """
    Crée le modèle EfficientNet-B4 avec une tête personnalisée.
    
    Architecture :
    - Backbone : EfficientNet-B4 pré-entraîné ImageNet
    - Dropout : 0.3
    - Couche finale : Linear → num_classes
    
    Args:
        num_classes: Nombre de catégories
        dropout_rate: Taux de dropout
    
    Returns:
        Modèle PyTorch
    """
    logger.info(f"🏗️  Création du modèle EfficientNet-B4 ({num_classes} classes)")
    
    # Charger le modèle pré-entraîné
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=dropout_rate
    )
    
    # Remplacer la tête avec dropout personnalisé
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"   Paramètres totaux    : {total_params:,}")
    logger.info(f"   Paramètres entraîn.  : {trainable_params:,}")
    
    return model


def geler_backbone(model: nn.Module) -> None:
    """
    Gèle tout le backbone (phase 1 : seule la tête est entraînée).
    """
    for param in model.parameters():
        param.requires_grad = False
    
    # Dégeler la tête (classifier)
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"❄️  Backbone gelé — {trainable:,} paramètres entraînables")


def degeler_derniers_blocs(model: nn.Module, nb_blocs: int = 2) -> None:
    """
    Dégèle les derniers blocs du backbone (phase 2).
    
    Args:
        model: Modèle EfficientNet
        nb_blocs: Nombre de blocs à dégeler
    """
    # EfficientNet-B4 a des blocs dans model.blocks
    blocks = list(model.blocks)
    total_blocks = len(blocks)
    
    # Dégeler les derniers nb_blocs
    for block in blocks[-(nb_blocs):]:
        for param in block.parameters():
            param.requires_grad = True
    
    # Dégeler aussi le conv_head et bn2 s'ils existent
    for name, param in model.named_parameters():
        if "conv_head" in name or "bn2" in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"🔓 Derniers {nb_blocs}/{total_blocks} blocs dégelés — "
        f"{trainable:,} paramètres entraînables"
    )


def degeler_tout(model: nn.Module) -> None:
    """
    Dégèle tout le réseau (phase 3 : fine-tuning complet).
    """
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🔓 Réseau entièrement dégelé — {trainable:,} paramètres entraînables")


def appliquer_unfreeze_progressif(model: nn.Module, epoch: int, config: Dict) -> None:
    """
    Applique la stratégie d'unfreeze progressif selon l'epoch.
    
    Phases :
    - Epochs 1-5   : seule la tête (classifier)
    - Epochs 6-15  : derniers 2 blocs + tête
    - Epochs 16-30 : tout le réseau
    
    Args:
        model: Modèle à modifier
        epoch: Numéro de l'epoch (1-indexed)
        config: Configuration des hyperparamètres
    """
    schedule = config["unfreeze_schedule"]
    
    if epoch == schedule["phase_1"]["start"]:
        logger.info(f"📌 Phase 1 : {schedule['phase_1']['description']}")
        geler_backbone(model)
    elif epoch == schedule["phase_2"]["start"]:
        logger.info(f"📌 Phase 2 : {schedule['phase_2']['description']}")
        degeler_derniers_blocs(model, nb_blocs=2)
    elif epoch == schedule["phase_3"]["start"]:
        logger.info(f"📌 Phase 3 : {schedule['phase_3']['description']}")
        degeler_tout(model)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy avec Label Smoothing.
    Réduit l'overconfidence du modèle.
    
    Label smoothing = 0.1 :
    - Cible douce : 0.9 pour la vraie classe, 0.1/(N-1) pour les autres
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EarlyStopping:
    """
    Early stopping pour arrêter l'entraînement si la val loss ne s'améliore plus.
    
    Patience : 7 epochs sans amélioration → arrêt
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"⏹️  Early stopping déclenché (patience={self.patience})"
                )
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Entraîne le modèle pour une epoch.
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (cuda/cpu)
        scaler: GradScaler pour mixed precision
        use_amp: Utiliser l'autocast
    
    Returns:
        Tuple (loss_moyenne, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Train", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Évalue le modèle sur le set de validation.
    
    Args:
        model: Modèle à évaluer
        val_loader: DataLoader de validation
        criterion: Fonction de perte
        device: Device
    
    Returns:
        Tuple (loss, top1_accuracy, top5_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        
        # Top-1 accuracy
        _, pred_top1 = outputs.max(1)
        correct_top1 += pred_top1.eq(labels).sum().item()
        
        # Top-5 accuracy
        _, pred_top5 = outputs.topk(min(5, outputs.size(1)), dim=1)
        correct_top5 += sum(
            labels[i].item() in pred_top5[i].tolist()
            for i in range(labels.size(0))
        )
    
    val_loss = running_loss / total
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    
    return val_loss, top1_acc, top5_acc


def sauvegarder_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    val_acc: float,
    config: Dict,
    is_best: bool = False
) -> None:
    """
    Sauvegarde un checkpoint du modèle.
    
    Args:
        model: Modèle entraîné
        optimizer: État de l'optimiseur
        scheduler: État du scheduler
        epoch: Numéro de l'epoch
        val_loss: Loss de validation
        val_acc: Accuracy de validation
        config: Configuration d'entraînement
        is_best: Si c'est le meilleur modèle
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "config": config,
    }
    
    # Sauvegarder le checkpoint courant
    path = MODELS_DIR / "checkpoint_last.pth"
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = MODELS_DIR / "efficientnet_b4_best.pth"
        torch.save(checkpoint, best_path)
        logger.info(f"💾 Meilleur modèle sauvegardé : {best_path}")


# ============================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================
def entrainer(config: Dict = CONFIG) -> Dict:
    """
    Pipeline d'entraînement complet avec unfreeze progressif.
    
    Étapes :
    1. Configurer le device (GPU/CPU)
    2. Charger les données (DataLoaders)
    3. Créer le modèle EfficientNet-B4
    4. Configurer l'optimiseur, loss, scheduler
    5. Boucle d'entraînement (30 epochs)
    6. Unfreeze progressif (3 phases)
    7. Early stopping (patience=7)
    8. Sauvegarder le meilleur modèle
    
    Args:
        config: Dictionnaire de configuration
    
    Returns:
        Historique d'entraînement (losses, accuracies)
    """
    logger.info("=" * 60)
    logger.info("🚀 ECommerce-IA — Entraînement EfficientNet-B4")
    logger.info("=" * 60)
    
    # Reproductibilité
    set_seed(config["seed"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device : {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Charger les données
    logger.info("\n📦 Chargement des données...")
    num_workers = 0 if device.type == "cpu" else config["num_workers"]
    
    train_loader, val_loader, test_loader, class_to_idx, num_classes = \
        get_dataloaders(
            batch_size=config["batch_size"],
            num_workers=num_workers,
            pin_memory=device.type == "cuda"
        )
    
    # Sauvegarder le mapping des classes
    config["num_classes"] = num_classes
    config["class_to_idx"] = class_to_idx
    
    # Créer le modèle
    model = creer_modele(num_classes, config["dropout_rate"])
    model = model.to(device)
    
    # Geler le backbone (phase 1)
    geler_backbone(model)
    
    # Loss avec label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=config["label_smoothing"])
    
    # Optimiseur AdamW
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Scheduler CosineAnnealing
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler() if config["use_amp"] and device.type == "cuda" else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    
    # Historique
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_top5_acc": [],
        "lr": [], "epoch_time": []
    }
    
    best_val_acc = 0.0
    
    logger.info(f"\n🏋️ Début de l'entraînement ({config['epochs']} epochs)")
    logger.info(f"   Batch size       : {config['batch_size']}")
    logger.info(f"   Learning rate    : {config['learning_rate']}")
    logger.info(f"   Weight decay     : {config['weight_decay']}")
    logger.info(f"   Label smoothing  : {config['label_smoothing']}")
    logger.info(f"   Mixed precision  : {config['use_amp']}")
    logger.info(f"   Early stopping   : patience={config['early_stopping_patience']}")
    logger.info("")
    
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        
        # Appliquer l'unfreeze progressif
        appliquer_unfreeze_progressif(model, epoch, config)
        
        # Mettre à jour l'optimiseur avec les nouveaux paramètres
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=scheduler.get_last_lr()[0] if epoch > 1 else config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Entraînement
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config["use_amp"]
        )
        
        # Validation
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Temps
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Historique
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_top1)
        history["val_top5_acc"].append(val_top5)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        
        # Log
        logger.info(
            f"Epoch {epoch:2d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Top1: {val_top1*100:.1f}% Top5: {val_top5*100:.1f}% | "
            f"LR: {current_lr:.6f} | {epoch_time:.0f}s"
        )
        
        # Sauvegarder le meilleur modèle
        is_best = val_top1 > best_val_acc
        if is_best:
            best_val_acc = val_top1
            logger.info(f"   🏆 Nouvelle meilleure accuracy : {best_val_acc*100:.2f}%")
        
        sauvegarder_checkpoint(
            model, optimizer, scheduler, epoch,
            val_loss, val_top1, config, is_best
        )
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"⏹️  Arrêt anticipé à l'epoch {epoch}")
            break
    
    # Résumé
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ DE L'ENTRAÎNEMENT")
    logger.info("=" * 60)
    logger.info(f"   Meilleure Val Accuracy : {best_val_acc*100:.2f}%")
    logger.info(f"   Epochs effectués       : {len(history['train_loss'])}")
    logger.info(f"   Temps total            : {sum(history['epoch_time']):.0f}s")
    logger.info(f"   Modèle sauvegardé      : {MODELS_DIR / 'efficientnet_b4_best.pth'}")
    logger.info("=" * 60)
    
    # Sauvegarder l'historique
    history_path = MODELS_DIR / "training_history.json"
    # Convertir les valeurs pour JSON
    history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
    history_json["best_val_accuracy"] = float(best_val_acc)
    history_json["config"] = {
        k: v for k, v in config.items() if k != "class_to_idx"
    }
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2)
    logger.info(f"💾 Historique sauvegardé : {history_path}")
    
    # Sauvegarder le mapping des classes
    mapping_path = MODELS_DIR / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ Prochaine étape : python src/evaluate.py (évaluation sur Test set)")
    
    return history


# ============================================
# Point d'entrée
# ============================================
if __name__ == "__main__":
    history = entrainer()
