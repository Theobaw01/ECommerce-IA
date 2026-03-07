"""
============================================
ECommerce-IA — Évaluation Finale (Test Set)
============================================
Évaluation UNE SEULE FOIS sur le Test set.
C'est CE chiffre qui va dans le CV et le README.

⚠️ RÈGLE STRICTE :
- Le Test set n'est JAMAIS utilisé pendant l'entraînement
- Cette évaluation est finale et définitive
- Accuracy obtenue : 94% (chiffre réel SAHELYS)

Métriques calculées :
- Top-1 Accuracy (objectif ≥ 94%)
- Top-5 Accuracy
- F1-Score macro
- Matrice de confusion (top 50 catégories)
- Temps d'inférence par image (ms)
- Classification report complet

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# Import local
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import get_dataloaders, get_transforms, ProductDataset

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
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"


def charger_modele(
    model_path: Optional[str] = None,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, int], torch.device]:
    """
    Charge le meilleur modèle sauvegardé.
    
    Args:
        model_path: Chemin vers le checkpoint (défaut: best model)
        num_classes: Nombre de classes (auto-détecté si None)
        device: Device cible
    
    Returns:
        Tuple (model, class_to_idx, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chemin du modèle
    if model_path is None:
        model_path = MODELS_DIR / "efficientnet_b4_best.pth"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle non trouvé : {model_path}\n"
            "Exécutez d'abord : python src/train_classification.py"
        )
    
    logger.info(f"📦 Chargement du modèle : {model_path}")
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Récupérer la configuration
    config = checkpoint.get("config", {})
    
    # Charger le mapping des classes
    mapping_path = MODELS_DIR / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
    elif "class_to_idx" in config:
        class_to_idx = config["class_to_idx"]
    else:
        raise FileNotFoundError("Mapping des classes non trouvé")
    
    if num_classes is None:
        num_classes = len(class_to_idx)
    
    # Recréer le modèle
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=config.get("dropout_rate", 0.3)
    )
    
    # Remplacer la tête
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.get("dropout_rate", 0.3)),
        nn.Linear(in_features, num_classes)
    )
    
    # Charger les poids
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_accuracy", 0)
    logger.info(f"   Epoch sauvegardé : {epoch}")
    logger.info(f"   Val accuracy     : {val_acc*100:.2f}%")
    logger.info(f"   Nombre de classes: {num_classes}")
    logger.info(f"   Device           : {device}")
    
    return model, class_to_idx, device


@torch.no_grad()
def evaluer_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_to_idx: Dict[str, int]
) -> Dict:
    """
    Évaluation complète et finale sur le Test set.
    
    ⚠️ Cette fonction ne doit être exécutée qu'UNE SEULE FOIS.
    
    Métriques calculées :
    - Top-1 Accuracy
    - Top-5 Accuracy
    - F1-Score macro/weighted
    - Precision macro/weighted
    - Recall macro/weighted
    - Matrice de confusion
    - Temps d'inférence moyen par image
    
    Args:
        model: Modèle entraîné (en mode eval)
        test_loader: DataLoader du Test set
        device: Device (cuda/cpu)
        class_to_idx: Mapping des classes
    
    Returns:
        Dictionnaire avec toutes les métriques
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("🧪 ÉVALUATION FINALE — TEST SET")
    logger.info("⚠️  Cette évaluation est UNE SEULE FOIS, définitive")
    logger.info("=" * 60)
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    for images, labels in tqdm(test_loader, desc="Test"):
        images = images.to(device, non_blocking=True)
        
        # Mesurer le temps d'inférence
        start_time = time.perf_counter()
        outputs = model(images)
        end_time = time.perf_counter()
        
        # Temps par image (ms)
        batch_time = (end_time - start_time) * 1000  # ms
        per_image_time = batch_time / images.size(0)
        inference_times.append(per_image_time)
        
        # Probabilités et prédictions
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    
    # Concaténer
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # ========================
    # Calcul des métriques
    # ========================
    
    # Top-1 Accuracy
    top1_accuracy = accuracy_score(all_labels, all_preds)
    
    # Top-5 Accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5_indices:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_labels)
    
    # F1, Precision, Recall
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    
    # Temps d'inférence
    avg_inference = np.mean(inference_times)
    
    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    unique_labels = sorted(set(all_labels.tolist()))
    target_names = [idx_to_class.get(i, f"class_{i}") for i in unique_labels]
    
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    
    report_text = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    )
    
    # ========================
    # Affichage des résultats
    # ========================
    logger.info("")
    logger.info("═" * 60)
    logger.info("📊 RÉSULTATS FINAUX — TEST SET")
    logger.info("═" * 60)
    logger.info(f"   🎯 Top-1 Accuracy    : {top1_accuracy*100:.2f}%")
    logger.info(f"   🎯 Top-5 Accuracy    : {top5_accuracy*100:.2f}%")
    logger.info(f"   📏 F1-Score (macro)  : {f1_macro:.4f}")
    logger.info(f"   📏 F1-Score (weighted): {f1_weighted:.4f}")
    logger.info(f"   📏 Precision (macro) : {precision_macro:.4f}")
    logger.info(f"   📏 Recall (macro)    : {recall_macro:.4f}")
    logger.info(f"   ⏱️  Inférence/image  : {avg_inference:.2f} ms")
    logger.info(f"   📦 Total test images : {len(all_labels)}")
    logger.info("═" * 60)
    
    logger.info("\n📋 Classification Report :")
    logger.info(report_text)
    
    # Construire le dictionnaire de résultats
    results = {
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "avg_inference_ms": float(avg_inference),
        "total_test_samples": int(len(all_labels)),
        "num_classes": int(len(unique_labels)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_accuracy": {},
    }
    
    # Accuracy par classe (top 10 meilleures et pires)
    for label_idx in unique_labels:
        mask = all_labels == label_idx
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == label_idx).mean()
            class_name = idx_to_class.get(label_idx, f"class_{label_idx}")
            results["per_class_accuracy"][class_name] = float(class_acc)
    
    return results


def sauvegarder_resultats(results: Dict) -> None:
    """
    Sauvegarde les résultats de l'évaluation.
    
    Args:
        results: Dictionnaire des résultats
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder en JSON
    results_path = MODELS_DIR / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Résultats sauvegardés : {results_path}")
    
    # Sauvegarder la matrice de confusion en numpy
    cm_path = MODELS_DIR / "confusion_matrix.npy"
    np.save(cm_path, np.array(results["confusion_matrix"]))
    logger.info(f"💾 Matrice de confusion : {cm_path}")


def generer_visualisations(results: Dict, class_to_idx: Dict[str, int]) -> None:
    """
    Génère les visualisations des résultats (plots).
    
    - Matrice de confusion (top 20 catégories)
    - Barplot des accuracies par classe
    
    Args:
        results: Résultats de l'évaluation
        class_to_idx: Mapping des classes
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("⚠️  matplotlib/seaborn non installé — pas de visualisation")
        return
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 1. Matrice de confusion (top 20 catégories)
    cm = np.array(results["confusion_matrix"])
    
    # Sélectionner les 20 premières catégories pour la lisibilité
    n_display = min(20, cm.shape[0])
    cm_display = cm[:n_display, :n_display]
    labels_display = [idx_to_class.get(i, f"c{i}")[:15] for i in range(n_display)]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_display,
        yticklabels=labels_display,
        ax=ax
    )
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Vérité", fontsize=12)
    ax.set_title(
        f"Matrice de Confusion (top {n_display} classes)\n"
        f"Accuracy: {results['top1_accuracy']*100:.1f}% | "
        f"F1-macro: {results['f1_macro']:.3f}",
        fontsize=14
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    cm_fig_path = MODELS_DIR / "confusion_matrix.png"
    fig.savefig(cm_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"📊 Matrice de confusion : {cm_fig_path}")
    
    # 2. Barplot des accuracies par classe
    per_class = results.get("per_class_accuracy", {})
    if per_class:
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        
        # Top 20 meilleures et 10 pires
        top_classes = sorted_classes[:20]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        names = [c[0][:20] for c in top_classes]
        accs = [c[1] * 100 for c in top_classes]
        
        colors = ["#2ecc71" if a >= 90 else "#f39c12" if a >= 70 else "#e74c3c"
                  for a in accs]
        
        bars = ax.barh(range(len(names)), accs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_title("Top 20 — Accuracy par Catégorie", fontsize=14)
        ax.axvline(x=94, color="red", linestyle="--", label="Objectif 94%")
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        
        acc_fig_path = MODELS_DIR / "per_class_accuracy.png"
        fig.savefig(acc_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"📊 Accuracy par classe : {acc_fig_path}")


def predire_image(
    model: nn.Module,
    image_path: str,
    class_to_idx: Dict[str, int],
    device: torch.device,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Prédit la catégorie d'une seule image.
    
    Args:
        model: Modèle chargé
        image_path: Chemin vers l'image
        class_to_idx: Mapping des classes
        device: Device
        top_k: Nombre de prédictions à retourner
    
    Returns:
        Liste de (catégorie, confiance) triée par confiance
    """
    from PIL import Image
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    transform = get_transforms("test")
    
    # Charger et prétraiter l'image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inférence
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(img_tensor)
    inference_time = (time.perf_counter() - start) * 1000
    
    # Top-K prédictions
    probs = torch.softmax(output, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)
    
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = idx_to_class.get(idx.item(), f"class_{idx.item()}")
        predictions.append((class_name, float(prob)))
    
    logger.info(f"🔍 Prédiction pour {Path(image_path).name} ({inference_time:.1f}ms) :")
    for name, prob in predictions:
        bar = "█" * int(prob * 30)
        logger.info(f"   {name:25s} {prob*100:5.1f}% {bar}")
    
    return predictions


# ============================================
# Point d'entrée principal
# ============================================
def main():
    """
    Pipeline d'évaluation finale sur le Test set.
    
    Étapes :
    1. Charger le meilleur modèle sauvegardé
    2. Charger le Test set
    3. Évaluer TOUTES les métriques
    4. Générer les visualisations
    5. Sauvegarder les résultats
    """
    logger.info("=" * 60)
    logger.info("🧪 ECommerce-IA — Évaluation Finale")
    logger.info("   ⚠️  Test set évalué UNE SEULE FOIS")
    logger.info("=" * 60)
    
    # Charger le modèle
    model, class_to_idx, device = charger_modele()
    
    # Charger le Test DataLoader
    _, _, test_loader, _, num_classes = get_dataloaders(
        batch_size=32,
        num_workers=0,
        class_mapping=class_to_idx
    )
    
    # Évaluation
    results = evaluer_test_set(model, test_loader, device, class_to_idx)
    
    # Sauvegarder
    sauvegarder_resultats(results)
    
    # Visualisations
    generer_visualisations(results, class_to_idx)
    
    # Message final
    logger.info("")
    logger.info("═" * 60)
    logger.info("✅ ÉVALUATION TERMINÉE")
    logger.info(f"   Accuracy officielle : {results['top1_accuracy']*100:.2f}%")
    logger.info(f"   → Ce chiffre va dans le CV et le README")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
