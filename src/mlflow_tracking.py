"""
============================================
ECommerce-IA — MLflow Experiment Tracking
============================================
Module de tracking MLOps pour le suivi des expériences
d'entraînement et d'évaluation des modèles IA.

Fonctionnalités :
- Tracking des hyperparamètres
- Logging des métriques (loss, accuracy, F1, etc.)
- Sauvegarde des artefacts (modèles, embeddings, plots)
- Comparaison d'expériences
- Registry de modèles

Compatible avec :
- EfficientNet-B4 (Classification CNN)
- ViT-Base/16 (Classification Transformer)
- NLP Engine (Intent, NER, Sentiment)
- Recommandation Hybride (SVD, Content-Based)

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

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
# Chemin du projet
# ============================================
ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = ROOT / "mlruns"

# ============================================
# Tentative d'import MLflow
# ============================================
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
    logger.info("MLflow disponible — tracking activé")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow non installé — tracking en mode fichier local")


# ============================================
# Classe ExperimentTracker
# ============================================
class ExperimentTracker:
    """
    Tracker d'expériences IA unifié.
    
    Utilise MLflow si disponible, sinon sauvegarde
    les métriques en JSON local (fallback).
    
    Usage:
        tracker = ExperimentTracker("classification_cnn")
        tracker.start_run("efficientnet_b4_v2")
        tracker.log_params({"lr": 3e-4, "epochs": 30})
        tracker.log_metrics({"accuracy": 0.8653, "top5": 0.9837})
        tracker.log_artifact("models/classification/efficientnet_b4_best.pth")
        tracker.end_run()
    """
    
    def __init__(self, experiment_name: str = "ecommerce_ia"):
        """
        Initialise le tracker.
        
        Args:
            experiment_name: Nom de l'expérience MLflow
        """
        self.experiment_name = experiment_name
        self.run_active = False
        self.run_name = None
        self.run_id = None
        self.start_time = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.params: Dict[str, Any] = {}
        self.artifacts: List[str] = []
        
        # Répertoire de fallback local
        self.local_dir = MLFLOW_DIR / experiment_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration MLflow si disponible
        if MLFLOW_AVAILABLE:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(MLFLOW_DIR))
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
            logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Démarre un nouveau run de tracking.
        
        Args:
            run_name: Nom du run (ex: "efficientnet_b4_v2")
            tags: Tags additionnels
        """
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        self.metrics_history = []
        self.params = {}
        self.artifacts = []
        
        if MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=self.run_name)
            self.run_id = mlflow.active_run().info.run_id
            
            # Tags par défaut
            default_tags = {
                "project": "ECommerce-IA",
                "author": "BAWANA Théodore",
                "company": "SAHELYS",
            }
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)
        else:
            self.run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_active = True
        logger.info(f"Run démarré: {self.run_name} (id: {self.run_id})")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Enregistre les hyperparamètres.
        
        Args:
            params: Dictionnaire de paramètres
                    Ex: {"lr": 3e-4, "epochs": 30, "batch_size": 32}
        """
        if not self.run_active:
            logger.warning("Aucun run actif — appeler start_run() d'abord")
            return
        
        self.params.update(params)
        
        if MLFLOW_AVAILABLE:
            # MLflow n'accepte que des strings, ints, floats
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        logger.info(f"Params enregistrés: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Enregistre des métriques.
        
        Args:
            metrics: Dictionnaire de métriques
                     Ex: {"loss": 0.42, "accuracy": 0.8653}
            step: Numéro d'étape (epoch, batch, etc.)
        """
        if not self.run_active:
            logger.warning("Aucun run actif — appeler start_run() d'abord")
            return
        
        record = {"step": step, "timestamp": time.time(), **metrics}
        self.metrics_history.append(record)
        
        if MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        
        if step is not None:
            logger.info(f"Step {step}: {metrics}")
    
    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        """
        Enregistre un artefact (fichier modèle, plot, etc.).
        
        Args:
            filepath: Chemin vers le fichier
            artifact_path: Sous-dossier dans les artefacts MLflow
        """
        if not self.run_active:
            logger.warning("Aucun run actif — appeler start_run() d'abord")
            return
        
        self.artifacts.append(filepath)
        
        if MLFLOW_AVAILABLE and os.path.exists(filepath):
            mlflow.log_artifact(filepath, artifact_path)
            logger.info(f"Artefact enregistré: {filepath}")
        else:
            logger.info(f"Artefact référencé: {filepath}")
    
    def log_model(self, model, model_name: str = "model"):
        """
        Enregistre un modèle PyTorch dans MLflow.
        
        Args:
            model: Modèle PyTorch
            model_name: Nom du modèle dans le registry
        """
        if not self.run_active:
            logger.warning("Aucun run actif — appeler start_run() d'abord")
            return
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.pytorch.log_model(model, model_name)
                logger.info(f"Modèle PyTorch enregistré: {model_name}")
            except Exception as e:
                logger.warning(f"Erreur log_model: {e}")
        else:
            logger.info(f"Modèle référencé (sans MLflow): {model_name}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        Termine le run en cours et sauvegarde le résumé.
        
        Args:
            status: Statut final ("FINISHED", "FAILED", "KILLED")
        """
        if not self.run_active:
            logger.warning("Aucun run actif")
            return
        
        duration = time.time() - self.start_time
        
        # Résumé du run
        summary = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment": self.experiment_name,
            "status": status,
            "duration_seconds": round(duration, 2),
            "params": self.params,
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "metrics_history": self.metrics_history,
            "artifacts": self.artifacts,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Sauvegarde locale (toujours)
        summary_path = self.local_dir / f"{self.run_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Fin du run MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("duration_seconds", duration)
            mlflow.end_run(status=status)
        
        self.run_active = False
        logger.info(f"Run terminé: {self.run_name} ({status}) — {duration:.1f}s")
        logger.info(f"Résumé sauvegardé: {summary_path}")
        
        return summary
    
    def get_best_run(self, metric: str = "accuracy", ascending: bool = False) -> Optional[Dict]:
        """
        Récupère le meilleur run selon une métrique.
        
        Args:
            metric: Nom de la métrique à optimiser
            ascending: True pour minimiser (ex: loss), False pour maximiser (ex: accuracy)
        
        Returns:
            Dict du meilleur run ou None
        """
        runs = []
        for path in self.local_dir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                run_data = json.load(f)
                final = run_data.get("final_metrics", {})
                if metric in final:
                    runs.append(run_data)
        
        if not runs:
            return None
        
        runs.sort(
            key=lambda r: r["final_metrics"].get(metric, 0),
            reverse=not ascending
        )
        return runs[0]
    
    def list_runs(self) -> List[Dict]:
        """Liste tous les runs de l'expérience."""
        runs = []
        for path in sorted(self.local_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        return runs


# ============================================
# Fonctions de tracking spécialisées
# ============================================

def track_classification_training(
    model_name: str,
    params: Dict[str, Any],
    epoch_metrics: List[Dict[str, float]],
    test_metrics: Dict[str, float],
    model_path: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Dict:
    """
    Track complet d'un entraînement de classification.
    
    Args:
        model_name: "efficientnet_b4" ou "vit_base_16"
        params: Hyperparamètres (lr, epochs, batch_size, etc.)
        epoch_metrics: Liste de métriques par epoch
        test_metrics: Métriques finales sur le test set
        model_path: Chemin du modèle sauvegardé (.pth)
        history_path: Chemin de l'historique (.json)
    
    Returns:
        Résumé du run
    """
    tracker = ExperimentTracker("classification")
    tracker.start_run(
        run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d')}",
        tags={"model_type": model_name, "task": "classification"}
    )
    
    # Hyperparamètres
    tracker.log_params(params)
    
    # Métriques par epoch
    for i, epoch_data in enumerate(epoch_metrics):
        tracker.log_metrics(epoch_data, step=i + 1)
    
    # Métriques finales
    tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    
    # Artefacts
    if model_path:
        tracker.log_artifact(model_path, "models")
    if history_path:
        tracker.log_artifact(history_path, "history")
    
    return tracker.end_run()


def track_nlp_evaluation(
    intent_accuracy: float,
    ner_f1: float,
    sentiment_accuracy: float,
    detailed_metrics: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Track d'une évaluation NLP.
    
    Args:
        intent_accuracy: Accuracy de la détection d'intent
        ner_f1: F1-score de l'extraction d'entités
        sentiment_accuracy: Accuracy de l'analyse de sentiment
        detailed_metrics: Métriques détaillées optionnelles
    
    Returns:
        Résumé du run
    """
    tracker = ExperimentTracker("nlp_evaluation")
    tracker.start_run(
        run_name=f"nlp_eval_{datetime.now().strftime('%Y%m%d')}",
        tags={"task": "nlp_evaluation"}
    )
    
    metrics = {
        "intent_accuracy": intent_accuracy,
        "ner_f1": ner_f1,
        "sentiment_accuracy": sentiment_accuracy,
    }
    
    if detailed_metrics:
        metrics.update(detailed_metrics)
    
    tracker.log_params({
        "num_intents": 15,
        "num_ner_types": 10,
        "sentiment_classes": 3,
        "preprocessing": "tokenize + lemmatize + stopwords",
    })
    
    tracker.log_metrics(metrics)
    
    return tracker.end_run()


def track_recommendation_evaluation(
    precision_at_k: float,
    recall_at_k: float,
    ndcg_at_k: float,
    coverage: float,
    k: int = 10,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Track d'une évaluation du système de recommandation.
    
    Args:
        precision_at_k: Precision@K
        recall_at_k: Recall@K
        ndcg_at_k: NDCG@K
        coverage: Taux de couverture du catalogue
        k: Nombre de recommandations
        weights: Poids des facteurs hybrides
    
    Returns:
        Résumé du run
    """
    tracker = ExperimentTracker("recommendation")
    tracker.start_run(
        run_name=f"recom_eval_{datetime.now().strftime('%Y%m%d')}",
        tags={"task": "recommendation_evaluation"}
    )
    
    default_weights = {
        "collaborative": 0.40,
        "content_based": 0.30,
        "geographic": 0.15,
        "price": 0.15,
    }
    
    tracker.log_params({
        "k": k,
        "algorithm": "hybrid_4_factors",
        **(weights or default_weights),
    })
    
    tracker.log_metrics({
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
        f"ndcg_at_{k}": ndcg_at_k,
        "coverage": coverage,
    })
    
    return tracker.end_run()


# ============================================
# Singleton global
# ============================================
_global_tracker: Optional[ExperimentTracker] = None


def get_tracker(experiment_name: str = "ecommerce_ia") -> ExperimentTracker:
    """
    Retourne le tracker global (singleton).
    
    Args:
        experiment_name: Nom de l'expérience
    
    Returns:
        Instance ExperimentTracker
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker(experiment_name)
    return _global_tracker


# ============================================
# Point d'entrée (démo)
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("ECommerce-IA — MLflow Experiment Tracking")
    print("=" * 60)
    
    # Demo: tracker de classification
    tracker = ExperimentTracker("demo_classification")
    tracker.start_run("efficientnet_b4_demo")
    
    tracker.log_params({
        "model": "efficientnet_b4",
        "lr": 3e-4,
        "epochs": 30,
        "batch_size": 32,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "label_smoothing": 0.1,
        "dropout": 0.3,
    })
    
    # Simulation de métriques d'entraînement
    for epoch in range(1, 6):
        tracker.log_metrics({
            "train_loss": 2.5 - epoch * 0.35,
            "val_loss": 2.7 - epoch * 0.30,
            "val_accuracy": 0.50 + epoch * 0.07,
            "val_top5": 0.80 + epoch * 0.03,
        }, step=epoch)
    
    # Métriques finales
    tracker.log_metrics({
        "test_accuracy_top1": 0.8653,
        "test_accuracy_top5": 0.9837,
        "test_images": 980,
    })
    
    summary = tracker.end_run()
    
    print(f"\nRésumé: {json.dumps(summary['final_metrics'], indent=2)}")
    print(f"Sauvegardé dans: {tracker.local_dir}")
    
    # Liste des runs
    runs = tracker.list_runs()
    print(f"\nNombre de runs: {len(runs)}")
    
    print("\n✅ MLflow tracking opérationnel")
