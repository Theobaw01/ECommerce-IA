"""
============================================
ECommerce-IA — NLP Evaluation Module
============================================
Évaluation complète du moteur NLP sur un jeu de test
de référence (gold standard).

Métriques évaluées :
- Intent Detection : Accuracy, F1 macro, Confusion Matrix
- NER : Precision, Recall, F1 par type d'entité
- Sentiment Analysis : Accuracy, F1, MAE du score
- Routing : Accuracy du mapping intent → module
- Latence : Temps de réponse moyen

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

# ============================================
# Configuration
# ============================================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================
# Jeu de test de référence (Gold Standard)
# ============================================
INTENT_TEST_SET: List[Dict[str, str]] = [
    # recherche_produit
    {"text": "Je cherche une robe rouge taille M", "intent": "recherche_produit"},
    {"text": "Montrez-moi les chaussures Nike", "intent": "recherche_produit"},
    {"text": "Avez-vous des sacs en cuir ?", "intent": "recherche_produit"},
    {"text": "Je voudrais voir les montres homme", "intent": "recherche_produit"},
    {"text": "Quels t-shirts avez-vous en stock ?", "intent": "recherche_produit"},
    # suivi_commande
    {"text": "Où en est ma commande CMD-2024-001 ?", "intent": "suivi_commande"},
    {"text": "Je veux suivre mon colis", "intent": "suivi_commande"},
    {"text": "Ma commande n'est pas arrivée", "intent": "suivi_commande"},
    {"text": "Quel est le statut de ma livraison ?", "intent": "suivi_commande"},
    # retour
    {"text": "Je souhaite retourner cet article", "intent": "retour"},
    {"text": "Comment faire un remboursement ?", "intent": "retour"},
    {"text": "L'article ne me convient pas, je veux un échange", "intent": "retour"},
    # livraison
    {"text": "Quels sont les délais de livraison ?", "intent": "livraison"},
    {"text": "Livrez-vous à Dakar ?", "intent": "livraison"},
    {"text": "Combien coûte la livraison express ?", "intent": "livraison"},
    # paiement
    {"text": "Je veux payer par carte bancaire Visa", "intent": "paiement"},
    {"text": "Acceptez-vous le paiement mobile Orange Money ?", "intent": "paiement"},
    {"text": "Le paiement à la livraison est possible ?", "intent": "paiement"},
    # recommandation
    {"text": "Que me conseillez-vous comme cadeau ?", "intent": "recommandation"},
    {"text": "Quels produits me recommandez-vous ?", "intent": "recommandation"},
    # compte
    {"text": "Je veux modifier mon mot de passe", "intent": "compte"},
    {"text": "Comment créer un compte ?", "intent": "compte"},
    # stock
    {"text": "Ce produit est-il disponible en taille L ?", "intent": "stock"},
    {"text": "Quand sera-t-il de retour en stock ?", "intent": "stock"},
    # promotion
    {"text": "Y a-t-il des soldes en ce moment ?", "intent": "promotion"},
    {"text": "Avez-vous un code promo ?", "intent": "promotion"},
    # garantie
    {"text": "Quelle est la garantie sur ce produit ?", "intent": "garantie"},
    # salutation
    {"text": "Bonjour !", "intent": "salutation"},
    {"text": "Bonsoir, j'ai besoin d'aide", "intent": "salutation"},
    # remerciement
    {"text": "Merci beaucoup pour votre aide", "intent": "remerciement"},
    {"text": "Parfait, merci !", "intent": "remerciement"},
    # plainte
    {"text": "C'est inadmissible, le service est horrible", "intent": "plainte"},
    {"text": "Je suis très mécontent de ma commande", "intent": "plainte"},
    # question_generale
    {"text": "Quels sont vos horaires d'ouverture ?", "intent": "question_generale"},
    {"text": "Où sont vos magasins ?", "intent": "question_generale"},
]

NER_TEST_SET: List[Dict[str, Any]] = [
    {
        "text": "Ma commande CMD-2024-12345 n'est pas arrivée",
        "entities": [{"type": "ORDER_ID", "value": "CMD-2024-12345"}],
    },
    {
        "text": "Je cherche un sac noir à 50 euros",
        "entities": [
            {"type": "COLOR", "value": "noir"},
            {"type": "PRICE", "value": "50"},
        ],
    },
    {
        "text": "Livraison à Paris taille XL",
        "entities": [
            {"type": "CITY", "value": "Paris"},
            {"type": "SIZE", "value": "XL"},
        ],
    },
    {
        "text": "Contactez-moi à test@email.com",
        "entities": [{"type": "EMAIL", "value": "test@email.com"}],
    },
    {
        "text": "Mon numéro est 06 12 34 56 78",
        "entities": [{"type": "PHONE", "value": "06 12 34 56 78"}],
    },
    {
        "text": "Commande du 15/03/2024",
        "entities": [{"type": "DATE", "value": "15/03/2024"}],
    },
    {
        "text": "Chaussures bleu taille 42 à 89.99€ livrées à Lyon",
        "entities": [
            {"type": "COLOR", "value": "bleu"},
            {"type": "SIZE", "value": "42"},
            {"type": "PRICE", "value": "89.99"},
            {"type": "CITY", "value": "Lyon"},
        ],
    },
]

SENTIMENT_TEST_SET: List[Dict[str, Any]] = [
    {"text": "Excellent service, je suis très satisfait !", "sentiment": "positif", "score_range": (0.3, 1.0)},
    {"text": "Produit de qualité, livraison rapide", "sentiment": "positif", "score_range": (0.1, 1.0)},
    {"text": "Le produit est arrivé", "sentiment": "neutre", "score_range": (-0.3, 0.3)},
    {"text": "Je demande des informations", "sentiment": "neutre", "score_range": (-0.3, 0.3)},
    {"text": "Très déçu, produit de mauvaise qualité", "sentiment": "n\u00e9gatif", "score_range": (-1.0, -0.1)},
    {"text": "C'est horrible, je veux un remboursement", "sentiment": "n\u00e9gatif", "score_range": (-1.0, -0.2)},
    {"text": "Le colis n'est pas arrivé, service nul", "sentiment": "n\u00e9gatif", "score_range": (-1.0, -0.1)},
    {"text": "Merci pour la livraison rapide, parfait", "sentiment": "positif", "score_range": (0.2, 1.0)},
]

ROUTING_TEST_SET: List[Dict[str, str]] = [
    {"intent": "recherche_produit", "expected_module": "recherche"},
    {"intent": "suivi_commande", "expected_module": "suivi"},
    {"intent": "recommandation", "expected_module": "recommandation"},
    {"intent": "stock", "expected_module": "stock"},
    {"intent": "salutation", "expected_module": "direct"},
    {"intent": "remerciement", "expected_module": "direct"},
    {"intent": "plainte", "expected_module": "escalade"},
]


# ============================================
# Classe NLPEvaluator
# ============================================
class NLPEvaluator:
    """Évaluateur complet du moteur NLP."""
    
    def __init__(self):
        """Initialise l'évaluateur avec le moteur NLP."""
        from src.nlp_engine import get_nlp_engine
        self.nlp = get_nlp_engine()
        self.results: Dict[str, Any] = {}
    
    # ─────────────────────────────────────
    # Intent Detection
    # ─────────────────────────────────────
    def evaluate_intent_detection(self) -> Dict[str, Any]:
        """
        Évalue la détection d'intent sur le jeu de test.
        
        Returns:
            Dict avec accuracy, f1_macro, confusion_matrix, per_class_metrics
        """
        y_true = []
        y_pred = []
        latencies = []
        
        for sample in INTENT_TEST_SET:
            t0 = time.time()
            result = self.nlp.detecter_intent(sample["text"])
            latencies.append(time.time() - t0)
            
            predicted = result["intent"]
            y_true.append(sample["intent"])
            y_pred.append(predicted)
        
        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0
        
        # Per-class metrics
        classes = sorted(set(y_true))
        per_class = {}
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": sum(1 for t in y_true if t == cls),
            }
        
        # F1 macro
        f1_macro = sum(m["f1"] for m in per_class.values()) / len(per_class) if per_class else 0
        
        # Confusion matrix (light)
        confusion = defaultdict(lambda: defaultdict(int))
        for t, p in zip(y_true, y_pred):
            confusion[t][p] += 1
        
        # Erreurs détaillées
        errors = [
            {"text": s["text"], "expected": t, "predicted": p}
            for s, t, p in zip(INTENT_TEST_SET, y_true, y_pred)
            if t != p
        ]
        
        result = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "num_samples": len(y_true),
            "num_correct": correct,
            "num_errors": len(errors),
            "per_class": per_class,
            "errors": errors,
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        }
        
        self.results["intent_detection"] = result
        return result
    
    # ─────────────────────────────────────
    # NER
    # ─────────────────────────────────────
    def evaluate_ner(self) -> Dict[str, Any]:
        """
        Évalue l'extraction d'entités nommées.
        
        Returns:
            Dict avec precision, recall, f1 globaux et par type
        """
        per_type_tp = Counter()
        per_type_fp = Counter()
        per_type_fn = Counter()
        latencies = []
        
        for sample in NER_TEST_SET:
            t0 = time.time()
            result = self.nlp.extraire_entites(sample["text"])
            latencies.append(time.time() - t0)
            
            # Entités prédites (set de tuples (type, value))
            predicted = set()
            for ent in result:
                predicted.add((ent["type"], ent.get("value", ent.get("valeur", ""))))
            
            # Entités attendues
            expected = set()
            for ent in sample["entities"]:
                expected.add((ent["type"], ent["value"]))
            
            # Calcul TP, FP, FN par type
            for ent_type, ent_val in predicted:
                if (ent_type, ent_val) in expected:
                    per_type_tp[ent_type] += 1
                else:
                    # Vérifier si le type est correct mais la valeur diffère
                    type_match = any(t == ent_type for t, v in expected)
                    if not type_match:
                        per_type_fp[ent_type] += 1
                    else:
                        # Type correct, valeur incorrecte — compter comme FP
                        per_type_fp[ent_type] += 1
            
            for ent_type, ent_val in expected:
                if (ent_type, ent_val) not in predicted:
                    per_type_fn[ent_type] += 1
        
        # Métriques par type
        all_types = sorted(set(list(per_type_tp.keys()) + list(per_type_fp.keys()) + list(per_type_fn.keys())))
        per_type_metrics = {}
        
        for ent_type in all_types:
            tp = per_type_tp[ent_type]
            fp = per_type_fp[ent_type]
            fn = per_type_fn[ent_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_type_metrics[ent_type] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }
        
        # Métriques globales (micro-average)
        total_tp = sum(per_type_tp.values())
        total_fp = sum(per_type_fp.values())
        total_fn = sum(per_type_fn.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        result = {
            "precision_micro": round(precision, 4),
            "recall_micro": round(recall, 4),
            "f1_micro": round(f1, 4),
            "per_type": per_type_metrics,
            "num_samples": len(NER_TEST_SET),
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        }
        
        self.results["ner"] = result
        return result
    
    # ─────────────────────────────────────
    # Sentiment Analysis
    # ─────────────────────────────────────
    def evaluate_sentiment(self) -> Dict[str, Any]:
        """
        Évalue l'analyse de sentiment.
        
        Returns:
            Dict avec accuracy, mae, per_class metrics
        """
        y_true = []
        y_pred = []
        score_errors = []
        latencies = []
        
        for sample in SENTIMENT_TEST_SET:
            t0 = time.time()
            result = self.nlp.analyser_sentiment(sample["text"])
            latencies.append(time.time() - t0)
            
            predicted_label = result["label"]
            predicted_score = result["score"]
            
            y_true.append(sample["sentiment"])
            y_pred.append(predicted_label)
            
            # Vérifier si le score est dans la plage attendue
            lo, hi = sample["score_range"]
            if not (lo <= predicted_score <= hi):
                score_errors.append({
                    "text": sample["text"],
                    "expected_range": sample["score_range"],
                    "actual_score": predicted_score,
                })
        
        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0
        
        # Per-class
        classes = sorted(set(y_true))
        per_class = {}
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": sum(1 for t in y_true if t == cls),
            }
        
        f1_macro = sum(m["f1"] for m in per_class.values()) / len(per_class) if per_class else 0
        
        result = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "num_samples": len(y_true),
            "num_correct": correct,
            "score_errors": score_errors,
            "per_class": per_class,
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        }
        
        self.results["sentiment"] = result
        return result
    
    # ─────────────────────────────────────
    # Routing
    # ─────────────────────────────────────
    def evaluate_routing(self) -> Dict[str, Any]:
        """
        Évalue le routing des intents vers les modules.
        
        Returns:
            Dict avec accuracy et détails
        """
        correct = 0
        errors = []
        
        for sample in ROUTING_TEST_SET:
            # Créer un texte fictif qui déclenche l'intent voulu
            # et vérifier que le router dirige vers le bon module
            routing = self.nlp.router_requete(
                # On passe directement l'intent pour tester le mapping
                sample.get("text", f"test {sample['intent']}")
            )
            
            predicted_module = routing.get("module", "")
            expected_module = sample["expected_module"]
            
            if predicted_module == expected_module:
                correct += 1
            else:
                errors.append({
                    "intent": sample["intent"],
                    "expected": expected_module,
                    "predicted": predicted_module,
                })
        
        accuracy = correct / len(ROUTING_TEST_SET) if ROUTING_TEST_SET else 0
        
        result = {
            "accuracy": round(accuracy, 4),
            "num_samples": len(ROUTING_TEST_SET),
            "num_correct": correct,
            "errors": errors,
        }
        
        self.results["routing"] = result
        return result
    
    # ─────────────────────────────────────
    # Full pipeline latency
    # ─────────────────────────────────────
    def evaluate_latency(self, n_iterations: int = 50) -> Dict[str, float]:
        """
        Mesure la latence du pipeline NLP complet.
        
        Args:
            n_iterations: Nombre d'itérations pour moyenner
        
        Returns:
            Dict avec avg, min, max, p95 latency en ms
        """
        test_texts = [s["text"] for s in INTENT_TEST_SET[:10]]
        latencies = []
        
        for _ in range(n_iterations):
            for text in test_texts:
                t0 = time.time()
                self.nlp.analyser(text)
                latencies.append((time.time() - t0) * 1000)
        
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        
        result = {
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "p95_ms": round(latencies[p95_idx], 2),
            "num_iterations": len(latencies),
        }
        
        self.results["latency"] = result
        return result
    
    # ─────────────────────────────────────
    # Full evaluation
    # ─────────────────────────────────────
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Lance l'évaluation complète de tous les composants NLP.
        
        Returns:
            Dict avec tous les résultats
        """
        logger.info("=" * 60)
        logger.info("ÉVALUATION COMPLÈTE DU MOTEUR NLP")
        logger.info("=" * 60)
        
        # 1. Intent Detection
        logger.info("\n📋 Évaluation Intent Detection...")
        intent_results = self.evaluate_intent_detection()
        logger.info(f"   Accuracy: {intent_results['accuracy']:.1%}")
        logger.info(f"   F1 macro: {intent_results['f1_macro']:.4f}")
        logger.info(f"   Erreurs: {intent_results['num_errors']}/{intent_results['num_samples']}")
        
        # 2. NER
        logger.info("\n🏷️ Évaluation NER...")
        ner_results = self.evaluate_ner()
        logger.info(f"   Precision: {ner_results['precision_micro']:.1%}")
        logger.info(f"   Recall: {ner_results['recall_micro']:.1%}")
        logger.info(f"   F1 micro: {ner_results['f1_micro']:.4f}")
        
        # 3. Sentiment
        logger.info("\n😊 Évaluation Sentiment...")
        sentiment_results = self.evaluate_sentiment()
        logger.info(f"   Accuracy: {sentiment_results['accuracy']:.1%}")
        logger.info(f"   F1 macro: {sentiment_results['f1_macro']:.4f}")
        
        # 4. Latency
        logger.info("\n⏱️ Évaluation Latence...")
        latency_results = self.evaluate_latency(n_iterations=20)
        logger.info(f"   Avg: {latency_results['avg_ms']:.2f} ms")
        logger.info(f"   P95: {latency_results['p95_ms']:.2f} ms")
        
        # Résumé
        summary = {
            "intent_detection": intent_results,
            "ner": ner_results,
            "sentiment": sentiment_results,
            "latency": latency_results,
            "overall": {
                "intent_accuracy": intent_results["accuracy"],
                "ner_f1": ner_results["f1_micro"],
                "sentiment_accuracy": sentiment_results["accuracy"],
                "avg_latency_ms": latency_results["avg_ms"],
            },
        }
        
        # Sauvegarde
        output_path = ROOT / "mlruns" / "nlp_evaluation_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 Rapport sauvegardé: {output_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("RÉSUMÉ")
        logger.info("=" * 60)
        logger.info(f"  Intent Accuracy : {intent_results['accuracy']:.1%}")
        logger.info(f"  NER F1 (micro)  : {ner_results['f1_micro']:.4f}")
        logger.info(f"  Sentiment Acc   : {sentiment_results['accuracy']:.1%}")
        logger.info(f"  Latence moy     : {latency_results['avg_ms']:.2f} ms")
        logger.info("=" * 60)
        
        return summary


# ============================================
# Point d'entrée
# ============================================
if __name__ == "__main__":
    evaluator = NLPEvaluator()
    results = evaluator.run_full_evaluation()
    
    print("\n✅ Évaluation NLP terminée")
