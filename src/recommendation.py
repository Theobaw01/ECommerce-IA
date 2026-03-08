"""
============================================
ECommerce-IA — Système de Recommandation Hybride
============================================
Système de recommandation à 4 facteurs pour personnaliser
l'expérience d'achat sur la plateforme e-commerce.

Architecture hybride :
- FACTEUR 1 (40%) : Historique d'achats — Collaborative Filtering (SVD)
- FACTEUR 2 (30%) : Similarité produits — Content-Based Filtering
- FACTEUR 3 (15%) : Distance géographique — Haversine
- FACTEUR 4 (15%) : Facteur prix — Budget utilisateur

Score final :
  score = 0.40 × historique + 0.30 × similarité +
          0.15 × géographique + 0.15 × prix

Métriques :
- Precision@K (K=5, K=10)
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Coverage

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import math
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

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
MODELS_DIR = PROJECT_ROOT / "models" / "recommendation"
DATA_DIR = PROJECT_ROOT / "data"

# Poids des facteurs
POIDS = {
    "historique": 0.40,
    "similarite": 0.30,
    "geographique": 0.15,
    "prix": 0.15
}

# Distance maximale en km
DISTANCE_MAX_KM = 50.0


# ============================================
# FACTEUR 1 — Collaborative Filtering (SVD)
# ============================================
class CollaborativeFilter:
    """
    Filtrage collaboratif basé sur SVD (Surprise library).
    "Les clients similaires ont aussi acheté..."
    
    Utilise la décomposition en valeurs singulières (SVD)
    pour prédire les notes/interactions utilisateur-produit.
    """
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr: float = 0.005):
        """
        Args:
            n_factors: Nombre de facteurs latents
            n_epochs: Nombre d'epochs d'entraînement
            lr: Learning rate
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        self.trainset = None
        self.is_fitted = False
    
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """
        Entraîne le modèle SVD sur les interactions utilisateur-produit.
        
        Args:
            interactions_df: DataFrame avec colonnes [user_id, product_id, rating]
        """
        try:
            from surprise import SVD, Dataset, Reader
            
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                interactions_df[["user_id", "product_id", "rating"]],
                reader
            )
            
            self.trainset = data.build_full_trainset()
            
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr,
                random_state=42
            )
            self.model.fit(self.trainset)
            self.is_fitted = True
            
            logger.info(
                f"✅ SVD entraîné : {self.trainset.n_users} utilisateurs, "
                f"{self.trainset.n_items} produits"
            )
        except ImportError:
            logger.warning("⚠️  scikit-surprise non installé — fallback moyenne")
            self._fit_fallback(interactions_df)
    
    def _fit_fallback(self, interactions_df: pd.DataFrame) -> None:
        """Fallback si Surprise n'est pas disponible."""
        self.user_means = interactions_df.groupby("user_id")["rating"].mean().to_dict()
        self.product_means = interactions_df.groupby("product_id")["rating"].mean().to_dict()
        self.global_mean = interactions_df["rating"].mean()
        self.is_fitted = True
        logger.info("✅ Fallback moyenne ajusté")
    
    def predict(self, user_id: str, product_id: str) -> float:
        """
        Prédit le score d'interaction utilisateur-produit.
        
        Args:
            user_id: Identifiant utilisateur
            product_id: Identifiant produit
        
        Returns:
            Score prédit entre 0 et 1
        """
        if not self.is_fitted:
            return 0.5
        
        if self.model is not None:
            try:
                prediction = self.model.predict(str(user_id), str(product_id))
                # Normaliser entre 0 et 1
                return max(0.0, min(1.0, (prediction.est - 1) / 4))
            except Exception:
                return 0.5
        else:
            # Fallback
            user_mean = self.user_means.get(user_id, self.global_mean)
            product_mean = self.product_means.get(product_id, self.global_mean)
            score = (user_mean + product_mean) / 2
            return max(0.0, min(1.0, (score - 1) / 4))
    
    def get_top_n(self, user_id: str, product_ids: List[str], n: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les top-N produits recommandés pour un utilisateur.
        
        Args:
            user_id: Identifiant utilisateur
            product_ids: Liste des produits candidats
            n: Nombre de recommandations
        
        Returns:
            Liste de (product_id, score) triée par score décroissant
        """
        scores = [(pid, self.predict(user_id, pid)) for pid in product_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
    
    def save(self, path: Optional[Path] = None) -> None:
        """Sauvegarde le modèle."""
        if path is None:
            path = MODELS_DIR / "svd_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"💾 Modèle SVD sauvegardé : {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CollaborativeFilter":
        """Charge un modèle sauvegardé."""
        if path is None:
            path = MODELS_DIR / "svd_model.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================
# FACTEUR 2 — Content-Based Filtering
# ============================================
class ContentBasedFilter:
    """
    Filtrage basé sur le contenu des produits.
    "Produits similaires à ce que vous regardez..."
    
    Features utilisées :
    - Catégorie (one-hot)
    - Prix (normalisé)
    - Marque (one-hot)
    - Embeddings visuels (si disponibles)
    """
    
    def __init__(self):
        """Initialise le moteur content-based (non entraîné)."""
        self.product_features = None
        self.product_ids = []
        self.similarity_matrix = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def fit(self, products_df: pd.DataFrame) -> None:
        """
        Construit la matrice de features et de similarité.
        
        Args:
            products_df: DataFrame des produits avec colonnes :
                         [id, categorie, prix, marque, ...]
        """
        self.product_ids = products_df["id"].astype(str).tolist()
        
        features_list = []
        
        # Feature 1 : Catégorie (one-hot encoding)
        if "categorie" in products_df.columns:
            cat_dummies = pd.get_dummies(products_df["categorie"], prefix="cat")
            features_list.append(cat_dummies.values)
        
        # Feature 2 : Prix normalisé
        if "prix" in products_df.columns:
            prix = products_df["prix"].fillna(0).values.reshape(-1, 1)
            prix_norm = self.scaler.fit_transform(prix)
            features_list.append(prix_norm)
        
        # Feature 3 : Marque (one-hot encoding)
        if "marque" in products_df.columns:
            marque_dummies = pd.get_dummies(
                products_df["marque"].fillna("unknown"), prefix="marque"
            )
            features_list.append(marque_dummies.values)
        
        # Feature 4 : Note moyenne
        if "note_moyenne" in products_df.columns:
            notes = products_df["note_moyenne"].fillna(3.0).values.reshape(-1, 1)
            notes_norm = notes / 5.0
            features_list.append(notes_norm)
        
        if features_list:
            self.product_features = np.hstack(features_list)
            # Calculer la matrice de similarité cosine
            self.similarity_matrix = cosine_similarity(self.product_features)
            self.is_fitted = True
            logger.info(
                f"✅ Content-Based : {len(self.product_ids)} produits, "
                f"{self.product_features.shape[1]} features"
            )
        else:
            logger.warning("⚠️  Aucune feature disponible pour le content-based")
    
    def get_similar(self, product_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les produits les plus similaires.
        
        Args:
            product_id: ID du produit source
            n: Nombre de produits similaires
        
        Returns:
            Liste de (product_id, score_similarite)
        """
        if not self.is_fitted:
            return []
        
        try:
            idx = self.product_ids.index(str(product_id))
        except ValueError:
            return []
        
        similarities = self.similarity_matrix[idx]
        # Exclure le produit lui-même
        similarities[idx] = -1
        
        top_indices = np.argsort(similarities)[::-1][:n]
        
        return [
            (self.product_ids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]
    
    def score(self, product_id_source: str, product_id_target: str) -> float:
        """
        Calcule le score de similarité entre deux produits.
        
        Returns:
            Score entre 0 et 1
        """
        if not self.is_fitted:
            return 0.0
        
        try:
            idx_source = self.product_ids.index(str(product_id_source))
            idx_target = self.product_ids.index(str(product_id_target))
            return float(max(0, self.similarity_matrix[idx_source][idx_target]))
        except (ValueError, IndexError):
            return 0.0


# ============================================
# FACTEUR 3 — Distance Géographique (Haversine)
# ============================================
class GeoFilter:
    """
    Calcul de la distance géographique entre utilisateur et vendeur.
    Utilise la formule de Haversine pour la distance sur la sphère terrestre.
    
    Pénalité : distance > 50km → score diminue progressivement.
    """
    
    RAYON_TERRE_KM = 6371.0
    
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calcule la distance en km entre deux points GPS (Haversine).
        
        Args:
            lat1, lon1: Coordonnées du point 1 (degrés)
            lat2, lon2: Coordonnées du point 2 (degrés)
        
        Returns:
            Distance en kilomètres
        """
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return GeoFilter.RAYON_TERRE_KM * c
    
    @staticmethod
    def score(
        user_lat: float, user_lon: float,
        vendor_lat: float, vendor_lon: float,
        max_distance: float = DISTANCE_MAX_KM
    ) -> float:
        """
        Calcule le score géographique (0 à 1).
        - Distance < max_distance → score élevé
        - Distance > max_distance → pénalité exponentielle
        
        Args:
            user_lat, user_lon: Position de l'utilisateur
            vendor_lat, vendor_lon: Position du vendeur
            max_distance: Distance maximale sans pénalité (km)
        
        Returns:
            Score entre 0 et 1
        """
        try:
            distance = GeoFilter.haversine(
                user_lat, user_lon, vendor_lat, vendor_lon
            )
            
            if distance <= max_distance:
                # Score linéaire décroissant jusqu'à max_distance
                return 1.0 - (distance / max_distance) * 0.3
            else:
                # Pénalité exponentielle au-delà
                penalty = math.exp(-(distance - max_distance) / max_distance)
                return max(0.0, 0.7 * penalty)
        except Exception:
            return 0.5  # Score neutre si coordonnées invalides


# ============================================
# FACTEUR 4 — Facteur Prix
# ============================================
class PriceFilter:
    """
    Score basé sur le budget estimé de l'utilisateur.
    
    - Prix dans la fourchette du budget → score élevé
    - Promotion/réduction → bonus de score
    - Historique de prix consultés pour estimer le budget
    """
    
    @staticmethod
    def score(
        product_price: float,
        user_budget_min: float,
        user_budget_max: float,
        has_promotion: bool = False,
        discount_percent: float = 0.0
    ) -> float:
        """
        Calcule le score prix pour un produit.
        
        Args:
            product_price: Prix du produit
            user_budget_min: Budget minimum de l'utilisateur
            user_budget_max: Budget maximum de l'utilisateur
            has_promotion: Le produit est en promotion
            discount_percent: Pourcentage de réduction
        
        Returns:
            Score entre 0 et 1
        """
        if user_budget_max <= 0:
            return 0.5
        
        # Prix effectif après réduction
        effective_price = product_price * (1 - discount_percent / 100)
        
        # Score de base selon la fourchette de budget
        if user_budget_min <= effective_price <= user_budget_max:
            # Dans le budget — score élevé
            range_size = user_budget_max - user_budget_min
            if range_size > 0:
                # Centre du budget = meilleur score
                center = (user_budget_min + user_budget_max) / 2
                distance_center = abs(effective_price - center) / (range_size / 2)
                score = 1.0 - 0.2 * distance_center
            else:
                score = 1.0
        elif effective_price < user_budget_min:
            # Moins cher que le budget — acceptable
            ratio = effective_price / user_budget_min
            score = max(0.3, ratio)
        else:
            # Plus cher que le budget — pénalité
            ratio = user_budget_max / effective_price
            score = max(0.0, ratio * 0.7)
        
        # Bonus promotion
        if has_promotion:
            score = min(1.0, score + 0.1)
        if discount_percent > 0:
            bonus = min(0.15, discount_percent / 100 * 0.3)
            score = min(1.0, score + bonus)
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def estimer_budget(
        historique_prix: List[float],
        percentile_bas: float = 25,
        percentile_haut: float = 75
    ) -> Tuple[float, float]:
        """
        Estime le budget de l'utilisateur à partir de son historique.
        
        Args:
            historique_prix: Liste des prix consultés/achetés
            percentile_bas: Percentile bas pour le budget minimum
            percentile_haut: Percentile haut pour le budget maximum
        
        Returns:
            Tuple (budget_min, budget_max)
        """
        if not historique_prix:
            return (0.0, 100.0)
        
        prices = np.array(historique_prix)
        budget_min = float(np.percentile(prices, percentile_bas))
        budget_max = float(np.percentile(prices, percentile_haut))
        
        # Marge de 20%
        budget_min *= 0.8
        budget_max *= 1.2
        
        return (budget_min, budget_max)


# ============================================
# MOTEUR DE RECOMMANDATION HYBRIDE
# ============================================
class HybridRecommender:
    """
    Moteur de recommandation hybride combinant 4 facteurs.
    
    score_final = (0.40 × score_historique) +
                  (0.30 × score_similarite) +
                  (0.15 × score_geographique) +
                  (0.15 × score_prix)
    """
    
    def __init__(
        self,
        poids: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            poids: Dictionnaire des poids par facteur
        """
        self.poids = poids or POIDS
        
        # Sous-modèles
        self.collaborative = CollaborativeFilter()
        self.content_based = ContentBasedFilter()
        self.geo_filter = GeoFilter()
        self.price_filter = PriceFilter()
        
        # Données
        self.products_df = None
        self.users_df = None
        self.is_fitted = False
        
        logger.info(f"🛍️  Recommandeur hybride initialisé (poids: {self.poids})")
    
    def fit(
        self,
        products_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Entraîne tous les sous-modèles.
        
        Args:
            products_df: DataFrame des produits
            interactions_df: DataFrame des interactions (user_id, product_id, rating)
            users_df: DataFrame des utilisateurs (optionnel)
        """
        logger.info("🏋️ Entraînement du système de recommandation...")
        
        self.products_df = products_df.copy()
        self.users_df = users_df.copy() if users_df is not None else None
        
        # Facteur 1 : Collaborative Filtering
        logger.info("   📌 Facteur 1 : Collaborative Filtering (SVD)")
        if len(interactions_df) > 0:
            self.collaborative.fit(interactions_df)
        
        # Facteur 2 : Content-Based
        logger.info("   📌 Facteur 2 : Content-Based Filtering")
        self.content_based.fit(products_df)
        
        self.is_fitted = True
        logger.info("✅ Système de recommandation entraîné")
    
    def recommander(
        self,
        user_id: str,
        n: int = 10,
        produit_consulte: Optional[str] = None,
        user_location: Optional[Tuple[float, float]] = None,
        user_budget: Optional[Tuple[float, float]] = None,
        historique_prix: Optional[List[float]] = None,
        exclure_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Génère les top-N recommandations pour un utilisateur.
        
        Args:
            user_id: Identifiant de l'utilisateur
            n: Nombre de recommandations
            produit_consulte: ID du dernier produit consulté (pour similarité)
            user_location: Coordonnées GPS (lat, lon)
            user_budget: Fourchette de budget (min, max)
            historique_prix: Liste des prix consultés (pour estimer le budget)
            exclure_ids: IDs de produits à exclure
        
        Returns:
            Liste de recommandations avec scores détaillés
        """
        if self.products_df is None:
            return []
        
        exclure = set(exclure_ids) if exclure_ids else set()
        
        # Estimer le budget si non fourni
        if user_budget is None and historique_prix:
            user_budget = self.price_filter.estimer_budget(historique_prix)
        elif user_budget is None:
            user_budget = (0, 999999)
        
        recommendations = []
        
        for _, product in self.products_df.iterrows():
            pid = str(product["id"])
            
            if pid in exclure:
                continue
            
            # Facteur 1 : Score historique (Collaborative)
            score_hist = self.collaborative.predict(user_id, pid)
            
            # Facteur 2 : Score similarité (Content-Based)
            if produit_consulte:
                score_sim = self.content_based.score(produit_consulte, pid)
            else:
                score_sim = 0.5  # Score neutre
            
            # Facteur 3 : Score géographique
            if user_location and "latitude" in product and "longitude" in product:
                try:
                    score_geo = self.geo_filter.score(
                        user_location[0], user_location[1],
                        float(product["latitude"]),
                        float(product["longitude"])
                    )
                except (ValueError, TypeError):
                    score_geo = 0.5
            else:
                score_geo = 0.5
            
            # Facteur 4 : Score prix
            if "prix" in product:
                has_promo = product.get("promotion", False)
                discount = product.get("discount_percent", 0)
                score_prix = self.price_filter.score(
                    float(product["prix"]),
                    user_budget[0], user_budget[1],
                    has_promo, discount
                )
            else:
                score_prix = 0.5
            
            # Score final pondéré
            score_final = (
                self.poids["historique"] * score_hist +
                self.poids["similarite"] * score_sim +
                self.poids["geographique"] * score_geo +
                self.poids["prix"] * score_prix
            )
            
            recommendations.append({
                "product_id": pid,
                "nom": product.get("nom", "Produit"),
                "categorie": product.get("categorie", ""),
                "prix": float(product.get("prix", 0)),
                "score_final": float(score_final),
                "facteurs": {
                    "historique": float(score_hist),
                    "similarite": float(score_sim),
                    "geographique": float(score_geo),
                    "prix": float(score_prix)
                }
            })
        
        # Trier par score final décroissant
        recommendations.sort(key=lambda x: x["score_final"], reverse=True)
        
        return recommendations[:n]
    
    def produits_similaires(self, product_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retourne les produits similaires (content-based pur).
        
        Args:
            product_id: ID du produit source
            n: Nombre de produits similaires
        
        Returns:
            Liste de produits similaires avec scores
        """
        similar = self.content_based.get_similar(product_id, n)
        
        results = []
        for pid, score in similar:
            product = self.products_df[self.products_df["id"].astype(str) == pid]
            if len(product) > 0:
                product = product.iloc[0]
                results.append({
                    "product_id": pid,
                    "nom": product.get("nom", "Produit"),
                    "categorie": product.get("categorie", ""),
                    "prix": float(product.get("prix", 0)),
                    "score_similarite": float(score)
                })
        
        return results
    
    def save(self, path: Optional[Path] = None) -> None:
        """Sauvegarde le système complet."""
        if path is None:
            path = MODELS_DIR / "recommender.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"💾 Recommandeur sauvegardé : {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HybridRecommender":
        """Charge un système sauvegardé."""
        if path is None:
            path = MODELS_DIR / "recommender.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================
# MÉTRIQUES D'ÉVALUATION
# ============================================
def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """Precision@K : proportion de produits pertinents dans le top-K."""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """Recall@K : proportion de produits pertinents retrouvés dans le top-K."""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """NDCG@K : mesure la qualité du classement des recommandations."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)
    
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def coverage(all_recommended: List[List[str]], total_products: int) -> float:
    """Coverage : diversité des recommandations (% produits couverts)."""
    unique_recommended = set()
    for recs in all_recommended:
        unique_recommended.update(recs)
    return len(unique_recommended) / total_products if total_products > 0 else 0.0


def evaluer_recommandations(
    recommender: HybridRecommender,
    test_interactions: pd.DataFrame,
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Évaluation complète du système de recommandation.
    
    Args:
        recommender: Système de recommandation entraîné
        test_interactions: Interactions de test
        k_values: Valeurs de K pour Precision/Recall@K
    
    Returns:
        Dictionnaire de métriques
    """
    logger.info("📊 Évaluation du système de recommandation...")
    
    metrics = {}
    all_recommendations = []
    
    users = test_interactions["user_id"].unique()
    
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id in users:
            # Produits pertinents (achetés/likés dans le test)
            relevant = test_interactions[
                test_interactions["user_id"] == user_id
            ]["product_id"].astype(str).tolist()
            
            if not relevant:
                continue
            
            # Recommandations
            recs = recommender.recommander(str(user_id), n=max(k_values))
            recommended_ids = [r["product_id"] for r in recs]
            all_recommendations.append(recommended_ids[:k])
            
            precisions.append(precision_at_k(recommended_ids, relevant, k))
            recalls.append(recall_at_k(recommended_ids, relevant, k))
            ndcgs.append(ndcg_at_k(recommended_ids, relevant, k))
        
        metrics[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    
    # Coverage
    total_products = len(recommender.products_df) if recommender.products_df is not None else 0
    metrics["coverage"] = coverage(all_recommendations, total_products)
    
    logger.info("📊 Métriques de recommandation :")
    for key, value in metrics.items():
        logger.info(f"   {key:20s} : {value:.4f}")
    
    return metrics


# ============================================
# DONNÉES SYNTHÉTIQUES POUR TEST/DÉMO
# ============================================
def generer_donnees_demo(
    nb_products: int = 100,
    nb_users: int = 50,
    nb_interactions: int = 500
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Génère des données synthétiques pour tester le système.
    
    Args:
        nb_products: Nombre de produits
        nb_users: Nombre d'utilisateurs
        nb_interactions: Nombre d'interactions
    
    Returns:
        Tuple (products_df, users_df, interactions_df)
    """
    np.random.seed(42)
    
    categories = [
        "Électronique", "Vêtements", "Maison", "Sport",
        "Alimentation", "Beauté", "Jouets", "Livres",
        "Auto", "Jardin"
    ]
    marques = [
        "Samsung", "Nike", "IKEA", "Adidas", "Apple",
        "Sony", "Zara", "Decathlon", "Bosch", "L'Oréal"
    ]
    
    # Produits
    products = {
        "id": [f"P{i:04d}" for i in range(nb_products)],
        "nom": [f"Produit {i}" for i in range(nb_products)],
        "categorie": np.random.choice(categories, nb_products),
        "prix": np.round(np.random.uniform(5, 500, nb_products), 2),
        "marque": np.random.choice(marques, nb_products),
        "stock": np.random.randint(0, 100, nb_products),
        "note_moyenne": np.round(np.random.uniform(1, 5, nb_products), 1),
        "latitude": np.round(np.random.uniform(48.0, 49.0, nb_products), 4),
        "longitude": np.round(np.random.uniform(2.0, 3.0, nb_products), 4),
        "promotion": np.random.choice([True, False], nb_products, p=[0.2, 0.8]),
        "discount_percent": np.random.choice([0, 5, 10, 15, 20, 30], nb_products)
    }
    products_df = pd.DataFrame(products)
    
    # Utilisateurs
    users = {
        "id": [f"U{i:04d}" for i in range(nb_users)],
        "nom": [f"Utilisateur {i}" for i in range(nb_users)],
        "latitude": np.round(np.random.uniform(48.5, 48.9, nb_users), 4),
        "longitude": np.round(np.random.uniform(2.2, 2.5, nb_users), 4),
        "budget_moyen": np.round(np.random.uniform(20, 300, nb_users), 2)
    }
    users_df = pd.DataFrame(users)
    
    # Interactions
    interactions = {
        "user_id": np.random.choice(users_df["id"], nb_interactions),
        "product_id": np.random.choice(products_df["id"], nb_interactions),
        "rating": np.round(np.random.uniform(1, 5, nb_interactions), 1),
        "type": np.random.choice(["vue", "like", "achat"], nb_interactions, p=[0.6, 0.25, 0.15])
    }
    interactions_df = pd.DataFrame(interactions)
    
    logger.info(
        f"✅ Données de démo : {nb_products} produits, "
        f"{nb_users} utilisateurs, {nb_interactions} interactions"
    )
    
    return products_df, users_df, interactions_df


# ============================================
# Point d'entrée principal
# ============================================
def main():
    """
    Pipeline complet du système de recommandation.
    
    1. Générer/charger les données
    2. Entraîner le système hybride
    3. Générer des recommandations
    4. Évaluer les métriques
    5. Sauvegarder le modèle
    """
    logger.info("=" * 60)
    logger.info("🛍️  ECommerce-IA — Système de Recommandation Hybride")
    logger.info("=" * 60)
    
    # Générer les données de démo
    products_df, users_df, interactions_df = generer_donnees_demo()
    
    # Séparer train/test interactions (80/20)
    n_train = int(len(interactions_df) * 0.8)
    train_inter = interactions_df.iloc[:n_train]
    test_inter = interactions_df.iloc[n_train:]
    
    # Créer et entraîner le recommandeur
    recommender = HybridRecommender()
    recommender.fit(products_df, train_inter, users_df)
    
    # Exemple de recommandation
    logger.info("")
    logger.info("📋 Exemple de recommandation pour U0001 :")
    logger.info("-" * 40)
    
    recs = recommender.recommander(
        user_id="U0001",
        n=5,
        user_location=(48.8, 2.3),
        user_budget=(20, 200)
    )
    
    for i, rec in enumerate(recs, 1):
        logger.info(
            f"   {i}. {rec['nom']:20s} | "
            f"Cat: {rec['categorie']:15s} | "
            f"Prix: {rec['prix']:7.2f}€ | "
            f"Score: {rec['score_final']:.3f}"
        )
        f = rec["facteurs"]
        logger.info(
            f"      → Hist: {f['historique']:.2f} | "
            f"Sim: {f['similarite']:.2f} | "
            f"Géo: {f['geographique']:.2f} | "
            f"Prix: {f['prix']:.2f}"
        )
    
    # Produits similaires
    logger.info("")
    logger.info("📋 Produits similaires à P0001 :")
    logger.info("-" * 40)
    similaires = recommender.produits_similaires("P0001", n=5)
    for i, s in enumerate(similaires, 1):
        logger.info(
            f"   {i}. {s['nom']:20s} | "
            f"Sim: {s['score_similarite']:.3f}"
        )
    
    # Évaluation
    logger.info("")
    metrics = evaluer_recommandations(recommender, test_inter)
    
    # Sauvegarder
    recommender.save()
    
    # Sauvegarder les métriques
    metrics_path = MODELS_DIR / "recommendation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"💾 Métriques sauvegardées : {metrics_path}")
    
    logger.info("")
    logger.info("✅ Système de recommandation prêt !")


if __name__ == "__main__":
    main()
