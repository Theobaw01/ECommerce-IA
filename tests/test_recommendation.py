"""
============================================
ECommerce-IA — Tests Unitaires : Recommandation Hybride
============================================
Tests pour le système de recommandation à 4 facteurs :
- CollaborativeFilter (SVD)
- ContentBasedFilter (TF-IDF + cosine)
- GeoFilter (Haversine)
- PriceFilter (budget)
- HybridRecommender (combinaison pondérée)

Auteur : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import pytest
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommendation import (
    CollaborativeFilter,
    ContentBasedFilter,
    GeoFilter,
    PriceFilter,
    HybridRecommender,
    POIDS,
    DISTANCE_MAX_KM,
)


# ============================================
# Fixtures
# ============================================
@pytest.fixture
def sample_interactions():
    """DataFrame d'interactions utilisateur-produit."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5,
                     1, 2, 3, 4, 5, 1, 2, 3],
        "product_id": [101, 102, 103, 101, 104, 102, 103, 105,
                        104, 105, 101, 103, 104, 105, 101, 103, 102,
                        105, 103, 104],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0,
                    4.0, 3.0, 5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0,
                    4.0, 5.0, 3.0],
    })


@pytest.fixture
def sample_products():
    """DataFrame de produits."""
    return pd.DataFrame({
        "product_id": [101, 102, 103, 104, 105],
        "nom": ["Chaussure Sport", "Robe Été", "Sac Cuir", "Montre Luxe", "T-Shirt Coton"],
        "description": [
            "Chaussure de sport confortable pour la course",
            "Robe d'été légère en coton fleuri",
            "Sac en cuir véritable italien marron",
            "Montre de luxe automatique en acier",
            "T-shirt en coton bio col rond blanc",
        ],
        "categorie": ["Sport", "Vêtement", "Accessoire", "Bijou", "Vêtement"],
        "prix": [89.0, 45.0, 150.0, 350.0, 25.0],
        "latitude": [48.85, 48.86, 43.60, 45.76, 48.85],
        "longitude": [2.35, 2.34, 1.44, 4.83, 2.35],
    })


@pytest.fixture
def collab_filter():
    return CollaborativeFilter(n_factors=10, n_epochs=10)


@pytest.fixture
def content_filter():
    return ContentBasedFilter()


@pytest.fixture
def geo_filter():
    return GeoFilter()


@pytest.fixture
def price_filter():
    return PriceFilter()


# ============================================
# Tests : Configuration des poids
# ============================================
class TestWeights:
    """Tests des poids du système hybride."""

    def test_weights_sum_to_one(self):
        total = sum(POIDS.values())
        assert abs(total - 1.0) < 0.01

    def test_historique_weight(self):
        assert POIDS["historique"] == 0.40

    def test_similarite_weight(self):
        assert POIDS["similarite"] == 0.30

    def test_geographique_weight(self):
        assert POIDS["geographique"] == 0.15

    def test_prix_weight(self):
        assert POIDS["prix"] == 0.15


# ============================================
# Tests : CollaborativeFilter (SVD)
# ============================================
class TestCollaborativeFilter:
    """Tests du filtrage collaboratif SVD."""

    def test_init(self, collab_filter):
        assert collab_filter.n_factors == 10
        assert collab_filter.n_epochs == 10
        assert collab_filter.is_fitted is False

    def test_fit(self, collab_filter, sample_interactions):
        collab_filter.fit(sample_interactions)
        # Soit SVD (model != None), soit fallback (is_fitted=True)
        assert collab_filter.is_fitted is True

    def test_predict_after_fit(self, collab_filter, sample_interactions):
        collab_filter.fit(sample_interactions)
        score = collab_filter.predict(user_id=1, product_id=101)
        assert isinstance(score, float)

    def test_predict_scores_in_range(self, collab_filter, sample_interactions):
        collab_filter.fit(sample_interactions)
        score = collab_filter.predict(user_id=1, product_id=101)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # normalized 0-1

    def test_predict_before_fit_returns_default(self, collab_filter):
        score = collab_filter.predict(user_id=1, product_id=101)
        assert isinstance(score, float)
        assert score == 0.5  # default when not fitted

    def test_get_top_n(self, collab_filter, sample_interactions):
        collab_filter.fit(sample_interactions)
        top = collab_filter.get_top_n(
            user_id="1",
            product_ids=["101", "102", "103", "104", "105"],
            n=3
        )
        assert isinstance(top, list)
        assert len(top) <= 3


# ============================================
# Tests : ContentBasedFilter
# ============================================
class TestContentBasedFilter:
    """Tests du filtrage basé sur le contenu."""

    def test_init(self, content_filter):
        assert content_filter.is_fitted is False

    def test_fit(self, content_filter, sample_products):
        # ContentBasedFilter needs 'id' column
        df = sample_products.rename(columns={"product_id": "id"})
        content_filter.fit(df)
        assert content_filter.is_fitted is True

    def test_similar_products(self, content_filter, sample_products):
        df = sample_products.rename(columns={"product_id": "id"})
        content_filter.fit(df)
        similar = content_filter.get_similar(product_id="101", n=3)
        assert isinstance(similar, list)
        assert len(similar) <= 3

    def test_similarity_scores_normalized(self, content_filter, sample_products):
        df = sample_products.rename(columns={"product_id": "id"})
        content_filter.fit(df)
        similar = content_filter.get_similar(product_id="101", n=5)
        for pid, score in similar:
            assert 0 <= score <= 1


# ============================================
# Tests : GeoFilter (Haversine)
# ============================================
class TestGeoFilter:
    """Tests du filtre géographique."""

    def test_haversine_distance(self, geo_filter):
        # Paris → Paris = 0 km
        dist = geo_filter.haversine(48.85, 2.35, 48.85, 2.35)
        assert dist == 0.0

    def test_haversine_known_distance(self, geo_filter):
        # Paris → Lyon ≈ 392 km
        dist = geo_filter.haversine(48.8566, 2.3522, 45.7640, 4.8357)
        assert 380 < dist < 410

    def test_haversine_symmetry(self, geo_filter):
        d1 = geo_filter.haversine(48.85, 2.35, 43.60, 1.44)
        d2 = geo_filter.haversine(43.60, 1.44, 48.85, 2.35)
        assert abs(d1 - d2) < 0.01

    def test_geo_score_close(self, geo_filter):
        # Même ville → score élevé
        score = geo_filter.score(48.85, 2.35, 48.86, 2.34)
        assert score > 0.9

    def test_geo_score_far(self, geo_filter):
        # Paris → Toulouse → score plus bas
        score = geo_filter.score(48.85, 2.35, 43.60, 1.44)
        assert score < 0.5


# ============================================
# Tests : PriceFilter
# ============================================
class TestPriceFilter:
    """Tests du filtre prix."""

    def test_exact_budget(self, price_filter):
        score = price_filter.score(100.0, 80.0, 120.0)
        assert score > 0.8

    def test_within_budget(self, price_filter):
        score = price_filter.score(80.0, 50.0, 150.0)
        assert score > 0.5

    def test_over_budget(self, price_filter):
        score = price_filter.score(300.0, 50.0, 100.0)
        assert score < 0.7

    def test_zero_budget(self, price_filter):
        score = price_filter.score(50.0, 0.0, 0.0)
        assert isinstance(score, float)


# ============================================
# Tests : HybridRecommender
# ============================================
class TestHybridRecommender:
    """Tests du système hybride complet."""

    def test_init(self):
        recommender = HybridRecommender()
        assert recommender is not None

    def test_hybrid_score_structure(self):
        recommender = HybridRecommender()
        # Le score hybride doit combiner les 4 facteurs
        assert hasattr(recommender, 'recommend') or hasattr(recommender, 'recommander')


# ============================================
# Tests : Métriques d'évaluation
# ============================================
class TestMetrics:
    """Tests des métriques d'évaluation."""

    def test_precision_at_k_perfect(self):
        """Precision@K parfait : toutes les reco sont pertinentes."""
        from src.recommendation import precision_at_k
        recommended = ["1", "2", "3", "4", "5"]
        relevant = ["1", "2", "3", "4", "5"]
        assert precision_at_k(recommended, relevant, k=5) == 1.0

    def test_precision_at_k_zero(self):
        """Precision@K nul : aucune reco n'est pertinente."""
        from src.recommendation import precision_at_k
        recommended = ["6", "7", "8", "9", "10"]
        relevant = ["1", "2", "3", "4", "5"]
        assert precision_at_k(recommended, relevant, k=5) == 0.0

    def test_recall_at_k(self):
        """Recall@K : fraction des pertinents retrouvés."""
        from src.recommendation import recall_at_k
        recommended = ["1", "2", "6", "7", "8"]
        relevant = ["1", "2", "3", "4", "5"]
        assert recall_at_k(recommended, relevant, k=5) == 0.4

    def test_ndcg_at_k_perfect(self):
        """NDCG parfait : ordre idéal."""
        from src.recommendation import ndcg_at_k
        recommended = ["1", "2", "3", "4", "5"]
        relevant = ["1", "2", "3", "4", "5"]
        score = ndcg_at_k(recommended, relevant, k=5)
        assert score == 1.0

    def test_coverage(self):
        """Coverage : diversité."""
        from src.recommendation import coverage
        all_recs = [["1", "2", "3"], ["2", "3", "4"], ["4", "5", "6"]]
        cov = coverage(all_recs, total_products=10)
        assert cov == 0.6
