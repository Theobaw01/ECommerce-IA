"""
============================================
ECommerce-IA — Tests Unitaires : Moteur NLP
============================================
Tests exhaustifs pour le pipeline NLP :
- Prétraitement (tokenisation, lemmatisation, stopwords)
- Détection d'intent (15 classes)
- NER (10 types d'entités)
- Analyse de sentiment (lexique + négation)
- Extraction de mots-clés (TF-IDF)
- Routing intelligent

Auteur : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import pytest
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp_engine import NLPEngine, STOPWORDS_FR, INTENT_PATTERNS, NER_PATTERNS


# ============================================
# Fixture : instance NLP
# ============================================
@pytest.fixture(scope="module")
def nlp():
    """Instance NLP partagée par tous les tests du module."""
    return NLPEngine()


# ============================================
# Tests : Configuration & Initialisation
# ============================================
class TestNLPInit:
    """Tests d'initialisation du moteur NLP."""

    def test_instance_creation(self, nlp):
        assert isinstance(nlp, NLPEngine)

    def test_intent_patterns_loaded(self, nlp):
        assert len(nlp.intent_patterns) == 15

    def test_ner_patterns_loaded(self, nlp):
        assert len(nlp.ner_patterns) == 10

    def test_stopwords_loaded(self, nlp):
        assert len(nlp.stopwords) > 100

    def test_stopwords_contains_common_words(self):
        for word in ["le", "la", "de", "et", "en", "est"]:
            assert word in STOPWORDS_FR

    def test_all_intents_present(self):
        expected = {
            "recherche_produit", "suivi_commande", "retour_remboursement",
            "livraison", "paiement", "recommandation", "compte", "stock",
            "promotion", "garantie", "vendeur", "salutation", "remerciement",
            "plainte", "question_generale",
        }
        assert set(INTENT_PATTERNS.keys()) == expected

    def test_all_ner_types_present(self):
        expected = {
            "ORDER_ID", "PRODUCT", "CATEGORY", "PRICE", "COLOR",
            "SIZE", "CITY", "EMAIL", "PHONE", "DATE",
        }
        assert set(NER_PATTERNS.keys()) == expected


# ============================================
# Tests : Prétraitement
# ============================================
class TestPreprocessing:
    """Tests du pipeline de prétraitement."""

    def test_basic_preprocessing(self, nlp):
        result = nlp.preprocesser("Bonjour, je cherche des chaussures !")
        assert "original" in result
        assert "normalise" in result
        assert "tokens" in result
        assert "tokens_filtres" in result
        assert "lemmes" in result

    def test_lowercasing(self, nlp):
        result = nlp.preprocesser("BONJOUR COMMENT ALLEZ-VOUS")
        assert result["normalise"] == "bonjour comment allez-vous"

    def test_tokenization(self, nlp):
        result = nlp.preprocesser("Je cherche un produit")
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) > 0

    def test_stopwords_removal(self, nlp):
        result = nlp.preprocesser("Je suis dans le magasin")
        # "magasin" devrait rester, stopwords supprimés
        filtered = result["tokens_filtres"]
        assert "le" not in filtered
        assert "je" not in filtered

    def test_lemmatization_plural(self, nlp):
        result = nlp.preprocesser("chaussures produits")
        lemmes = result["lemmes"]
        assert "chaussure" in lemmes or "produit" in lemmes

    def test_empty_input(self, nlp):
        result = nlp.preprocesser("")
        assert result["tokens"] == []

    def test_unicode_normalization(self, nlp):
        result = nlp.preprocesser("L'été «magnifique»")
        assert isinstance(result["normalise"], str)

    def test_whitespace_normalization(self, nlp):
        result = nlp.preprocesser("  trop   d'espaces   ici  ")
        assert "  " not in result["normalise"]


# ============================================
# Tests : Détection d'Intent
# ============================================
class TestIntentDetection:
    """Tests de détection d'intention (15 classes)."""

    @pytest.mark.parametrize("text,expected_intent", [
        ("Je cherche des chaussures Nike", "recherche_produit"),
        ("Où en est ma commande CMD-2024-001 ?", "suivi_commande"),
        ("Je veux retourner ce produit défectueux", "retour_remboursement"),
        ("Quels sont vos délais de livraison ?", "livraison"),
        ("Je veux payer par carte bancaire Visa", "paiement"),
        ("Recommandez-moi un cadeau pour Noël", "recommandation"),
        ("Comment créer mon compte ?", "compte"),
        ("Ce produit est-il en stock ?", "stock"),
        ("Y a-t-il des promotions en cours ?", "promotion"),
        ("Le produit est sous garantie ?", "garantie"),
        ("Je voudrais devenir vendeur", "vendeur"),
        ("Bonjour !", "salutation"),
        ("Merci beaucoup !", "remerciement"),
        ("C'est une arnaque, je suis furieux !", "plainte"),
    ])
    def test_intent_detection(self, nlp, text, expected_intent):
        result = nlp.detecter_intent(text)
        assert result["intent"] == expected_intent, (
            f"Pour '{text}': attendu '{expected_intent}', obtenu '{result['intent']}'"
        )

    def test_intent_output_structure(self, nlp):
        result = nlp.detecter_intent("Bonjour")
        assert "intent" in result
        assert "confidence" in result
        assert "all_intents" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1

    def test_confidence_top1_highest(self, nlp):
        result = nlp.detecter_intent("Je cherche des chaussures rouges à moins de 50€")
        all_intents = result["all_intents"]
        assert all_intents[0]["confidence"] >= all_intents[1]["confidence"]

    def test_top_k_returns_multiple(self, nlp):
        result = nlp.detecter_intent("Je cherche un produit", top_k=5)
        assert len(result["all_intents"]) >= 3

    def test_intent_with_empty_text(self, nlp):
        result = nlp.detecter_intent("")
        assert "intent" in result


# ============================================
# Tests : NER (Named Entity Recognition)
# ============================================
class TestNER:
    """Tests d'extraction d'entités nommées (10 types)."""

    def test_extract_order_id(self, nlp):
        entities = nlp.extraire_entites("Ma commande CMD-2024-001 n'est pas arrivée")
        types = [e["type"] for e in entities]
        assert "ORDER_ID" in types

    def test_extract_price(self, nlp):
        entities = nlp.extraire_entites("Je cherche un produit à moins de 100€")
        types = [e["type"] for e in entities]
        values = [e["value"] for e in entities if e["type"] == "PRICE"]
        assert "PRICE" in types
        assert any("100" in v for v in values)

    def test_extract_color(self, nlp):
        entities = nlp.extraire_entites("Je veux un sac noir et un pantalon bleu")
        colors = [e["value"] for e in entities if e["type"] == "COLOR"]
        assert "noir" in colors
        assert "bleu" in colors

    def test_extract_size(self, nlp):
        entities = nlp.extraire_entites("Disponible en taille 42 ?")
        types = [e["type"] for e in entities]
        assert "SIZE" in types

    def test_extract_city(self, nlp):
        entities = nlp.extraire_entites("Livraison à Dakar possible ?")
        cities = [e["value"] for e in entities if e["type"] == "CITY"]
        assert "Dakar" in cities

    def test_extract_email(self, nlp):
        entities = nlp.extraire_entites("Contactez-moi à test@example.com")
        emails = [e["value"] for e in entities if e["type"] == "EMAIL"]
        assert "test@example.com" in emails

    def test_extract_phone(self, nlp):
        entities = nlp.extraire_entites("Appelez-moi au 06 12 34 56 78")
        types = [e["type"] for e in entities]
        assert "PHONE" in types

    def test_extract_date(self, nlp):
        entities = nlp.extraire_entites("Commande passée le 15/03/2024")
        types = [e["type"] for e in entities]
        assert "DATE" in types

    def test_multiple_entities(self, nlp):
        text = "Je cherche des chaussures rouges taille 42 à moins de 100€ à Dakar"
        entities = nlp.extraire_entites(text)
        types = {e["type"] for e in entities}
        assert len(types) >= 3  # COLOR, SIZE, PRICE, CITY

    def test_no_entities(self, nlp):
        entities = nlp.extraire_entites("Bonjour")
        assert isinstance(entities, list)

    def test_entity_structure(self, nlp):
        entities = nlp.extraire_entites("commande CMD-2024-001")
        if entities:
            e = entities[0]
            assert "type" in e
            assert "value" in e
            assert "start" in e
            assert "end" in e

    def test_entities_sorted_by_position(self, nlp):
        text = "chaussures rouges taille 42 à 50€ à Dakar"
        entities = nlp.extraire_entites(text)
        if len(entities) >= 2:
            positions = [e["start"] for e in entities]
            assert positions == sorted(positions)


# ============================================
# Tests : Analyse de Sentiment
# ============================================
class TestSentimentAnalysis:
    """Tests d'analyse de sentiment (lexique + négation)."""

    def test_positive_sentiment(self, nlp):
        result = nlp.analyser_sentiment("Excellent produit, super qualité, merci !")
        assert result["label"] == "positif"
        assert result["score"] > 0

    def test_negative_sentiment(self, nlp):
        result = nlp.analyser_sentiment("Horrible, produit cassé et service nul !")
        assert result["label"] == "négatif"
        assert result["score"] < 0

    def test_neutral_sentiment(self, nlp):
        result = nlp.analyser_sentiment("Quels sont vos horaires d'ouverture ?")
        assert result["label"] == "neutre"

    def test_negation_handling(self, nlp):
        result_pos = nlp.analyser_sentiment("C'est excellent")
        result_neg = nlp.analyser_sentiment("Ce n'est pas excellent")
        # La négation devrait diminuer le score
        assert result_neg["score"] < result_pos["score"]

    def test_sentiment_score_range(self, nlp):
        result = nlp.analyser_sentiment("Un produit banal")
        assert -1 <= result["score"] <= 1

    def test_sentiment_output_structure(self, nlp):
        result = nlp.analyser_sentiment("Bonjour")
        assert "label" in result
        assert "score" in result
        assert "details" in result
        assert result["label"] in {"positif", "négatif", "neutre"}

    def test_strong_negative(self, nlp):
        result = nlp.analyser_sentiment(
            "C'est une arnaque scandaleux, je suis furieux et mécontent !"
        )
        assert result["label"] == "négatif"
        assert result["score"] < -0.3

    def test_mixed_sentiment(self, nlp):
        result = nlp.analyser_sentiment("Le produit est bon mais la livraison était lente")
        # Score should be close to neutral
        assert isinstance(result["score"], float)


# ============================================
# Tests : Extraction de Mots-clés
# ============================================
class TestKeywordExtraction:
    """Tests d'extraction de mots-clés (TF-IDF simplifié)."""

    def test_keyword_extraction(self, nlp):
        keywords = nlp.extraire_mots_cles(
            "Je cherche des chaussures de sport Nike rouges confortables"
        )
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_keyword_structure(self, nlp):
        keywords = nlp.extraire_mots_cles("Un excellent produit de qualité")
        if keywords:
            kw = keywords[0]
            assert "mot" in kw
            assert "score" in kw

    def test_keywords_sorted_by_score(self, nlp):
        keywords = nlp.extraire_mots_cles(
            "Les chaussures Nike sont les meilleures chaussures du marché"
        )
        if len(keywords) >= 2:
            scores = [k["score"] for k in keywords]
            assert scores == sorted(scores, reverse=True)

    def test_empty_keywords(self, nlp):
        keywords = nlp.extraire_mots_cles("")
        assert isinstance(keywords, list)


# ============================================
# Tests : Analyse complète (pipeline)
# ============================================
class TestFullAnalysis:
    """Tests du pipeline NLP complet."""

    def test_full_analysis(self, nlp):
        result = nlp.analyser("Je cherche des chaussures Nike rouges à moins de 100€")
        assert "texte" in result
        assert "preprocessed" in result
        assert "intent" in result
        assert "entities" in result
        assert "sentiment" in result
        assert "keywords" in result

    def test_empty_analysis(self, nlp):
        result = nlp.analyser("")
        assert result["intent"]["intent"] == "question_generale"
        assert result["intent"]["confidence"] == 0.0

    def test_none_analysis(self, nlp):
        result = nlp.analyser(None)
        assert result["intent"]["intent"] == "question_generale"


# ============================================
# Tests : Routing Intelligent
# ============================================
class TestRouting:
    """Tests du routing intelligent des requêtes."""

    def test_routing_search(self, nlp):
        route = nlp.router_requete("Je cherche un produit à acheter")
        assert route["module"] == "recherche"
        assert route["action"] == "search_products"

    def test_routing_recommendation(self, nlp):
        route = nlp.router_requete("Recommandez-moi un cadeau de Noël")
        assert route["module"] == "recommandation"

    def test_routing_order_tracking(self, nlp):
        route = nlp.router_requete("Où en est ma commande CMD-2024-001 ?")
        assert route["module"] == "suivi"

    def test_routing_greeting(self, nlp):
        route = nlp.router_requete("Bonjour !")
        assert route["module"] == "direct"

    def test_routing_complaint_escalation(self, nlp):
        route = nlp.router_requete(
            "C'est une arnaque horrible, service scandaleux et inacceptable !"
        )
        assert route["escalade"] is True

    def test_routing_rag_required(self, nlp):
        route = nlp.router_requete("Quels sont vos délais de livraison ?")
        assert route["requires_rag"] is True

    def test_routing_output_structure(self, nlp):
        route = nlp.router_requete("Bonjour")
        assert "module" in route
        assert "action" in route
        assert "nlp_analysis" in route
        assert "requires_rag" in route
        assert "escalade" in route


# ============================================
# Tests : Singleton
# ============================================
class TestSingleton:
    """Tests du pattern singleton."""

    def test_singleton_returns_same_instance(self):
        from src.nlp_engine import get_nlp_engine
        engine1 = get_nlp_engine()
        engine2 = get_nlp_engine()
        assert engine1 is engine2


# ============================================
# Tests de robustesse
# ============================================
class TestRobustness:
    """Tests de robustesse et cas limites."""

    def test_very_long_text(self, nlp):
        text = "chaussure " * 500
        result = nlp.analyser(text)
        assert result is not None

    def test_special_characters(self, nlp):
        result = nlp.analyser("@#$%^&*()!?")
        assert result is not None

    def test_numbers_only(self, nlp):
        result = nlp.analyser("12345 67890")
        assert result is not None

    def test_unicode_text(self, nlp):
        result = nlp.analyser("Café résumé naïf Noël")
        assert result is not None

    def test_mixed_language(self, nlp):
        result = nlp.analyser("Hello bonjour I need des chaussures")
        assert result is not None
