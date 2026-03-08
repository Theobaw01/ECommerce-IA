"""
============================================
ECommerce-IA — Moteur NLP (Natural Language Processing)
============================================
Pipeline NLP complet pour l'analyse sémantique des requêtes
utilisateur dans le chatbot e-commerce.

Architecture NLP :
- Prétraitement : tokenisation, lemmatisation, stopwords
- Détection d'intent : classification de l'intention utilisateur
- NER (Named Entity Recognition) : extraction d'entités
- Analyse de sentiment : positif / négatif / neutre
- Extraction de mots-clés : TF-IDF sur la requête

Intents détectés :
- recherche_produit, suivi_commande, retour_remboursement,
  livraison, paiement, recommandation, compte, stock,
  promotion, garantie, vendeur, salutation, remerciement,
  plainte, question_generale

Entités extraites :
- PRODUCT, ORDER_ID, CATEGORY, BRAND, PRICE, COLOR,
  SIZE, CITY, EMAIL, PHONE, DATE

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

# ============================================
# Configuration
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# STOPWORDS FRANÇAIS
# ============================================
STOPWORDS_FR = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "est", "que", "qui", "dans", "pour", "pas", "sur", "au", "aux",
    "avec", "ce", "ces", "son", "sa", "ses", "se", "il", "elle",
    "nous", "vous", "ils", "elles", "mon", "ma", "mes", "ton", "ta",
    "tes", "ne", "plus", "par", "été", "être", "avoir", "fait",
    "comme", "ou", "mais", "donc", "car", "ni", "si", "je", "tu",
    "on", "me", "te", "lui", "leur", "y", "dont", "où", "tout",
    "tous", "toute", "toutes", "autre", "autres", "même", "aussi",
    "très", "bien", "peu", "trop", "ici", "là", "cela", "cette",
    "cet", "quel", "quelle", "quels", "quelles", "comment", "quand",
    "pourquoi", "combien", "chaque", "quelque", "quelques", "aucun",
    "aucune", "encore", "entre", "vers", "chez", "sans", "sous",
    "après", "avant", "contre", "depuis", "pendant", "puis",
    "alors", "ainsi", "votre", "vos", "notre", "nos", "à", "d",
    "l", "s", "n", "c", "j", "m", "t", "qu", "ai", "a", "ont",
    "été", "suis", "es", "sommes", "êtes", "sont", "avons", "avez",
}


# ============================================
# PATTERNS D'INTENTS (règles + scoring)
# ============================================
INTENT_PATTERNS = {
    "recherche_produit": {
        "keywords": [
            "cherche", "recherche", "trouver", "produit", "acheter",
            "article", "disponible", "catalogue", "achète", "veux",
            "besoin", "chercher", "similaire", "photo", "image",
            "ressemble", "genre", "type", "modèle", "référence",
        ],
        "patterns": [
            r"je\s+(?:cherche|veux|voudrais|souhaite)\s+(?:un|une|des|le|la)",
            r"(?:avez|proposez).vous\s+(?:des|un|une)",
            r"(?:où|ou)\s+(?:trouver|acheter)",
            r"(?:montrez|affichez|conseillez).moi",
        ],
        "priority": 1.0,
    },
    "suivi_commande": {
        "keywords": [
            "commande", "suivi", "suivre", "colis", "livré", "expédié",
            "tracking", "numéro", "statut", "état", "reçu", "arrivée",
            "expédition", "envoi", "envoyé", "reception",
        ],
        "patterns": [
            r"(?:suivi|suivre|état|statut)\s+(?:de\s+)?(?:ma|la|une)\s+commande",
            r"commande\s+(?:n[°o]?\s*|numéro\s*|#\s*)?[\w\-]+",
            r"où\s+en\s+est\s+(?:ma|la)\s+commande",
            r"(?:reçu|arrivé|livré)\s+(?:ma|la|mon)",
        ],
        "priority": 1.2,
    },
    "retour_remboursement": {
        "keywords": [
            "retour", "retourner", "rembourser", "remboursement",
            "échanger", "échange", "renvoyer", "renvoi", "annuler",
            "annulation", "défectueux", "cassé", "abîmé", "problème",
            "endommagé", "satisfait",
        ],
        "patterns": [
            r"(?:retourner|renvoyer|échanger)\s+(?:un|une|le|la|mon|ma)",
            r"(?:demander|obtenir|avoir)\s+(?:un\s+)?remboursement",
            r"(?:produit|article)\s+(?:défectueux|cassé|abîmé|endommagé)",
            r"pas\s+satisfait",
        ],
        "priority": 1.1,
    },
    "livraison": {
        "keywords": [
            "livraison", "livrer", "délai", "délais", "frais",
            "gratuit", "express", "standard", "point", "relais",
            "click", "collect", "adresse", "domicile", "jour",
            "rapide", "lent", "temps",
        ],
        "patterns": [
            r"(?:délai|frais|mode|option)\s+(?:de\s+)?livraison",
            r"(?:livré|livrer)\s+(?:en|sous|dans)",
            r"livraison\s+(?:gratuite|express|standard|rapide)",
            r"(?:combien|quel)\s+(?:de\s+)?temps\s+(?:pour\s+)?(?:la\s+)?livr",
        ],
        "priority": 1.0,
    },
    "paiement": {
        "keywords": [
            "paiement", "payer", "carte", "paypal", "virement",
            "prix", "coût", "facturer", "facture", "bancaire",
            "visa", "mastercard", "prélèvement", "crédit", "3x",
            "plusieurs", "fois",
        ],
        "patterns": [
            r"(?:moyen|mode|option)\s+(?:de\s+)?paiement",
            r"(?:payer|régler)\s+(?:en|par|avec)",
            r"(?:carte|visa|mastercard|paypal|virement)",
            r"paiement\s+(?:en\s+)?(?:plusieurs|3|4)\s*(?:x|fois)",
        ],
        "priority": 1.0,
    },
    "recommandation": {
        "keywords": [
            "recommander", "recommandation", "suggestion", "conseil",
            "conseiller", "suggérer", "proposer", "idée", "cadeau",
            "populaire", "tendance", "meilleur", "top", "avis",
            "noter", "note", "personnalisé", "préférence",
        ],
        "patterns": [
            r"(?:recommandez|conseillez|suggérez|proposez).(?:moi|nous)",
            r"(?:quel|quelle)\s+(?:est|sont)\s+(?:le|la|les)\s+meilleur",
            r"(?:idée|cadeau|suggestion)\s+(?:de|pour)",
            r"(?:que|quoi)\s+(?:me|nous)\s+(?:recommandez|conseillez)",
        ],
        "priority": 1.0,
    },
    "compte": {
        "keywords": [
            "compte", "inscription", "inscrire", "connecter", "connexion",
            "mot", "passe", "password", "profil", "email", "modifier",
            "supprimer", "désinscrire", "données", "personnel",
        ],
        "patterns": [
            r"(?:créer|modifier|supprimer)\s+(?:mon|un)\s+compte",
            r"(?:mot\s+de\s+passe|password)\s+(?:oublié|changer|modifier)",
            r"(?:me\s+)?(?:connecter|inscrire|déconnecter)",
            r"(?:mon|le)\s+(?:profil|compte)",
        ],
        "priority": 1.0,
    },
    "stock": {
        "keywords": [
            "stock", "disponible", "disponibilité", "rupture",
            "réapprovisionnement", "quantité", "restant", "épuisé",
        ],
        "patterns": [
            r"(?:en\s+)?stock",
            r"(?:est|sont).(?:il|elle|ils|elles)\s+disponible",
            r"rupture\s+de\s+stock",
            r"(?:combien|quantité)\s+(?:il\s+)?(?:reste|reste-t-il|en\s+stock)",
        ],
        "priority": 1.1,
    },
    "promotion": {
        "keywords": [
            "promotion", "promo", "solde", "réduction", "code",
            "coupon", "remise", "offre", "pourcentage", "rabais",
            "bon", "réduire", "moins", "cher",
        ],
        "patterns": [
            r"(?:code|coupon|bon)\s+(?:promo|de\s+réduction|promotionnel)",
            r"(?:promotion|solde|offre)\s+(?:en\s+)?(?:cours|actuelle)",
            r"(?:réduction|remise|rabais)",
        ],
        "priority": 1.0,
    },
    "garantie": {
        "keywords": [
            "garantie", "garanti", "panne", "réparation", "réparer",
            "défaut", "sav", "service", "après", "vente", "warranty",
        ],
        "patterns": [
            r"(?:sous\s+)?garantie",
            r"(?:service\s+)?après.vente",
            r"(?:réparation|réparer|panne)",
        ],
        "priority": 1.0,
    },
    "vendeur": {
        "keywords": [
            "vendeur", "vendre", "vente", "boutique", "magasin",
            "espace", "commissio", "partenaire", "marchand",
        ],
        "patterns": [
            r"devenir\s+vendeur",
            r"(?:espace|profil)\s+vendeur",
            r"(?:ouvrir|créer)\s+(?:une\s+)?boutique",
        ],
        "priority": 0.9,
    },
    "salutation": {
        "keywords": [
            "bonjour", "bonsoir", "salut", "hello", "hi", "hey",
            "coucou", "bonne", "journée",
        ],
        "patterns": [
            r"^(?:bonjour|bonsoir|salut|hello|hey|coucou|hi)\s*[!.?]*$",
        ],
        "priority": 0.5,
    },
    "remerciement": {
        "keywords": [
            "merci", "remercie", "bravo", "super", "excellent",
            "parfait", "génial", "top", "cool", "thanks",
        ],
        "patterns": [
            r"^(?:merci|thanks|bravo|super|excellent|parfait)\s*[!.]*$",
            r"(?:merci|remercie)\s+(?:beaucoup|bien|infiniment)",
        ],
        "priority": 0.5,
    },
    "plainte": {
        "keywords": [
            "plainte", "réclamation", "mécontent", "furieux", "honteux",
            "inacceptable", "scandaleux", "horrible", "déplorable",
            "nul", "catastrophe", "arnaque", "escroquerie", "volé",
        ],
        "patterns": [
            r"(?:je\s+)?(?:suis\s+)?(?:très\s+)?(?:mécontent|furieux|déçu|insatisfait)",
            r"(?:porter|déposer|faire)\s+(?:une\s+)?(?:plainte|réclamation)",
            r"(?:c'est|c est)\s+(?:une?\s+)?(?:arnaque|honte|scandale)",
        ],
        "priority": 1.3,  # Priorité élevée pour les plaintes
    },
    "question_generale": {
        "keywords": [
            "question", "renseignement", "information", "savoir",
            "comment", "pourquoi", "quand", "combien", "quel",
            "expliquer", "aide", "aider",
        ],
        "patterns": [
            r"^(?:comment|pourquoi|quand|combien|quel|quelle|quels|quelles)",
            r"(?:j'ai|j ai)\s+une?\s+question",
            r"(?:pouvez|pourriez).vous\s+(?:m'|me\s+)?(?:expliquer|aider|dire)",
        ],
        "priority": 0.3,  # Faible priorité (fallback)
    },
}


# ============================================
# PATTERNS NER (Named Entity Recognition)
# ============================================
NER_PATTERNS = {
    "ORDER_ID": [
        r"(?:commande|order|cmd)\s*(?:n[°o]?\s*|numéro\s*|#\s*)([\w\-]+)",
        r"(CMD[\-_]?\d{4}[\-_]\d{3,})",
        r"#\s*(\d{5,})",
    ],
    "PRODUCT": [
        r"(?:le|la|un|une|mon|ma|ce|cette)\s+([\w\-]+(?:\s+[\w\-]+){0,3})\s+(?:que|qui|est|a)",
    ],
    "CATEGORY": [
        r"(?:catégorie|rayon|section)\s+[\"']?([\w\s]+?)[\"']?(?:\s|$|\.)",
        r"(?:en|de|du)\s+((?:électronique|vêtement|chaussure|sport|maison|"
        r"beauté|jouet|livre|auto|jardin|alimentation)s?)",
    ],
    "PRICE": [
        r"(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)",
        r"(?:prix|coût|budget)\s+(?:de\s+)?(\d+(?:[.,]\d{1,2})?)",
        r"(?:moins|plus)\s+(?:de\s+)?(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?)",
        r"(?:entre\s+)?(\d+)\s*(?:€|euros?)?\s*(?:et|à|-)\s*(\d+)\s*(?:€|euros?)",
    ],
    "COLOR": [
        r"\b(noir|blanc|rouge|bleu|vert|jaune|orange|rose|violet|marron|"
        r"gris|beige|bordeaux|turquoise|doré|argenté|kaki|corail|navy)\b",
    ],
    "SIZE": [
        r"(?:taille|pointure|size)\s*([\w\d]+)",
        r"\b([XSML]{1,3}L?)\b",
        r"(?:taille|pointure)\s*(\d{2,3})",
    ],
    "CITY": [
        r"(?:à|de|vers|sur)\s+(Paris|Lyon|Marseille|Toulouse|Nice|Nantes|"
        r"Strasbourg|Montpellier|Bordeaux|Lille|Rennes|Reims|Toulon|"
        r"Grenoble|Dijon|Angers|Dakar|Abidjan|Brazzaville|Kinshasa|"
        r"Douala|Yaoundé|Bamako|Ouagadougou|Niamey|Lomé|Cotonou)",
    ],
    "EMAIL": [
        r"([\w.+-]+@[\w-]+\.[\w.]+)",
    ],
    "PHONE": [
        r"(\+?\d{1,3}[\s.-]?\d{2,3}[\s.-]?\d{2,3}[\s.-]?\d{2,3}[\s.-]?\d{2,3})",
        r"(0\d[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2})",
    ],
    "DATE": [
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        r"(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
        r"août|septembre|octobre|novembre|décembre)\s+\d{4})",
    ],
}


# ============================================
# LEXIQUE DE SENTIMENT
# ============================================
SENTIMENT_POSITIF = {
    "excellent": 2.0, "parfait": 2.0, "super": 1.5, "génial": 2.0,
    "formidable": 2.0, "magnifique": 2.0, "bravo": 1.5, "merci": 1.0,
    "satisfait": 1.5, "content": 1.5, "heureux": 1.5, "ravi": 2.0,
    "incroyable": 2.0, "fantastique": 2.0, "bien": 1.0, "bon": 1.0,
    "bonne": 1.0, "top": 1.5, "cool": 1.0, "adoré": 2.0, "adore": 2.0,
    "aime": 1.0, "recommande": 1.5, "rapide": 1.0, "efficace": 1.5,
    "pratique": 1.0, "utile": 1.0, "agréable": 1.0, "qualité": 1.0,
    "fiable": 1.5, "confiance": 1.0, "impressionnant": 2.0,
}

SENTIMENT_NEGATIF = {
    "horrible": -2.0, "terrible": -2.0, "nul": -2.0, "mauvais": -1.5,
    "déteste": -2.0, "déçu": -1.5, "mécontent": -2.0, "furieux": -2.5,
    "inacceptable": -2.5, "scandaleux": -2.5, "honteux": -2.0,
    "arnaque": -2.5, "escroquerie": -2.5, "volé": -2.0, "pire": -2.0,
    "lent": -1.0, "cassé": -1.5, "défectueux": -1.5, "abîmé": -1.5,
    "endommagé": -1.5, "problème": -1.0, "erreur": -1.0, "bug": -1.0,
    "impossible": -1.5, "compliqué": -1.0, "cher": -0.5, "long": -0.5,
    "attendre": -0.5, "déplorable": -2.0, "catastrophe": -2.0,
    "jamais": -1.0, "aucun": -0.5, "plainte": -1.5, "réclamation": -1.0,
}

# Négations qui inversent le sentiment
NEGATIONS = {"ne", "n", "pas", "plus", "aucun", "aucune", "jamais", "rien", "sans"}


# ============================================
# MOTEUR NLP
# ============================================
class NLPEngine:
    """
    Moteur NLP complet pour l'analyse des requêtes e-commerce.
    
    Pipeline :
    1. Prétraitement (normalisation, tokenisation)
    2. Détection d'intent (classification d'intention)
    3. Extraction d'entités (NER)
    4. Analyse de sentiment
    5. Extraction de mots-clés (TF-IDF simplifié)
    
    Usage :
        nlp = NLPEngine()
        result = nlp.analyser("Je cherche des chaussures Nike rouges à moins de 100€")
        # {
        #   "intent": "recherche_produit",
        #   "intent_confidence": 0.85,
        #   "entities": [
        #     {"type": "BRAND", "value": "Nike", "start": 30, "end": 34},
        #     {"type": "COLOR", "value": "rouges", "start": 35, "end": 41},
        #     {"type": "PRICE", "value": "100", "start": 52, "end": 55}
        #   ],
        #   "sentiment": {"label": "neutre", "score": 0.0},
        #   "keywords": ["chaussures", "nike", "rouges"],
        #   "tokens": [...]
        # }
    """
    
    def __init__(self):
        """Initialise le moteur NLP."""
        self.intent_patterns = INTENT_PATTERNS
        self.ner_patterns = NER_PATTERNS
        self.stopwords = STOPWORDS_FR
        
        logger.info("🧠 Moteur NLP initialisé")
        logger.info(f"   Intents    : {len(self.intent_patterns)}")
        logger.info(f"   Entités    : {len(self.ner_patterns)} types")
        logger.info(f"   Stopwords  : {len(self.stopwords)} mots")
    
    # ============================================
    # PRÉTRAITEMENT
    # ============================================
    def preprocesser(self, texte: str) -> Dict[str, Any]:
        """
        Prétraite le texte : normalisation, tokenisation, lemmatisation simplifiée.
        
        Pipeline :
        1. Normalisation Unicode
        2. Mise en minuscules (sauf noms propres)
        3. Tokenisation
        4. Suppression des stopwords
        5. Lemmatisation basique (règles suffixales)
        
        Args:
            texte: Texte brut de l'utilisateur
        
        Returns:
            {
                "original": str,
                "normalise": str,
                "tokens": List[str],
                "tokens_filtres": List[str],
                "lemmes": List[str]
            }
        """
        original = texte.strip()
        
        # Normalisation Unicode basique
        normalise = original.lower()
        normalise = re.sub(r"[''`]", "'", normalise)
        normalise = re.sub(r"[«»""\"]+", '"', normalise)
        normalise = re.sub(r"\s+", " ", normalise)
        
        # Tokenisation
        tokens = re.findall(r"[\w'àâäéèêëïîôùûüÿçœæ]+", normalise)
        
        # Filtrage stopwords
        tokens_filtres = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        # Lemmatisation simplifiée (règles suffixales françaises)
        lemmes = [self._lemmatiser(t) for t in tokens_filtres]
        
        return {
            "original": original,
            "normalise": normalise,
            "tokens": tokens,
            "tokens_filtres": tokens_filtres,
            "lemmes": lemmes,
        }
    
    def _lemmatiser(self, mot: str) -> str:
        """
        Lemmatisation basique par règles suffixales.
        
        Transformations :
        - Pluriels → singulier (chaussures → chaussure)
        - Féminins → masculin (livraisons → livraison)  
        - Verbes → infinitif approché
        
        Args:
            mot: Token à lemmatiser
        
        Returns:
            Lemme approximatif
        """
        # Pluriels en -aux → -al
        if mot.endswith("aux") and len(mot) > 4:
            return mot[:-3] + "al"
        # Pluriels en -eaux → -eau
        if mot.endswith("eaux") and len(mot) > 5:
            return mot[:-1]
        # Pluriels en -s (sauf certains)
        if mot.endswith("s") and len(mot) > 3 and not mot.endswith(("is", "us", "as", "os")):
            return mot[:-1]
        # -ment → supprimer (adverbes)
        if mot.endswith("ment") and len(mot) > 6:
            return mot[:-4]
        # -tion → garder tel quel (noms)
        if mot.endswith("tion"):
            return mot
        # -ement, -ement
        if mot.endswith("ement") and len(mot) > 7:
            return mot[:-5] + "er"
        
        return mot
    
    # ============================================
    # DÉTECTION D'INTENT
    # ============================================
    def detecter_intent(self, texte: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Détecte l'intention de l'utilisateur par scoring hybride
        (mots-clés + patterns regex).
        
        Algorithme de scoring :
        1. Score mots-clés = (nb_hits / nb_keywords) × priority
        2. Score regex = nb_patterns_matchés × 0.3 × priority
        3. Score final = keywords_score + regex_score
        4. Normalisation softmax sur les scores
        
        Args:
            texte: Texte de l'utilisateur
            top_k: Nombre d'intents à retourner
        
        Returns:
            {
                "intent": str,           # Intent principal
                "confidence": float,     # Score de confiance [0, 1]
                "all_intents": [         # Top-K intents
                    {"intent": str, "confidence": float}, ...
                ]
            }
        """
        texte_lower = texte.lower()
        scores = {}
        
        for intent_name, config in self.intent_patterns.items():
            score = 0.0
            
            # Score mots-clés
            keywords = config["keywords"]
            hits = sum(1 for kw in keywords if kw in texte_lower)
            if keywords:
                keyword_score = (hits / len(keywords)) * config["priority"]
                # Bonus si plusieurs hits
                if hits >= 3:
                    keyword_score *= 1.5
                score += keyword_score
            
            # Score patterns regex
            for pattern in config.get("patterns", []):
                try:
                    if re.search(pattern, texte_lower):
                        score += 0.3 * config["priority"]
                except re.error:
                    pass
            
            scores[intent_name] = score
        
        # Normalisation softmax
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                # Softmax avec température
                exp_scores = {}
                temperature = 0.5
                for k, v in scores.items():
                    exp_scores[k] = math.exp(v / temperature)
                total = sum(exp_scores.values())
                normalized = {k: v / total for k, v in exp_scores.items()}
            else:
                normalized = {k: 1.0 / len(scores) for k in scores}
        else:
            normalized = {"question_generale": 1.0}
        
        # Trier par score
        sorted_intents = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        
        top_intent = sorted_intents[0]
        
        return {
            "intent": top_intent[0],
            "confidence": round(float(top_intent[1]), 4),
            "all_intents": [
                {"intent": name, "confidence": round(float(conf), 4)}
                for name, conf in sorted_intents[:top_k]
            ],
        }
    
    # ============================================
    # EXTRACTION D'ENTITÉS (NER)
    # ============================================
    def extraire_entites(self, texte: str) -> List[Dict[str, Any]]:
        """
        Extrait les entités nommées du texte par patterns regex.
        
        Types d'entités :
        - ORDER_ID : numéros de commande (CMD-2024-001, #12345)
        - PRODUCT  : noms de produits
        - CATEGORY : catégories de produits
        - PRICE    : montants en euros
        - COLOR    : couleurs
        - SIZE     : tailles
        - CITY     : villes
        - EMAIL    : adresses email
        - PHONE    : numéros de téléphone
        - DATE     : dates
        
        Args:
            texte: Texte de l'utilisateur
        
        Returns:
            Liste de {type, value, start, end}
        """
        entites = []
        seen_spans = set()
        
        for entity_type, patterns in self.ner_patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, texte, re.IGNORECASE):
                        # Prendre le premier groupe capturé, ou le match entier
                        if match.groups():
                            value = match.group(1)
                            start = match.start(1)
                            end = match.end(1)
                        else:
                            value = match.group(0)
                            start = match.start(0)
                            end = match.end(0)
                        
                        # Éviter les doublons de span
                        span_key = (start, end)
                        if span_key not in seen_spans:
                            seen_spans.add(span_key)
                            entites.append({
                                "type": entity_type,
                                "value": value.strip(),
                                "start": start,
                                "end": end,
                            })
                except re.error:
                    pass
        
        # Trier par position
        entites.sort(key=lambda x: x["start"])
        
        return entites
    
    # ============================================
    # ANALYSE DE SENTIMENT
    # ============================================
    def analyser_sentiment(self, texte: str) -> Dict[str, Any]:
        """
        Analyse le sentiment du texte par lexique pondéré
        avec gestion des négations.
        
        Algorithme :
        1. Tokeniser le texte
        2. Détecter les fenêtres de négation (3 tokens après une négation)
        3. Scorer chaque token via le lexique
        4. Inverser le score si dans une fenêtre de négation
        5. Agréger en score global → label
        
        Échelle : [-1, +1] → négatif / neutre / positif
        
        Args:
            texte: Texte à analyser
        
        Returns:
            {
                "label": "positif" | "neutre" | "négatif",
                "score": float (-1 à +1),
                "details": {
                    "mots_positifs": [...],
                    "mots_negatifs": [...],
                    "negations": [...],
                }
            }
        """
        tokens = re.findall(r"[\w'àâäéèêëïîôùûüÿçœæ]+", texte.lower())
        
        score_total = 0.0
        mots_positifs = []
        mots_negatifs = []
        negations_trouvees = []
        
        # Fenêtre de négation
        negation_active = False
        negation_distance = 0
        NEGATION_WINDOW = 3
        
        for token in tokens:
            # Détecter les négations
            if token in NEGATIONS:
                negation_active = True
                negation_distance = 0
                negations_trouvees.append(token)
                continue
            
            # Compter la distance depuis la dernière négation
            if negation_active:
                negation_distance += 1
                if negation_distance > NEGATION_WINDOW:
                    negation_active = False
            
            # Scorer le token
            if token in SENTIMENT_POSITIF:
                score = SENTIMENT_POSITIF[token]
                if negation_active:
                    score = -score * 0.8  # Inverser + atténuer
                    mots_negatifs.append(f"ne...{token}")
                else:
                    mots_positifs.append(token)
                score_total += score
            
            elif token in SENTIMENT_NEGATIF:
                score = SENTIMENT_NEGATIF[token]
                if negation_active:
                    score = -score * 0.5  # Inverser + atténuer davantage
                    mots_positifs.append(f"ne...{token}")
                else:
                    mots_negatifs.append(token)
                score_total += score
        
        # Normaliser entre -1 et 1
        if tokens:
            score_normalise = max(-1.0, min(1.0, score_total / max(len(tokens) * 0.3, 1)))
        else:
            score_normalise = 0.0
        
        # Déterminer le label
        if score_normalise > 0.15:
            label = "positif"
        elif score_normalise < -0.15:
            label = "négatif"
        else:
            label = "neutre"
        
        return {
            "label": label,
            "score": round(float(score_normalise), 4),
            "details": {
                "mots_positifs": mots_positifs,
                "mots_negatifs": mots_negatifs,
                "negations": negations_trouvees,
            },
        }
    
    # ============================================
    # EXTRACTION DE MOTS-CLÉS (TF-IDF simplifié)
    # ============================================
    def extraire_mots_cles(
        self,
        texte: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extrait les mots-clés les plus importants du texte
        via un scoring TF-IDF simplifié.
        
        Score = TF × IDF_approx
        - TF = fréquence du terme dans le texte
        - IDF_approx = log(1 / fréquence_langue_estimée)
        
        Args:
            texte: Texte à analyser
            top_k: Nombre de mots-clés à retourner
        
        Returns:
            Liste de {mot, score, frequence}
        """
        tokens = re.findall(r"[\w'àâäéèêëïîôùûüÿçœæ]+", texte.lower())
        tokens_filtres = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        if not tokens_filtres:
            return []
        
        # TF (Term Frequency)
        counter = Counter(tokens_filtres)
        total = len(tokens_filtres)
        
        # IDF approximatif (mots rares = score plus élevé)
        # Les mots courants de notre domaine ont un IDF plus bas
        mots_domaine_courants = {
            "produit", "commande", "livraison", "compte", "aide",
            "acheter", "recherche", "question", "bonjour", "comment",
        }
        
        keywords = []
        for mot, freq in counter.items():
            tf = freq / total
            
            # IDF approximatif basé sur la longueur et la rareté
            if mot in mots_domaine_courants:
                idf = 0.5
            elif len(mot) > 8:
                idf = 2.0  # Mots longs = probablement spécifiques
            elif len(mot) > 5:
                idf = 1.5
            else:
                idf = 1.0
            
            score = tf * idf
            keywords.append({
                "mot": mot,
                "score": round(float(score), 4),
                "frequence": freq,
            })
        
        # Trier par score décroissant
        keywords.sort(key=lambda x: x["score"], reverse=True)
        
        return keywords[:top_k]
    
    # ============================================
    # ANALYSE COMPLÈTE
    # ============================================
    def analyser(self, texte: str) -> Dict[str, Any]:
        """
        Analyse NLP complète d'un texte utilisateur.
        
        Exécute le pipeline complet :
        1. Prétraitement
        2. Détection d'intent
        3. Extraction d'entités (NER)
        4. Analyse de sentiment
        5. Extraction de mots-clés
        
        Args:
            texte: Texte brut de l'utilisateur
        
        Returns:
            Résultat complet de l'analyse NLP
        """
        if not texte or not texte.strip():
            return {
                "texte": "",
                "preprocessed": {},
                "intent": {"intent": "question_generale", "confidence": 0.0, "all_intents": []},
                "entities": [],
                "sentiment": {"label": "neutre", "score": 0.0, "details": {}},
                "keywords": [],
            }
        
        # Pipeline NLP
        preprocessed = self.preprocesser(texte)
        intent = self.detecter_intent(texte)
        entities = self.extraire_entites(texte)
        sentiment = self.analyser_sentiment(texte)
        keywords = self.extraire_mots_cles(texte)
        
        return {
            "texte": texte,
            "preprocessed": preprocessed,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "keywords": keywords,
        }
    
    # ============================================
    # ROUTING INTELLIGENT
    # ============================================
    def router_requete(self, texte: str) -> Dict[str, Any]:
        """
        Analyse la requête et détermine le module cible
        pour le traitement (chatbot, recommandation, recherche, etc.).
        
        Routing :
        - recherche_produit  → module recherche/classification
        - recommandation     → module recommandation
        - suivi_commande     → module suivi
        - stock              → module stock
        - salutation/remerciement → réponse directe
        - plainte            → escalade humain
        - autre              → chatbot RAG
        
        Args:
            texte: Requête utilisateur
        
        Returns:
            {
                "module": str,
                "action": str,
                "nlp_analysis": Dict,
                "requires_rag": bool,
                "escalade": bool
            }
        """
        analyse = self.analyser(texte)
        intent = analyse["intent"]["intent"]
        confidence = analyse["intent"]["confidence"]
        sentiment = analyse["sentiment"]
        
        # Mapping intent → module
        ROUTING_MAP = {
            "recherche_produit": ("recherche", "search_products"),
            "recommandation": ("recommandation", "get_recommendations"),
            "suivi_commande": ("suivi", "track_order"),
            "stock": ("stock", "check_stock"),
            "salutation": ("direct", "greet"),
            "remerciement": ("direct", "thank"),
            "plainte": ("escalade", "handle_complaint"),
            "retour_remboursement": ("chatbot_rag", "handle_return"),
            "livraison": ("chatbot_rag", "delivery_info"),
            "paiement": ("chatbot_rag", "payment_info"),
            "compte": ("chatbot_rag", "account_info"),
            "garantie": ("chatbot_rag", "warranty_info"),
            "promotion": ("chatbot_rag", "promo_info"),
            "vendeur": ("chatbot_rag", "seller_info"),
            "question_generale": ("chatbot_rag", "general_qa"),
        }
        
        module, action = ROUTING_MAP.get(intent, ("chatbot_rag", "general_qa"))
        
        # Override : si sentiment très négatif → escalade
        escalade = False
        if sentiment["score"] < -0.5:
            escalade = True
            module = "escalade"
            action = "handle_negative_sentiment"
        
        # Déterminer si le RAG est nécessaire
        requires_rag = module == "chatbot_rag"
        
        return {
            "module": module,
            "action": action,
            "nlp_analysis": analyse,
            "requires_rag": requires_rag,
            "escalade": escalade,
        }


# ============================================
# SINGLETON
# ============================================
_nlp_instance: Optional[NLPEngine] = None


def get_nlp_engine() -> NLPEngine:
    """Retourne l'instance singleton du moteur NLP."""
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = NLPEngine()
    return _nlp_instance


# ============================================
# DÉMONSTRATION
# ============================================
def main():
    """Démonstration du moteur NLP."""
    
    nlp = NLPEngine()
    
    tests = [
        "Bonjour, je cherche des chaussures Nike rouges à moins de 100€",
        "Où en est ma commande CMD-2024-001 ?",
        "Je suis très mécontent, le produit est arrivé cassé !",
        "Pouvez-vous me recommander un cadeau pour Noël ?",
        "Quels sont vos délais de livraison à Dakar ?",
        "Comment retourner un article défectueux ?",
        "Merci beaucoup, vous êtes super efficaces !",
        "Ce produit est-il disponible en taille 42 bleu ?",
        "Je voudrais payer en 3 fois par carte Visa",
        "C'est une arnaque, je n'ai jamais reçu ma commande !",
    ]
    
    print("=" * 80)
    print("🧠 ECommerce-IA — Moteur NLP — Démonstration")
    print("=" * 80)
    
    for texte in tests:
        print(f"\n{'─' * 80}")
        print(f"📝 Entrée : \"{texte}\"")
        
        result = nlp.analyser(texte)
        
        print(f"   🎯 Intent     : {result['intent']['intent']} "
              f"(confiance: {result['intent']['confidence']:.1%})")
        print(f"   😊 Sentiment  : {result['sentiment']['label']} "
              f"(score: {result['sentiment']['score']:+.2f})")
        
        if result["entities"]:
            entities_str = ", ".join(
                f"{e['type']}={e['value']}" for e in result["entities"]
            )
            print(f"   🏷️  Entités    : {entities_str}")
        
        if result["keywords"]:
            kw_str = ", ".join(k["mot"] for k in result["keywords"][:5])
            print(f"   🔑 Mots-clés  : {kw_str}")
        
        # Routing
        route = nlp.router_requete(texte)
        print(f"   🔀 Module     : {route['module']} → {route['action']}")
        if route["escalade"]:
            print(f"   ⚠️  ESCALADE HUMAIN requise")
    
    print(f"\n{'=' * 80}")
    print("✅ Moteur NLP opérationnel !")


if __name__ == "__main__":
    main()
