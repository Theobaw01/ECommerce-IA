"""
============================================
ECommerce-IA — Chatbot RAG Service Client
============================================
Chatbot intelligent basé sur RAG (Retrieval-Augmented Generation)
pour le service client de la plateforme e-commerce.

Architecture :
- LangChain pour l'orchestration du pipeline RAG
- ChromaDB pour le vector store (stockage des embeddings)
- sentence-transformers/all-MiniLM-L6-v2 pour les embeddings
- Mistral-7B-Instruct (HuggingFace API) pour la génération
- Fallback : google/flan-t5-large (local, léger)

Pipeline RAG :
1. Indexation : documents → chunks → embeddings → ChromaDB
2. Requête : question → embedding → recherche top-3 → contexte
3. Génération : contexte + question → réponse structurée

Fonctionnalités :
- Réponses questions produits
- Vérification stock
- Suivi commande simulé
- Recommandations personnalisées
- Escalade humain si confiance < 0.6
- Historique conversation (session)

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# NLP Engine intégré
from src.nlp_engine import get_nlp_engine, NLPEngine

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
MODELS_DIR = PROJECT_ROOT / "models" / "chatbot"
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"

# Configuration du chatbot
CHATBOT_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "generation_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "fallback_model": "google/flan-t5-large",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k_retrieval": 3,
    "temperature": 0.3,
    "max_new_tokens": 500,
    "confidence_threshold": 0.6,  # Escalade humain si < 0.6
    "chroma_collection": "ecommerce_kb",
}


# ============================================
# BASE DE CONNAISSANCES
# ============================================
def creer_base_connaissances() -> List[Dict[str, str]]:
    """
    Crée la base de connaissances du chatbot e-commerce.
    Contient les FAQ, politiques, et informations catalogue.
    
    Returns:
        Liste de documents {titre, contenu, categorie}
    """
    documents = [
        # === FAQ PRODUITS ===
        {
            "titre": "Comment rechercher un produit ?",
            "contenu": (
                "Vous pouvez rechercher un produit de plusieurs façons sur notre plateforme : "
                "1. Utilisez la barre de recherche en haut de la page pour chercher par nom ou catégorie. "
                "2. Utilisez notre recherche visuelle IA : uploadez une photo d'un produit et notre "
                "système de classification visuelle (94% de précision) trouvera des produits similaires. "
                "3. Parcourez nos catégories dans le menu principal. "
                "4. Consultez les recommandations personnalisées sur votre page d'accueil."
            ),
            "categorie": "faq"
        },
        {
            "titre": "Comment fonctionne la recherche par image ?",
            "contenu": (
                "Notre système de recherche par image utilise un modèle EfficientNet-B4 entraîné "
                "sur plus de 3 000 images de produits avec une précision de 94%. "
                "Pour l'utiliser : 1. Cliquez sur l'icône appareil photo dans la barre de recherche. "
                "2. Uploadez ou prenez une photo du produit recherché. "
                "3. Notre IA analyse l'image et identifie la catégorie du produit. "
                "4. Les produits correspondants sont affichés avec un score de confiance."
            ),
            "categorie": "faq"
        },
        {
            "titre": "Quels moyens de paiement acceptez-vous ?",
            "contenu": (
                "Nous acceptons les moyens de paiement suivants : "
                "- Carte bancaire (Visa, Mastercard, CB) "
                "- PayPal "
                "- Virement bancaire "
                "- Paiement en 3x sans frais pour les commandes supérieures à 100€ "
                "Tous les paiements sont sécurisés via un protocole SSL 256 bits."
            ),
            "categorie": "paiement"
        },
        {
            "titre": "Comment suivre ma commande ?",
            "contenu": (
                "Pour suivre votre commande : "
                "1. Connectez-vous à votre compte. "
                "2. Allez dans 'Mes commandes' depuis votre profil. "
                "3. Cliquez sur la commande concernée pour voir le suivi détaillé. "
                "Les statuts possibles sont : En préparation, Expédié, En livraison, Livré. "
                "Vous recevrez également des emails de notification à chaque changement de statut. "
                "Le numéro de suivi est disponible dès l'expédition du colis."
            ),
            "categorie": "commande"
        },
        # === POLITIQUE DE LIVRAISON ===
        {
            "titre": "Délais et frais de livraison",
            "contenu": (
                "Nos délais et frais de livraison : "
                "- Livraison standard (3-5 jours ouvrés) : 4.99€, gratuite dès 50€ d'achat "
                "- Livraison express (1-2 jours ouvrés) : 9.99€ "
                "- Point relais (3-5 jours ouvrés) : 3.99€ "
                "- Click & Collect (disponible sous 2h) : gratuit "
                "Les vendeurs locaux (moins de 50 km) peuvent proposer une livraison le jour même. "
                "Les délais sont indicatifs et peuvent varier selon la disponibilité du produit."
            ),
            "categorie": "livraison"
        },
        {
            "titre": "Livraison internationale",
            "contenu": (
                "Nous livrons dans toute l'Union Européenne et dans certains pays hors UE. "
                "Délais : 5-10 jours ouvrés selon la destination. "
                "Frais : calculés automatiquement en fonction du poids et de la destination. "
                "Les droits de douane peuvent s'appliquer pour les destinations hors UE."
            ),
            "categorie": "livraison"
        },
        # === POLITIQUE DE RETOUR ===
        {
            "titre": "Politique de retour et remboursement",
            "contenu": (
                "Notre politique de retour : "
                "- Vous disposez de 30 jours après réception pour retourner un produit. "
                "- Le produit doit être dans son emballage d'origine, non utilisé et non endommagé. "
                "- Les frais de retour sont à la charge du client, sauf en cas de produit défectueux. "
                "- Le remboursement est effectué sous 14 jours après réception du retour. "
                "- Le remboursement se fait sur le moyen de paiement initial. "
                "Pour initier un retour, rendez-vous dans 'Mes commandes' et cliquez sur 'Retourner'."
            ),
            "categorie": "retour"
        },
        {
            "titre": "Échange de produit",
            "contenu": (
                "Pour échanger un produit : "
                "1. Initiez un retour depuis 'Mes commandes'. "
                "2. Sélectionnez 'Échange' comme motif. "
                "3. Choisissez le nouveau produit souhaité (taille, couleur, etc.). "
                "4. L'échange est gratuit si le prix est identique. "
                "Si le nouveau produit est plus cher, la différence sera facturée."
            ),
            "categorie": "retour"
        },
        # === GARANTIE ===
        {
            "titre": "Garantie des produits",
            "contenu": (
                "Tous nos produits bénéficient : "
                "- D'une garantie légale de conformité de 2 ans. "
                "- D'une garantie des vices cachés. "
                "- Certains produits ont une garantie constructeur supplémentaire. "
                "En cas de panne ou défaut, contactez notre service client avec votre numéro "
                "de commande et une description du problème. "
                "Nous organisons la réparation ou le remplacement du produit."
            ),
            "categorie": "garantie"
        },
        # === COMPTE UTILISATEUR ===
        {
            "titre": "Création et gestion de compte",
            "contenu": (
                "Pour créer un compte : "
                "1. Cliquez sur 'S'inscrire' en haut à droite. "
                "2. Renseignez votre email, nom et mot de passe. "
                "3. Validez votre email via le lien reçu. "
                "Avantages d'un compte : historique de commandes, recommandations personnalisées, "
                "liste de favoris, suivi de livraison, et offres exclusives."
            ),
            "categorie": "compte"
        },
        # === RECOMMANDATIONS ===
        {
            "titre": "Comment fonctionnent les recommandations ?",
            "contenu": (
                "Notre système de recommandation IA utilise 4 facteurs pour personnaliser "
                "vos suggestions : "
                "1. Historique d'achats (40%) : basé sur vos achats précédents et ceux "
                "de clients similaires. "
                "2. Similarité produits (30%) : produits similaires à ceux que vous consultez. "
                "3. Proximité géographique (15%) : favorise les vendeurs proches de chez vous. "
                "4. Budget (15%) : adapté à votre fourchette de prix habituelle. "
                "Les recommandations s'améliorent au fur et à mesure de votre utilisation."
            ),
            "categorie": "recommandation"
        },
        # === VENDEURS ===
        {
            "titre": "Devenir vendeur sur la plateforme",
            "contenu": (
                "Pour devenir vendeur : "
                "1. Créez un compte vendeur depuis 'Espace Vendeur'. "
                "2. Soumettez les documents nécessaires (SIRET, RIB, pièce d'identité). "
                "3. Validation sous 48h par notre équipe. "
                "4. Publiez vos produits avec photos et descriptions. "
                "Commission : 8-15% selon la catégorie. "
                "Outils disponibles : dashboard de ventes, analytique, gestion de stock."
            ),
            "categorie": "vendeur"
        },
        # === PROMOTIONS ===
        {
            "titre": "Promotions et codes promo",
            "contenu": (
                "Pour utiliser un code promo : "
                "1. Ajoutez les produits à votre panier. "
                "2. Saisissez le code dans le champ 'Code promo' à l'étape de paiement. "
                "3. La réduction s'applique automatiquement. "
                "Les codes promo ne sont pas cumulables sauf indication contraire. "
                "Retrouvez nos offres en cours sur la page 'Promotions'."
            ),
            "categorie": "promotion"
        },
    ]
    
    return documents


# ============================================
# CHATBOT RAG
# ============================================
class EcommerceChatbot:
    """
    Chatbot RAG pour le service client e-commerce.
    
    Pipeline :
    1. Indexation des documents dans ChromaDB
    2. Recherche sémantique des documents pertinents
    3. Génération de réponse contextuelle
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le chatbot.
        
        Args:
            config: Configuration du chatbot
        """
        self.config = config or CHATBOT_CONFIG
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.chain = None
        self.sessions: Dict[str, List[Dict]] = {}
        self.conversation_memory: Dict[str, List[str]] = {}  # Mémoire multi-turn
        self.is_initialized = False
        
        # Intégrer le moteur NLP
        try:
            self.nlp_engine = get_nlp_engine()
            logger.info("🧠 Moteur NLP intégré au chatbot")
        except Exception as e:
            logger.warning(f"⚠️  NLP non disponible : {e}")
            self.nlp_engine = None
        
        logger.info("🤖 Initialisation du chatbot RAG...")
    
    def initialiser(self, documents: Optional[List[Dict]] = None) -> None:
        """
        Initialise le pipeline RAG complet.
        
        Étapes :
        1. Charger le modèle d'embeddings
        2. Indexer les documents dans ChromaDB
        3. Configurer le modèle de génération
        4. Construire la chaîne RAG LangChain
        
        Args:
            documents: Liste de documents à indexer (optionnel)
        """
        if documents is None:
            documents = creer_base_connaissances()
        
        # Étape 1 : Embeddings
        self._initialiser_embeddings()
        
        # Étape 2 : Indexation
        self._indexer_documents(documents)
        
        # Étape 3 : Modèle de génération
        self._initialiser_llm()
        
        # Étape 4 : Chaîne RAG
        self._construire_chain()
        
        self.is_initialized = True
        logger.info("✅ Chatbot RAG prêt !")
    
    def _initialiser_embeddings(self) -> None:
        """Initialise le modèle d'embeddings."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config["embedding_model"],
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"✅ Embeddings chargés : {self.config['embedding_model']}")
        except ImportError:
            logger.warning("⚠️  LangChain non disponible — mode fallback")
            self.embeddings = None
    
    def _indexer_documents(self, documents: List[Dict]) -> None:
        """
        Indexe les documents dans ChromaDB.
        
        Découpe les documents en chunks de 512 tokens,
        calcule les embeddings et les stocke dans ChromaDB.
        
        Args:
            documents: Liste de documents à indexer
        """
        if self.embeddings is None:
            logger.warning("⚠️  Embeddings non disponibles — indexation en mémoire")
            self.documents_store = documents
            return
        
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            from langchain.schema import Document
            
            # Découpage en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Convertir en documents LangChain
            langchain_docs = []
            for doc in documents:
                text = f"{doc['titre']}\n\n{doc['contenu']}"
                metadata = {
                    "titre": doc["titre"],
                    "categorie": doc.get("categorie", "general")
                }
                chunks = text_splitter.create_documents(
                    [text],
                    metadatas=[metadata]
                )
                langchain_docs.extend(chunks)
            
            logger.info(f"📄 {len(documents)} documents → {len(langchain_docs)} chunks")
            
            # Créer le vector store ChromaDB
            persist_dir = str(MODELS_DIR / "chroma_db")
            
            self.vector_store = Chroma.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                persist_directory=persist_dir,
                collection_name=self.config["chroma_collection"]
            )
            
            logger.info(f"✅ ChromaDB indexé : {persist_dir}")
            
        except ImportError as e:
            logger.warning(f"⚠️  Erreur d'import : {e}")
            self.documents_store = documents
    
    def _initialiser_llm(self) -> None:
        """
        Initialise le modèle de génération de texte.
        
        Essaie dans l'ordre :
        1. Mistral-7B-Instruct via HuggingFace Inference API
        2. Flan-T5-Large en local (fallback léger)
        3. Mode template (sans LLM)
        """
        # Essayer HuggingFace Inference API
        hf_api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
        
        if hf_api_key:
            try:
                from langchain_community.llms import HuggingFaceHub
                
                self.llm = HuggingFaceHub(
                    repo_id=self.config["generation_model"],
                    huggingfacehub_api_token=hf_api_key,
                    model_kwargs={
                        "temperature": self.config["temperature"],
                        "max_new_tokens": self.config["max_new_tokens"],
                    }
                )
                logger.info(f"✅ LLM via API : {self.config['generation_model']}")
                return
            except Exception as e:
                logger.warning(f"⚠️  API HuggingFace échouée : {e}")
        
        # Essayer en local
        try:
            from transformers import pipeline as hf_pipeline
            from langchain_community.llms import HuggingFacePipeline
            
            pipe = hf_pipeline(
                "text2text-generation",
                model=self.config["fallback_model"],
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"]
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"✅ LLM local : {self.config['fallback_model']}")
            return
        except Exception as e:
            logger.warning(f"⚠️  LLM local échoué : {e}")
        
        # Mode sans LLM
        logger.info("ℹ️  Mode template (sans LLM) — réponses basées sur la recherche")
        self.llm = None
    
    def _construire_chain(self) -> None:
        """Construit la chaîne RAG LangChain."""
        if self.vector_store is None or self.llm is None:
            logger.info("ℹ️  Chaîne RAG en mode simplifié")
            return
        
        try:
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            
            # Template de prompt en français
            prompt_template = """Tu es un assistant service client pour une plateforme e-commerce.
Utilise le contexte suivant pour répondre à la question du client.
Si tu ne connais pas la réponse, dis-le poliment et propose de transférer vers un agent humain.
Réponds toujours en français, de manière professionnelle et concise.

Contexte :
{context}

Question du client : {question}

Réponse :"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Créer la chaîne RAG
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": self.config["top_k_retrieval"]}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            logger.info("✅ Chaîne RAG construite")
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur chaîne RAG : {e}")
            self.chain = None
    
    def rechercher_contexte(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Recherche les documents pertinents pour une question.
        
        Args:
            question: Question de l'utilisateur
            top_k: Nombre de documents à retourner
        
        Returns:
            Liste de documents pertinents avec scores
        """
        if self.vector_store is not None:
            try:
                results = self.vector_store.similarity_search_with_score(
                    question, k=top_k
                )
                return [
                    {
                        "contenu": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    }
                    for doc, score in results
                ]
            except Exception as e:
                logger.warning(f"⚠️  Erreur recherche : {e}")
        
        # Fallback : recherche par mots-clés
        if hasattr(self, "documents_store"):
            return self._recherche_mots_cles(question, top_k)
        
        return []
    
    def _recherche_mots_cles(self, question: str, top_k: int) -> List[Dict]:
        """Recherche de fallback par mots-clés."""
        mots = question.lower().split()
        resultats = []
        
        for doc in self.documents_store:
            texte = f"{doc['titre']} {doc['contenu']}".lower()
            score = sum(1 for mot in mots if mot in texte) / max(len(mots), 1)
            resultats.append({
                "contenu": f"{doc['titre']}\n{doc['contenu']}",
                "metadata": {"categorie": doc.get("categorie", "general")},
                "score": score
            })
        
        resultats.sort(key=lambda x: x["score"], reverse=True)
        return resultats[:top_k]
    
    def generer_reponse(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère une réponse à une question utilisateur.
        
        Pipeline NLP + RAG :
        1. Analyse NLP : intent, entités, sentiment
        2. Routing intelligent selon l'intent
        3. Construction du contexte avec mémoire conversationnelle
        4. Recherche RAG dans ChromaDB (top-3 chunks)
        5. Génération de réponse contextuelle (LLM ou template)
        6. Scoring de confiance
        7. Escalade humain si confiance < 0.6 ou sentiment négatif
        
        Args:
            question: Question de l'utilisateur
            session_id: ID de session pour l'historique
        
        Returns:
            Dictionnaire avec réponse, confiance, sources, analyse NLP
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Initialiser la session si nécessaire
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        # ── ÉTAPE 1 : Analyse NLP ──
        nlp_result = None
        intent_info = {"intent": "question_generale", "confidence": 0.0}
        sentiment_info = {"label": "neutre", "score": 0.0}
        entities = []
        keywords = []
        
        if self.nlp_engine:
            nlp_result = self.nlp_engine.analyser(question)
            intent_info = nlp_result.get("intent", intent_info)
            sentiment_info = nlp_result.get("sentiment", sentiment_info)
            entities = nlp_result.get("entities", [])
            keywords = nlp_result.get("keywords", [])
            
            logger.info(
                f"🧠 NLP → Intent: {intent_info['intent']} "
                f"({intent_info['confidence']:.0%}) | "
                f"Sentiment: {sentiment_info['label']} | "
                f"Entités: {len(entities)}"
            )
        
        # ── ÉTAPE 2 : Gestion de la mémoire conversationnelle ──
        # Garder les 5 derniers échanges comme contexte
        self.conversation_memory[session_id].append(question)
        if len(self.conversation_memory[session_id]) > 10:
            self.conversation_memory[session_id] = self.conversation_memory[session_id][-10:]
        
        # Enrichir la requête avec le contexte conversationnel
        enriched_query = question
        if len(self.conversation_memory[session_id]) > 1:
            previous = self.conversation_memory[session_id][-3:-1]
            enriched_query = " ".join(previous) + " " + question
        
        # Ajouter la question à l'historique
        self.sessions[session_id].append({
            "role": "user",
            "content": question,
            "nlp": {
                "intent": intent_info["intent"],
                "intent_confidence": intent_info["confidence"],
                "sentiment": sentiment_info["label"],
                "sentiment_score": sentiment_info["score"],
                "entities": entities,
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # ── ÉTAPE 3 : Routing intelligent ──
        intent = intent_info["intent"]
        
        # Réponses directes pour salutations/remerciements
        if intent == "salutation" and intent_info["confidence"] > 0.3:
            reponse = self._reponse_salutation()
            sources = []
            confidence = 0.95
        elif intent == "remerciement" and intent_info["confidence"] > 0.3:
            reponse = self._reponse_remerciement()
            sources = []
            confidence = 0.95
        elif intent == "suivi_commande":
            # Extraire l'ID de commande des entités
            order_ids = [e["value"] for e in entities if e["type"] == "ORDER_ID"]
            order_id = order_ids[0] if order_ids else "inconnue"
            suivi = self.suivre_commande(order_id)
            reponse = suivi["message"]
            sources = ["Système de suivi interne"]
            confidence = 0.90
        elif intent == "stock":
            product_names = [e["value"] for e in entities if e["type"] == "PRODUCT"]
            pid = product_names[0] if product_names else "inconnu"
            stock = self.verifier_stock(pid)
            reponse = stock["message"]
            sources = ["Système de gestion de stock"]
            confidence = 0.90
        else:
            # ── ÉTAPE 4 : Recherche RAG dans ChromaDB ──
            context_docs = self.rechercher_contexte(enriched_query)
            
            # Calculer la confiance (basée sur le score de recherche)
            if context_docs:
                confidence = max(doc["score"] for doc in context_docs)
                # Normaliser entre 0 et 1
                confidence = min(1.0, max(0.0, 1.0 - confidence))  # ChromaDB: distance → plus bas = mieux
            else:
                confidence = 0.0
            
            # ── ÉTAPE 5 : Génération de réponse ──
            if self.chain is not None:
                try:
                    result = self.chain({"query": enriched_query})
                    reponse = result["result"]
                    sources = [
                        doc.page_content[:100]
                        for doc in result.get("source_documents", [])
                    ]
                except Exception as e:
                    logger.warning(f"⚠️  Erreur génération : {e}")
                    reponse, sources = self._generer_reponse_template(question, context_docs)
            else:
                reponse, sources = self._generer_reponse_template(question, context_docs)
        
        # ── ÉTAPE 6 : Ajustements post-traitement ──
        # Escalade humain si confiance < seuil OU sentiment très négatif
        escalade = confidence < self.config["confidence_threshold"]
        if sentiment_info["score"] < -0.5:
            escalade = True
            reponse += (
                "\n\n😟 Je comprends votre frustration. "
                "Je vous met en relation avec un conseiller humain "
                "qui pourra traiter votre demande en priorité."
            )
        elif escalade:
            reponse += (
                "\n\n🔄 Je ne suis pas totalement sûr de cette réponse. "
                "Souhaitez-vous être mis en contact avec un conseiller humain ?"
            )
        
        # Construire la réponse enrichie
        response = {
            "session_id": session_id,
            "question": question,
            "reponse": reponse,
            "confiance": float(confidence),
            "sources": sources,
            "escalade_humain": escalade,
            "timestamp": datetime.now().isoformat(),
            # Données NLP pour le frontend
            "nlp": {
                "intent": intent_info["intent"],
                "intent_confidence": float(intent_info["confidence"]),
                "sentiment": sentiment_info["label"],
                "sentiment_score": float(sentiment_info["score"]),
                "entities": entities,
                "keywords": [k["mot"] for k in keywords[:5]] if keywords else [],
            }
        }
        
        # Ajouter à l'historique
        self.sessions[session_id].append({
            "role": "assistant",
            "content": reponse,
            "confiance": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _reponse_salutation(self) -> str:
        """Génère une réponse de salutation."""
        import random
        salutations = [
            "Bonjour ! 👋 Comment puis-je vous aider aujourd'hui ?",
            "Bonjour et bienvenue ! 🛍️ Je suis votre assistant IA. Que puis-je faire pour vous ?",
            "Bonjour ! 😊 N'hésitez pas à me poser vos questions sur nos produits, livraisons, ou votre compte.",
        ]
        return random.choice(salutations)
    
    def _reponse_remerciement(self) -> str:
        """Génère une réponse aux remerciements."""
        import random
        remerciements = [
            "De rien ! 😊 N'hésitez pas si vous avez d'autres questions.",
            "Avec plaisir ! 🎉 Je suis là pour vous aider.",
            "Je vous en prie ! Bonne continuation sur notre plateforme. 🛍️",
        ]
        return random.choice(remerciements)
    
    def _generer_reponse_template(
        self,
        question: str,
        context_docs: List[Dict]
    ) -> Tuple[str, List[str]]:
        """
        Génère une réponse basée sur les templates (fallback sans LLM).
        
        Args:
            question: Question utilisateur
            context_docs: Documents de contexte
        
        Returns:
            Tuple (réponse, sources)
        """
        if not context_docs:
            return (
                "Je suis désolé, je n'ai pas trouvé d'information correspondant "
                "à votre question. Pouvez-vous reformuler ou contacter notre "
                "service client au 01 23 45 67 89 ?",
                []
            )
        
        # Utiliser le meilleur document comme base de réponse
        best_doc = context_docs[0]
        reponse = f"Voici les informations que j'ai trouvées :\n\n{best_doc['contenu']}"
        
        if len(context_docs) > 1:
            reponse += "\n\nInformations complémentaires :"
            for doc in context_docs[1:]:
                if doc["score"] > 0.3:
                    reponse += f"\n• {doc['contenu'][:200]}..."
        
        sources = [doc["contenu"][:100] for doc in context_docs]
        
        return reponse, sources
    
    def get_historique(self, session_id: str) -> List[Dict]:
        """
        Retourne l'historique de conversation d'une session.
        
        Args:
            session_id: ID de la session
        
        Returns:
            Liste des messages de la session
        """
        return self.sessions.get(session_id, [])
    
    def verifier_stock(self, product_id: str) -> Dict[str, Any]:
        """
        Vérifie la disponibilité d'un produit en stock.
        (Simulation — à connecter à la base de données réelle)
        
        Args:
            product_id: ID du produit
        
        Returns:
            Informations de stock
        """
        import random
        random.seed(hash(product_id))
        
        stock = random.randint(0, 50)
        return {
            "product_id": product_id,
            "en_stock": stock > 0,
            "quantite": stock,
            "message": (
                f"✅ Ce produit est en stock ({stock} unités disponibles)."
                if stock > 0 else
                "❌ Ce produit est actuellement en rupture de stock. "
                "Nous pouvons vous notifier de sa disponibilité."
            )
        }
    
    def suivre_commande(self, order_id: str) -> Dict[str, Any]:
        """
        Retourne le suivi d'une commande (simulation).
        
        Args:
            order_id: ID de la commande
        
        Returns:
            Informations de suivi
        """
        import random
        random.seed(hash(order_id))
        
        statuts = [
            "En préparation", "Expédié",
            "En transit", "En livraison", "Livré"
        ]
        statut_idx = random.randint(0, len(statuts) - 1)
        
        return {
            "order_id": order_id,
            "statut": statuts[statut_idx],
            "date_estimee": "2026-03-15",
            "transporteur": random.choice(["Colissimo", "Chronopost", "UPS"]),
            "message": f"📦 Votre commande {order_id} est actuellement : {statuts[statut_idx]}."
        }
    
    def nouvelle_session(self) -> str:
        """Crée une nouvelle session de chat."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id
    
    def sauvegarder(self, path: Optional[Path] = None) -> None:
        """Sauvegarde les sessions et la configuration."""
        if path is None:
            path = MODELS_DIR / "chatbot_sessions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": self.config,
            "sessions": self.sessions,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Sessions sauvegardées : {path}")


# ============================================
# SAUVEGARDER LA BASE DE CONNAISSANCES
# ============================================
def sauvegarder_base_connaissances() -> None:
    """
    Sauvegarde la base de connaissances en fichiers JSON
    pour permettre l'indexation ultérieure.
    """
    KB_DIR.mkdir(parents=True, exist_ok=True)
    documents = creer_base_connaissances()
    
    kb_path = KB_DIR / "knowledge_base.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Base de connaissances sauvegardée : {kb_path}")
    logger.info(f"   {len(documents)} documents indexés")


# ============================================
# Point d'entrée principal
# ============================================
def main():
    """
    Démonstration du chatbot RAG.
    
    1. Créer la base de connaissances
    2. Initialiser le chatbot
    3. Tester des conversations
    """
    logger.info("=" * 60)
    logger.info("🤖 ECommerce-IA — Chatbot RAG Service Client")
    logger.info("=" * 60)
    
    # Sauvegarder la base de connaissances
    sauvegarder_base_connaissances()
    
    # Initialiser le chatbot
    chatbot = EcommerceChatbot()
    chatbot.initialiser()
    
    # Tests de conversation
    questions_test = [
        "Comment puis-je suivre ma commande ?",
        "Quels sont les délais de livraison ?",
        "Comment fonctionne la recherche par image ?",
        "Quelle est votre politique de retour ?",
        "Comment fonctionnent les recommandations personnalisées ?",
    ]
    
    session_id = chatbot.nouvelle_session()
    
    logger.info("")
    logger.info("💬 Démonstration de conversation")
    logger.info("=" * 60)
    
    for question in questions_test:
        logger.info(f"\n👤 Client : {question}")
        
        response = chatbot.generer_reponse(question, session_id)
        
        logger.info(f"🤖 Bot (confiance: {response['confiance']:.1%}) :")
        logger.info(f"   {response['reponse'][:300]}...")
        
        if response["escalade_humain"]:
            logger.info("   ⚠️  → Escalade vers agent humain")
        
        logger.info(f"   📚 Sources : {len(response['sources'])} documents")
    
    # Test vérification stock
    logger.info("")
    logger.info("📦 Test vérification stock :")
    stock = chatbot.verifier_stock("P0001")
    logger.info(f"   {stock['message']}")
    
    # Test suivi commande
    logger.info("")
    logger.info("📦 Test suivi commande :")
    suivi = chatbot.suivre_commande("CMD-2024-001")
    logger.info(f"   {suivi['message']}")
    
    # Historique
    historique = chatbot.get_historique(session_id)
    logger.info(f"\n📝 Historique : {len(historique)} messages dans la session")
    
    # Sauvegarder
    chatbot.sauvegarder()
    
    logger.info("")
    logger.info("✅ Chatbot RAG opérationnel !")


if __name__ == "__main__":
    main()
