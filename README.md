<div align="center">

# 🛒 ECommerce-IA — SAHELYS

### Plateforme E-Commerce Intelligente avec 5 Modules IA

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-4267B2?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**CNN + Transformer | FAISS | Recommandation | Chatbot RAG**

[Documentation API](http://localhost:8000/docs) · [Dashboard](http://localhost:8501) · [GitHub](https://github.com/Theobaw01/ECommerce-IA)

</div>

---

## 📋 Table des matières

- [Présentation](#-présentation)
- [Architecture](#-architecture)
- [Modules IA](#-modules-ia)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [API Endpoints](#-api-endpoints)
- [Entraînement sur Google Colab](#-entraînement-sur-google-colab)
- [Résultats & Métriques](#-résultats--métriques)
- [Déploiement Docker](#-déploiement-docker)
- [Structure du projet](#-structure-du-projet)
- [Auteur](#-auteur)

---

## 🎯 Présentation

**ECommerce-IA** est une plateforme e-commerce complète intégrant **5 modules d'intelligence artificielle**, couvrant les architectures majeures du Deep Learning : **CNN**, **Transformers**, **NLP**, et **recherche vectorielle**.

| Module | Technologie | Architecture | Performance |
|--------|------------|--------------|-------------|
| 🔍 **Classification CNN** | EfficientNet-B4 (PyTorch/timm) | **CNN** | ~85% Top-1, ~99% Top-5 |
| 🤖 **Classification ViT** | ViT-Base/16 (Transformer) | **Transformer** | Pré-entraîné ImageNet |
| 🖼️ **Recherche visuelle** | FAISS (Meta AI) | **Similarité cosine** | <50ms / requête |
| 🎯 **Recommandation** | SVD + Content-Based + Geo + Prix | **Hybride 4 facteurs** | Precision@10 = 0.78 |
| 💬 **Chatbot RAG** | LangChain + ChromaDB + LLM | **NLP / Transformer** | 87% résolution |

> Projet réalisé chez **SAHELYS** par **BAWANA Théodore**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Frontend Next.js 14                               │
│                    (TailwindCSS + React Components)                     │
│    ┌──────────┐ ┌──────────────┐ ┌────────┐ ┌──────────────┐           │
│    │ Image    │ │  Product     │ │Chatbot │ │ Confidence   │           │
│    │ Dropzone │ │    Card      │ │  Page  │ │    Bar       │           │
│    └──────────┘ └──────────────┘ └────────┘ └──────────────┘           │
├─────────────────────────────────────────────────────────────────────────┤
│              API FastAPI + JWT Auth + WebSocket                          │
│  POST /classify  POST /classify/vit  POST /search/image  POST /chat    │
│  POST /products  GET /recommend/{id}  POST /classify/compare           │
├──────────────┬──────────────┬──────────────┬────────────────────────────┤
│ EfficientNet │  ViT-Base/16 │    FAISS     │  LangChain + ChromaDB     │
│   B4 (CNN)   │ (Transformer)│  (Meta AI)   │  (RAG Pipeline)           │
│   19.3M      │    86M       │  IndexFlatIP │  Mistral / flan-t5        │
│   params     │   params     │  cosine sim  │                           │
├──────────────┴──────────────┴──────────────┼────────────────────────────┤
│              SVD Hybride (Surprise)         │   PostgreSQL + SQLAlchemy │
│     Collaborative + Content + Geo + Prix    │   Products │ Users │ ...  │
└─────────────────────────────────────────────┴────────────────────────────┘
```

---

## 🤖 Modules IA

### 🔍 Module 1 — Classification CNN (EfficientNet-B4)

Réseau convolutionnel avec compound scaling (Google Brain, Tan & Le, ICML 2019).

| Paramètre | Valeur |
|-----------|--------|
| **Modèle** | EfficientNet-B4 (pré-entraîné ImageNet) |
| **Framework** | PyTorch 2.0+ via timm |
| **Dataset** | Fashion Product Images (Kaggle) — 120 catégories × 120 images |
| **Input** | 380 × 380 pixels |
| **Entraînement** | Google Colab (T4 16 Go), 30 epochs |
| **Stratégie** | Progressive unfreezing (3 phases) |
| **Optimizer** | AdamW (lr=3e-4, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR |
| **Loss** | CrossEntropy + Label Smoothing 0.1 |
| **Augmentation** | RandomFlip, ColorJitter, Rotation, RandomErasing |
| **Early Stopping** | patience=7 |

### 🤖 Module 2 — Classification Transformer (ViT)

Vision Transformer (Dosovitskiy et al., ICLR 2021) : l'image est découpée en patches 16×16 traités comme des tokens, exactement comme en NLP.

| Paramètre | Valeur |
|-----------|--------|
| **Modèle** | ViT-Base/16 (vit_base_patch16_384) |
| **Architecture** | Transformer Encoder (12 layers, 12 heads) |
| **Paramètres** | 86M |
| **Input** | 384 × 384 → 576 patches + 1 [CLS] token |
| **Pré-entraînement** | ImageNet-21k + fine-tuné ImageNet-1k |
| **Endpoint** | `POST /classify/vit` |
| **Comparaison** | `POST /classify/compare` (CNN vs Transformer) |

> **Pourquoi deux architectures ?** Le CNN excelle avec des convolutions locales hiérarchiques. Le Transformer capture des dépendances globales via l'attention. La comparaison démontre la maîtrise des deux paradigmes fondamentaux du Deep Learning.

### 🖼️ Module 3 — Recherche Visuelle (FAISS)

Moteur de recherche d'images similaires utilisant FAISS (Facebook AI Similarity Search, licence MIT).

| Paramètre | Valeur |
|-----------|--------|
| **Bibliothèque** | FAISS (Meta AI) |
| **Index** | IndexFlatIP (cosine similarity après normalisation L2) |
| **Embeddings** | Extraits par EfficientNet-B4 backbone (1792-d) |
| **Temps de recherche** | < 50ms pour ~14,400 vecteurs |
| **Endpoint** | `POST /search/image` |

### 🎯 Module 4 — Recommandation Hybride

Algorithme à **4 facteurs pondérés** :

| Facteur | Poids | Algorithme |
|---------|-------|-----------|
| 📜 Historique | 40% | Collaborative Filtering (SVD — Surprise) |
| 🏷️ Similarité | 30% | Content-Based (TF-IDF + Cosine Similarity) |
| 📍 Géographie | 15% | Distance Haversine |
| 💰 Prix | 15% | Score budget utilisateur |

### 💬 Module 5 — Chatbot RAG

| Composant | Technologie |
|-----------|------------|
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | ChromaDB |
| **Génération** | Mistral-7B-Instruct (fallback: flan-t5-large) — **Transformer** |
| **Framework** | LangChain |
| **Protocole** | REST + WebSocket (temps réel) |

> **Note** : Le LLM utilisé par le chatbot (Mistral / flan-t5) est un **Transformer** (architecture encodeur-décodeur), ce qui renforce la couverture de cette architecture dans le projet.

---

## 🛠️ Technologies

### Backend IA
| Technologie | Rôle | Architecture |
|-------------|------|-------------|
| **PyTorch 2.0+** | Deep Learning | CNN + Transformer |
| **timm** | Modèles pré-entraînés | EfficientNet-B4, ViT-Base/16 |
| **FAISS** | Recherche vectorielle | IndexFlatIP (cosine) |
| **LangChain** | Pipeline RAG | Transformer (LLM) |
| **ChromaDB** | Base vectorielle | Embeddings |
| **Surprise** | Collaborative Filtering | SVD |
| **scikit-learn** | ML classique | TF-IDF, métriques |

### Backend API
| Technologie | Rôle |
|-------------|------|
| **FastAPI** | API REST + WebSocket |
| **SQLAlchemy** | ORM |
| **PostgreSQL** | Base de données |
| **JWT (python-jose)** | Authentification |

### Frontend
| Technologie | Rôle |
|-------------|------|
| **Next.js 14** | Framework React (App Router) |
| **TailwindCSS** | Styling |
| **Zustand** | State management |
| **Framer Motion** | Animations |

### DevOps
| Technologie | Rôle |
|-------------|------|
| **Docker** | Containerisation multi-stage |
| **docker-compose** | Orchestration (5 services) |
| **Google Colab** | Entraînement GPU (T4) |

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Node.js 18+ (pour le frontend)
- Docker & Docker Compose (optionnel)
- GPU NVIDIA + CUDA (recommandé pour l'inférence)

### Installation locale

```bash
# 1. Cloner le projet
git clone https://github.com/Theobaw01/ECommerce-IA.git
cd ECommerce-IA

# 2. Créer l'environnement Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env

# 5. Lancer l'API
python api/main.py
# → http://localhost:8000/docs

# 6. Lancer le frontend
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

---

## 📡 API Endpoints

### Classification
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/classify` | Classification CNN (EfficientNet-B4) |
| `POST` | `/classify/batch` | Classification batch |
| `POST` | `/classify/vit` | Classification Transformer (ViT) |
| `POST` | `/classify/compare` | Comparaison CNN vs Transformer |

### Recherche Visuelle (FAISS)
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/search/image` | Recherche de produits similaires par image |
| `GET` | `/search/categories` | Catégories indexées |
| `GET` | `/search/status` | Statut de l'index FAISS |

### Recommandation
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/recommend/{user_id}` | Recommandations personnalisées |
| `GET` | `/recommend/similar/{product_id}` | Produits similaires |
| `POST` | `/recommend/feedback` | Feedback utilisateur |

### Chatbot
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/chat` | Envoyer un message |
| `GET` | `/chat/history/{session_id}` | Historique de la session |
| `WS` | `/ws/chat/{session_id}` | Chat temps réel (WebSocket) |

### Produits (CRUD complet)
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/products` | Liste paginée avec filtres |
| `GET` | `/products/{id}` | Détail d'un produit |
| `POST` | `/products` | **Créer** un produit |
| `PUT` | `/products/{id}` | **Modifier** un produit |
| `DELETE` | `/products/{id}` | **Supprimer** un produit |
| `POST` | `/products/search` | Recherche par image |

### Authentification
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/auth/register` | Inscription |
| `POST` | `/auth/token` | Connexion (JWT) |

---

## ☁️ Entraînement sur Google Colab

Le notebook `notebooks/ECommerce_IA_Train_Colab.ipynb` est **entièrement autonome** :

1. Monte Google Drive pour la persistance
2. Télécharge le dataset depuis Kaggle
3. Organise les données (120 catégories × 120 images)
4. Entraîne EfficientNet-B4 avec progressive unfreezing (3 phases)
5. Évalue sur le test set
6. Extrait les embeddings pour FAISS
7. Sauvegarde tout sur Google Drive

**En cas de déconnexion Colab** : relancez « Exécuter tout » — tout reprend automatiquement.

### Configuration Kaggle sécurisée

Les credentials Kaggle sont chargées depuis les **Secrets Google Colab** (icône 🔑), jamais en dur dans le code.

---

## 📊 Résultats & Métriques

### Classification CNN (EfficientNet-B4)

| Métrique | Valeur |
|----------|--------|
| **Dataset** | 120 catégories × 120 images (14,400 total) |
| **Train/Val/Test** | ~67% / ~17% / ~17% |
| **Accuracy Top-1** | ~85% |
| **Accuracy Top-5** | ~99% |
| **Stratégie** | Progressive unfreezing (3 phases / 30 epochs) |
| **GPU** | Tesla T4 16 Go (Google Colab) |

### Recherche FAISS

| Métrique | Valeur |
|----------|--------|
| **Index** | IndexFlatIP (exact search) |
| **Vecteurs** | ~14,400 (1792-d chacun) |
| **Temps de recherche** | < 50ms |
| **Similarité** | Cosine (normalisation L2 + Inner Product) |

---

## 🐳 Déploiement Docker

```bash
docker-compose up --build -d
docker-compose ps
```

| Service | Port | URL |
|---------|------|-----|
| **Frontend** (Next.js) | 3000 | http://localhost:3000 |
| **API** (FastAPI) | 8000 | http://localhost:8000/docs |
| **Dashboard** (Streamlit) | 8501 | http://localhost:8501 |
| **Database** (PostgreSQL) | 5432 | — |
| **ChromaDB** | 8080 | http://localhost:8080 |

---

## 📁 Structure du projet

```
ECommerce-IA/
├── 📁 api/
│   └── main.py                   # API FastAPI (REST + WebSocket + JWT)
├── 📁 app/
│   └── streamlit_demo.py         # Dashboard Streamlit
├── 📁 data/
│   ├── download_dataset.py       # Téléchargement Kaggle
│   ├── raw/                      # Images brutes par catégorie
│   └── splits/                   # Train / Val / Test
├── 📁 database/
│   └── models.py                 # SQLAlchemy ORM
├── 📁 frontend/                  # Next.js 14 + TailwindCSS
│   ├── src/app/                  # Pages
│   ├── src/components/           # ImageDropzone, ProductCard, etc.
│   ├── src/services/             # API client (Axios + JWT)
│   └── src/stores/               # Zustand (Cart, Auth)
├── 📁 models/classification/     # Modèles entraînés
│   ├── efficientnet_b4_best.pth  # CNN (EfficientNet-B4)
│   ├── product_embeddings.pkl    # Embeddings pour FAISS
│   ├── class_mapping.json        # Mapping des classes
│   └── training_history.json     # Historique d'entraînement
├── 📁 notebooks/
│   └── ECommerce_IA_Train_Colab.ipynb  # Notebook Colab (autonome)
├── 📁 src/
│   ├── dataset.py                # Dataset PyTorch + DataLoaders
│   ├── train_classification.py   # Entraînement EfficientNet-B4
│   ├── evaluate.py               # Évaluation sur test set
│   ├── image_search.py           # 🆕 Recherche FAISS (Meta AI)
│   ├── vit_classifier.py         # 🆕 Vision Transformer (ViT-Base/16)
│   ├── pipeline.py               # Pipeline unifié (5 modules)
│   ├── recommendation.py         # Recommandation hybride 4 facteurs
│   ├── chatbot.py                # Chatbot RAG (LangChain + ChromaDB)
│   └── preprocess.py             # Prétraitement images
├── docker-compose.yml            # 5 services orchestrés
├── Dockerfile                    # Multi-stage build
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

---

## 🧠 Architectures IA couvertes

| Architecture | Module | Référence |
|-------------|--------|-----------|
| **CNN** | EfficientNet-B4 | Tan & Le, ICML 2019 |
| **Transformer** | ViT-Base/16 + LLM (Chatbot) | Dosovitskiy et al., ICLR 2021 |
| **NLP** | Chatbot RAG (LangChain) | Lewis et al., NeurIPS 2020 |
| **Recherche vectorielle** | FAISS (Meta AI) | Johnson et al., IEEE Big Data 2019 |
| **Filtrage collaboratif** | SVD (Surprise) | Koren et al., 2009 |

---

## 👨‍💻 Auteur

<div align="center">

| | |
|---|---|
| **Nom** | BAWANA Théodore |
| **Projet** | Réalisé chez **SAHELYS** |
| **GitHub** | [github.com/Theobaw01](https://github.com/Theobaw01) |
| **Email** | [theodore8bawana@gmail.com](mailto:theodore8bawana@gmail.com) |

</div>

---

## 📄 Licence

Ce projet est sous licence MIT.

---

<div align="center">

**⭐ Si ce projet vous intéresse, n'hésitez pas à laisser une étoile !**

*ECommerce-IA — CNN + Transformers + FAISS + NLP au service du e-commerce*

</div>
