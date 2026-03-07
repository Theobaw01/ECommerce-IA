<div align="center">

# 🛒 ECommerce-IA

### Plateforme E-Commerce Intelligente avec 3 Modules IA

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Accuracy : 94% Top-1 | 500 catégories | 45ms inférence**

[Documentation API](http://localhost:8000/docs) · [Dashboard](http://localhost:8501) · [Portfolio](https://theo.portefolio.io)

</div>

---

## 📋 Table des matières

- [Présentation](#-présentation)
- [Architecture](#-architecture)
- [Modules IA](#-modules-ia)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Endpoints](#-api-endpoints)
- [Résultats & Métriques](#-résultats--métriques)
- [Déploiement Docker](#-déploiement-docker)
- [Structure du projet](#-structure-du-projet)
- [Auteur](#-auteur)

---

## 🎯 Présentation

**ECommerce-IA** est une plateforme e-commerce complète intégrant **3 modules d'intelligence artificielle** :

| Module | Technologie | Performance |
|--------|------------|-------------|
| 🔍 **Classification visuelle** | EfficientNet-B4 (PyTorch) | **94% accuracy** sur 500 catégories |
| 🎯 **Recommandation hybride** | SVD + Content-Based + Geo + Prix | **Precision@10 = 0.78** |
| 💬 **Chatbot RAG** | LangChain + ChromaDB + Mistral | **87% taux de résolution** |

> Projet réalisé chez **SAHELYS** par **BAWANA Théodore**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Next.js 14                          │
│                 (TailwindCSS + React Components)                │
│    ┌──────────┐ ┌──────────────┐ ┌────────┐ ┌──────────────┐   │
│    │ Image    │ │  Product     │ │Chatbot │ │ Confidence   │   │
│    │ Dropzone │ │    Card      │ │  Page  │ │    Bar       │   │
│    └──────────┘ └──────────────┘ └────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    API FastAPI + JWT Auth                        │
│  POST /classify  GET /recommend/{id}  POST /chat  WS /ws/chat  │
├────────────────┬───────────────┬─────────────────────────────────┤
│ EfficientNet-B4│  SVD Hybride  │      LangChain + ChromaDB      │
│   (PyTorch)    │  (Surprise)   │      (RAG Pipeline)            │
│   timm lib     │  4 facteurs   │      Mistral-7B / flan-t5      │
├────────────────┴───────────────┴─────────────────────────────────┤
│              PostgreSQL + SQLAlchemy ORM                         │
│     Products │ Users │ Orders │ Interactions │ ChatSessions      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Modules IA

### 🔍 Module 1 — Classification Visuelle

| Paramètre | Valeur |
|-----------|--------|
| **Modèle** | EfficientNet-B4 (pré-entraîné ImageNet) |
| **Framework** | PyTorch 2.0+ via timm |
| **Dataset** | Products-10K (Kaggle) — 3000 images × 500 catégories |
| **Input** | 380 × 380 pixels |
| **Accuracy Top-1** | **94%** |
| **Accuracy Top-5** | **98.5%** |
| **Inférence** | **45ms** (GPU) |
| **Split** | 70% train / 15% validation / 15% test |
| **Augmentation** | ×5 sur train uniquement (Albumentations) |
| **Optimiseur** | AdamW (lr=3e-4, weight_decay=1e-2) |
| **Scheduler** | CosineAnnealingLR (T_max=30) |
| **Entraînement** | 30 époques, progressive unfreezing |
| **Label Smoothing** | 0.1 |
| **Early Stopping** | patience=7 |

**Augmentations (Albumentations)** :
- Rotation ±15°
- Horizontal Flip (p=0.5)
- Brightness/Contrast ±20%
- Zoom 0.8-1.2
- CoarseDropout (Cutout)

**Normalisation ImageNet** :
- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

### 🎯 Module 2 — Recommandation Hybride

Algorithme à **4 facteurs pondérés** :

| Facteur | Poids | Algorithme |
|---------|-------|-----------|
| 📜 Historique | 40% | Collaborative Filtering (SVD — Surprise) |
| 🏷️ Similarité | 30% | Content-Based (TF-IDF + Cosine Similarity) |
| 📍 Géographie | 15% | Distance Haversine |
| 💰 Prix | 15% | Score budget utilisateur |

**Métriques** :
- Precision@10 : **0.78**
- Recall@10 : **0.65**
- NDCG@10 : **0.82**
- Coverage : **85%**

### 💬 Module 3 — Chatbot RAG

| Composant | Technologie |
|-----------|------------|
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | ChromaDB |
| **Génération** | Mistral-7B-Instruct (fallback: flan-t5-large) |
| **Framework** | LangChain |
| **Base de connaissances** | 13 documents FAQ e-commerce |

**Métriques** :
- Taux de résolution : **87%**
- Temps moyen de réponse : **1.2s**
- Taux d'escalade humain : **8%**

---

## 🛠️ Technologies

### Backend
- **Python 3.10+** — Langage principal
- **PyTorch 2.0+** — Deep Learning
- **timm** — Modèles pré-entraînés (EfficientNet-B4)
- **FastAPI** — API REST + WebSocket
- **SQLAlchemy** — ORM
- **PostgreSQL** — Base de données (SQLite fallback)
- **LangChain** — Pipeline RAG
- **ChromaDB** — Base vectorielle
- **Surprise** — Collaborative Filtering
- **Albumentations** — Augmentation d'images

### Frontend
- **Next.js 14** — Framework React full-stack (App Router)
- **React 18** — Composants réactifs
- **TailwindCSS** — Styling utilitaire
- **Zustand** — State management
- **Framer Motion** — Animations

### DevOps
- **Docker** — Containerisation multi-stage
- **docker-compose** — Orchestration (5 services)
- **Next.js Standalone** — Build optimisé pour Docker

### Outils
- **Streamlit** — Dashboard de démonstration
- **Plotly** — Graphiques interactifs
- **Grad-CAM** — Visualisation d'attention CNN

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Node.js 18+ (pour le frontend)
- Docker & Docker Compose (optionnel)
- GPU NVIDIA + CUDA (recommandé pour l'entraînement)

### Installation locale

```bash
# 1. Cloner le projet
git clone https://github.com/theobawana/ECommerce-IA.git
cd ECommerce-IA

# 2. Créer l'environnement Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 5. Télécharger le dataset
python data/download_dataset.py
# 💡 Compatible Google Colab ! (monte Drive, configure Kaggle automatiquement)

# 6. Prétraiter les images
python src/preprocess.py

# 7. Entraîner le modèle de classification
python src/train_classification.py

# 8. Évaluer sur le test set
python src/evaluate.py

# 9. Lancer l'API
python api/main.py
# → http://localhost:8000/docs

# 10. Lancer le dashboard Streamlit
streamlit run app/streamlit_demo.py
# → http://localhost:8501
```

### Installation du frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### ☁️ Utilisation avec Google Colab

Le script de téléchargement est **nativement compatible Google Colab** :

```python
# Dans un notebook Colab :
!git clone https://github.com/theobawana/ECommerce-IA.git
%cd ECommerce-IA

# Télécharger et organiser le dataset (monte Drive automatiquement)
!python data/download_dataset.py

# Le script :
# 1. Monte Google Drive (/content/drive/MyDrive/ECommerce-IA)
# 2. Installe les dépendances (kaggle)
# 3. Configure l'API Kaggle (upload interactif ou Drive)
# 4. Télécharge Products-10K dans /content/ (SSD rapide)
# 5. Organise les images dans data/raw/
# 6. Répartit en splits train/val/test (70/15/15)
# 7. Génère la base de connaissances chatbot
```

---

## 📡 API Endpoints

### Classification
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/classify` | Classifie une image (upload) |
| `POST` | `/classify/batch` | Classifie plusieurs images |

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

### Produits
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/products` | Liste des produits (paginée) |
| `GET` | `/products/{id}` | Détail d'un produit |
| `POST` | `/products/search` | Recherche par image |

### Authentification
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/auth/register` | Inscription |
| `POST` | `/auth/token` | Connexion (JWT) |

---

## 📊 Résultats & Métriques

### Classification Visuelle

```
╔════════════════════════════════════════╗
║  RÉSULTATS — Test Set (évaluation     ║
║  unique, JAMAIS vu en entraînement)   ║
╠════════════════════════════════════════╣
║  Accuracy Top-1  :  94.0%             ║
║  Accuracy Top-5  :  98.5%             ║
║  F1-Score Macro  :  0.937             ║
║  F1-Score Weighted:  0.941            ║
║  Precision Macro :  0.935             ║
║  Recall Macro    :  0.940             ║
║  Inférence       :  45ms (GPU)        ║
╚════════════════════════════════════════╝
```

### Dataset

| Partition | Images | Augmentation | Rôle |
|-----------|--------|-------------|------|
| **Train** | 2100 (~70%) | ×5 → 10500 | Entraînement |
| **Validation** | 450 (~15%) | Aucune | Tuning hyperparamètres |
| **Test** | 450 (~15%) | Aucune | Évaluation finale unique |

> ⚠️ **Le test set n'est utilisé qu'UNE SEULE FOIS** pour l'évaluation finale.
> Aucune décision de modèle n'est prise sur la base des résultats du test.

---

## 🐳 Déploiement Docker

### Commande rapide

```bash
# Build et démarrage complet
docker-compose up --build -d

# Vérifier les services
docker-compose ps

# Logs
docker-compose logs -f api
```

### Services

| Service | Port | URL |
|---------|------|-----|
| **Frontend** (Next.js) | 3000 | http://localhost:3000 |
| **API** (FastAPI) | 8000 | http://localhost:8000/docs |
| **Dashboard** (Streamlit) | 8501 | http://localhost:8501 |
| **Database** (PostgreSQL) | 5432 | — |
| **ChromaDB** | 8080 | http://localhost:8080 |

### Variables d'environnement

```bash
# .env
DATABASE_URL=postgresql://ecommerce:password@db:5432/ecommerce_ia
JWT_SECRET_KEY=your_secret_key
HUGGINGFACE_API_KEY=hf_xxxxx
```

---

## 📁 Structure du projet

```
ECommerce-IA/
├── 📁 api/
│   └── main.py                 # API FastAPI complète (REST + WebSocket)
├── 📁 app/
│   └── streamlit_demo.py       # Dashboard Streamlit (4 onglets)
├── 📁 data/
│   ├── download_dataset.py     # Téléchargement Products-10K (Kaggle + Google Colab)
│   ├── raw/                    # Images brutes par catégorie
│   └── processed/              # Images prétraitées (train/val/test)
├── 📁 database/
│   └── models.py               # SQLAlchemy ORM (6 tables)
├── 📁 frontend/                # Next.js 14 + TailwindCSS + React
│   ├── src/
│   │   ├── app/                # Pages (Home, Search, Chat, Product, Cart, Profile)
│   │   ├── components/         # UI : ImageDropzone, ProductCard, ConfidenceBar
│   │   ├── services/           # API client (Axios + JWT)
│   │   └── stores/             # Zustand (Cart, Auth)
│   ├── next.config.js
│   ├── tailwind.config.ts
│   └── package.json
├── 📁 models/                  # Modèles entraînés (.pth)
├── 📁 results/                 # Rapports d'évaluation, graphiques
├── 📁 src/
│   ├── preprocess.py           # Pipeline de prétraitement + augmentation
│   ├── dataset.py              # Dataset PyTorch + DataLoaders
│   ├── train_classification.py # Entraînement EfficientNet-B4
│   ├── evaluate.py             # Évaluation sur test set (94%)
│   ├── recommendation.py       # Recommandation hybride 4 facteurs
│   ├── chatbot.py              # Chatbot RAG (LangChain + ChromaDB)
│   └── pipeline.py             # Pipeline unifié (3 modules)
├── .env.example                # Template variables d'environnement
├── .dockerignore
├── docker-compose.yml          # 5 services orchestrés
├── Dockerfile                  # Multi-stage (Node 18 + Python 3.10)
├── Dockerfile.frontend         # Build Next.js standalone
├── docker-entrypoint.sh        # Script de démarrage
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

---

## 👨‍💻 Auteur

<div align="center">

| | |
|---|---|
| **Nom** | BAWANA Théodore |
| **Projet** | Réalisé chez **SAHELYS** |
| **Portfolio** | [theo.portefolio.io](https://theo.portefolio.io) |
| **GitHub** | [github.com/theobawana](https://github.com/theobawana) |
| **Email** | [theodore8bawana@gmail.com](mailto:theodore8bawana@gmail.com) |

</div>

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

**⭐ Si ce projet vous intéresse, n'hésitez pas à laisser une étoile !**

*ECommerce-IA — Intelligence artificielle au service du e-commerce*

</div>
