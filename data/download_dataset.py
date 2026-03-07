"""
============================================
ECommerce-IA — Téléchargement du Dataset
============================================
Notebook/Script optimisé pour Google Colab.

Télécharge le dataset Products-10K (Fashion Product Images)
depuis Kaggle et organise les fichiers dans les dossiers du projet.

Dataset : paramaggarwal/fashion-product-images-dataset
Source  : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Stratégie :
- Monter Google Drive (si Colab)
- Installer & configurer l'API Kaggle
- Télécharger le dataset complet
- Sélectionner 3 000 images (500 catégories × 6 images)
- Organiser dans data/raw/ avec un dossier par catégorie
- Séparer dans data/splits/ (train 70% / val 15% / test 15%)

Usage dans Google Colab :
    !python data/download_dataset.py

Auteur  : BAWANA Théodore — Projet réalisé chez SAHELYS
"""

import os
import sys
import json
import shutil
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

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
# Détection Google Colab
# ============================================
IS_COLAB = False
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    pass

# ============================================
# Chemins du projet
# ============================================
if IS_COLAB:
    # Monter Google Drive automatiquement
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

    # Le projet est dans Google Drive
    PROJECT_ROOT = Path("/content/drive/MyDrive/ECommerce-IA")
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Google Colab détecté — projet dans : {PROJECT_ROOT}")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
MODELS_DIR = PROJECT_ROOT / "models"

KAGGLE_DATASET = "paramaggarwal/fashion-product-images-dataset"

# Paramètres de sélection
NB_CATEGORIES = 500
IMAGES_PAR_CATEGORIE = 6   # 500 × 6 = 3 000 images
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def creer_structure_dossiers() -> None:
    """
    Crée toute l'arborescence du projet dans Google Drive (ou local).
    Garantit que chaque dossier existe avant d'y écrire.
    """
    dossiers = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        SPLITS_DIR / "train",
        SPLITS_DIR / "val",
        SPLITS_DIR / "test",
        KNOWLEDGE_BASE_DIR,
        MODELS_DIR / "classification",
        MODELS_DIR / "recommendation",
        MODELS_DIR / "chatbot",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "notebooks",
    ]
    for d in dossiers:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"✅ Structure de dossiers créée sous : {PROJECT_ROOT}")


def installer_dependances_colab() -> None:
    """
    Installe les paquets nécessaires dans Google Colab.
    Colab a déjà PyTorch/numpy/pandas mais il manque kaggle, timm, etc.
    """
    if not IS_COLAB:
        return

    logger.info("📦 Installation des dépendances Colab...")
    os.system("pip install -q kaggle")
    logger.info("✅ Dépendances installées.")


def configurer_kaggle_colab(kaggle_json_path: Optional[str] = None) -> bool:
    """
    Configure l'API Kaggle dans Google Colab.

    3 méthodes supportées (dans l'ordre de priorité) :
    1. Upload interactif du fichier kaggle.json
    2. Chemin vers kaggle.json dans Google Drive
    3. Variables d'environnement KAGGLE_USERNAME / KAGGLE_KEY

    Args:
        kaggle_json_path: (optionnel) chemin absolu vers kaggle.json
                          ex: "/content/drive/MyDrive/kaggle.json"

    Returns:
        True si l'authentification réussit
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_file = kaggle_dir / "kaggle.json"

    # ---- Méthode 1 : fichier déjà en place ou fourni via chemin ----
    if kaggle_json_path and Path(kaggle_json_path).exists():
        shutil.copy(kaggle_json_path, kaggle_file)
        os.chmod(str(kaggle_file), 0o600)
        logger.info(f"✅ kaggle.json copié depuis : {kaggle_json_path}")

    elif kaggle_file.exists():
        logger.info("✅ kaggle.json déjà présent.")

    # ---- Méthode 2 : chercher dans Google Drive ----
    elif IS_COLAB:
        drive_candidates = [
            Path("/content/drive/MyDrive/kaggle.json"),
            Path("/content/drive/MyDrive/Colab/kaggle.json"),
            Path("/content/drive/MyDrive/.kaggle/kaggle.json"),
        ]
        found = False
        for candidate in drive_candidates:
            if candidate.exists():
                shutil.copy(str(candidate), str(kaggle_file))
                os.chmod(str(kaggle_file), 0o600)
                logger.info(f"✅ kaggle.json trouvé dans Drive : {candidate}")
                found = True
                break

        if not found:
            # ---- Méthode 3 : upload interactif ----
            logger.info("📎 Veuillez uploader votre fichier kaggle.json :")
            logger.info("   (Kaggle → Settings → API → Create New Token)")
            try:
                from google.colab import files
                uploaded = files.upload()   # popup d'upload
                if "kaggle.json" in uploaded:
                    with open(str(kaggle_file), "wb") as f:
                        f.write(uploaded["kaggle.json"])
                    os.chmod(str(kaggle_file), 0o600)
                    logger.info("✅ kaggle.json uploadé avec succès.")
                else:
                    logger.warning("⚠️  Fichier kaggle.json non détecté dans l'upload.")
                    return False
            except Exception as e:
                logger.error(f"❌ Upload échoué : {e}")
                return False

    # ---- Méthode 4 : variables d'environnement ----
    if not kaggle_file.exists():
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            logger.info("✅ Credentials Kaggle via variables d'environnement.")
        else:
            logger.error("❌ Impossible de configurer Kaggle.")
            return False

    # Tester l'authentification
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info("✅ API Kaggle authentifiée avec succès.")
        return True
    except Exception as e:
        logger.error(f"❌ Authentification Kaggle échouée : {e}")
        return False


def telecharger_dataset() -> Path:
    """
    Télécharge le dataset Fashion Product Images depuis Kaggle.
    Utilise le cache Colab /content/ pour éviter de saturer le Drive.

    Returns:
        Path vers le dossier contenant les données téléchargées
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Télécharger dans /content (rapide, SSD Colab) plutôt que Drive
    if IS_COLAB:
        download_dir = Path("/content/kaggle_download")
    else:
        download_dir = DATA_DIR / "kaggle_download"

    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📥 Téléchargement du dataset : {KAGGLE_DATASET}")
    logger.info(f"   Destination : {download_dir}")

    api = KaggleApi()
    api.authenticate()

    try:
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(download_dir),
            unzip=True
        )
        logger.info("✅ Dataset téléchargé et décompressé.")
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement : {e}")
        logger.info("💡 Téléchargez manuellement :")
        logger.info(f"   https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        logger.info(f"   Et dézippez dans : {download_dir}")
        raise

    return download_dir


def charger_metadata(download_dir: Path) -> pd.DataFrame:
    """
    Charge les métadonnées du dataset (styles.csv).
    """
    csv_candidates = list(download_dir.rglob("styles.csv"))
    if not csv_candidates:
        csv_candidates = list(download_dir.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {download_dir}"
        )

    csv_path = csv_candidates[0]
    logger.info(f"📄 Métadonnées : {csv_path}")

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(csv_path, error_bad_lines=False)

    logger.info(f"   {len(df)} produits | Colonnes : {list(df.columns)}")

    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    return df


def trouver_images(download_dir: Path) -> Dict[str, Path]:
    """
    Localise toutes les images du dataset.
    """
    images_dict = {}
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for img_path in download_dir.rglob("*"):
        if img_path.suffix.lower() in extensions:
            images_dict[img_path.stem] = img_path

    logger.info(f"🖼️  {len(images_dict)} images trouvées")
    return images_dict


def selectionner_sous_ensemble(
    df: pd.DataFrame,
    images_dict: Dict[str, Path],
    nb_categories: int = NB_CATEGORIES,
    images_par_cat: int = IMAGES_PAR_CATEGORIE,
    seed: int = SEED
) -> Tuple[pd.DataFrame, str]:
    """
    Sélectionne un sous-ensemble équilibré :
    nb_categories catégories × images_par_cat images = 3 000.
    """
    random.seed(seed)

    # Trouver la colonne de catégorie
    cat_col = None
    for col in ["articleType", "subCategory", "masterCategory", "category"]:
        if col in df.columns:
            cat_col = col
            break
    if cat_col is None:
        for col in df.columns:
            if df[col].dtype == "object" and df[col].nunique() > 10:
                cat_col = col
                break
    if cat_col is None:
        raise ValueError("Impossible de trouver une colonne de catégorie")

    logger.info(f"📊 Colonne catégorie : '{cat_col}' ({df[cat_col].nunique()} uniques)")

    # Filtrer les produits ayant une image
    df_with_img = df[df["id"].isin(images_dict.keys())].copy()
    logger.info(f"   {len(df_with_img)} produits avec image")

    cat_counts = df_with_img[cat_col].value_counts()
    categories_valides = cat_counts[cat_counts >= images_par_cat].index.tolist()
    logger.info(f"   {len(categories_valides)} catégories avec ≥{images_par_cat} images")

    categories_sel = categories_valides[:nb_categories]
    nb_cat_final = len(categories_sel)
    if nb_cat_final < nb_categories:
        logger.warning(f"⚠️  {nb_cat_final}/{nb_categories} catégories disponibles")

    selection = []
    for cat in categories_sel:
        cat_df = df_with_img[df_with_img[cat_col] == cat]
        echantillon = cat_df.sample(n=min(images_par_cat, len(cat_df)), random_state=seed)
        selection.append(echantillon)

    df_selection = pd.concat(selection, ignore_index=True)
    logger.info(f"✅ Sous-ensemble : {nb_cat_final} catégories, {len(df_selection)} images")

    return df_selection, cat_col


def organiser_images(
    df_selection: pd.DataFrame,
    images_dict: Dict[str, Path],
    cat_col: str
) -> None:
    """
    Copie les images dans data/raw/<catégorie>/ puis les répartit
    dans data/splits/{train,val,test}/<catégorie>/.
    """
    logger.info(f"📂 Organisation des images dans {RAW_DIR}")

    # Nettoyer raw
    if RAW_DIR.exists():
        for item in RAW_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

    compteur, erreurs = 0, 0

    for _, row in tqdm(df_selection.iterrows(), total=len(df_selection),
                       desc="Copie des images"):
        product_id = str(row["id"])
        categorie = str(row[cat_col]).strip()
        categorie_clean = categorie.replace("/", "_").replace("\\", "_")
        categorie_clean = categorie_clean.replace(" ", "_").lower()

        cat_dir = RAW_DIR / categorie_clean
        cat_dir.mkdir(parents=True, exist_ok=True)

        if product_id in images_dict:
            src = images_dict[product_id]
            dst = cat_dir / f"{product_id}{src.suffix}"
            try:
                shutil.copy2(str(src), str(dst))
                compteur += 1
            except Exception as e:
                logger.warning(f"⚠️  Erreur copie {product_id} : {e}")
                erreurs += 1
        else:
            erreurs += 1

    logger.info(f"✅ {compteur} images copiées ({erreurs} erreurs)")


def repartir_splits() -> None:
    """
    Répartit les images de data/raw/ dans data/splits/{train,val,test}/
    avec un ratio 70 / 15 / 15 (stratifié par catégorie).

    Règles strictes :
    - Pas de fuite de données entre les splits
    - Augmentation appliquée UNIQUEMENT sur train (fait dans preprocess.py)
    - Le test set ne sera utilisé qu'UNE SEULE FOIS
    """
    logger.info("📊 Répartition train / val / test...")

    random.seed(SEED)

    # Nettoyer les splits existants
    for split in ["train", "val", "test"]:
        split_dir = SPLITS_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0}

    categories = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

    for cat_dir in tqdm(categories, desc="Split par catégorie"):
        cat_name = cat_dir.name
        images = sorted(list(cat_dir.glob("*")))
        images = [img for img in images if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

        random.shuffle(images)
        n = len(images)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        # Le reste va dans test
        split_indices = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, split_images in split_indices.items():
            dst_dir = SPLITS_DIR / split_name / cat_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(str(img), str(dst_dir / img.name))
                stats[split_name] += 1

    total = sum(stats.values())
    logger.info(f"✅ Splits créés :")
    logger.info(f"   Train : {stats['train']} ({stats['train']/total:.0%})")
    logger.info(f"   Val   : {stats['val']}   ({stats['val']/total:.0%})")
    logger.info(f"   Test  : {stats['test']}  ({stats['test']/total:.0%})")

    # Sauvegarder les stats de split
    split_stats_path = DATA_DIR / "split_stats.json"
    with open(split_stats_path, "w", encoding="utf-8") as f:
        json.dump({**stats, "total": total, "ratio": f"{TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}"}, f, indent=2)
    logger.info(f"💾 Stats sauvegardées : {split_stats_path}")


def sauvegarder_metadata(df_selection: pd.DataFrame, cat_col: str) -> None:
    """
    Sauvegarde les métadonnées, le mapping des classes et les statistiques.
    """
    meta_path = DATA_DIR / "metadata_selection.csv"
    df_selection.to_csv(meta_path, index=False)
    logger.info(f"💾 Métadonnées : {meta_path}")

    categories = sorted(df_selection[cat_col].unique().tolist())
    class_mapping = {cat: idx for idx, cat in enumerate(categories)}

    mapping_path = DATA_DIR / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Mapping : {mapping_path} ({len(class_mapping)} classes)")

    stats = {
        "total_images": len(df_selection),
        "nb_categories": len(categories),
        "images_par_categorie": IMAGES_PAR_CATEGORIE,
        "categories": categories,
        "dataset_source": KAGGLE_DATASET,
        "seed": SEED,
    }
    stats_path = DATA_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Statistiques : {stats_path}")


def generer_base_connaissances() -> None:
    """
    Génère la base de connaissances initiale pour le chatbot RAG.
    Fichiers texte dans data/knowledge_base/.
    """
    documents = {
        "livraison.txt": (
            "Politique de livraison ECommerce-IA\n\n"
            "Livraison standard : 3 à 5 jours ouvrés, offerte dès 50€ d'achat.\n"
            "Livraison express : 24h, 9,99€.\n"
            "Livraison internationale : 7 à 14 jours, tarif calculé au poids.\n"
            "Suivi en temps réel disponible dans l'espace client.\n"
            "Points relais disponibles en France métropolitaine."
        ),
        "retours.txt": (
            "Politique de retours et remboursements\n\n"
            "Retour gratuit sous 30 jours après réception.\n"
            "Le produit doit être dans son emballage d'origine, non porté.\n"
            "Remboursement sous 5 à 10 jours ouvrés après réception du retour.\n"
            "Échange possible pour une autre taille ou couleur.\n"
            "Étiquette de retour prépayée envoyée par email."
        ),
        "paiement.txt": (
            "Moyens de paiement acceptés\n\n"
            "Carte bancaire : Visa, Mastercard, American Express.\n"
            "PayPal et Apple Pay.\n"
            "Paiement en 3x ou 4x sans frais dès 100€ (via Alma).\n"
            "Virement bancaire pour les commandes professionnelles.\n"
            "Transactions sécurisées par chiffrement SSL 256 bits."
        ),
        "garantie.txt": (
            "Garanties produits\n\n"
            "Garantie légale de conformité : 2 ans.\n"
            "Garantie commerciale étendue disponible (+1 an, 5€).\n"
            "SAV joignable par chat, email ou téléphone.\n"
            "Remplacement ou remboursement en cas de défaut de fabrication."
        ),
        "compte.txt": (
            "Gestion de compte\n\n"
            "Inscription gratuite avec email ou OAuth Google/GitHub.\n"
            "Historique de commandes complet.\n"
            "Liste de souhaits et alertes de prix.\n"
            "Suppression du compte sur simple demande (RGPD)."
        ),
        "recommandations.txt": (
            "Système de recommandation\n\n"
            "Algorithme hybride à 4 facteurs :\n"
            "- Historique d'achats (40%) : collaborative filtering SVD\n"
            "- Similarité produits (30%) : content-based filtering\n"
            "- Proximité géographique (15%) : distance Haversine\n"
            "- Budget utilisateur (15%) : filtrage par gamme de prix\n"
            "Les recommandations sont recalculées en temps réel."
        ),
        "promotions.txt": (
            "Offres et promotions\n\n"
            "Soldes saisonnières : jusqu'à -60%.\n"
            "Ventes flash quotidiennes de 12h à 14h.\n"
            "Code parrainage : 10€ offerts pour le parrain et le filleul.\n"
            "Newsletter : offres exclusives et accès anticipé."
        ),
    }

    for filename, content in documents.items():
        filepath = KNOWLEDGE_BASE_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info(f"📚 Base de connaissances : {len(documents)} documents dans {KNOWLEDGE_BASE_DIR}")


def generer_dataset_demo() -> None:
    """
    Génère un mini-dataset de démonstration (images synthétiques)
    si le téléchargement Kaggle échoue.
    """
    logger.info("🔧 Génération du dataset de démonstration...")

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.error("❌ pip install Pillow numpy")
        return

    categories_demo = [
        "tshirt", "pantalon", "robe", "chaussures", "sac",
        "montre", "lunettes", "chapeau", "ceinture", "echarpe",
        "veste", "short", "jupe", "pull", "chemise",
        "sandales", "baskets", "bottes", "manteau", "gilet"
    ]
    images_par_cat = 6
    random.seed(SEED)
    np.random.seed(SEED)

    class_mapping = {}

    for idx, cat in enumerate(categories_demo):
        cat_dir = RAW_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        class_mapping[cat] = idx

        for i in range(images_par_cat):
            color = np.array([
                (idx * 37 + i * 13) % 256,
                (idx * 53 + i * 29) % 256,
                (idx * 71 + i * 41) % 256
            ], dtype=np.uint8)
            img_array = np.full((380, 380, 3), color, dtype=np.uint8)
            noise = np.random.randint(-30, 30, (380, 380, 3), dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array)
            img.save(cat_dir / f"demo_{cat}_{i:03d}.jpg")

    mapping_path = DATA_DIR / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    total = len(categories_demo) * images_par_cat
    stats = {
        "total_images": total,
        "nb_categories": len(categories_demo),
        "images_par_categorie": images_par_cat,
        "categories": categories_demo,
        "dataset_source": "demo_synthétique",
        "seed": SEED,
        "note": "Dataset démo — remplacer par Products-10K pour le vrai entraînement"
    }
    stats_path = DATA_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Demo : {len(categories_demo)} catégories, {total} images")


# ============================================
# Point d'entrée principal
# ============================================
def main():
    """
    Pipeline complet (Google Colab compatible) :
    1. Créer la structure de dossiers (Drive ou local)
    2. Installer les dépendances Colab
    3. Configurer & authentifier Kaggle
    4. Télécharger le dataset Products-10K
    5. Sélectionner 3 000 images (500 cat × 6)
    6. Organiser dans data/raw/
    7. Répartir dans data/splits/ (70/15/15)
    8. Sauvegarder métadonnées + mapping
    9. Générer la base de connaissances chatbot
    """
    logger.info("=" * 60)
    logger.info("🚀 ECommerce-IA — Téléchargement du Dataset")
    if IS_COLAB:
        logger.info("   🟢 Mode Google Colab")
    else:
        logger.info("   🔵 Mode local")
    logger.info("=" * 60)

    # Étape 1 : structure
    creer_structure_dossiers()

    # Étape 2 : dépendances Colab
    installer_dependances_colab()

    # Vérifier si déjà téléchargé
    existing_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()] if RAW_DIR.exists() else []
    if len(existing_dirs) > 5 and "--force" not in sys.argv:
        logger.info(f"📂 Dataset déjà présent ({len(existing_dirs)} catégories)")
        logger.info("   Ajoutez --force pour re-télécharger")

        # Vérifier si les splits existent
        if not (SPLITS_DIR / "train").exists() or len(list((SPLITS_DIR / "train").iterdir())) == 0:
            repartir_splits()

        generer_base_connaissances()
        return

    # Étape 3-6 : Kaggle
    if configurer_kaggle_colab():
        try:
            download_dir = telecharger_dataset()
            df = charger_metadata(download_dir)
            images = trouver_images(download_dir)

            if len(images) == 0:
                raise FileNotFoundError("Aucune image trouvée")

            df_selection, cat_col = selectionner_sous_ensemble(df, images)
            organiser_images(df_selection, images, cat_col)
            sauvegarder_metadata(df_selection, cat_col)

            # Nettoyage du dossier temporaire
            logger.info("🧹 Nettoyage des fichiers temporaires...")
            shutil.rmtree(download_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"❌ Erreur Kaggle : {e}")
            logger.info("🔄 Fallback → dataset de démonstration")
            generer_dataset_demo()
    else:
        logger.info("🔄 Kaggle non configuré → dataset de démonstration")
        generer_dataset_demo()

    # Étape 7 : splits
    repartir_splits()

    # Étape 8 : base de connaissances
    generer_base_connaissances()

    # ---- Résumé ----
    nb_images = sum(1 for _ in RAW_DIR.rglob("*.jpg"))
    nb_categories = len([d for d in RAW_DIR.iterdir() if d.is_dir()])

    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ DU DATASET")
    logger.info("=" * 60)
    logger.info(f"   Images totales : {nb_images}")
    logger.info(f"   Catégories     : {nb_categories}")
    logger.info(f"   Raw            : {RAW_DIR}")
    logger.info(f"   Splits         : {SPLITS_DIR}")
    logger.info(f"   Knowledge Base : {KNOWLEDGE_BASE_DIR}")
    if IS_COLAB:
        logger.info(f"   Google Drive   : {PROJECT_ROOT}")
    logger.info("=" * 60)
    logger.info("✅ Prochaine étape : python src/preprocess.py")


if __name__ == "__main__":
    main()
