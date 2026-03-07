"""
Script de téléchargement robuste avec suivi de progression.
Télécharge le dataset Fashion Product Images (small) depuis Kaggle.
"""
import os
import sys
import zipfile
import requests
from pathlib import Path

# Config
DOWNLOAD_DIR = Path(r"C:\Users\user\Desktop\smartmarket\ECommerce-IA\data\kaggle_download")
ZIP_PATH = DOWNLOAD_DIR / "fashion-product-images-small.zip"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Auth Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("[OK] Kaggle authentifié")

# Obtenir l'URL de téléchargement
print("[...] Téléchargement du dataset 'fashion-product-images-small'...")
print(f"[...] Destination: {DOWNLOAD_DIR}")

try:
    api.dataset_download_files(
        'paramaggarwal/fashion-product-images-small',
        path=str(DOWNLOAD_DIR),
        unzip=False,
        quiet=False
    )
    print("[OK] Téléchargement terminé!")
except Exception as e:
    print(f"[ERREUR] {e}")
    sys.exit(1)

# Trouver le zip
zips = list(DOWNLOAD_DIR.glob("*.zip"))
if not zips:
    print("[ERREUR] Aucun zip trouvé")
    sys.exit(1)

zip_path = zips[0]
print(f"[...] Décompression de {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.0f} MB)...")

with zipfile.ZipFile(str(zip_path), 'r') as zf:
    zf.extractall(str(DOWNLOAD_DIR))
    
print("[OK] Décompression terminée!")

# Supprimer le zip
zip_path.unlink()
print("[OK] Zip supprimé")

# Lister le contenu
for item in sorted(DOWNLOAD_DIR.iterdir()):
    if item.is_dir():
        count = sum(1 for _ in item.rglob("*") if _.is_file())
        print(f"  📁 {item.name}/ ({count} fichiers)")
    else:
        print(f"  📄 {item.name} ({item.stat().st_size / 1024:.0f} KB)")

print("\n[DONE] Dataset prêt!")
