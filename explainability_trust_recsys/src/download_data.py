import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from src.config import DATA_DIR, MOVIELENS_1M_URL

def download_and_extract(dest_dir: Path = DATA_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ml-1m.zip"
    extract_dir = dest_dir / "ml-1m"

    if extract_dir.exists() and (extract_dir / "ratings.dat").exists():
        print(f"[OK] Dataset already extracted at: {extract_dir}")
        return extract_dir

    if not zip_path.exists():
        print(f"[DL] Downloading MovieLens 1M -> {zip_path}")
        urlretrieve(MOVIELENS_1M_URL, zip_path)

    print(f"[EXTRACT] Extracting -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # The zip extracts folder named "ml-1m"
    if not extract_dir.exists():
        raise FileNotFoundError("Extraction failed: ml-1m folder not found.")

    print("[DONE] Download + extraction complete.")
    return extract_dir

if __name__ == "__main__":
    download_and_extract()
