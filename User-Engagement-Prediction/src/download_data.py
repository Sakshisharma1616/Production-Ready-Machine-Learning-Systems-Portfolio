import zipfile
import requests
from pathlib import Path
from src.config import RAW_DIR, MOVIELENS_SMALL_URL
def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
def main() -> None:
    zip_path = RAW_DIR / "ml-latest-small.zip"
    extract_dir = RAW_DIR / "ml-latest-small"
    print(f"Downloading: {MOVIELENS_SMALL_URL}")
    download_file(MOVIELENS_SMALL_URL, zip_path)
    print(f"Unzipping to: {extract_dir}")
    unzip(zip_path, extract_dir)
    print("Done ✅")
if __name__ == "__main__":
    main()
