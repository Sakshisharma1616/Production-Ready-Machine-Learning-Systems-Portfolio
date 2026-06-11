from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
from config import Paths, DataConfig
from utils import ensure_dir

def fetch_adult_from_openml(name: str) -> pd.DataFrame:
    """
    Fetch dataset from OpenML using sklearn.
    Returns a single DataFrame with features + target.
    """
    bunch = fetch_openml(name=name, as_frame=True)
    df = bunch.frame.copy()
    return df

def main(out_path: Path) -> None:
    paths = Paths()
    cfg = DataConfig()

    ensure_dir(paths.data_dir)

    df = fetch_adult_from_openml(cfg.openml_name)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved raw data: {out_path} | shape={df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=str(Paths().data_dir / "raw_adult.csv"),
        help="Output CSV path for raw dataset",
    )
    args = parser.parse_args()
    main(Path(args.out))
