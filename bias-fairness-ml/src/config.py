from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    models_dir: Path = root / "models"
    reports_dir: Path = root / "reports"

@dataclass(frozen=True)
class DataConfig:
    # OpenML dataset name for Adult Census style income prediction.
    # We'll fetch through sklearn's fetch_openml.
    openml_name: str = "adult"
    target_col: str = "class"  # adult dataset target typically "class"
    # Sensitive features to audit
    sensitive_cols: tuple[str, ...] = ("sex", "race")

@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    # Baseline model choice
    # We'll use LogisticRegression for interpretability + strong baseline.
    max_iter: int = 200
