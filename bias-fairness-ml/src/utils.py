from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def pretty_print_dict(d: Dict[str, Any]) -> None:
    print(json.dumps(d, indent=2))
