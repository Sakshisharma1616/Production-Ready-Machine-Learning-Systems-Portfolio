from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RANDOM_SEED = 42
# Recommender params
NO_COMPONENTS = 64
LEARNING_RATE = 0.05
EPOCHS = 20
NUM_THREADS = 4
# Recommendation settings
TOP_K = 10

