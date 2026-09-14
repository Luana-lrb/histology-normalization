from pathlib import Path
import torch

# ============================================================
# Diretórios do Projeto
# ============================================================

# Pasta raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"

SPLITS_DIR = SRC_DIR / "deep_learning" / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# Dataset
# ============================================================
TOP_K_NORMALIZATIONS = 3

RAW_DATASET_DIR = DATA_DIR / "raw"

PROCESSED_DATASET_DIR = DATA_DIR / "processed"
    
# ============================================================
# Data Augmentation
# ============================================================

USE_AUGMENTATION = True

# ============================================================
# Modelo
# ============================================================

MODEL_NAME = "efficientnet_b0"

NUM_CLASSES = 4

CLASS_NAMES = [
    "healthy",
    "mild",
    "moderate",
    "severe"
]

IMAGE_SIZE = 224

PRETRAINED = True

# ============================================================
# Treinamento
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 4

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 8

# ============================================================
# Divisão do Dataset
# ============================================================

TRAIN_SIZE = 0.70

VALIDATION_SIZE = 0.15

TEST_SIZE = 0.15

# ============================================================
# Reprodutibilidade
# ============================================================

SEEDS = [42, 68, 93]

# ============================================================
# Hardware
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)