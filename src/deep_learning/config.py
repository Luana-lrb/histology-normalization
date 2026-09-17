from pathlib import Path # Para trabalhar com os caminhos
import torch # Para verificar se tem GPU

# ============================================================
# Diretórios do Projeto
# ============================================================

# Pasta raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # .parent sobe uma pasta

SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"

SPLITS_DIR = SRC_DIR / "deep_learning" / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# Dataset
# ============================================================
TOP_K_NORMALIZATIONS = 3 # Selecionamos as 3 melhores para o exp 2

RAW_DATASET_DIR = DATA_DIR / "raw"

PROCESSED_DATASET_DIR = DATA_DIR / "processed"
    
NUM_CLASSES = 4

# Importante para a indexação das classes
CLASS_NAMES = [
    "healthy",
    "mild",
    "moderate",
    "severe"
]

IMAGE_SIZE = 224

PRETRAINED = True # Se vai ser pre treinado no ImageNet

# ============================================================
# Treinamento
# ============================================================

BATCH_SIZE = 32 # A rede recebe 32 imagens por vez
# Se são 320 imagens -> 10 baches = 1 época

NUM_WORKERS = 4 # Quantos processos auxiliares do DataLoader prepararão as imagens

EPOCHS = 50 # limite máximo

LEARNING_RATE = 1e-4 # controla o tamanho das atualizações dos pesos da rede

WEIGHT_DECAY = 1e-4 # evitar que os pesos cresçam excessivamente

PATIENCE = 8 # Early Stopping - se a validação não melhorar por 8 épocas

# ============================================================
# Divisão do Dataset
# ============================================================

TRAIN_SIZE = 0.70

VALIDATION_SIZE = 0.15

TEST_SIZE = 0.15

# ============================================================
# Reprodutibilidade
# ============================================================

SPLIT_SEED = 42
SEEDS = [42, 68, 93] # Para fazer média e desvio padrão depois
# Não é cross - validation! Usamos os mesmos train, validation e test

# ============================================================
# Hardware
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
