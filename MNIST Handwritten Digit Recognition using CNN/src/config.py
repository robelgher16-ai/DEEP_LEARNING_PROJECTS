from pathlib import Path
import torch

# ==========================
# Project Paths
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create folders automatically
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Dataset
# ==========================
IMAGE_SIZE = 28
NUM_CLASSES = 10
CHANNELS = 1

# ==========================

# ==========================
# Training
# ==========================
BATCH_SIZE = 64
VALID_SPLIT = 0.10
SEED = 42
# ==========================
# Training Hyperparameters
# ==========================
EPOCHS = 20
LEARNING_RATE = 0.001
# Learning Rate Scheduler
STEP_SIZE = 3
GAMMA = 0.1

WEIGHT_DECAY = 1e-4
# Model
DROPOUT = 0.50
# ==========================
# Early Stopping
PATIENCE = 3
# ==========================
# Device
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"