import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# Dataset parameters
IMAGE_SIZE = (512, 512)  # Target image size
BANDS = ['B2', 'B3', 'B4', 'B8']  # Sentinel-2 bands: Blue, Green, Red, NIR
N_CHANNELS = len(BANDS)
BATCH_SIZE = 16
TEST_SIZE = 0.2  # Validation split ratio
SEED = 42

# Model parameters
MODEL_NAME = "unet"  # Options: "unet", "deeplab", "random_forest"
BACKBONE = "resnet34"  # For segmentation models
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 5  # Early stopping patience
MAX_EPOCHS = 50

# Augmentation parameters
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotate": True,
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2)
}

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)