from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

PROCESSED_FULLY_CLOUDED_DATA_PATH = PROCESSED_DATA_PATH / "data" / "fully_clouded"
PROCESSED_FREE_CLOUDED_DATA_PATH = PROCESSED_DATA_PATH / "data" / "cloud_free"
PROCESSED_PARTIALLY_CLOUDED_DATA_PATH = PROCESSED_DATA_PATH / "data" / "partially_clouded"

PROCESSED_FULLY_CLOUDED_MASK_PATH = PROCESSED_DATA_PATH / "masks" / "fully_clouded"
PROCESSED_FREE_CLOUDED_MASK_PATH = PROCESSED_DATA_PATH / "masks" / "cloud_free"
PROCESSED_PARTIALLY_CLOUDED_MASK_PATH = PROCESSED_DATA_PATH / "masks" / "partially_clouded"

PROCESSED_MISCLASSIFIED_DATA_PATH = PROCESSED_DATA_PATH / "data" / "misclassified"
PROCESSED_MISCLASSIFIED_MASK_PATH = PROCESSED_DATA_PATH / "masks" / "misclassified"

REVIEW_DIR = PROCESSED_DATA_PATH / "review"

MODEL_PATH = ROOT_DIR / "outputs" / "models" / "Misclassification_model.pth"

def create_dirs():
    paths = [
        PROCESSED_DATA_PATH,
        PROCESSED_FULLY_CLOUDED_DATA_PATH,
        PROCESSED_FREE_CLOUDED_DATA_PATH,
        PROCESSED_PARTIALLY_CLOUDED_DATA_PATH,
        PROCESSED_FULLY_CLOUDED_MASK_PATH,
        PROCESSED_FREE_CLOUDED_MASK_PATH,
        PROCESSED_PARTIALLY_CLOUDED_MASK_PATH,
        PROCESSED_MISCLASSIFIED_DATA_PATH,
        PROCESSED_MISCLASSIFIED_MASK_PATH,
        REVIEW_DIR
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        
        
class Classical_Model_Config:
    # Paths
    PROCESSED_DATA = Path("../data/processed")
    OUTPUT_PATH = Path("../outputs")
    MODEL_PATH = OUTPUT_PATH / "models"
    FEATURES_PATH = OUTPUT_PATH / "features"
    
    # Create directories
    for path in [MODEL_PATH, FEATURES_PATH, OUTPUT_PATH / "logs", OUTPUT_PATH / "predictions"]:
        os.makedirs(path, exist_ok=True)
    
    # Data processing
    TILE_SIZE = 128  # Process images in smaller tiles to save memory
    MAX_SAMPLES_PER_IMAGE = 8192  # Cap samples from each image
    SAMPLES_PER_CATEGORY = 300  # Balance samples across categories
    PRECOMPUTE_FEATURES = True  # Store features on disk
    
    # Training
    BATCH_SIZE = 512  # Pixels per batch
    N_EPOCHS = 20
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    CROSS_VAL_FOLDS = 3 if os.cpu_count() >= 4 else 1  # Number of cross-validation folds
    
    # Early stopping
    EARLY_STOP_PATIENCE = 4
    EARLY_STOP_DELTA = 0.001
    
    # Feature engineering
    SPECTRAL_INDICES = True
    SPATIAL_WINDOW_SIZES = [3, 5]  # Multi-scale spatial features
    TEXTURE_WINDOW = 5
    FEATURE_SELECTION = True  # Use feature selection
    MAX_FEATURES = 30  # Max number of features to keep
    
    # Model parameters
    SGD_PARAMS = {
        'loss': 'modified_huber',
        'penalty': 'elasticnet',
        'alpha': 0.0005,
        'l1_ratio': 0.15,
        'learning_rate': 'adaptive',
        'eta0': 0.02,
        'max_iter': 1,
        'class_weight': 'balanced',
        'n_jobs': min(4, os.cpu_count())
    }
    
    RF_PARAMS = {
        'n_estimators': 30,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 0.5,
        'bootstrap': True,
        'class_weight': 'balanced',
        'n_jobs': min(4, os.cpu_count())
    }
    
    SVM_PARAMS = {
        'penalty': 'l2',
        'loss': 'squared_hinge',
        'dual': False,
        'C': 0.1,
        'class_weight': 'balanced',
        'max_iter': 500
    }

    # Post-processing
    POST_PROCESS = True
    MORPHOLOGY_SIZE = 3
    PROBABILITY_THRESHOLD = 0.4  # Lower than 0.5 to catch more clouds

config = Classical_Model_Config()
