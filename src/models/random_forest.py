# %% [markdown]
# # Classical Model Experiments for Cloud Masking
# 
# This notebook explores classical machine learning approaches for cloud masking using 4-band (RGB + IR) satellite imagery with categorized cloud coverage.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import equalize_hist
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            confusion_matrix, roc_auc_score,
                            classification_report)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import time
import json
from pathlib import Path

# %%
# Configuration
DATA_PATH = Path("../data/raw")
PROCESSED_PATH = Path("../data/processed")
OUTPUT_PATH = Path("../outputs")
MODEL_PATH = OUTPUT_PATH / "models"
os.makedirs(MODEL_PATH, exist_ok=True)

# Cloud coverage categories
CLOUD_CATEGORIES = ['cloud_free', 'partially_clouded', 'fully_clouded']

# Model parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,          # Limited for model size
    'min_samples_split': 5,   # Increased for regularization
    'min_samples_leaf': 5,    # Increased for regularization
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
    'verbose': 1,
    'class_weight': None      # We'll handle weights manually
}

# %%
# Helper functions
def load_tiff_image(path):
    """Load a TIFF image with all bands"""
    return tiff.imread(path)

def display_image_and_mask(image, mask=None, bands=[3,2,1], figsize=(10, 10), title=None):
    """Display an image (RGB or IR) and optionally its mask"""
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    
    if len(image.shape) == 2:  # Single band
        plt.imshow(image, cmap='gray')
    else:
        # For RGB display (assuming bands are in order R,G,B,IR)
        rgb = image[:,:,bands]
        # Scale to 0-1 and enhance contrast
        rgb = equalize_hist(rgb)
        plt.imshow(rgb)
    
    if mask is not None:
        plt.imshow(mask, alpha=0.3, cmap='jet')
    plt.axis('off')
    plt.show()

def calculate_ndvi(image):
    """Calculate NDVI from 4-band image (R,G,B,IR)"""
    red = image[:, :, 0].astype('float32')
    nir = image[:, :, 3].astype('float32')
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def calculate_ndwi(image):
    """Calculate NDWI from 4-band image (R,G,B,IR)"""
    green = image[:, :, 1].astype('float32')
    nir = image[:, :, 3].astype('float32')
    ndwi = (green - nir) / (green + nir + 1e-6)
    return ndwi

def calculate_cloud_index(image):
    """Calculate a simple cloud index (CI)"""
    blue = image[:, :, 2].astype('float32')
    nir = image[:, :, 3].astype('float32')
    ci = (blue - nir) / (blue + nir + 1e-6)
    return ci

def calculate_ndsi(image):
    """Normalized Difference Snow Index (using IR as SWIR substitute)"""
    green = image[:, :, 1].astype('float32')
    nir = image[:, :, 3].astype('float32')
    return (green - nir) / (green + nir + 1e-6)

def extract_features(image):
    """Extract features from 4-band image for pixel-wise classification"""
    features = []
    
    # Original bands
    for b in range(image.shape[2]):
        features.append(image[:, :, b].ravel())
    
    # Spectral indices
    features.append(calculate_ndvi(image).ravel())
    features.append(calculate_ndwi(image).ravel())
    features.append(calculate_cloud_index(image).ravel())
    features.append(calculate_ndsi(image).ravel())
    
    # Stack all features
    return np.column_stack(features)

# %%
# Data loading with cloud coverage categories
print("Loading data with cloud coverage categories...")

def get_image_mask_pairs(base_path):
    """Get paired image and mask paths from categorized directories"""
    image_mask_pairs = []
    
    for category in CLOUD_CATEGORIES:
        image_dir = base_path / "data" / category
        mask_dir = base_path / "masks" / category
        
        if not image_dir.exists() or not mask_dir.exists():
            print(f"Warning: Missing directory for category {category}")
            continue
            
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        
        # Verify matching pairs
        assert len(image_files) == len(mask_files), f"Mismatch in {category}: {len(image_files)} images vs {len(mask_files)} masks"
        for img, msk in zip(image_files, mask_files):
            assert img == msk, f"Mismatched files in {category}: {img} vs {msk}"
            image_mask_pairs.append((image_dir / img, mask_dir / msk, category))
    
    return image_mask_pairs

# Get all pairs from processed data (already categorized)
image_mask_pairs = get_image_mask_pairs(PROCESSED_PATH)

# Display samples from each category
for category in CLOUD_CATEGORIES:
    category_pairs = [p for p in image_mask_pairs if p[2] == category]
    if category_pairs:
        img_path, mask_path, _ = category_pairs[0]
        image = load_tiff_image(img_path)
        mask = load_tiff_image(mask_path)
        display_image_and_mask(image, mask, bands=[3,2,1], 
                            title=f"Sample: {category.replace('_', ' ').title()}")

# %%
# Data preprocessing with balanced sampling
print("Preprocessing data with balanced sampling...")

def preprocess_balanced_data(image_mask_pairs, samples_per_category=100000):
    """Load and preprocess data with balanced sampling from each category"""
    X = []
    y = []
    categories = []
    
    # Group pairs by category
    category_groups = {}
    for img_path, mask_path, category in image_mask_pairs:
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append((img_path, mask_path))
    
    # Process each category in batches
    for category, pairs in category_groups.items():
        print(f"\nProcessing {len(pairs)} {category} images...")
        category_samples = 0
        
        for img_path, mask_path in tqdm(pairs):
            # Load image and mask
            image = load_tiff_image(img_path)
            mask = load_tiff_image(mask_path)
            
            # Normalize image (0-1)
            image = image.astype('float32') / np.iinfo(image.dtype).max
            
            # Extract features and labels
            features = extract_features(image)
            labels = mask.ravel()
            
            # Random subsampling if needed
            if samples_per_category:
                n_samples = min(samples_per_category - category_samples, features.shape[0])
                if n_samples <= 0:
                    continue
                    
                idx = np.random.choice(features.shape[0], n_samples, replace=False)
                features = features[idx]
                labels = labels[idx]
                category_samples += n_samples
            
            X.append(features)
            y.append(labels)
            categories.extend([category] * len(labels))
            
            if samples_per_category and category_samples >= samples_per_category:
                break
    
    # Combine all samples
    X = np.vstack(X) if len(X) > 1 else X[0]
    y = np.concatenate(y)
    categories = np.array(categories)
    
    return X, y, categories

# Balanced sampling (100k pixels per category)
X, y, categories = preprocess_balanced_data(image_mask_pairs, samples_per_category=100000)

print(f"\nTotal samples: {X.shape[0]}")
print(f"Feature dimension: {X.shape[1]}")
print("Class distribution:")
print(pd.Series(y).value_counts())
print("\nCategory distribution:")
print(pd.Series(categories).value_counts())

# Plot class distribution
plt.figure(figsize=(10,5))
plt.hist(y, bins=20)
plt.title('Pixel Class Distribution')
plt.xlabel('Class (0=No Cloud, 1=Cloud)')
plt.ylabel('Count')
plt.show()

# %%
# Train-test split preserving category distribution
print("Splitting data into train, validation, and test sets...")

# First split into train+val and test
X_train_val, X_test, y_train_val, y_test, cat_train_val, cat_test = train_test_split(
    X, y, categories, test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, stratify=y)

# Then split train+val into train and val
X_train, X_val, y_train, y_val, cat_train, cat_val = train_test_split(
    X_train_val, y_train_val, cat_train_val, 
    test_size=VAL_SIZE/(1-TEST_SIZE), 
    random_state=RANDOM_STATE, 
    stratify=y_train_val)

print(f"Train shape: {X_train.shape} (categories: {np.unique(cat_train, return_counts=True)})")
print(f"Validation shape: {X_val.shape} (categories: {np.unique(cat_val, return_counts=True)})")
print(f"Test shape: {X_test.shape} (categories: {np.unique(cat_test, return_counts=True)})")

# %%
# Model training - Random Forest with combined weights
print("Training Random Forest classifier...")

# 1. Calculate category weights (image-level balance)
category_counts = pd.Series(cat_train).value_counts()
category_weights = (1 / (category_counts + 1e-6))  # Add small epsilon
category_weights = category_weights / category_weights.sum()
category_weights = category_weights.to_dict()

# 2. Calculate class weights (pixel-level balance)
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# 3. Combine weights
sample_weights = np.ones_like(y_train, dtype='float32')
for idx, (category, label) in enumerate(zip(cat_train, y_train)):
    sample_weights[idx] = category_weights[category] * class_weight_dict[label]

print("\nWeight Statistics:")
print(f"Category weights: {category_weights}")
print(f"Class weights: {class_weight_dict}")
print(f"Sample weights - Min: {sample_weights.min():.4f}, Max: {sample_weights.max():.4f}")
print(f"Sample weights - Mean: {sample_weights.mean():.4f}, Std: {sample_weights.std():.4f}")

# Initialize and train model
rf_model = RandomForestClassifier(**RF_PARAMS)

start_time = time.time()
rf_model.fit(X_train, y_train, sample_weight=sample_weights)
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

# %%
# Enhanced model evaluation
def evaluate_model(model, X, y, categories=None, set_name="Validation"):
    """Comprehensive model evaluation with category breakdown"""
    start_time = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    inference_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred),
        'classification_report': classification_report(y, y_pred, target_names=['No Cloud', 'Cloud'])
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_proba)
    
    # Print results
    print(f"\n{set_name} Set Evaluation:")
    print(f"Inference time: {inference_time:.4f} sec")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Category-wise metrics
    if categories is not None:
        print("\nCategory-wise Performance:")
        category_results = []
        for category in np.unique(categories):
            mask = categories == category
            cat_metrics = {
                'category': category,
                'accuracy': accuracy_score(y[mask], y_pred[mask]),
                'precision': precision_score(y[mask], y_pred[mask]),
                'recall': recall_score(y[mask], y_pred[mask]),
                'f1': f1_score(y[mask], y_pred[mask]),
                'support': sum(mask)
            }
            if y_proba is not None:
                cat_metrics['roc_auc'] = roc_auc_score(y[mask], y_proba[mask])
            
            category_results.append(cat_metrics)
            print(f"{category:20s} - Accuracy: {cat_metrics['accuracy']:.4f}, F1: {cat_metrics['f1']:.4f}, Support: {cat_metrics['support']}")
    
    metrics.update({
        'inference_time': inference_time,
        'category_results': category_results if categories is not None else None
    })
    
    return metrics

# Evaluate on all sets
print("\nModel Evaluation Results:")
train_metrics = evaluate_model(rf_model, X_train, y_train, cat_train, "Train")
val_metrics = evaluate_model(rf_model, X_val, y_val, cat_val, "Validation")
test_metrics = evaluate_model(rf_model, X_test, y_test, cat_test, "Test")

# %%
# Feature importance analysis
print("Feature importances:")
feature_names = ['Red', 'Green', 'Blue', 'IR', 'NDVI', 'NDWI', 'CloudIndex', 'NDSI']
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances, 'Std': std})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'], yerr=feature_importance['Std'])
plt.title("Feature Importances")
plt.ylabel("Mean Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "feature_importances.png")
plt.show()

# %%
# Model persistence and logging
print("Saving model and logs...")

# Save model
model_filename = MODEL_PATH / "random_forest_cloud_masking.joblib"
joblib.dump(rf_model, model_filename)

# Prepare log entry
log_entry = {
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    'model': 'RandomForest',
    'training_time_sec': training_time,
    'model_size_mb': Path(model_filename).stat().st_size/(1024*1024),
    'dataset_stats': {
        'total_samples': X.shape[0],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0],
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'category_distribution': dict(zip(*np.unique(categories, return_counts=True))),
    },
    'weights': {
        'category_weights': category_weights,
        'class_weights': class_weight_dict,
        'sample_weight_stats': {
            'min': float(sample_weights.min()),
            'max': float(sample_weights.max()),
            'mean': float(sample_weights.mean()),
            'std': float(sample_weights.std())
        }
    },
    'metrics': {
        'train': {k: v for k, v in train_metrics.items() 
                 if k not in ['confusion_matrix', 'classification_report', 'category_results']},
        'validation': {k: v for k, v in val_metrics.items() 
                      if k not in ['confusion_matrix', 'classification_report', 'category_results']},
        'test': {k: v for k, v in test_metrics.items() 
                if k not in ['confusion_matrix', 'classification_report', 'category_results']}
    },
    'feature_importances': feature_importance.to_dict('records'),
    'parameters': RF_PARAMS,
    'inference_speed': {
        'pixels_per_sec': X_val.shape[0]/val_metrics['inference_time'],
        'time_per_pixel': val_metrics['inference_time']/X_val.shape[0]
    }
}

# Save logs as JSON
log_file = OUTPUT_PATH / "logs" / "model_logs.json"
os.makedirs(log_file.parent, exist_ok=True)
with open(log_file, 'a') as f:
    json.dump(log_entry, f, indent=2)
    f.write("\n")

print(f"Model saved to {model_filename}")
print(f"Logs saved to {log_file}")

# %%
# Inference function for full images
def predict_full_image(model, image_path, output_path=None):
    """Run inference on a full image and save results"""
    # Load and preprocess image
    image = load_tiff_image(image_path)
    original_shape = image.shape[:2]
    image_norm = image.astype('float32') / np.iinfo(image.dtype).max
    
    # Process in batches if large
    batch_size = 100000
    mask_flat = np.zeros(image_norm.shape[0] * image_norm.shape[1], dtype='uint8')
    
    for i in range(0, mask_flat.shape[0], batch_size):
        # Get batch of features
        batch_slice = slice(i, min(i + batch_size, mask_flat.shape[0]))
        features = extract_features(image_norm.reshape(-1, 4)[batch_slice])
        
        # Predict
        mask_flat[batch_slice] = model.predict(features)
    
    # Reshape to original image dimensions
    mask = mask_flat.reshape(original_shape)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image (false color)
    rgb = image_norm[:,:,[3,2,1]]  # IR, R, G
    rgb = equalize_hist(rgb)
    ax1.imshow(rgb)
    ax1.set_title("Original Image (False Color)")
    ax1.axis('off')
    
    # Prediction
    ax2.imshow(rgb)
    ax2.imshow(mask, alpha=0.3, cmap='jet')
    ax2.set_title("Cloud Mask Prediction")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save results
    if output_path:
        os.makedirs(output_path.parent, exist_ok=True)
        # Save mask
        tiff.imwrite(output_path.with_suffix('.tif'), mask)
        # Save visualization
        plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
        print(f"Saved results to {output_path}")
    
    plt.show()
    return mask

# Test inference on samples from each category
for category in CLOUD_CATEGORIES:
    category_pairs = [p for p in image_mask_pairs if p[2] == category]
    if category_pairs:
        img_path, _, _ = category_pairs[0]
        output_path = OUTPUT_PATH / "predictions" / f"rf_pred_{category}.tif"
        print(f"\nPredicting {category} image: {img_path.name}")
        _ = predict_full_image(rf_model, img_path, output_path)

# %%
# Create run_classical_inference.py content
inference_script_content = f'''#!/usr/bin/env python3
"""
Classical Model Inference Script
Usage: python run_classical_inference.py <input_image_path> <output_path>
"""

import sys
import tifffile as tiff
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm

def load_model():
    """Load the trained model"""
    model_path = Path("{MODEL_PATH.absolute()}") / "random_forest_cloud_masking.joblib"
    return joblib.load(model_path)

def extract_features(image):
    """Feature extraction matching training"""
    features = []
    # Original bands
    for b in range(image.shape[2]):
        features.append(image[:, :, b].ravel())
    # Spectral indices
    ndvi = (image[:,:,3] - image[:,:,0]) / (image[:,:,3] + image[:,:,0] + 1e-6)
    ndwi = (image[:,:,1] - image[:,:,3]) / (image[:,:,1] + image[:,:,3] + 1e-6)
    ci = (image[:,:,2] - image[:,:,3]) / (image[:,:,2] + image[:,:,3] + 1e-6)
    ndsi = (image[:,:,1] - image[:,:,3]) / (image[:,:,1] + image[:,:,3] + 1e-6)
    features.extend([ndvi.ravel(), ndwi.ravel(), ci.ravel(), ndsi.ravel()])
    return np.column_stack(features)

def predict_image(model, image_path, output_path):
    """Run inference on an image"""
    # Load and normalize image
    image = tiff.imread(image_path)
    image = image.astype('float32') / np.iinfo(image.dtype).max
    original_shape = image.shape[:2]
    
    # Process in batches to reduce memory usage
    batch_size = 100000
    total_pixels = original_shape[0] * original_shape[1]
    mask_flat = np.zeros(total_pixels, dtype='uint8')
    
    print(f"Processing {total_pixels:,} pixels...")
    for i in tqdm(range(0, total_pixels, batch_size)):
        batch_slice = slice(i, min(i + batch_size, total_pixels))
        batch_features = extract_features(image.reshape(-1, 4)[batch_slice])
        mask_flat[batch_slice] = model.predict(batch_features)
    
    # Save result
    tiff.imwrite(output_path, mask_flat.reshape(original_shape))
    print(f"Saved prediction to {{output_path}}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_classical_inference.py <input_image_path> <output_path>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file {{input_path}} not found")
        sys.exit(1)
    
    model = load_model()
    predict_image(model, input_path, output_path)
'''

# Save inference script
inference_script_path = Path("run_classical_inference.py")
with open(inference_script_path, 'w') as f:
    f.write(inference_script_content)

print(f"\nCreated inference script at {inference_script_path}")

# %%
# Next steps and conclusions
print("""
## Final Results and Recommendations

1. Model Performance Summary:
   - Test Accuracy: {test_accuracy:.4f}
   - Test F1 Score: {test_f1:.4f}
   - Processing Speed: ~{speed:,.0f} pixels/sec
   - Model Size: {model_size:.1f} MB

2. Key Observations:
   - Most important features: {top_features}
   - Best performance on: {best_category}
   - Most challenging: {hardest_category}

3. Recommended Next Steps:
   - Experiment with larger training sets
   - Try ensemble methods (XGBoost, LightGBM)
   - Add spatial features (texture, context windows)
   - Compare with deep learning approaches
   - Deploy using the generated inference script

4. Production Notes:
   - Memory-efficient batch processing implemented
   - Comprehensive logging available
   - Model handles both pixel-level and category-level imbalances
""".format(
    test_accuracy=test_metrics['accuracy'],
    test_f1=test_metrics['f1'],
    speed=X_val.shape[0]/val_metrics['inference_time'],
    model_size=Path(model_filename).stat().st_size/(1024*1024),
    top_features=', '.join(feature_importance['Feature'].head(3).tolist()),
    best_category=max(test_metrics['category_results'], key=lambda x: x['f1'])['category'],
    hardest_category=min(test_metrics['category_results'], key=lambda x: x['f1'])['category']
))