import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from src.config import *
from src.utils import rle_encode
import os

def load_model(model_path):
    """Load trained model"""
    return tf.keras.models.load_model(model_path, compile=False)

def preprocess_image(image_path):
    """Preprocess single image for inference"""
    with rasterio.open(image_path) as img:
        image = img.read([1,2,3,4])  # Read all 4 bands
        image = np.moveaxis(image, 0, -1)  # Change to HWC format
    
    # Normalize
    p1, p99 = np.percentile(image, (1, 99), axis=(0,1))
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def predict_mask(model, image):
    """Predict mask for single image"""
    pred = model.predict(image)
    pred = (pred > 0.5).astype(np.uint8)  # Threshold at 0.5
    return pred[0]  # Remove batch dimension

def run_inference(test_dir, output_csv):
    """Run inference on test set and save predictions"""
    # Load model
    model_path = MODEL_DIR / f"{MODEL_NAME}_best.h5"
    model = load_model(model_path)
    
    # Get test images
    test_images = sorted(list(test_dir.glob("*.tiff")))
    
    # Prepare DataFrame for predictions
    results = []
    
    for img_path in tqdm(test_images, desc="Processing test images"):
        # Load and preprocess image
        image = preprocess_image(img_path)
        
        # Predict mask
        pred_mask = predict_mask(model, image)
        
        # Resize to 512x512 if needed
        if pred_mask.shape[:2] != IMAGE_SIZE:
            pred_mask = tf.image.resize(pred_mask, IMAGE_SIZE, method="nearest")
            pred_mask = (pred_mask.numpy() > 0.5).astype(np.uint8)
        
        # Flatten mask and encode
        flat_mask = pred_mask.squeeze().flatten()
        rle = rle_encode(flat_mask)
        
        # Save results
        results.append({
            "id": img_path.stem,
            "segmentation": rle
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, 
                       help="Directory containing test images")
    parser.add_argument("--output_csv", type=str, default="submission.csv",
                       help="Output CSV file path")
    args = parser.parse_args()
    
    run_inference(Path(args.test_dir), args.output_csv)