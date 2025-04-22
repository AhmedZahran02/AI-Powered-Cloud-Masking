import os
import time
import psutil
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from src.preprocessing import extract_features, calculate_spectral_indices
from src.models.ensemble import EnsembleModel
from pathlib import Path
from src.visualization import plot_image_and_mask
from src.utils import load_and_normalize_tiff
from src.config import config
from src.utils import generate_tiles
import numpy as np
from scipy import ndimage
from src.rle_encoder_decoder import rle_encode
from skimage.transform import resize

# Ensure logs directory exists
os.makedirs("outputs/logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    filename="outputs/logs/classical_model_logs.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_classical_model_stats(model):
    logging.info("Logging Classical Model Stats...")

    if hasattr(model, 'coef_'):
        n_params = model.coef_.size
        logging.info(f"Number of Parameters: {n_params}")
    else:
        logging.info("Model has no coef_ attribute to determine parameter count.")

    process = psutil.Process()
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / (1024 ** 2)
    logging.info(f"Memory usage (RSS): {memory_mb:.2f} MB")

def predict_full_image(model, image_path):
    """Predict cloud mask for a full image with tiling"""
    # Load image
    image = load_and_normalize_tiff(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Process image in tiles to save memory
    img_tiles, coords = generate_tiles(image, config.TILE_SIZE)
    h, w = image.shape[1], image.shape[2]
    
    # Initialize full mask
    full_mask = np.zeros((h, w), dtype=np.float32)
    
    # Process each tile
    for tile, (y, x) in zip(img_tiles, coords):
        # Extract features
        indices = calculate_spectral_indices(tile)
        features = extract_features(tile, indices)
        
        # Get probabilities
        proba = model.predict_proba(features)
        
        # Reshape to tile shape
        tile_mask = proba.reshape(config.TILE_SIZE, config.TILE_SIZE)
        
        # Place in full mask
        full_mask[y:y+config.TILE_SIZE, x:x+config.TILE_SIZE] = tile_mask
    
    # Apply threshold and post-processing
    # binary_mask = (full_mask > config.PROBABILITY_THRESHOLD).astype(np.uint8)
    
    # if config.POST_PROCESS:
    #     # Remove small isolated pixels (noise)
    #     binary_mask = ndimage.binary_opening(
    #         binary_mask, 
    #         structure=np.ones((config.MORPHOLOGY_SIZE, config.MORPHOLOGY_SIZE))
    #     )
        
    #     # Fill small holes
    #     binary_mask = ndimage.binary_closing(
    #         binary_mask, 
    #         structure=np.ones((config.MORPHOLOGY_SIZE, config.MORPHOLOGY_SIZE))
    #     )
    
    if config.POST_PROCESS:
        high_threshold = 0.75
        low_threshold = 1.0  # Adjust based on your proba_map scale
        cloud_free_threshold = 0.05

        strong_clouds = full_mask > high_threshold
        weak_clouds = (full_mask > low_threshold) & (full_mask <= high_threshold)
        cloud_free = full_mask < cloud_free_threshold

        # Small dilation on strong clouds
        struct = ndimage.generate_binary_structure(2, 2)
        dilated_strong = ndimage.binary_dilation(strong_clouds, structure=struct, iterations=1)

        # Connect weak clouds near strong clouds
        final_mask = dilated_strong.copy()
        labeled_weak, num_labels = ndimage.label(weak_clouds)

        for label in range(1, num_labels + 1):
            weak_region = (labeled_weak == label)
            if np.any(dilated_strong & weak_region):
                final_mask |= weak_region

        # Fill small holes in detected cloud regions
        final_mask = ndimage.binary_fill_holes(final_mask)

        # Remove small isolated patches (e.g., < 20 pixels)
        labeled_final, num_labels = ndimage.label(final_mask)
        component_sizes = np.bincount(labeled_final.ravel())
        small_components = component_sizes < 20
        small_components[0] = False  # background stays
        final_mask[small_components[labeled_final]] = False
        
        final_mask[cloud_free] = False
        
        # Optional smoothing
        final_mask = ndimage.binary_closing(
            final_mask, structure=np.ones((config.MORPHOLOGY_SIZE, config.MORPHOLOGY_SIZE))
        )

        binary_mask = final_mask.astype(np.uint8)
        
    else:
        binary_mask = (full_mask > config.PROBABILITY_THRESHOLD).astype(np.uint8)

    
    return image,binary_mask

def predict_classical(model: EnsembleModel, test_folder, output_csv, visualize=False):
    submissions = []
    start_time = time.time()

    prediction_dir = Path("outputs/predictions")
    if visualize:
        os.makedirs(prediction_dir, exist_ok=True)

    # Load all image paths from the test folder
    test_folder_path = Path(test_folder)
    image_files = list(test_folder_path.glob("*.tif"))

    for image_file in tqdm(image_files, desc="Classical Model Prediction"):
        image_id = image_file.stem
        
        # Predict
        t0 = time.time()
        image,predicted_mask = predict_full_image(model, image_file)
        mask_resized = resize(predicted_mask, (512, 512), order=0, mode='reflect', preserve_range=True)
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
        elapsed = time.time() - t0

        logging.info(f"Inference for image {image_id} took {elapsed*1000:.2f} ms")

        rle = rle_encode(mask_resized)
        submissions.append({
                "id": image_file.stem,
                "segmentation": rle
            })

        # Visualization
        if visualize:
                vis_path = prediction_dir / f"{image_file.stem}_prediction.png"
                plot_image_and_mask(image, mask_resized, title="Prediction Visualization", save_path=vis_path)

    total_time = time.time() - start_time
    avg_time = total_time / len(image_files)

    logging.info(f"Total inference time: {total_time:.2f} seconds")
    logging.info(f"Average inference time per image: {avg_time*1000:.2f} ms")

    df = pd.DataFrame(submissions)
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")

def main(test_folder, output_csv, model_path,visualize=False):
    print("Loading trained classical model...")

    model = EnsembleModel.load(model_path)
    log_classical_model_stats(model)
    
    predict_classical(model,test_folder, output_csv,visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, required=True, help="Path to folder with .tif images")
    parser.add_argument("--model_path", type=str, default="outputs\\models\\Classical_model.joblib", help="Path to trained model")
    parser.add_argument("--output_csv", type=str, default="classical_submission.csv", help="Path to output CSV")
    parser.add_argument("--visualize", action="store_true", help="Save prediction visualizations")

    args = parser.parse_args()
    main(args.test_folder, args.output_csv, args.model_path, args.visualize)
