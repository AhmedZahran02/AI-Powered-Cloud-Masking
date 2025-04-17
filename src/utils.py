import rasterio # type: ignore
import numpy as np

def load_tiff_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
    return image

def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        if src.count > 1:
            print(f"Warning: Mask {mask_path} has multiple bands ({src.count}). Using only the first band.")
        mask = src.read(1)  # Single-channel mask
    return mask

def load_and_normalize_tiff(image_path):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # shape: (C, H, W)

    # Normalize each band like in your dataset loader
    for i in range(image.shape[0]):
        band = image[i]
        if band.max() > band.min():
            image[i] = (band - band.min()) / (band.max() - band.min())
        else:
            image[i] = 0

    return image