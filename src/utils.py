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

def generate_tiles(image, tile_size=128, overlap=0):
    """Split image into tiles for memory-efficient processing"""
    if len(image.shape) == 3:  # Multi-band image
        h, w = image.shape[1], image.shape[2]
        tiles = []
        coords = []
        
        for y in range(0, h - overlap, tile_size - overlap):
            if y + tile_size > h:
                y = h - tile_size
                
            for x in range(0, w - overlap, tile_size - overlap):
                if x + tile_size > w:
                    x = w - tile_size
                    
                tile = image[:, y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                coords.append((y, x))
                
        return tiles, coords
    else:  # Single band (mask)
        h, w = image.shape
        tiles = []
        coords = []
        
        for y in range(0, h - overlap, tile_size - overlap):
            if y + tile_size > h:
                y = h - tile_size
                
            for x in range(0, w - overlap, tile_size - overlap):
                if x + tile_size > w:
                    x = w - tile_size
                    
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                coords.append((y, x))
                
        return tiles, coords