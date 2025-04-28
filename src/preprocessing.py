import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter
from scipy import ndimage
from src.config import config

def calculate_spectral_indices(image):
    bands = []
    for i in range(image.shape[0]):
        band = image[i].astype('float32')
        band = np.nan_to_num(band, nan=0.0)
        bands.append(band)
    
    if len(bands) == 4:
        blue, green, red, nir = bands
    else:
        raise ValueError(f"Expected 4 bands, got {len(bands)}")
    
    indices = {}
    eps = 1e-6 
    
    # Standard indices
    with np.errstate(divide='ignore', invalid='ignore'):
        # NDVI (Normalized Difference Vegetation Index)
        indices['ndvi'] = np.nan_to_num((nir - red) / (nir + red + eps), nan=0.0)
        
        # NDWI (Normalized Difference Water Index)
        indices['ndwi'] = np.nan_to_num((green - nir) / (green + nir + eps), nan=0.0)
        
        # Enhanced Vegetation Index (EVI)
        indices['evi'] = np.nan_to_num(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps), nan=0.0)
        
        # Visible Atmospherically Resistant Index (VARI)
        indices['vari'] = np.nan_to_num((green - red) / (green + red - blue + eps), nan=0.0)
        
        # Simple Ratio (SR)
        indices['sr'] = np.nan_to_num(nir / (red + eps), nan=0.0)
        
        # Shadow index
        indices['si'] = np.nan_to_num((1 - blue) * (1 - green) * (1 - red), nan=0.0)
        
        # Normalized Difference Built-up Index (NDBI)
        if nir is not None:
            indices['ndbi'] = np.nan_to_num((nir - green) / (nir + green + eps), nan=0.0)
    
    # Clip extreme values
    for key in indices:
        indices[key] = np.clip(indices[key], -1, 1)
    
    return indices

def calculate_spatial_features(band, window_size):
    # Mean and variance
    mean = uniform_filter(band, size=window_size)
    mean_sq = uniform_filter(band**2, size=window_size)
    variance = np.maximum(mean_sq - mean**2, 0.0)  # Ensure non-negative
    std = np.sqrt(variance)
    
    # Min, max, range
    min_val = minimum_filter(band, size=window_size)
    max_val = maximum_filter(band, size=window_size)
    range_val = max_val - min_val
    
    # Edge detection (approximate gradient magnitude)
    sobel_h = ndimage.sobel(band, axis=0)
    sobel_v = ndimage.sobel(band, axis=1)
    edge = np.sqrt(sobel_h**2 + sobel_v**2)
    
    return np.stack([mean, std, min_val, max_val, range_val, edge], axis=-1)

def calculate_texture_features(band, window_size):
    # Calculate GLCM texture features
    features = []
    
    # entropy
    entropy = uniform_filter(band * np.log(band + 1e-10), size=window_size)
    features.append(entropy)
    
    # Contrast
    mean = uniform_filter(band, size=window_size)
    variance = uniform_filter((band - mean)**2, size=window_size)
    features.append(variance)
    
    # Homogeneity
    min_val = minimum_filter(band, size=window_size)
    max_val = maximum_filter(band, size=window_size)
    range_val = max_val - min_val + 1e-6
    homogeneity = 1.0 / range_val
    features.append(homogeneity)
    
    return np.stack(features, axis=-1)

def extract_features(image, indices=None):
    if image.shape[0] != 4:
        raise ValueError(f"Expected 4 bands, got {image.shape[0]}")
    
    # Calculate spectral indices if not provided
    if indices is None and config.SPECTRAL_INDICES:
        indices = calculate_spectral_indices(image)
    
    # Determine feature counts
    n_bands = image.shape[0]
    n_spectral = len(indices) if indices else 0
    n_spatial_per_band = 6  # mean, std, min, max, range, edge
    n_texture_per_band = 3  # entropy, variance, homogeneity
    n_scales = len(config.SPATIAL_WINDOW_SIZES)
    
    # Calculate total feature count
    total_features = (
        n_bands +  # Raw bands
        n_spectral +  # Spectral indices
        n_bands * n_spatial_per_band * n_scales +  # Spatial features
        n_bands * n_texture_per_band  # Texture features
    )
    
    # Transpose to (H, W, C) for easier processing
    image = np.transpose(image, (1, 2, 0))
    h, w, c = image.shape
    n_pixels = h * w
    
    # Initialize feature array
    features = np.zeros((n_pixels, total_features), dtype='float32')
    
    # 1. Raw bands
    features[:, 0:c] = image.reshape(n_pixels, c)
    col_idx = c
    
    # 2. Spectral indices
    if indices:
        for idx_name, idx_values in indices.items():
            features[:, col_idx] = idx_values.ravel()
            col_idx += 1
    
    # 3. Multi-scale spatial features
    for band_idx in range(c):
        band = image[:, :, band_idx]
        
        for scale_idx, window_size in enumerate(config.SPATIAL_WINDOW_SIZES):
            spatial = calculate_spatial_features(band, window_size)
            start_col = col_idx + band_idx * n_spatial_per_band * n_scales + scale_idx * n_spatial_per_band
            end_col = start_col + n_spatial_per_band
            features[:, start_col:end_col] = spatial.reshape(n_pixels, n_spatial_per_band)
    
    # Update column index
    col_idx = c + n_spectral + c * n_spatial_per_band * n_scales
    
    # 4. Texture features
    for band_idx in range(c):
        band = image[:, :, band_idx]
        texture = calculate_texture_features(band, config.TEXTURE_WINDOW)
        start_col = col_idx + band_idx * n_texture_per_band
        end_col = start_col + n_texture_per_band
        features[:, start_col:end_col] = texture.reshape(n_pixels, n_texture_per_band)
    
    # handle NaNs or infs
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features