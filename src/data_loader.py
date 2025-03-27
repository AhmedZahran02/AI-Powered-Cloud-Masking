import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import albumentations as A
from src.config import *

class CloudDataset(Sequence):
    """Data loader for cloud segmentation dataset"""
    
    def __init__(self, image_paths, mask_paths, batch_size=BATCH_SIZE, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        
        # Define augmentations
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5),
        ])
        
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        images = []
        masks = []
        
        for i in batch_indices:
            # Load image (all bands)
            with rasterio.open(self.image_paths[i]) as img:
                image = img.read([1,2,3,4])  # Read all 4 bands
                image = np.moveaxis(image, 0, -1)  # Change to HWC format
            
            # Load mask
            with rasterio.open(self.mask_paths[i]) as mask:
                mask = mask.read(1)
                mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
            
            # Normalize image
            image = self.normalize_image(image)
            
            # Apply augmentations
            if self.augment:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            images.append(image)
            masks.append(mask)
        
        return np.array(images), np.array(masks)
    
    def normalize_image(self, image):
        """Normalize image to 0-1 range per band"""
        # Clip to 1-99 percentiles to remove outliers
        p1, p99 = np.percentile(image, (1, 99), axis=(0,1))
        image = np.clip(image, p1, p99)
        # Min-max normalization
        image = (image - p1) / (p99 - p1 + 1e-8)
        return image
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        np.random.shuffle(self.indices)

def prepare_datasets(data_dir=RAW_DATA_DIR):
    """Prepare train/validation datasets"""
    image_paths = sorted(list((data_dir / "images").glob("*.tiff")))
    mask_paths = sorted(list((data_dir / "masks").glob("*.tiff")))
    
    # Split into train/val
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, 
        test_size=TEST_SIZE, 
        random_state=SEED
    )
    
    return train_img, val_img, train_mask, val_mask