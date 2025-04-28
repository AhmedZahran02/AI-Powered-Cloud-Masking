import rasterio # type: ignore
import os
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import thread_map
from pathlib import Path
from src.utils import load_and_normalize_tiff,load_mask,generate_tiles
from src.preprocessing import calculate_spectral_indices,extract_features
from src.config import config

class Augment:
    def __init__(self):
        self.augmentations = [
            self.horizontal_flip,
            self.vertical_flip,
            self.rotate_90
        ]
    
    def __call__(self, image, mask):
        for aug in self.augmentations:
            if random.random() > 0.5:
                image, mask = aug(image, mask)
        return image, mask
    
    @staticmethod
    def horizontal_flip(image, mask):
        return np.flip(image, axis=2).copy(), np.flip(mask, axis=2).copy()
    
    @staticmethod
    def vertical_flip(image, mask):
        return np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
    
    @staticmethod
    def rotate_90(image, mask):
        return np.rot90(image, k=1, axes=(1,2)).copy(), np.rot90(mask, k=1, axes=(1,2)).copy()
    
class CloudDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels=None, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels if labels is not None else [None] * len(image_paths)
        self.transform = transform
        self.augmentor = Augment() if augment else None
        self._validate_paths()

    def _validate_paths(self):
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read().astype(np.float32)

            # Normalize bands
            for i in range(image.shape[0]):
                band = image[i]
                if band.max() > band.min():
                    image[i] = (band - band.min()) / (band.max() - band.min())
                else:
                    image[i] = 0

            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)
                mask = (mask > 0.5).astype(np.float32)

            if self.augmentor:
                image, mask = self.augmentor(image, mask)

            return (
                torch.from_numpy(image),
                torch.from_numpy(mask),
                self.labels[idx]  # can be None if labels not provided
            )

        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return torch.zeros((4, 512, 512)), torch.zeros((1, 512, 512)), None
        
def prepare_datasets(data_dir, val_split=0.1, test_split=0.1):
    subfolders = ['fully_clouded', 'cloud_free', 'partially_clouded']
    image_paths, mask_paths, labels = [], [], []

    for idx, folder in enumerate(subfolders):
        image_subdir = os.path.join(data_dir, 'data', folder)
        mask_subdir = os.path.join(data_dir, 'masks', folder)

        if not os.path.exists(image_subdir) or not os.path.exists(mask_subdir):
            raise FileNotFoundError(f"Missing {folder} data or masks")

        images = sorted([
            os.path.join(image_subdir, f)
            for f in os.listdir(image_subdir) if f.endswith('.tif')
        ])
        masks = sorted([
            os.path.join(mask_subdir, f)
            for f in os.listdir(mask_subdir) if f.endswith('.tif')
        ])

        image_paths.extend(images)
        mask_paths.extend(masks)
        labels.extend([idx] * len(images))  # label per class

    # Stratified split
    train_img, test_img, train_mask, test_mask, train_lbls, test_lbls = train_test_split(
        image_paths, mask_paths, labels, test_size=test_split, stratify=labels, random_state=42)

    train_img, val_img, train_mask, val_mask, train_lbls, val_lbls = train_test_split(
    train_img, train_mask, train_lbls,test_size=val_split / (1 - test_split),stratify=train_lbls,random_state=42
    )

    print(f"Train: {len(train_img)}, Val: {len(val_img)}, Test: {len(test_img)}")

    return (
    CloudDataset(train_img, train_mask, train_lbls, augment=True),
    CloudDataset(val_img, val_mask, val_lbls),
    CloudDataset(test_img, test_mask, test_lbls)
    )
 
class ClassicalCloudDataset:
    def __init__(self, image_pairs, mode='train', feature_cache_dir=None):
        self.image_pairs = image_pairs
        self.mode = mode
        self.feature_cache_dir = feature_cache_dir
        if feature_cache_dir:
            os.makedirs(feature_cache_dir, exist_ok=True)
        self.class_weights = None
    
    def _get_cache_path(self, img_path):
        #Get path for cached features
        if self.feature_cache_dir:
            img_name = img_path.stem
            return Path(self.feature_cache_dir) / f"{img_name}_features.npz"
        return None
    
    def _stratified_sampling(self, mask, n_samples, random_state=None):
        # Balanced sampling preserving class ratio
        if not isinstance(mask, np.ndarray):
            raise ValueError("Mask must be a numpy array")
        if n_samples <= 0:
            return np.array([], dtype=np.int32)

        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        flat_mask = mask.ravel()
        cloud_idx = np.where(flat_mask == 1)[0]
        clear_idx = np.where(flat_mask == 0)[0]

        # Calculate sample counts
        n_cloud = min(len(cloud_idx), max(int(n_samples * 0.5), 100))
        n_clear = min(len(clear_idx), max(n_samples - n_cloud, 0))

        # Adjust sample counts
        if n_clear < (n_samples - n_cloud):
            n_cloud = min(len(cloud_idx), max(n_samples - n_clear, 0))

        # Perform sampling
        sampled_cloud = np.array([], dtype=np.int32)
        sampled_clear = np.array([], dtype=np.int32)

        if n_cloud > 0 and len(cloud_idx) > 0:
            sampled_cloud = rng.choice(cloud_idx, size=n_cloud, replace=False)
            sampled_cloud = sampled_cloud.astype(np.int32)

        if n_clear > 0 and len(clear_idx) > 0:
            sampled_clear = rng.choice(clear_idx, size=n_clear, replace=False)
            sampled_clear = sampled_clear.astype(np.int32)

        # Combine results
        combined = np.concatenate([sampled_cloud, sampled_clear])
        return combined.astype(np.int32) if len(combined) > 0 else np.array([], dtype=np.int32)
    
    def process_image_pair(self, img_path, mask_path, category, random_state=None):
        # Process a single image-mask pair, with caching
        cache_path = self._get_cache_path(img_path)

        # Try to load from cache
        if cache_path and cache_path.exists() and config.PRECOMPUTE_FEATURES:
            try:
                cached = np.load(cache_path)
                features = cached['features']
                labels = cached['labels']
                return features, labels
            except Exception as e:
                print(f"Cache read error for {img_path.name}: {e}")

        # Load data
        image = load_and_normalize_tiff(img_path)
        if image is None:
            return None, None

        mask = load_mask(mask_path)
        if mask is None:
            return None, None

        # Split into tiles for memory efficiency
        img_tiles, img_coords = generate_tiles(image, config.TILE_SIZE)
        mask_tiles, _ = generate_tiles(mask, config.TILE_SIZE)

        all_features = []
        all_labels = []

        # Process each tile
        for img_tile, mask_tile in zip(img_tiles, mask_tiles):
            # Calculate indices once per tile
            indices = calculate_spectral_indices(img_tile)

            # Extract features
            features = extract_features(img_tile, indices)

            # Get labels
            labels = mask_tile.ravel()

            # Sampling for balanced classes
            if self.mode == 'train':
                max_samples = min(config.MAX_SAMPLES_PER_IMAGE // len(img_tiles), len(labels))
                if max_samples > 0:  # Only sample if we have samples left
                    idx = self._stratified_sampling(mask_tile, max_samples, random_state)
                    if len(idx) > 0:  # Check we got valid indices
                        idx = np.asarray(idx, dtype=np.int32)
                        features = features[idx]
                        labels = labels[idx]

            if len(features) > 0 and len(labels) > 0:  # Only add if we have data
                all_features.append(features)
                all_labels.append(labels)

        # Combine results from all tiles
        if all_features:
            features = np.vstack(all_features)
            labels = np.hstack(all_labels)

            # Save to cache
            if cache_path and config.PRECOMPUTE_FEATURES:
                try:
                    np.savez_compressed(cache_path, features=features, labels=labels)
                except Exception as e:
                    print(f"Cache write error for {img_path.name}: {e}")

            return features, labels
        return None, None
    
    def batch_generator(self, random_state=None):
        # Generate batches of features and labels
        
        # Process all image pairs in parallel
        all_features = []
        all_labels = []

        # Create wrapper function for thread_map
        def process_wrapper(args):
            img_path, mask_path, category = args
            return self.process_image_pair(img_path, mask_path, category, random_state=random_state)

        # Use thread_map for parallel processing with progress bar
        max_workers = min(os.cpu_count(), 16)
        results = thread_map(process_wrapper,
                            self.image_pairs,
                            max_workers=max_workers,
                            desc=f"Processing {self.mode} data")

        # Collect results and verify feature dimensions
        expected_features = None
        for features, labels in results:
            if features is not None and labels is not None:
                if expected_features is None:
                    expected_features = features.shape[1]
                elif features.shape[1] != expected_features:
                    print(f"Warning: Feature dimension mismatch {features.shape[1]} vs {expected_features}, skipping")
                    continue

                all_features.append(features)
                all_labels.append(labels)

        if not all_features:
            print(f"No valid features found in {self.mode} dataset!")
            return

        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)

        if self.mode != 'train':
            if len(y) > config.MAX_SAMPLES_PER_IMAGE * 10:
                idx = np.random.choice(len(y), config.MAX_SAMPLES_PER_IMAGE * 10, replace=False)
                X = X[idx]
                y = y[idx]
            yield X, y
            return

        # For training => shuffle and batch
        indices = np.arange(len(y))
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        rng.shuffle(indices)

        # Generate batches
        start_idx = 0
        while start_idx < len(indices):
            batch_indices = indices[start_idx:start_idx + config.BATCH_SIZE]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            yield X_batch, y_batch
            start_idx += config.BATCH_SIZE
    
    def calculate_class_weights(self):
        # Compute global class weights
        class_counts = {0: 0, 1: 0}
        
        for _, mask_path, _ in self.image_pairs:
            mask = load_mask(mask_path)
            if mask is not None:
                unique, counts = np.unique(mask, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    if cls in class_counts:
                        class_counts[cls] += cnt
        
        total = sum(class_counts.values())
        if total > 0 and class_counts[0] > 0 and class_counts[1] > 0:
            self.class_weights = {
                0: total / (2 * class_counts[0]),
                1: total / (2 * class_counts[1])
            }
        else:
            print("Warning: Unable to calculate class weights, using default")
            self.class_weights = {0: 1.0, 1: 1.0}
        
        return self.class_weights   