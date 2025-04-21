import rasterio # type: ignore
import os
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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
    