import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

def split_dataset(data_dir, val_ratio=0.2):
    """Split dataset into training and validation sets."""
    all_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) 
                if f.endswith('.tiff')]
    
    train_files, val_files = train_test_split(all_files, test_size=val_ratio, random_state=42)
    
    # Create directories
    os.makedirs(os.path.join(data_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'masks'), exist_ok=True)
    
    # Move files to appropriate directories
    for file_list, target_dir in [(train_files, 'train'), (val_files, 'val')]:
        for file in file_list:
            img_src = os.path.join(data_dir, 'images', file)
            mask_src = os.path.join(data_dir, 'masks', file)
            
            img_dst = os.path.join(data_dir, target_dir, 'images', file)
            mask_dst = os.path.join(data_dir, target_dir, 'masks', file)
            
            os.symlink(img_src, img_dst)
            os.symlink(mask_src, mask_dst)
    
    return train_files, val_files

class DataGenerator(tf.keras.utils.Sequence):
    """Data generator for training and validation."""
    def __init__(self, data_dir, file_list, batch_size=8, dim=(512, 512), 
                 shuffle=True, augment=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.floor(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        file_list_temp = [self.file_list[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(file_list_temp)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, file_list_temp):
        """Generates data containing batch_size samples."""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 4))
        y = np.empty((self.batch_size, *self.dim, 1))
        
        # Generate data
        for i, file in enumerate(file_list_temp):
            # Load image and mask
            img_path = os.path.join(self.data_dir, 'images', file)
            mask_path = os.path.join(self.data_dir, 'masks', file)
            
            image = load_tiff_image(img_path)
            mask = load_tiff_image(mask_path)
            
            # Transpose and resize image
            image = np.transpose(image, (1, 2, 0))  # CHW to HWC
            image = cv2.resize(image, self.dim)
            
            # Resize mask
            mask = cv2.resize(mask[0], self.dim)
            mask = np.expand_dims(mask, axis=-1)
            
            # Apply augmentation if enabled
            if self.augment:
                image, mask = self._augment(image, mask)
                
            # Normalize image
            image = image / 255.0
                
            # Store sample
            X[i,] = image
            y[i,] = mask / 255.0  # Normalize mask
        
        return X, y
    
    def _augment(self, image, mask):
        """Apply augmentation to image and mask."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random rotation
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
        
        # Random brightness adjustment (only for RGB channels)
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image[:, :, :3] = np.clip(image[:, :, :3] * brightness_factor, 0, 1)
        
        return image, mask