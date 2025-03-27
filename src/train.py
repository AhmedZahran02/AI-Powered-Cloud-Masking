import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)
from src.data_loader import CloudDataset, prepare_datasets
from src.models.unet import build_unet
from src.models.deeplab import build_deeplabv3
from src.evaluate import dice_coef
from src.config import *
import os
import datetime

def train_model():
    # Prepare datasets
    train_img, val_img, train_mask, val_mask = prepare_datasets()
    
    # Create data generators
    train_gen = CloudDataset(train_img, train_mask, batch_size=BATCH_SIZE, augment=True)
    val_gen = CloudDataset(val_img, val_mask, batch_size=BATCH_SIZE, augment=False)
    
    # Build model
    if MODEL_NAME == "unet":
        model = build_unet()
    elif MODEL_NAME == "deeplab":
        model = build_deeplabv3()
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_DIR / f"{MODEL_NAME}_best.h5"),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_dice_coef",
            patience=PATIENCE,
            mode="max",
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_dice_coef",
            factor=0.1,
            patience=PATIENCE//2,
            verbose=1,
            mode="max",
            min_lr=1e-6
        ),
        TensorBoard(
            log_dir=str(OUTPUT_DIR / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=MAX_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(str(MODEL_DIR / f"{MODEL_NAME}_final.h5"))
    
    return history

if __name__ == "__main__":
    # Set GPU memory growth to avoid allocation errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    train_model()