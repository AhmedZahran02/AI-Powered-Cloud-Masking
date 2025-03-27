from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, 
    UpSampling2D, concatenate, BatchNormalization, 
    Activation, Dropout
)
from tensorflow.keras.optimizers import Adam
from src.config import *

def conv_block(input_tensor, num_filters):
    """Convolutional block with two conv layers"""
    x = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

def build_unet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS)):
    """Build UNet model"""
    inputs = Input(input_shape)
    
    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = conv_block(p4, 512)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 256)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 128)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 64)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 32)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model