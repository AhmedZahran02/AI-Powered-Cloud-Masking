import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss (1 - dice coefficient)"""
    return 1 - dice_coef(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1):
    """Intersection over Union (IoU) metric"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate_model(model, generator):
    """Evaluate model on given generator"""
    dice_scores = []
    iou_scores = []
    
    for i in range(len(generator)):
        x, y_true = generator[i]
        y_pred = model.predict(x)
        
        # Calculate metrics for each sample in batch
        for j in range(y_true.shape[0]):
            dice = dice_coef(y_true[j], y_pred[j]).numpy()
            iou = iou_score(y_true[j], y_pred[j]).numpy()
            dice_scores.append(dice)
            iou_scores.append(iou)
    
    return {
        "dice_coef": np.mean(dice_scores),
        "iou_score": np.mean(iou_scores)
    }

class DiceCoefficient(Metric):
    """Keras metric for Dice coefficient"""
    def __init__(self, name='dice_coef', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_coef = self.add_weight(name='dc', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        dc = dice_coef(y_true, y_pred)
        self.dice_coef.assign_add(dc)
        self.count.assign_add(1.)

    def result(self):
        return self.dice_coef / self.count

    def reset_states(self):
        self.dice_coef.assign(0.)
        self.count.assign(0.)